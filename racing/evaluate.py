"""Future offline prediction metrics and analysis of real driving logs."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from .config import dump_json, sha256, contract_hash
from .data import read_jsonl, read_prepared


def summarize(logdir, annotations=None):
    path = Path(logdir)
    run = json.loads((path/"run.json").read_text(encoding="utf-8"))
    cfg = run["config"]
    rows = list(read_jsonl(path/"decisions.jsonl"))
    decisions = [r for r in rows if r["kind"] == "decision"]
    events = [r["event"] for r in rows if r["kind"] == "event"]
    laps, crossings = [], {}
    for e in events:
        if e.get("msg_type") != "collision_with_starting_line": continue
        line, t = e["starting_line_index"], float(e["timeStamp"])
        if line in crossings and t-crossings[line] >= cfg["evaluation"]["minimum_lap_s"]:
            laps.append({"start": crossings[line], "end": t, "seconds": t-crossings[line], "line": line})
        crossings[line] = t
    speeds = [r["speed"] for r in decisions]
    latency = [r["inference_ms"] for r in decisions if r.get("inference_ms") is not None]
    elapsed = speed_integral = steering_variation = 0.0
    offtrack_events, was_out, known_cte = 0, False, 0
    limit = cfg["evaluation"]["cte_limit"]
    for prev, cur in zip(decisions, decisions[1:]):
        dt = cur["sim_time"]-prev["sim_time"]
        if 0 < dt <= cfg["data"]["max_gap_s"] and prev["sequence_id"] == cur["sequence_id"]:
            elapsed += dt
            speed_integral += dt*(prev["speed"]+cur["speed"])/2
            steering_variation += abs(cur["command"]["steering"]-prev["command"]["steering"])
    if limit is not None:
        for r in decisions:
            cte = r["telemetry"].get("cte")
            if cte is None: was_out = False; continue
            known_cte += 1
            out = abs(float(cte)) > limit
            if out and not was_out: offtrack_events += 1
            was_out = out
    report = {"mode": run["mode"], "checkpoint_sha256": run["checkpoint_sha256"], "frames": len(decisions), "speed_mean_time_weighted": speed_integral/elapsed if elapsed else None, "speed_max": max(speeds) if speeds else None, "steering_total_variation_per_s": steering_variation/elapsed if elapsed else None, "inference_ms_p50_p95": np.percentile(latency, [50,95]).tolist() if latency else None, "decision_age_ms_p95": float(np.percentile([r["decision_age_s"]*1000 for r in decisions],95)) if decisions else None, "collision_events": sum(e.get("msg_type") == "hit" for e in events), "offtrack_events_cte_proxy": offtrack_events if known_cte else None, "cte_coverage": known_cte/len(decisions) if decisions else 0, "candidate_laps": laps, "verified_attempts": None, "success_rate": None, "lap_mean_s": None, "lap_best_s": None, "official_score": None, "note": "Starting-line crossings alone do not prove complete valid laps. Use reviewed annotations; missing metrics remain null."}
    if annotations:
        ann = json.loads(Path(annotations).read_text(encoding="utf-8"))
        attempts = ann["attempts"]
        if not attempts: raise ValueError("Annotations need at least one attempted lap")
        good = []
        for row in attempts:
            if type(row["success"]) is not bool: raise ValueError("success must be a boolean")
            if row["success"]:
                value = float(row["lap_seconds"])
                if not np.isfinite(value) or value <= 0: raise ValueError("Invalid lap duration")
                good.append(value)
        report.update(verified_attempts=len(attempts), success_rate=len(good)/len(attempts), lap_mean_s=float(np.mean(good)) if good else None, lap_best_s=min(good) if good else None)
        if all("offtrack_events" in row for row in attempts):
            counts=[row["offtrack_events"] for row in attempts]
            if not all(type(v) is int and v >= 0 for v in counts): raise ValueError("Invalid annotated offtrack count")
            report["offtrack_events_reviewed"] = sum(counts)
        report["annotations_sha256"] = sha256(annotations)
    return report


def offline(checkpoint, dataset):
    import torch
    from .inference import load_checkpoint
    from .dataset import DrivingDataset
    model, cfg, _ = load_checkpoint(checkpoint)
    torch.set_num_threads(cfg["training"]["threads"])
    root = Path(dataset)
    manifest, splits = read_prepared(root, cfg)
    rows = splits["test"]
    loader = torch.utils.data.DataLoader(DrivingDataset(rows,cfg), batch_size=64)
    predictions, targets, masks = [], [], []
    with torch.inference_mode():
        for x,speed,y,mask in loader:
            predictions.append(model(x,speed).numpy()); targets.append(y.numpy()); masks.append(mask.numpy())
    pred,y,mask = map(np.concatenate,(predictions,targets,masks))
    def metric(select):
        if not select.any(): return None
        return float(np.abs(pred[select,0]-y[select,0]).mean())
    return {"checkpoint_sha256": sha256(checkpoint), "n": len(y), "mae_by_horizon": [float((np.abs(pred[:,i]-y[:,i])*mask[:,i]).sum()/mask[:,i].sum()) if mask[:,i].sum() else None for i in range(y.shape[1])], "straight_mae": metric(abs(y[:,0]) <= .05), "turn_mae": metric(abs(y[:,0]) > .2), "recovery_mae": metric(np.array([r["recovery"] for r in rows],dtype=bool)), "note": "Offline imitation errors do not measure closed-loop driving success."}


def main():
    p = argparse.ArgumentParser(description="Future evaluation; no simulator is launched")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("offline"); q.add_argument("--checkpoint", required=True); q.add_argument("--dataset", required=True)
    q = sub.add_parser("summarize"); q.add_argument("--logdir", required=True); q.add_argument("--annotations"); q.add_argument("--score-plugin", help="Official scoring module:function, only after rules are supplied")
    for q in sub.choices.values(): q.add_argument("--out", required=True)
    a = p.parse_args()
    report = offline(a.checkpoint,a.dataset) if a.command == "offline" else summarize(a.logdir,a.annotations)
    if a.command == "summarize" and a.score_plugin:
        module, fn = a.score_plugin.split(":")
        report["official_score"] = getattr(importlib.import_module(module),fn)(report)
    dump_json(a.out,report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
