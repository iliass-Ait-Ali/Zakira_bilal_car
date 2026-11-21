"""Read-only audit of real baseline sessions, images and matching driving logs."""
import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import numpy as np
from .config import load_config, dump_json
from .data import load_sessions, read_jsonl, split_groups
from .laps import validate_laps


def stats(values):
    return {k:float(v) for k,v in zip(("min","p05","median","p95","max"), np.percentile(values,[0,5,50,95,100]))} if values else None


def audit(root, cfg, out, logs="runs", visual=False):
    root, out = Path(root), Path(out)
    samples, rejected, summaries = load_sessions(root, cfg)
    by_session = {s["session"]:s for s in summaries}
    candidates = defaultdict(list)
    for path in Path(logs).glob("**/run.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("group_id"): candidates[meta["group_id"]].append(path.parent)
    examples, intervals, all_steering, all_speed = [], [], [], []
    for path in sorted(root.glob("*/records.jsonl")):
        meta = json.loads(path.with_name("session.json").read_text(encoding="utf-8"))
        rows = list(read_jsonl(path)); summary = by_session[meta["session_id"]]
        if not rows: continue
        sequence_ids = {r["sequence_id"] for r in rows}
        hashes = {r["image"] for r in rows}
        actual_images = {str(p.relative_to(path.parent)).replace("\\","/") for p in (path.parent/"images").glob("*") if p.is_file()}
        tel = [r["telemetry"] for r in rows]
        steering = [r["command"]["steering"] for r in rows]
        speed = [r["speed"] for r in rows]
        times = [r["sim_time"] for r in rows]
        gaps = np.diff(times)
        all_steering.extend(steering); all_speed.extend(speed)
        summary.update(images_on_disk=len(actual_images), orphan_images=sorted(actual_images-hashes),
                       missing_images=sorted(hashes-actual_images), source_counts=dict(Counter(r["source"] for r in rows)),
                       sim_duration_s=times[-1]-times[0], first_sim_time=times[0],
                       wall_interval=[rows[0]["received_monotonic"],rows[-1]["sent_monotonic"]],
                       gap_seconds=stats(gaps.tolist()), gaps_above_limit=int(sum(gaps>cfg["data"]["max_gap_s"])),
                       steering=stats(steering), speed=stats(speed), throttle=stats([r["command"]["throttle"] for r in rows]),
                       brake=stats([r["command"]["brake"] for r in rows]),
                       braking_frames=sum(r["command"]["brake"]>0 for r in rows),
                       decision_age_s=stats([r["decision_age_s"] for r in rows]),
                       collision_frames=sum(t.get("hit") not in ("none","",None) for t in tel),
                       max_abs_cte=max(abs(t["cte"]) for t in tel), nodes_observed=len({t["activeNode"] for t in tel}),
                       total_nodes=sorted({t["totalNodes"] for t in tel}), recoveries=sum(r["recovery"] for r in rows),
                       baseline_config=meta["config"].get("baseline"), matched_runs=[])
        intervals.append((meta["group_id"],*summary["wall_interval"]))
        for directory in candidates[meta["group_id"]]:
            run_rows = list(read_jsonl(directory/"decisions.jsonl"))
            decisions = [r for r in run_rows if r.get("kind")=="decision" and r["sequence_id"] in sequence_ids]
            if not decisions: continue
            index = {(r["sequence_id"],r["frame_id"]):r for r in decisions}
            mismatches = []
            for line,r in enumerate(rows,1):
                reference = index.get((r["sequence_id"],r["frame_id"]))
                if reference is None or any(r[k]!=reference[k] for k in ("command","speed","source","sim_time","received_monotonic","sent_monotonic","telemetry")):
                    mismatches.append(line)
            end = directory/"end.json"
            validation = validate_laps(run_rows, cfg["data"]["max_abs_cte"])
            summary["matched_runs"].append({"path":str(directory), "matching_decisions":len(decisions),
                    "record_mismatch_lines":mismatches, "fault":json.loads(end.read_text())["fault"] if end.exists() else "unknown",
                    "validated_laps":validation["validated_laps"], "lap_seconds":[l["lap_seconds"] for l in validation["laps"] if l["validated"]]})
        selections=[min(rows,key=lambda r:r["command"]["steering"]),max(rows,key=lambda r:r["command"]["steering"]),
                    max(rows,key=lambda r:abs(r["telemetry"]["cte"])),min(rows,key=lambda r:abs(r["command"]["steering"]))]
        examples.extend((meta["group_id"],path.parent/r["image"],r) for r in selections)
        logging.info("%s: %s accepted, %s matched driving logs",meta["group_id"],summary["accepted"],len(summary["matched_runs"]))
    pixels = defaultdict(list)
    for s in samples: pixels[s["pixel_sha256"][-1]].append((s["group_id"],s["session_id"],s["images"][-1]))
    duplicated = {key:values for key,values in pixels.items() if len(values)>1}
    cross_group = {key:values for key,values in duplicated.items() if len({v[0] for v in values})>1}
    try:
        splits,mapping = split_groups(samples,cfg)
        split_counts={name:len(rows) for name,rows in splits.items()}
        split_error=None
    except ValueError as error: mapping,split_counts,split_error={},{},str(error)
    overlap=[]
    for i,(group,a,b) in enumerate(intervals):
        for other,c,d in intervals[i+1:]:
            if group!=other and max(a,c)<=min(b,d): overlap.append([group,other])
    report={"sessions":summaries, "session_count":len(summaries), "groups":sorted({s["group"] for s in summaries}),
            "records":sum(s["records"] for s in summaries), "accepted":sum(s["accepted"] for s in summaries),
            "prepared_samples_before_split":len(samples), "rejected":len(rejected), "rejection_reasons":dict(Counter(r["reason"] for r in rejected)),
            "source_policy":["baseline"], "split_counts":split_counts, "split_groups":mapping, "split_error":split_error,
            "overlapping_group_wall_intervals":overlap, "unique_pixel_images":len(pixels),
            "duplicate_image_excess":sum(len(v)-1 for v in duplicated.values()),
            "cross_group_exact_duplicate_hashes":len(cross_group),
            "cross_group_duplicate_examples":list(cross_group.items())[:10],
            "steering_histogram":{"edges":cfg["data"]["steering_bins"],"counts":np.histogram(all_steering,bins=cfg["data"]["steering_bins"])[0].tolist()},
            "speed":stats(all_speed), "horizon_supervision":np.array([r["mask"] for r in samples]).sum(axis=0).astype(int).tolist() if samples else [],
            "limits":"Independent acquisition sessions are not independent driving conditions. Matching timestamps refer to this machine's monotonic clock. CTE is a conservative quality proxy, not a calibrated track-boundary measurement."}
    dump_json(out/"audit.json",report); dump_json(out/"rejected.json",rejected)
    if visual:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        fig,axes=plt.subplots(len(summaries),4,figsize=(12,3.2*len(summaries)),squeeze=False)
        for ax,(group,path,row) in zip(axes.flat,examples):
            with Image.open(path) as image: ax.imshow(image)
            ax.axis("off"); ax.set_title(f"{group}  t={row['sim_time']:.1f}s\ns={row['command']['steering']:.2f} CTE={row['telemetry']['cte']:.2f}",fontsize=9)
        fig.tight_layout(); fig.savefig(out/"sample_review.png",dpi=110); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ("sessions","cross_group_duplicate_examples")},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions",default="data/sessions"); parser.add_argument("--config",default="configs/race.json")
    parser.add_argument("--out",default="data/audit/baseline_v1"); parser.add_argument("--logs",default="runs"); parser.add_argument("--visual",action="store_true")
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format="%(message)s")
    audit(args.sessions,load_config(args.config),args.out,args.logs,args.visual)
