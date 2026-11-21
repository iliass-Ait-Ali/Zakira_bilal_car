"""Conservative lap evidence from repeated start-line events AND ordered nodes."""
import argparse
import json
import math
from pathlib import Path
from .config import dump_json


AUTONOMOUS_SOURCES = ("baseline", "cnn")


def validate_laps(rows, cte_limit=1.0, allowed_sources=("baseline",), cte_center=0.0):
    """allowed_sources defaults to the expert only; pass ("cnn",) to evaluate the CNN.
    Manual driving is never accepted, and a lap must come from a single source."""
    allowed_sources = tuple(allowed_sources)
    if not allowed_sources or any(s not in AUTONOMOUS_SOURCES for s in allowed_sources):
        raise ValueError(f"allowed_sources must be a non-empty subset of {AUTONOMOUS_SOURCES}")
    decisions = [r for r in rows if r.get("kind") == "decision"]
    crossings = [r["event"] for r in rows if r.get("kind") == "event" and
                 r["event"].get("msg_type") == "collision_with_starting_line"]
    anchors = [e for e in crossings if e["starting_line_index"] == 0]
    laps = []
    for first, last in zip(anchors, anchors[1:]):
        segment = [r for r in decisions if first["received_monotonic"] <= r["received_monotonic"] <= last["received_monotonic"]]
        if not segment: continue
        total = int(segment[0]["telemetry"]["totalNodes"])
        nodes = [int(r["telemetry"]["activeNode"]) for r in segment]
        steps = [(b-a+total//2) % total-total//2 for a,b in zip(nodes,nodes[1:])]
        skipped = [step for step in steps if abs(step)>1]
        duration = last["timeStamp"]-first["timeStamp"]
        hits = sorted(set(str(r["telemetry"]["hit"]) for r in segment if r["telemetry"].get("hit") not in (None,"none","")))
        hit_events = [r for r in rows if r.get("kind") == "event" and r["event"].get("msg_type") == "hit" and
                      first["received_monotonic"] <= r["event"]["received_monotonic"] <= last["received_monotonic"]]
        cte = max(abs(r["telemetry"]["cte"]-cte_center) for r in segment)
        closure = math.hypot(segment[-1]["telemetry"]["pos_x"]-segment[0]["telemetry"]["pos_x"],
                             segment[-1]["telemetry"]["pos_z"]-segment[0]["telemetry"]["pos_z"])
        reasons = []
        if len(set(nodes)) != total: reasons.append("not_all_nodes_observed")
        if any(int(r["telemetry"]["totalNodes"]) != total for r in segment): reasons.append("node_count_changed")
        if skipped or any(s<0 for s in steps): reasons.append("nonconsecutive_or_reverse_nodes")
        if not total-2 <= sum(steps) <= total+1: reasons.append("progress_not_one_circuit")
        if duration < 10: reasons.append("implausibly_short")
        if hits or hit_events: reasons.append("collision")
        if cte > cte_limit: reasons.append("cte_exceeded_conservative_limit")
        if closure > .3: reasons.append("positions_do_not_close")
        sources = sorted(set(r["source"] for r in segment))
        if any(s not in allowed_sources for s in sources): reasons.append("nonautonomous_source")
        if len(sources) > 1: reasons.append("mixed_sources")
        if min(r["speed"] for r in segment) < .15: reasons.append("vehicle_stopped")
        if max(r["decision_age_s"] for r in segment) > .15: reasons.append("stale_decision")
        laps.append({"validated": not reasons, "reasons": reasons, "lap_seconds": duration, "sources": sources,
                     "start_event": first, "finish_event": last, "observed_nodes": len(set(nodes)),
                     "total_nodes": total, "net_node_progress": sum(steps), "node_sequence": list(dict.fromkeys(nodes)),
                     "max_abs_cte": cte, "cte_limit": cte_limit, "cte_center": cte_center, "closure_distance_telemetry_units": closure,
                     "minimum_speed": min(r["speed"] for r in segment), "maximum_speed": max(r["speed"] for r in segment),
                     "maximum_decision_age_s": max(r["decision_age_s"] for r in segment), "hits": hits,
                     "decisions": len(segment), "sim_time_start": segment[0]["sim_time"], "sim_time_end": segment[-1]["sim_time"]})
    return {"validated_laps": sum(l["validated"] for l in laps), "allowed_sources": list(allowed_sources),
            "laps": laps, "crossings": crossings,
            "decision_count": len(decisions), "duration_sim_s": decisions[-1]["sim_time"]-decisions[0]["sim_time"] if decisions else 0,
            "max_abs_cte": max((abs(r["telemetry"]["cte"]) for r in decisions), default=0),
            "method": "Consecutive index-0 crossings; all nodes in forward order; closed position; no hits/stops/manual/stale decisions; conservative CTE bound. CTE is simulator units, not a calibrated track-width measurement."}


def analyze(directory, plot=False, allowed_sources=("baseline",), cte_center=0.0, cte_limit=1.0):
    directory = Path(directory)
    rows = [json.loads(s) for s in (directory/"decisions.jsonl").read_text().splitlines()]
    report = validate_laps(rows, cte_limit=cte_limit, allowed_sources=allowed_sources, cte_center=cte_center)
    end = directory/"end.json"
    report["run_fault"] = json.loads(end.read_text())["fault"] if end.exists() else "still_running"
    dump_json(directory/"lap_validation.json", report)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = [r for r in rows if r["kind"] == "decision"]
        fig, ax = plt.subplots(1,3,figsize=(15,4))
        points=ax[0].scatter([r["telemetry"]["pos_x"] for r in d], [r["telemetry"]["pos_z"] for r in d],
                            c=[r["sim_time"] for r in d],s=3,cmap="viridis")
        ax[0].set(xlabel="pos_x (télémétrie)",ylabel="pos_z (télémétrie)",title="Trajectoire mesurée",aspect="equal")
        fig.colorbar(points,ax=ax[0],label="Temps simulateur (s)")
        ax[1].plot([r["sim_time"] for r in d],[r["telemetry"]["activeNode"] for r in d]); ax[1].set(xlabel="Temps (s)",ylabel="activeNode",title="Progression sur les 147 points")
        ax[2].plot([r["sim_time"] for r in d],[r["telemetry"]["cte"] for r in d]); ax[2].axhline(1,color="red",linestyle="--"); ax[2].axhline(-1,color="red",linestyle="--")
        ax[2].set(xlabel="Temps (s)",ylabel="CTE (unités simulateur)",title="Écart au chemin de référence")
        fig.tight_layout(); fig.savefig(directory/"trajectory.png",dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ("laps","crossings")},ensure_ascii=False))
    for lap in report["laps"]: print(json.dumps({k:v for k,v in lap.items() if k not in ("node_sequence","start_event","finish_event")},ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("directory"); parser.add_argument("--plot",action="store_true")
    parser.add_argument("--sources",default="baseline",help="Comma-separated accepted lap sources: baseline, cnn (manual is never accepted)")
    parser.add_argument("--cte-center",type=float,default=0.0,help="Reference offset of the lane centre for this track (warren is about -2.1)")
    parser.add_argument("--cte-limit",type=float,default=1.0,help="Maximum |CTE - centre| for a validated lap")
    args=parser.parse_args(); analyze(args.directory,args.plot,tuple(s.strip() for s in args.sources.split(",") if s.strip()),args.cte_center,args.cte_limit)
