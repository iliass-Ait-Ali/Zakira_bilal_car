"""Summarize a driving run on any track: progress along nodes before the first hit / large CTE."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in (d/"decisions.jsonl").read_text().splitlines()]
dec = [r for r in rows if r["kind"] == "decision"]
if not dec: print(json.dumps({"run": str(d), "decisions": 0})); sys.exit()
total = int(dec[0]["telemetry"].get("totalNodes", 0) or 0)
first_bad = None
for i, r in enumerate(dec):
    t = r["telemetry"]
    if t.get("hit") not in (None, "none", "") or abs(float(t.get("cte", 0))) > 3:
        first_bad = i; break
clean = (dec[:first_bad] if first_bad is not None else dec) or dec[:1]
nodes = [int(r["telemetry"].get("activeNode", 0)) for r in clean]
steps = [(b-a+total//2) % total-total//2 for a, b in zip(nodes, nodes[1:])] if total else []
progress = sum(s for s in steps if s > 0)
end = json.loads((d/"end.json").read_text())["fault"] if (d/"end.json").exists() else None
print(json.dumps({"run": str(d), "seconds_clean": round(clean[-1]["sim_time"]-dec[0]["sim_time"], 1),
  "seconds_total": round(dec[-1]["sim_time"]-dec[0]["sim_time"], 1), "first_problem": None if first_bad is None else dec[first_bad]["telemetry"].get("hit"),
  "nodes_progress_clean": progress, "total_nodes": total, "laps_equiv": round(progress/total, 2) if total else None,
  "mean_speed": round(sum(r["speed"] for r in clean)/len(clean), 2), "max_abs_cte_clean": round(max(abs(float(r["telemetry"].get("cte", 0))) for r in clean), 2), "fault": end}))
