"""Summarize one CNN run on any track for the final table (JSON line).
Usage: python tools/final_eval.py <run_dir> <scene>"""
import json, math, sys
from pathlib import Path
d, scene = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in (d/"decisions.jsonl").read_text().splitlines()]
dec = [r for r in rows if r["kind"] == "decision"]
t0 = dec[0]["sim_time"]
hit = next((r for r in dec if r["telemetry"].get("hit") not in (None, "none", "")), None)
hit_events = [r for r in rows if r["kind"] == "event" and r["event"].get("msg_type") == "hit"]
first_hit_t = min([hit["sim_time"]-t0] if hit else [] + [], default=None)
clean = [r for r in dec if hit is None or r["sim_time"] < hit["sim_time"]]
total = int(dec[0]["telemetry"]["totalNodes"]); n = [int(r["telemetry"]["activeNode"]) for r in clean]
steps = [(b-a+total//2) % total-total//2 for a, b in zip(n, n[1:])]
stuck = sum(1 for r in clean if r["speed"] < .15)/max(len(clean), 1)
end = json.loads((d/"end.json").read_text())["fault"] if (d/"end.json").exists() else None
print(json.dumps({"scene": scene, "run": d.name, "seconds": round(dec[-1]["sim_time"]-t0, 1),
                  "seconds_without_collision": round((clean[-1]["sim_time"]-t0) if clean else 0, 1),
                  "collision": None if hit is None else str(hit["telemetry"]["hit"]),
                  "laps_progress": round(sum(steps)/total, 2), "stopped_fraction": round(stuck, 2),
                  "mean_speed": round(sum(r["speed"] for r in clean)/max(len(clean), 1), 2), "fault": end}))
