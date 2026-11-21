"""Exit 0 if a recorded teacher session is clean (no collision, not stuck), else 1.
Usage: python tools/check_session.py data/sessions/<id>"""
import json, sys
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/records.jsonl")]
t0 = rows[0]["sim_time"]
hits = sum(1 for r in rows if (r.get("telemetry") or {}).get("hit") not in (None, "none", ""))
late = [r for r in rows if r["sim_time"]-t0 > 3]
stuck = sum(1 for r in late if r["speed"] < .1)/max(len(late), 1)
ok = len(rows) > 4000 and hits == 0 and stuck < .05
print(f"{sys.argv[1]} records={len(rows)} hits={hits} stuck={stuck:.0%} -> {'keep' if ok else 'EXCLUDE'}")
sys.exit(0 if ok else 1)
