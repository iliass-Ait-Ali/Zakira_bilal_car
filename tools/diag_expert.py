"""Show what the expert saw just before it failed, from a diagnostic recording.
Usage: python tools/diag_expert.py <session_dir> <config> <out.png> [index]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from racing.baseline import BaselinePilot, lane_rows, yellow_runs
from racing.config import load_config
s, cfg_path, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = load_config(cfg_path); p = cfg["baseline"]
recs = [json.loads(l) for l in (s/"records.jsonl").read_text().splitlines()]
def bad(r):
    t = r["telemetry"]
    return r.get("lane_lost") and not r.get("coasting") or t.get("hit") not in (None, "none", "")
idx = int(sys.argv[4]) if len(sys.argv) > 4 else next((i for i, r in enumerate(recs) if i > 40 and bad(r)), len(recs)-1)
print("failure index", idx, "t", round(recs[idx]["sim_time"]-recs[0]["sim_time"], 2), "node", recs[idx]["telemetry"]["activeNode"])
for r in recs[max(0, idx-30):idx+3:3]:
    print(round(r["sim_time"]-recs[0]["sim_time"], 2), r.get("lane_confidence"), round(r.get("lane_error", 0) or 0, 3), r.get("coasting"), round(r["speed"], 2), round(r["command"]["steering"], 2), round(r["telemetry"]["cte"], 2))
picks = [max(0, idx-k) for k in (28, 20, 14, 10, 6, 3, 1, 0)]
sheet = Image.new("RGB", (4*260, 2*260), "white")
for n, i in enumerate(picks):
    im = Image.open(s/recs[i]["image"]).convert("RGB"); a = np.asarray(im); d = ImageDraw.Draw(im)
    if p.get("guide") == "yellow":
        for y, rs in yellow_runs(a, p.get("yellow_pale", False)):
            for x in rs: d.point((x, y), fill=(0, 255, 255))
    else:
        for y, l, r in lane_rows(a, None, p.get("paint", "white"), p.get("white_min", 180), p.get("white_spread", 65), p.get("yellow_ratio", .4)):
            if l is not None: d.rectangle((l-1, y-1, l+1, y+1), outline=(0, 255, 255))
            if r is not None: d.rectangle((r-1, y-1, r+1, y+1), outline=(255, 0, 0))
    d.text((3, 3), f"i{i} e{recs[i].get('lane_error', 0) or 0:.2f} s{recs[i]['command']['steering']:.2f}", fill=(255, 0, 0))
    sheet.paste(im, ((n % 4)*260, (n//4)*260))
sheet.save(out)
