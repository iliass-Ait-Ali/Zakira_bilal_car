"""Build a reference line (x, z points, one per simulator node) for guide="path".

Usage: python tools/build_path.py <out.json> <records.jsonl or decisions.jsonl>... [--from-cte] [--smooth N] [--upsample K]
Each point is the median car position seen at that activeNode, ignoring frames with a
collision. With --from-cte the simulator path itself is used (position minus the
cross-track error to the right). Nodes never visited are filled by interpolation.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("logs", nargs="+")
p.add_argument("--from-cte", action="store_true"); p.add_argument("--min-speed", type=float, default=.2); p.add_argument("--smooth", type=int, default=2)
p.add_argument("--upsample", type=int, default=1, help="Points per node in the output (smoother aim on tracks with few nodes; then the line no longer maps to activeNode)")
a = p.parse_args()
by_node, total = defaultdict(list), None
for log in a.logs:
    for line in open(log, encoding="utf-8"):
        r = json.loads(line)
        t = r.get("telemetry")
        if not t or t.get("hit") not in ("none", None) or t.get("speed", 0) < a.min_speed: continue
        total = t["totalNodes"]
        x, z = float(t["pos_x"]), float(t["pos_z"])
        if a.from_cte:
            yaw = math.radians(float(t["yaw"])); c = float(t["cte"])
            x, z = x-c*math.cos(yaw), z+c*math.sin(yaw)
        by_node[int(t["activeNode"])].append((x, z))
known = sorted(by_node)
pts = np.full((total, 2), np.nan)
for k in known: pts[k] = np.median(np.asarray(by_node[k]), axis=0)
idx = np.arange(total)
for d in range(2):  # circular interpolation of unvisited nodes
    pts[:, d] = np.interp(idx, known, pts[known, d], period=total)
if a.smooth:
    w = 2*a.smooth+1
    pts = np.stack([np.convolve(np.r_[pts[-a.smooth:, d], pts[:, d], pts[:a.smooth, d]], np.ones(w)/w, "valid") for d in range(2)], 1)
visited = [k in by_node for k in range(total)]
if a.upsample > 1:  # periodic linear interpolation between nodes, then one more smoothing pass
    t = np.arange(total*a.upsample)/a.upsample
    pts = np.stack([np.interp(t, idx, pts[:, d], period=total) for d in range(2)], 1)
    w = a.upsample+1
    pts = np.stack([np.convolve(np.r_[pts[-w:, d], pts[:, d], pts[:w, d]], np.ones(2*w+1)/(2*w+1), "valid") for d in range(2)], 1)
    visited = None
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"points": pts.round(4).tolist(), "visited_nodes": len(known), "visited": visited, "total_nodes": total, "from_cte": a.from_cte}), encoding="utf-8")
print(f"{a.out}: {len(known)}/{total} nodes visited")
