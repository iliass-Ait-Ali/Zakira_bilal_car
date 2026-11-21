"""Off-road check that the collision sensor misses (painted lines, open floor).
Fixed circuits: distance in metres from the car to the nearest position a teacher
drove through (all clean teacher frames of data/sessions). Off road = farther than
--margin metres. Random circuits (generated_road, generated_track: a new layout at
every launch): |cte - centre| compared with 1.5 x the teachers' 99th percentile (median over sessions).
A run is flagged OFF when it spends more than 1 s off road (the first 3 s, the spawn spot, are ignored).
Usage: python tools/offtrack_check.py results/final_X.jsonl [...] [--margin 0.3]"""
import argparse, glob, json
import numpy as np
import torch
RANDOM = {"generated_road", "generated_track"}
a = argparse.ArgumentParser(); a.add_argument("results", nargs="+"); a.add_argument("--margin", type=float, default=.3); a = a.parse_args()
ref = json.load(open("configs/train_v13_all.json"))["data"].get("cte_reference", {})
pos, cte = {}, {}
for f in glob.glob("data/sessions/*/session.json"):
    scene = json.load(open(f))["config"]["runtime"]["scene"]
    c = ref.get(scene, {}).get("center", 0.0)
    mine = []
    for r in map(json.loads, open(f.replace("session.json", "records.jsonl"))):
        t = r.get("telemetry")
        if not t or t.get("hit") not in (None, "none", "") or r.get("lane_lost") or r["speed"] < .1: continue
        pos.setdefault(scene, []).append((float(t["pos_x"]), float(t["pos_z"])))
        mine.append(abs(float(t["cte"])-c))
    if mine: cte.setdefault(scene, []).append(np.percentile(mine, 99))
pos = {s: torch.tensor(np.unique(np.round(np.asarray(v), 2), axis=0), dtype=torch.float32) for s, v in pos.items()}
# median over sessions: one session with a broken simulator path (huge cte) must not widen the limit
cte_limit = {s: 1.5*float(np.median(v)) for s, v in cte.items()}
summary = {}
for path in a.results:
    print("==", path)
    for line in open(path):
        r = json.loads(line); s = r["scene"]; c = ref.get(s, {}).get("center", 0.0)
        d = [x for x in map(json.loads, open(f"runs/final_eval/{r['run']}/decisions.jsonl")) if x.get("kind") == "decision" and x.get("telemetry")]
        d = [x for x in d if x["sim_time"]-d[0]["sim_time"] >= 3.0]  # the first 3 s are the simulator's spawn spot, not the car's driving
        if s in RANDOM:
            e = np.array([abs(float(x["telemetry"]["cte"])-c) for x in d]); lim = cte_limit[s]; unit = "cte"
        else:
            car = torch.tensor([(float(x["telemetry"]["pos_x"]), float(x["telemetry"]["pos_z"])) for x in d], dtype=torch.float32)
            e = torch.cat([torch.cdist(chunk, pos[s]).min(1).values for chunk in car.split(512)]).numpy(); lim = a.margin; unit = "m"
        off = float((e > lim).sum())/20
        note = ""
        if s in RANDOM and np.percentile(e, 95) > 20*lim:
            # The simulator's own path is broken on this layout (huge cte while the car drives
            # normally, as in teacher session road_04): the cte says nothing, look at the snapshots.
            off, note = 0.0, "  [cte du simulateur cassé : jugé sur les images]"
        if r.get("fault") or r["seconds"] < 115:  # the run stopped early (technical fault): not a valid 2-minute test
            note += f"  [essai interrompu à {r['seconds']} s : {r.get('fault')}]"
        ok = off <= 1 and r["collision"] is None and not r.get("fault") and r["seconds"] >= 115
        summary.setdefault(path, []).append(ok)
        print(f"  {s:20s} {'PASS' if ok else 'FAIL'}  off-road {off:5.1f}s  (p95 {np.percentile(e,95):.2f} max {e.max():.2f} {unit}, limit {lim:.2f})  collision={r['collision']}{note}")
    print(f"  -> {sum(summary[path])}/{len(summary[path])} circuits clean (no collision AND on the road)")
