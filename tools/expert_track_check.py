"""Quick expert run summary on any track (progress, hits, lane loss, stops)."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = [json.loads(l) for l in (d/'decisions.jsonl').read_text().splitlines()]
dec = [r for r in rows if r['kind'] == 'decision']
tot = int(dec[0]['telemetry']['totalNodes']); n = [int(r['telemetry']['activeNode']) for r in dec]
st = [(b-a+tot//2) % tot-tot//2 for a, b in zip(n, n[1:])]
hits = sorted({str(r['telemetry'].get('hit')) for r in dec if r['telemetry'].get('hit') not in (None, 'none', '')})
lost = sum(1 for r in dec if r.get('lane_lost'))/len(dec)
sp = [r['speed'] for r in dec]; cte = sorted(abs(r['telemetry']['cte']) for r in dec)
cross = sum(1 for r in rows if r['kind'] == 'event' and r['event'].get('msg_type') == 'collision_with_starting_line')
print(f"prog={sum(st)/tot:.2f} laps hits={hits} lane_lost={lost:.0%} speed={sum(sp)/len(sp):.2f} stopped={sum(s<.2 for s in sp)/len(sp):.0%} "
      f"cte_med={cte[len(cte)//2]:.2f} cte_p95={cte[int(len(cte)*.95)]:.2f} cte_max={cte[-1]:.2f} crossings={cross} secs={dec[-1]['sim_time']-dec[0]['sim_time']:.0f} run={d.name}")
