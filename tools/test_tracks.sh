#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for scene in generated_road warehouse sparkfun_avc generated_track waveshare mini_monaco warren circuit_launch mountain_track; do
  python - "$scene" <<'PY'
import json, sys
c = json.load(open('configs/race_cnn_v2_a.json')); c['runtime']['scene'] = sys.argv[1]
json.dump(c, open(f'configs/tracks/cnn_v2_a_{sys.argv[1]}.json', 'w'), indent=2)
PY
  echo "=== $scene"
  out=$(bash tools/sim_run.sh runs/track_tests cnn -- --mode cnn --checkpoint runs/cnn_fast_v2/best.pt --config configs/tracks/cnn_v2_a_$scene.json --seconds 60 --snapshots 4)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r')
  echo "$out" | grep -o 'ERROR.*'
  [ -n "$run" ] && .venv/Scripts/python.exe tools/summarize_track.py "$run"
done
echo ALL_DONE
