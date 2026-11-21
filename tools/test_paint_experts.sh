#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for scene in ${@:-generated_track waveshare mountain_track circuit_launch warehouse sparkfun_avc}; do echo "=== $scene"
  out=$(bash tools/sim_run.sh runs/track_tests baseline -- --mode baseline --config configs/tracks/expert_paint_${scene}_slow.json --seconds 120 --snapshots 6); run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
done
echo ALL_DONE
