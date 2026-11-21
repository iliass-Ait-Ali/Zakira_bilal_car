#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while tasklist | grep -qi donkey_sim; do sleep 5; done   # wait for the collection batch
for scene in generated_track mountain_track sparkfun_avc; do echo "=== $scene"
  out=$(bash tools/sim_run.sh runs/track_tests baseline -- --mode baseline --config configs/tracks/expert_cte_${scene}_slow.json --record --baseline-validated --sessions data/diag --group diag --seconds 120)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"; echo "SESSION=data/diag/$(ls -t data/diag | head -1)"
done
echo ALL_DONE
