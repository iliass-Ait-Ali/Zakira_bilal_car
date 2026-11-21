#!/usr/bin/env bash
# Records circuit_launch with the pure-pursuit teacher (guide="path").
cd "$(dirname "$0")/.."
for spec in claunch_p1:expert_path_circuit_launch_slow claunch_p2:expert_path_circuit_launch_mid claunch_p3:expert_path_circuit_launch_slow; do
  g=${spec%%:*}; cfg=${spec##*:}
  echo "=== $g $cfg"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/$cfg.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  s=$(ls -t data/sessions | head -1); echo "session $g $s $(wc -l < data/sessions/$s/records.jsonl) records, valid $(grep -c '"label_valid": true' data/sessions/$s/records.jsonl)"
done
echo ALL_DONE
