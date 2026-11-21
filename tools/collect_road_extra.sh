#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in road_04:mid road_05:slow; do
  g=${spec%%:*}; k=${spec##*:}
  echo "=== $g"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/expert_yellow_generated_road_$k.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  s=$(ls -t data/sessions | head -1); echo "session $g $s $(wc -l < data/sessions/$s/records.jsonl) records, valid $(grep -c '"label_valid": true' data/sessions/$s/records.jsonl)"
done
echo ALL_DONE
