#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in monaco_01:mini_monaco:slow:0 road_01:generated_road:slow:-2.2 monaco_02:mini_monaco:mid:0 road_02:generated_road:mid:-2.2 monaco_03:mini_monaco:slow:0 road_03:generated_road:slow:-2.2 monaco_04:mini_monaco:mid:0; do
  IFS=: read g scene k cc <<< "$spec"
  echo "=== $g $scene $k"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/expert_yellow_${scene}_$k.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  .venv/Scripts/python.exe -m racing.laps "$run" --cte-center $cc --cte-limit 1.75 | grep -o '"validated": [a-z]*, "reasons": [^]]*], "lap_seconds": [0-9.]*'
  s=$(ls -t data/sessions | head -1); echo "session $g $s $(wc -l < data/sessions/$s/records.jsonl) records, valid $(grep -c '"label_valid": true' data/sessions/$s/records.jsonl)"
done
echo ALL_DONE
