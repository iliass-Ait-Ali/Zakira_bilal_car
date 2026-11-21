#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in warren_01:slow warren_02:mid warren_03:slow warren_04:mid; do
  g=${spec%%:*}; k=${spec##*:}
  echo "=== $g $k"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/expert_warren_$k.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  .venv/Scripts/python.exe -m racing.laps "$run" --cte-center -2.1 --cte-limit 1.75 | grep -o '"validated": [a-z]*, "reasons": [^]]*], "lap_seconds": [0-9.]*'
  s=$(ls -t data/sessions | head -1); echo "session $s $(wc -l < data/sessions/$s/records.jsonl) records, valid $(grep -c '"label_valid": true' data/sessions/$s/records.jsonl)"
done
echo ALL_DONE
