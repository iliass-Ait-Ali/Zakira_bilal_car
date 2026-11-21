#!/usr/bin/env bash
cd "$(dirname "$0")/.."
CK=${1:-runs/cnn_multi4_v4/best.pt}
for spec in main:150:0 warren:150:-2.1 mini_monaco:150:0 generated_road:150:-2.2 mountain_track:90:0; do IFS=: read name secs cc <<< "$spec"; echo "=== $name"
  out=$(bash tools/sim_run.sh runs/cnn_drive cnn -- --mode cnn --checkpoint $CK --config configs/tracks/cnn_v4_$name.json --seconds $secs --snapshots 5); run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  lim=1.75; [ "$name" = main ] && lim=1.0
  .venv/Scripts/python.exe -m racing.laps "$run" --sources cnn --cte-center $cc --cte-limit $lim | grep -o '"validated": [a-z]*, "reasons": [^]]*], "lap_seconds": [0-9.]*'
done
echo ALL_DONE
