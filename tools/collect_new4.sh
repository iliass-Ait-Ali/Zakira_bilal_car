#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in gtrack_01:expert_paint_generated_track_slow wshare_01:expert_paint_waveshare_slow whouse_01:expert_cte_warehouse_slow claunch_01:expert_cte_circuit_launch_slow \
            gtrack_02:expert_paint_generated_track_mid wshare_02:expert_paint_waveshare_mid whouse_02:expert_cte_warehouse_mid claunch_02:expert_cte_circuit_launch_mid \
            gtrack_03:expert_paint_generated_track_slow wshare_03:expert_paint_waveshare_slow whouse_03:expert_cte_warehouse_slow claunch_03:expert_cte_circuit_launch_slow; do
  g=${spec%%:*}; cfg=${spec##*:}
  echo "=== $g $cfg"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/$cfg.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  s=$(ls -t data/sessions | head -1); echo "session $g $s $(wc -l < data/sessions/$s/records.jsonl) records, valid $(grep -c '"label_valid": true' data/sessions/$s/records.jsonl)"
done
echo ALL_DONE
