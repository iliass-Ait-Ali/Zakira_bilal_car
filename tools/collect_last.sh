#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in gtrack_c1:expert_cte_generated_track_slow mount_01:expert_cte_mountain_track_slow spark_01:expert_cte_sparkfun_avc_slow whouse_04:expert_cte_warehouse_slow \
            gtrack_c2:expert_cte_generated_track_mid mount_02:expert_cte_mountain_track_mid spark_02:expert_cte_sparkfun_avc_slow \
            gtrack_c3:expert_cte_generated_track_slow mount_03:expert_cte_mountain_track_slow spark_03:expert_cte_sparkfun_avc_slow; do
  g=${spec%%:*}; cfg=${spec##*:}
  echo "=== $g $cfg"
  out=$(bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/$cfg.json --record --baseline-validated --group $g --seconds 300)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
  .venv/Scripts/python.exe tools/expert_track_check.py "$run"
  s=$(ls -t data/sessions | head -1); echo "session $g $s $(wc -l < data/sessions/$s/records.jsonl) records"
done
echo ALL_DONE
