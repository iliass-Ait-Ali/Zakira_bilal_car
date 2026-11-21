#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for spec in fast_01:a fast_02:m fast_03:a fast_04:m fast_05:a; do
  g=${spec%%:*}; k=${spec##*:}
  echo "=== $g config expert_fast_$k ==="
  bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/expert_fast_$k.json --record --baseline-validated --group $g --seconds 300 | grep -v SUMMARY | cut -c1-260
done
echo ALL_DONE
