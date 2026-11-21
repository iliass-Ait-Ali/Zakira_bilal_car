#!/usr/bin/env bash
# Usage: tools/diag_run.sh <config> <seconds>  -> records every frame into data/diag (never used for training)
cd "$(dirname "$0")/.."
cfg=$1; secs=${2:-120}
out=$(bash tools/sim_run.sh runs/track_tests baseline -- --mode baseline --config $cfg --record --baseline-validated --sessions data/diag --group diag --seconds $secs)
run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r'); echo "$out" | grep -o 'ERROR.*'
.venv/Scripts/python.exe tools/expert_track_check.py "$run"
echo "SESSION=data/diag/$(ls -t data/diag | head -1)"
