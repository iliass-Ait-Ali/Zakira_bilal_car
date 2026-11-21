#!/usr/bin/env bash
# Usage: tools/final_eval.sh <checkpoint> <config_prefix> [seconds] [tag]  -> results/final_<name>[_tag].jsonl
cd "$(dirname "$0")/.."
CK=$1; PREFIX=$2; SECS=${3:-120}; TAG=${4:+_$4}
[ -f "$CK" ] || { echo "missing checkpoint $CK"; exit 1; }
mkdir -p results; OUT=results/final_$(basename $(dirname $CK))$TAG.jsonl; : > $OUT
for scene in roboracingleague_1 warren mini_monaco generated_road generated_track waveshare warehouse circuit_launch mountain_track sparkfun_avc; do
  cfg=configs/tracks/${PREFIX}_$scene.json
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  out=$(bash tools/sim_run.sh runs/final_eval cnn -- --mode cnn --checkpoint $CK --config $cfg --seconds $SECS --snapshots 10)
  run=$(echo "$out" | grep -o 'RUN=.*' | sed 's/RUN=//' | tr -d '\r')
  .venv/Scripts/python.exe tools/final_eval.py "$run" $scene | tee -a $OUT
done
echo "written $OUT"
