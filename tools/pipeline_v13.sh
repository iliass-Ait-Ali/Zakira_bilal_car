#!/usr/bin/env bash
# v13: new waveshare data (pursuit teacher) + the network also learns the pedal (throttle/brake).
# Runs detached so it survives an editor restart. Progress: logs/pipeline_v13.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
log(){ echo "$(date +%H:%M) $*" >> logs/pipeline_v13.log; }
log START
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/gpu_v13.log; sleep 15; done ) & GPUMON=$!
if [ ! -f data/prepared/all10_v6/dataset.json ]; then
  rm -rf data/prepared/all10_v6
  $PY -m racing.data --sessions data/sessions --out data/prepared/all10_v6 --config configs/train_v13_all.json > /dev/null 2>logs/prep_v13.err; log "prep exit=$?"
fi
if [ ! -f runs/cnn_all10_v13/checkpoint_hashes.json ]; then
  rm -rf runs/cnn_all10_v13
  $PY -m racing.train --config configs/train_v13_all.json --dataset data/prepared/all10_v6 --out runs/cnn_all10_v13 --cache-dir data/cache --gpu-data --start-training > logs/train_v13.log 2>&1; log "train exit=$? $(tail -1 runs/cnn_all10_v13/metrics.csv)"
fi
kill $GPUMON
for pass in "cnn_v13n r1" "cnn_v13 r1" "cnn_v13n r2" "cnn_v13 r2"; do
  set -- $pass
  out=results/final_cnn_all10_v13$([ $1 = cnn_v13n ] && echo _net)_$2.jsonl
  out_tag=$([ $1 = cnn_v13n ] && echo net_$2 || echo $2)
  if [ -f "$out" ] && [ $(wc -l < "$out") -ge 10 ]; then continue; fi
  bash tools/final_eval.sh runs/cnn_all10_v13/best.pt $1 120 $out_tag > logs/eval_v13_$out_tag.log 2>&1; log "eval $1 $2 done"
done
log "ALL DONE"
