#!/usr/bin/env bash
# Prepares and trains CNN v11 (v9 + noise-injection data on circuit_launch, warehouse, main), then evaluates it twice on the 10 circuits.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -m racing.data --sessions data/sessions --out data/prepared/all10_v4 --config configs/train_v11_all.json > /dev/null 2>logs/prep_v11.err; echo "prep v11 exit=$?"
$PY -m racing.train --config configs/train_v11_all.json --dataset data/prepared/all10_v4 --out runs/cnn_all10_v11 --cache-dir data/cache --gpu-data --start-training > logs/train_v11.log 2>&1; echo "train v11 exit=$?"
tail -2 runs/cnn_all10_v11/metrics.csv
for r in r1 r2; do bash tools/final_eval.sh runs/cnn_all10_v11/best.pt cnn_v11 120 $r > logs/eval_v11_$r.log 2>&1; echo "eval v11 $r done"; done
