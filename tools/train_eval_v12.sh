#!/usr/bin/env bash
# Prepares and trains CNN v12 (v11 + noise-injection data on generated_track), then evaluates it twice on the 10 circuits.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -m racing.data --sessions data/sessions --out data/prepared/all10_v5 --config configs/train_v12_all.json > /dev/null 2>logs/prep_v12.err; echo "prep v12 exit=$?"
$PY -m racing.train --config configs/train_v12_all.json --dataset data/prepared/all10_v5 --out runs/cnn_all10_v12 --cache-dir data/cache --gpu-data --start-training > logs/train_v12.log 2>&1; echo "train v12 exit=$?"
tail -2 runs/cnn_all10_v12/metrics.csv
for r in r1 r2; do bash tools/final_eval.sh runs/cnn_all10_v12/best.pt cnn_v12 120 $r > logs/eval_v12_$r.log 2>&1; echo "eval v12 $r done"; done
