#!/usr/bin/env bash
# Prepares and trains CNN v9 and CNN+LSTM v10 on all 10 circuits (with the noise-injection sparkfun data).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY -m racing.data --sessions data/sessions --out data/prepared/all10_v3 --config configs/train_v9_all.json > /dev/null 2>logs/prep_v9.err; echo "prep v9 exit=$?"
( $PY -m racing.train --config configs/train_v9_all.json --dataset data/prepared/all10_v3 --out runs/cnn_all10_v9 --cache-dir data/cache --gpu-data --start-training > logs/train_v9.log 2>&1; echo "train v9 exit=$?" ) &
$PY -m racing.data --sessions data/sessions --out data/prepared/all10_lstm_v2 --config configs/train_v10_lstm_all.json > /dev/null 2>logs/prep_v10.err; echo "prep v10 exit=$?"
wait
$PY -m racing.train --config configs/train_v10_lstm_all.json --dataset data/prepared/all10_lstm_v2 --out runs/cnn_lstm_all10_v10 --cache-dir data/cache --gpu-data --start-training > logs/train_v10.log 2>&1; echo "train v10 exit=$?"
