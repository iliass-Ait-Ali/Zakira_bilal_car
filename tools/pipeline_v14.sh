#!/usr/bin/env bash
# v14 = v13 + "pedal DART" sessions (throttle cut/boost pushes) on the 10 circuits, so the
# network also learns to recover its speed (e.g. push harder when slow on a hill).
# Runs detached. Progress: logs/pipeline_v14.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
log(){ echo "$(date +%H:%M) $*" >> logs/pipeline_v14.log; }
log START
# 1) let the v13 comparison pass (old speed rule) finish, then stop the v13 pipeline
until grep -q "eval cnn_v13 r1 done" logs/pipeline_v13.log; do sleep 20; done
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*pipeline_v13*' -or \$_.CommandLine -like '*final_eval.sh*' -or \$_.CommandLine -like '*racing.drive*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
sleep 3; taskkill //IM donkey_sim.exe //F > /dev/null 2>&1; sleep 5
rm -f results/final_cnn_all10_v13_net_r2.jsonl
log "v13 stopped after its rule pass"
# 2) record the pedal-DART sessions
if ! grep -q "collect done" logs/pipeline_v14.log 2>/dev/null || true; then :; fi
for spec in mount_q1:mountain_track mount_q2:mountain_track spark_q1:sparkfun_avc spark_q2:sparkfun_avc gtrack_q1:generated_track claunch_q1:circuit_launch whouse_q1:warehouse wshare_q1:waveshare main_q1:roboracingleague_1 warren_q1:warren monaco_q1:mini_monaco road_q1:generated_road; do
  g=${spec%%:*}; scene=${spec##*:}
  if grep -q "\"group_id\": \"$g\"" data/sessions/*/session.json 2>/dev/null; then continue; fi
  bash tools/sim_run.sh runs/driving baseline -- --mode baseline --config configs/tracks/expert_pedal_$scene.json --record --baseline-validated --group $g --seconds 300 > logs/collect_$g.log 2>&1
  s=$(grep -l "\"group_id\": \"$g\"" data/sessions/*/session.json | head -1 | xargs dirname)
  if [ -n "$s" ] && $PY tools/check_session.py $s >> logs/pipeline_v14.log; then log "$g kept"; else mkdir -p data/excluded_sessions/failed; [ -n "$s" ] && mv $s data/excluded_sessions/failed/; log "$g EXCLUDED"; fi
done
log "collect done"
# 3) train (old caches removed first: the disk is small)
ls -t data/cache/*.npy 2>/dev/null | tail -n +2 | xargs -r rm -f
if [ ! -f data/prepared/all10_v7/dataset.json ]; then rm -rf data/prepared/all10_v7; $PY -m racing.data --sessions data/sessions --out data/prepared/all10_v7 --config configs/train_v14_all.json > /dev/null 2>logs/prep_v14.err; log "prep exit=$?"; fi
if [ ! -f runs/cnn_all10_v14/checkpoint_hashes.json ]; then rm -rf runs/cnn_all10_v14; $PY -m racing.train --config configs/train_v14_all.json --dataset data/prepared/all10_v7 --out runs/cnn_all10_v14 --cache-dir data/cache --gpu-data --start-training > logs/train_v14.log 2>&1; log "train exit=$? $(tail -1 runs/cnn_all10_v14/metrics.csv)"; fi
# 4) evaluate: network pedal twice, old rule once
for pass in "cnn_v14n net_r1" "cnn_v14n net_r2" "cnn_v14 r1"; do
  set -- $pass; out=results/final_cnn_all10_v14_$2.jsonl
  if [ -f "$out" ] && [ $(wc -l < "$out") -ge 10 ]; then continue; fi
  bash tools/final_eval.sh runs/cnn_all10_v14/best.pt $1 120 $2 > logs/eval_v14_$2.log 2>&1
  $PY tools/offtrack_check.py $out >> logs/pipeline_v14.log 2>&1; log "eval $1 $2 done"
done
log "ALL DONE"
