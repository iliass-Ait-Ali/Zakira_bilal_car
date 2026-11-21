#!/usr/bin/env bash
# Usage: tools/sim_run.sh <logdir> <sources> -- <racing.drive args...>
# Launches DonkeySim, waits for port 9091, runs racing.drive, then validates laps.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$1"; SOURCES="$2"; shift 3
cd "$ROOT/DonkeySimWin/DonkeySimWin" && cmd //c start "" donkey_sim.exe
cd "$ROOT"
for i in $(seq 1 60); do
  (echo > /dev/tcp/127.0.0.1/9091) 2>/dev/null && break
  sleep 1
done
sleep 3
PY="$ROOT/.venv/Scripts/python.exe"
OUT=$("$PY" -m racing.drive --connect --logdir "$LOGDIR" "$@" 2>&1)
echo "$OUT" | tail -4
RUN=$(echo "$OUT" | grep -o 'Run saved: .*' | sed 's/Run saved: //' | tr -d '\r')
[ -z "$RUN" ] && RUN=$(echo "$OUT" | grep -o 'Protocol logs: .*' | sed 's/Protocol logs: //' | tr -d '\r')
echo "RUN=$RUN"
"$PY" -m racing.laps "$RUN" --sources "$SOURCES" --plot 2>&1 | tail -6
sleep 2
tasklist | grep -qi donkey_sim && taskkill //IM donkey_sim.exe //F >/dev/null 2>&1
exit 0
