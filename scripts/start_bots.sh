#!/bin/bash
# Start the swing bot and the intraday engine if they are not already running. Safe to run repeatedly.
cd "$(dirname "$0")/.." || exit 1
unset OPENROUTER_API_KEY  # a stale value in the shell would override the key in .env
start() {  # $1 = log file, rest = arguments to main.py
  local log="$1"; shift
  if pgrep -f "venv/bin/python -u main.py $*" >/dev/null; then echo "already running: $*"; return; fi
  mkdir -p logs
  nohup ./venv/bin/python -u main.py "$@" >> "logs/$log" 2>&1 &
  echo "started: $*"
}
start bot.log --loop 30 --live
start intraday.log --intraday --live
