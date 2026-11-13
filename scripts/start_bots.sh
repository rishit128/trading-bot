#!/bin/bash
# Start the swing bot and the intraday engine if they are not already running. Safe to run repeatedly.
cd "$(dirname "$0")/.." || exit 1
unset OPENROUTER_API_KEY  # a stale value in the shell would override the key in .env
start() {  # $1 = log file, rest = arguments to main.py
  local log="$1"; shift
  # anchored: an unanchored pattern also matches any shell whose command line merely mentions the bot
  if pgrep -f "^\./venv/bin/python -u main.py $*\$" >/dev/null; then echo "already running: $*"; return; fi
  mkdir -p logs
  HEARTBEAT_FILE="logs/heartbeat-${log%.log}" nohup ./venv/bin/python -u main.py "$@" >> "logs/$log" 2>&1 &  # read by scripts/watchdog.py
  echo "started: $*"
}
start bot.log --loop 30 --live --holdout  # same flags as the AWS deployment had
start intraday.log --intraday --live
