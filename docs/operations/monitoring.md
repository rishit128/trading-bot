# Monitoring and incident response

"Running" is not "working". The bots run on the owner's laptop (started by `scripts/start_bots.sh`, and by cron at boot).

| Layer | Detects | Where it reports |
| --- | --- | --- |
| **Process check** (`pgrep -af main.py`) | A bot crashed or was never started | You run it; `./scripts/start_bots.sh` starts whatever is missing |
| **Heartbeat** (`src/monitoring/heartbeat.py`, written when `HEARTBEAT_FILE` is set) | A loop that is **stuck** (nothing recorded for 75 min swing / 15 min intraday) or whose **last 3 cycles all failed** | The heartbeat file; not set by default on the laptop |
| **The bot's own alerts** (Telegram) | Risk halts and their clearing, a drawdown pause ending, `CYCLE FAILED`, `ERROR processing <stock>`, `UNIVERSE SCAN FAILED`, `LLM UNAVAILABLE` and recovery, every order outcome, every intraday entry/exit/square-off, a weekly learning report | Your Telegram chat |

There is no watchdog now: nothing messages you if the laptop sleeps, loses internet or a bot dies. Check `pgrep -af main.py`
when Telegram goes quiet during market hours. Secrets are never logged (the HTTP library's request logging is switched off
because the Telegram token is part of every URL).

## Looking at the state

```
pgrep -af main.py                       # both bots running?
tail -n 50 logs/bot.log                 # swing bot
tail -n 50 logs/intraday.log            # intraday engines
python main.py --report                 # swing account; --intraday-report for the three intraday accounts
```

Telegram: `/status`, `/positions`, `/pnl`, `/intraday`, `/gapgo`, `/gaponly`. Reports:
`scripts/paper_report.py` (every trade), `scripts/compare_gap.py` (gap-and-go against its control),
`scripts/reconcile_paper.py` (does the ledger tie out; non-zero exit on drift).

## What each alert means and what to do

| Alert | Meaning | Do |
| --- | --- | --- |
| `CYCLE FAILED (will retry in N min)` | One whole swing cycle raised; it retries by itself | Only act if it repeats |
| `LLM UNAVAILABLE` | Every AI model failed this cycle; the bot is holding, no new buys | Check `scripts/test_openrouter_api.py`; free models are rate-limited and sometimes withdrawn |
| `RISK HALT: ...` | Daily-loss or drawdown limit reached; new buys blocked | Expected protection. A drawdown halt ends by itself after `DRAWDOWN_PAUSE_DAYS`, or send `/rebase` |
| `ERROR processing <stock>` | One stock failed; the others went on | Usually that stock's data; the same stock failing for days needs a look |
| `the 09:30 step started late` | The intraday engine was not running at 09:30, so that day's entries may be missed | Check the laptop was awake; restart the bots |
| No Telegram messages at all | Bot token wrong (HTTP 401 in `logs/bot.log`), or a second process is polling the same bot (a `409` conflict) | Fix `TELEGRAM_BOT_TOKEN` in `.env` and restart, or stop the other poller |

## Not covered (be aware)

- **Laptop sleep, shutdown or lost internet** is not alarmed; those sessions are simply missed.
- **Stale market data** is handled per stock (the stock waits), not as a process-level alarm.
- **Reconciliation** is a command you run (`scripts/reconcile_paper.py`), not a scheduled alert.
- **Backups**: the `.db` files in the project folder are the only copy of the trading history; copy them somewhere safe now and then.
