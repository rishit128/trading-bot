# Monitoring and incident response

"Running" is not "working". Four independent layers watch the bots, from the process outward.

| Layer | Detects | Where it reports |
| --- | --- | --- |
| **Docker restart policy** (`unless-stopped`) | The process crashed | Restarts it; `docker inspect` shows the restart count |
| **Heartbeat health check** (`src/monitoring/heartbeat.py`) | A loop that is **stuck** (nothing recorded for 75 min swing / 15 min intraday) or whose **last 3 cycles all failed** (Yahoo down, a bug, a broker error) | `docker compose ps` shows `(unhealthy)`; the watchdog alerts |
| **Watchdog** (`deploy/watchdog.sh`, cron every 15 min on the server) | A container not running, a container unhealthy, or the swing bot silent for 75 min | One Telegram message when it goes wrong and one when it is back |
| **The bot's own alerts** (Telegram) | Risk halts and their clearing, a drawdown pause ending, `CYCLE FAILED`, `ERROR processing <stock>`, `UNIVERSE SCAN FAILED`, `LLM UNAVAILABLE` and recovery, every order outcome, every intraday entry/exit/square-off, a weekly learning report | Your Telegram chat |

Logs are JSON in the containers (`LOG_FORMAT=json`): `sudo docker logs --since 1h <container>`. Secrets are never logged (the HTTP
library's request logging is switched off because the Telegram token is part of every URL).

## Looking at the state

```
sudo docker compose -f docker-compose.aws.yml ps                       # Up / healthy / unhealthy
sudo docker exec ai-trading-bot-trading-bot-1 cat /data/heartbeat-swing     # last cycle, ok or not, failures in a row
sudo docker exec ai-trading-bot-intraday-bot-1 cat /data/heartbeat-intraday
cat ~/ai-trading-bot/.deployed-commit                                  # what is running (after the first deploy with the new script)
```

Telegram: `/status`, `/positions`, `/pnl`, `/intraday`, `/gapgo`, `/gaponly`. Reports on the server:
`scripts/paper_report.py` (every trade), `scripts/compare_gap.py` (gap-and-go against its control),
`scripts/reconcile_paper.py` (does the ledger tie out; non-zero exit on drift).

## What each alert means and what to do

| Alert | Meaning | Do |
| --- | --- | --- |
| `WATCHDOG: ... is unhealthy` | The loop stopped reporting, or 3+ cycles failed in a row | Read the last cycle's detail in the heartbeat file and the logs; Yahoo or OpenRouter outages clear by themselves; a code error needs a fix or a rollback ([deployment](deployment.md)) |
| `WATCHDOG: ... is not running` | The container exited and Docker is not restarting it | `docker compose up -d`; look at `docker logs` for the crash |
| `CYCLE FAILED (will retry in N min)` | One whole swing cycle raised; it retries by itself | Only act if it repeats (the heartbeat turns unhealthy at 3) |
| `LLM UNAVAILABLE` | Every AI model failed this cycle; the bot is holding, no new buys | Check `scripts/test_openrouter_api.py`; free models are rate-limited and sometimes withdrawn |
| `RISK HALT: ...` | Daily-loss or drawdown limit reached; new buys blocked | Expected protection. A drawdown halt ends by itself after `DRAWDOWN_PAUSE_DAYS`, or send `/rebase` |
| `ERROR processing <stock>` | One stock failed; the others went on | Usually that stock's data; the same stock failing for days needs a look |
| No Telegram messages at all | Bot token wrong, or a second process is polling the same bot (a `409` conflict) | The bot says so once in chat; stop the other poller (an old deploy on another computer) |

## Not covered (be aware)

- **Stale market data** is handled per stock (the stock waits), not as a process-level alarm: a data source that is down for
  hours shows up as repeated `WAIT`s and failed cycles, which the heartbeat turns into `unhealthy`.
- **Reconciliation** is a command you run (`scripts/reconcile_paper.py`), not a scheduled alert.
- **Disk space and memory** are not alarmed. At the last check the server had 66% of a 20 GB disk used and the two containers
  used about 200 MB each of 1.9 GB. The image is now built once, but the two older copies stay on disk until they are removed
  (`sudo docker image prune -a` once the new containers are healthy: it frees about 3 GB).
