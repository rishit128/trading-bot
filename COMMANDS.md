# Commands cheat sheet

Everything you'd actually type, in one place. The bots run on this laptop. Run from the project folder
(`/home/rishit/trading bt/trading-bot`) with `source venv/bin/activate` run first, unless a section says otherwise. Full explanations are in
`README.md` — this file is just "what do I type."

## Running the bots (on this laptop)

```
./scripts/start_bots.sh                    # start the swing bot and the intraday bot in the background (safe to re-run)
pgrep -af main.py                          # are both running? (--loop 30 --live --holdout and --intraday --live)
tail -f logs/bot.log                       # watch the swing bot live (Ctrl+C stops watching, not the bot)
tail -f logs/intraday.log                  # watch the intraday engines (ORB, gap-and-go, gap-only)
pkill -f "main.py --loop"; pkill -f "main.py --intraday"                              # stop both
pkill -f "main.py --loop"; pkill -f "main.py --intraday"; sleep 5; ./scripts/start_bots.sh   # restart (after changing .env or code)
```
Cron starts both about a minute after the laptop boots. Keep the laptop on, awake and online from 09:15 to 15:30 on trading
days: the bots only trade while it is running. Settings are in `/home/rishit/trading bt/.env` (one folder above this one).

## Telegram (no terminal needed)

```
/status              equity, cash, mode
/positions           open positions with profit/loss
/history              closed trades
/pnl                  profit/loss by day and week, both accounts
/intraday             intraday (breakout) account and open positions; /intraday_history for its closed trades
/gapgo                gap-and-go paper account and open positions; /gapgo_history for its closed trades
/gaponly              gap-only control account (gap-and-go without the breakout); /gaponly_history
/news                 news-catalyst paper account (exchange filings read by the AI); /news_history
/portfolio            read your real Integrated account (read-only); it will ask for the OTP here
/research_start       each morning (~9:00): log in once (OTP here); the broker's research calls are read every 10 min until 15:35
                      and used by the swing bot (extra candidates + a line in the AI prompt) and the breakout engine
/research_stop        end the research watch early (logs out)
/pause                stop placing new orders
/resume               allow orders again
```
Automatic messages: a weekly learning report (Mondays: outcomes labelled, AI vs the plain rule, holdout reserve),

## Checking things on this PC

```
python main.py --report          # paper account: equity, trades, win rate, fees, exits
python main.py --positions       # every open position (P&L, buy date), closed-trade history, overall status
python main.py --screen          # scan all NSE stocks, print today's top candidates (~1 min)
python main.py --check           # test keys and connectivity (OpenRouter, broker, Yahoo, Telegram)
python main.py --graph           # print the LangGraph workflow diagrams
```

## Running the bot by hand, in a terminal (what start_bots.sh does for you)

```
python main.py                    # one dry-run cycle: scan + AI + risk check, records decisions, trades nothing
python main.py --loop 30          # repeat every 30 min while NSE is open (dry run)
python main.py --loop 30 --live   # same, but trade in the built-in paper simulator
python main.py --intraday [--live]     # intraday breakout engine, its own paper account
python main.py --intraday-report       # the intraday accounts (breakout, gap-and-go, gap-only, news): positions, history, status
```
**Don't start a second copy while the background bots are running** — they'd fight over the same Telegram bot and databases.

## The real-portfolio agent, on this PC 

```
python -m src.portfolio setup              # mobile, MPIN, customer ID -> Windows Credential Manager
python -m src.portfolio check              # what is stored, masked
python -m src.portfolio forget [--profile]  # delete the stored details (and the browser profile)
python main.py --portfolio --show          # run it once without the trading bot running
```

## Tests and code checks (before trusting a change)

```
python -m pytest                 # ~855 offline tests (no network, no keys), a few minutes; add --cov for coverage
python -m ruff check .           # lint (rules and settings: pyproject.toml)
python -m ruff format --check .  # formatting; `python -m ruff format .` fixes it
python -m mypy                   # type-checks the whole application
python -m bandit -r src main.py -q            # static security checks
python -m pip_audit -r requirements.txt --no-deps   # known vulnerabilities in the pinned dependencies
```

## Docker, locally on this PC (an alternative to start_bots.sh; not used right now)

```
docker compose up -d --build     # paper trading in the built-in simulator, every 30 min while NSE is open
docker compose logs -f           # what it's doing
docker compose exec trading-bot python main.py --report
docker compose down              # stop (the database volume, and your paper history, survives)
```

## Research and analysis scripts

```
scripts/test_openrouter_api.py             # tries each configured AI model with a real call (safe)
scripts/reconcile_paper.py                 # rebuilds cash from recorded fills, flags any mismatch
scripts/paper_report.py                    # full history, both accounts: every trade, P&L by day/week/total
scripts/analyze_mistakes.py                # judges every analysed stock against what then happened
scripts/intraday_archive.py status|refresh|import-cache   # the archive of 5-minute bars (fills itself while the engine runs)
scripts/compare_gap.py                     # gap-and-go vs its gap-only control: per trade, date by date, costs, drawdown
scripts/angel_report.py [--days 5]          # Angel One vs Yahoo bars per session (timing, agreement) and the clean-session count
scripts/news_report.py [--sample 30]       # news-catalyst gate 1: do material filings move as the AI said? + labels to check by hand
scripts/intraday_data_quality.py           # the intraday engines' live data vs the final bars (verdict after 10 sessions)
scripts/backtest_intraday.py [--strategy gapgo|gaponly] [--bars-file F]   # the LIVE engine replayed over stored sessions (PAPER_INITIAL_CASH by default: Rs 25,000)
scripts/audit_intraday.py                  # does the intraday rule beat a random entry? which filters add anything?
scripts/research_intraday.py               # five pre-declared intraday ideas vs a random null; re-run as the archive grows
scripts/backtest_live_config.py [--index 0|50|100|500]   # the live swing setup on 10 years of history (0 = every NSE stock)
```
Run any of these with `python scripts/<name>.py` (some accept `--help` for their options).
