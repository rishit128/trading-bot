# Commands cheat sheet

Everything you'd actually type, in one place. Run from the project folder (`C:\Users\RISHIT\Downloads\AI trading bot`)
in a terminal with `venv\Scripts\activate` run first, unless a section says otherwise. Full explanations are in
`README.md` — this file is just "what do I type."

## The AWS deployment (the one that's actually running, 24/7)

```
./deploy/deploy.sh                          # deploy your current local code changes to AWS (safe to re-run anytime)
ssh -i deploy/ai-trading-bot-key.pem ubuntu@35.154.232.64                          # log in to the server
ssh -i deploy/ai-trading-bot-key.pem ubuntu@35.154.232.64 "cd ai-trading-bot && sudo docker compose -f docker-compose.aws.yml logs -f"      # watch it live
```
Once logged in via `ssh` above, run these **on the server**. Two containers run: `trading-bot` (the swing bot, every 30
min, plus the real-portfolio agent) and `intraday-bot` (the opening-range breakout engine, its own paper account).
```
cd ai-trading-bot
sudo docker compose -f docker-compose.aws.yml ps                              # is it running? (both containers)
sudo docker compose -f docker-compose.aws.yml logs -f                         # watch both live (Ctrl+C to stop watching)
sudo docker compose -f docker-compose.aws.yml logs -f intraday-bot            # watch just the intraday engine
sudo docker compose -f docker-compose.aws.yml exec trading-bot python main.py --report        # paper account summary
sudo docker compose -f docker-compose.aws.yml exec trading-bot python main.py --positions     # open positions & history
sudo docker compose -f docker-compose.aws.yml exec trading-bot python main.py --check         # test all connections
sudo docker compose -f docker-compose.aws.yml exec intraday-bot python main.py --intraday-report  # intraday account status
sudo docker compose -f docker-compose.aws.yml exec trading-bot python scripts/paper_report.py       # full P&L history, both accounts
sudo docker compose -f docker-compose.aws.yml exec trading-bot python -m src.portfolio setup  # one-time: save your Integrated login
sudo docker compose -f docker-compose.aws.yml exec trading-bot python -m src.portfolio check  # what's saved (masked)
sudo docker compose -f docker-compose.aws.yml restart                         # restart both
sudo docker compose -f docker-compose.aws.yml down                            # stop both (history is kept, safe)
```

## Telegram (no terminal needed)

```
/status              equity, cash, mode
/positions           open positions with profit/loss
/history              closed trades
/pnl                  profit/loss by day and week, both accounts
/portfolio            read your real Integrated account (read-only); it will ask for the OTP here
/pause                stop placing new orders
/resume               allow orders again
```
Automatic messages: a weekly learning report (Mondays: outcomes labelled, AI vs the plain rule, holdout reserve),
and WATCHDOG alerts from the server itself if a bot container is down or silent for 75 minutes.

## Checking things on this PC

```
python main.py --report          # paper account: equity, trades, win rate, fees, exits
python main.py --positions       # every open position (P&L, buy date), closed-trade history, overall status
python main.py --screen          # scan all NSE stocks, print today's top candidates (~1 min)
python main.py --check           # test keys and connectivity (OpenRouter, broker, Yahoo, Telegram)
python main.py --graph           # print the LangGraph workflow diagrams
```

## Running the bot locally (only if you're NOT relying on the AWS one for something)

```
python main.py                    # one dry-run cycle: scan + AI + risk check, records decisions, trades nothing
python main.py --loop 30          # repeat every 30 min while NSE is open (dry run)
python main.py --loop 30 --live   # same, but trade in the built-in paper simulator
python main.py --intraday [--live]     # intraday breakout engine, its own paper account
python main.py --intraday-report       # the intraday account's positions, history and status
```
**Don't run this at the same time as the AWS deployment** — they'd fight over the same Telegram bot. Stop one first.

## The real-portfolio agent, on this PC (only needed if you're not using the AWS one)

```
python -m src.portfolio setup              # mobile, MPIN, customer ID -> Windows Credential Manager
python -m src.portfolio check              # what is stored, masked
python -m src.portfolio forget [--profile]  # delete the stored details (and the browser profile)
python main.py --portfolio --show          # run it once without the trading bot running
```

## Tests and code checks (before trusting a change)

```
python -m pytest              # ~780 offline tests (no network, no keys), a couple of minutes
python -m pyflakes main.py src scripts tests
python -m mypy                # type-checks the whole application
```

## Docker, locally on this PC (an alternative to running python main.py directly — not needed if using AWS)

```
docker compose up -d --build     # paper trading in the built-in simulator, every 30 min while NSE is open
docker compose logs -f           # what it's doing
docker compose exec trading-bot python main.py --report
docker compose down              # stop (the database volume, and your paper history, survives)
```

## Research and analysis scripts

```
scripts/test_openrouter_api.py             # tries each configured AI model with a real call (safe)
scripts/seed_setup_memory.py build|check|import   # rebuild 10y of setups for the learning memory; `check` = the
                                           #   walk-forward gate (FAILED 2026-10-01, so LEARNING_SEED stays off)
scripts/reconcile_paper.py                 # rebuilds cash from recorded fills, flags any mismatch
scripts/paper_report.py                    # full history, both accounts: every trade, P&L by day/week/total
scripts/analyze_mistakes.py                # judges every analysed stock against what then happened
scripts/intraday_archive.py status|refresh|import-cache   # the archive of 5-minute bars (fills itself while the engine runs)
scripts/backtest_intraday.py               # the LIVE engine replayed over the archived sessions (Rs 20,000 account by default)
scripts/audit_intraday.py                  # does the intraday rule beat a random entry? which filters add anything?
scripts/research_intraday.py               # Phase 2: five pre-declared intraday ideas vs a random null, with the kill criterion
scripts/analyze_patterns.py                # which entry conditions the closed trades actually won on
scripts/research_cash_sweep.py             # the idle-cash sweep test (--universe nse for all NSE stocks)
scripts/research_decomposition.py          # pure momentum to the live setup, one rule at a time (--confirm for the live-account check)
scripts/validate_walk_forward.py SYMBOL    # walk-forward, out-of-sample check of one decision phase
```
Run any of these with `python scripts/<name>.py` (some accept `--help` for their options).

## AWS account — where things live

- **Region:** ap-south-1 (Mumbai) — the console must be switched to this region to see the instance.
- **Instance:** `ai-trading-bot`, static IP `35.154.232.64`.
- **Login secret:** AWS Secrets Manager → `ai-trading-bot/integrated-portfolio` (never shown in plain text there).
- **SSH key:** `deploy/ai-trading-bot-key.pem` in this folder — the only copy; don't delete it or you lose server access.
