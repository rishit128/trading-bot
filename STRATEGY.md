# Trading Strategy (as implemented)

Defaults chosen by the builder, NOT by the account owner: review them, then change via `.env` (see `.env.example`).

## Market and universe
- **India (default):** all NSE main-board EQ-series stocks (~2,300), scanned once per day.
- Filters: price >= Rs 100, average daily traded value >= Rs 10 crore, >= 200 days of history, daily volatility <= 4%,
  price > MA50 > MA200, RSI(14) < 70. Ranked by 63-day return / volatility; top 15 analysed, plus current holdings.
- `UNIVERSE=watchlist` uses a fixed `WATCHLIST` instead.

## Entry
- Technical agent (AI over price, MA50, MA200, RSI14, volume of the **last completed session**) proposes BUY / SELL / HOLD
  with a confidence; long only.
- Multiple agents combine by role: leads must agree, advisors confirm or veto (`src/engine/strategy.py`). Minimum confidence 0.60.

## Exits (changed after a 9-year test, see below)
- **Protective stop -8%** (broker-side; was -2%) and effectively **no profit target** (+100%; was +5%): winners run.
- **Deterministic trend exit** (`TREND_EXIT=true`): a held stock is sold when its last completed close is below its
  200-day average. Plain code, like the risk engine; it does not depend on the AI happening to say SELL.
- The AI can still say SELL. Re-buy cooldown 24h so a stop-out is not bought straight back.

## Risk limits (the risk engine is the only source of order size)
- 5% of equity per stock; 80% total exposure; at most 10 open positions.
- New buys halt at -2% on the day or -20% from peak equity (the -20% halt does not reset by itself). `/pause` blocks all orders.

## Acceptance criteria (set before testing)
Signal research (`scripts/research_signals.py`), all seven required: net CAGR > 10%; Sharpe > 0.5; held-out final 3 years
Sharpe > 0.5 and CAGR > 0; Sharpe above Nifty 50 buy-and-hold; profitable in >= 60% of calendar years; max drawdown better
than -30% (relaxed from the plan's -20% up front, since no long-only Indian equity strategy avoids that through 2020);
still profitable with costs doubled.

## Evidence

### 1. Signal research: 10 years, Nifty 500, Indian costs (0.12%/side), trade one day after signal
Sep 2016 - Sep 2026, today's Nifty 500 members (498 stocks), parameters fixed from the literature, last 3 years held out.

| | CAGR | Sharpe | max DD | held-out 3y CAGR / Sharpe | vs equal-weight basket (CAGR/yr, info ratio) |
|---|---|---|---|---|---|
| Nifty 50 index | 10.5% | 0.70 | -38.4% | 5.1% / 0.44 | |
| **Nifty 500 index (real)** | 12.1% | 0.78 | -38.3% | 9.1% / 0.68 | |
| Equal-weight basket of today's members | 21.5% | 1.16 | -42.7% | 20.3% / 1.13 | |
| Momentum 12-1 (top 20, monthly) | 32.7% | 1.32 | -41.3% | 28.2% / 1.06 | +11.2%, 0.80 |
| Low volatility (top 20, monthly) | 14.2% | 1.12 | -25.9% | 8.3% / 0.78 | -7.3%, -0.60 |
| Donchian 55/20 breakout | 18.1% | 1.09 | -39.6% | 18.2% / 1.04 | -3.4%, -0.36 |
| **Current trend rule, no stops** | 21.6% | 1.24 | -32.7% | 18.6% / 1.05 | **+0.2%, -0.02** |
| RSI(2) dip in uptrend | -1.2% | 0.02 | -49.1% | -6.8% / -0.31 | -22.7%, -2.28 |
| Nifty 50 above 200-day, else cash | 5.7% | 0.57 | -24.7% | 4.7% / 0.50 | |

- **Survivorship bias is about 9.4 points a year**: the equal-weight basket of today's members earned 21.5%, the real
  Nifty 500 index 12.1%. Every stock-picking number above is inflated by it; only differences from the basket (same bias) mean anything.
- Pre-declared checklist: only low volatility passed all seven, but against the same-bias basket it *lags* by 7.3 points a
  year, so that pass is an artefact of the bias. Momentum, Donchian and the trend rule failed the drawdown rule.
- Reading: **the bot's entry rule adds nothing over just holding the stock universe** (+0.2%/yr) beyond a shallower
  drawdown. Momentum is the only candidate with relative outperformance (+11%/yr, held-out +8%/yr) but its risk-adjusted
  edge disappears out of sample (held-out Sharpe -0.07 vs the basket) and it is not survivorship-safe. Nothing here is a proven edge.

### 2. Exit study: why the original 2%/5% bracket lost money (Nifty 50, Oct 2017 - Sep 2026, event simulator, Indian costs)
Same entry rule, same stocks (45 usable), 5% positions, max 10 open; halves split at 2022-04-07.

| Exit style | Full | 1st half | 2nd half | trades | max DD | Sharpe |
|---|---|---|---|---|---|---|
| Original: 2% stop / 5% target / 10 days | -20.2% | -20.2% | -13.0% | 1,951 | -20.4% | -0.73 |
| 8% stop / 20% target / 60 days | +55.4% | +31.7% | +12.2% | 606 | -13.1% | 0.71 |
| Trend exit only, no stop or target | +133.9% | +58.6% | +34.1% | 179 | -18.7% | 0.88 |
| **8% protective stop + trend exit (now the default)** | **+143.2%** | +60.6% | +36.8% | 204 | -16.4% | 0.94 |
| 8% / 20%, no time limit | +59.1% | +30.4% | +22.7% | 450 | -13.8% | 0.74 |

The last two rows were added after seeing the first three, to mirror what the live bot can do (no time exit; broker-side
stop). The ordering (tighter is worse) held in both halves and is a comparison inside one fixed set of stocks, so
survivorship bias does not explain it.

Failure analysis of the original exits (P&L in Rs on a Rs 10 lakh account): 1,243 stop-outs averaged -2.37% and cost
Rs 1.30M; 548 targets averaged +4.30% and earned Rs 1.03M; 996 trades lasted 2 days or less (-Rs 393k); stops hit after a
**median of 1 day** (inside normal 2-4% daily noise). The bot's own -20% drawdown limit then halted it in 2022 for good.
Zero-cost the original still only made +7.9%: costs (~0.24% round trip on ~3,700 trades) did the rest.

### 3. Earlier one-year tests
- India, trend rule with the old 2%/5% exits, Nifty 50: -6.96% with costs, -2.44% without; Nifty 500: -12.47% / -5.81%.
- US, AI technical agent, 5 stocks, 420 signals, 2025-09 to 2026-09, old exits: -0.02%, 28% win rate vs 28.6% break-even.

## What is and is not established
- **Established:** tight 2%/5% exits destroyed value in every window tested; a wide protective stop with a trend exit is
  far better on the same entries in both halves of 9 years.
- **Not established:** any edge from the entry signal; anything about the AI agent on Indian stocks (all Indian evidence
  uses the plain rule as a stand-in); results without survivorship bias; live behaviour (circuit limits, real fills).
- The exit change does not make this a validated strategy. Still paper / dry run only, never real money.

## Round 2 research (2026-09-21) and the ranking change
Ten years of Nifty 500 daily data, same pre-declared rules as `scripts/research_signals.py`. Excess is CAGR over an equal-weight basket of the same stocks (which carries ~9 points/yr of survivorship bias).
- Current trend rule: +0.1%/yr excess. Adds nothing over holding the basket.
- 12-1 momentum, top 20 monthly: +11.2%/yr excess, information ratio 0.80, +7.9%/yr in the last 3 years. Max drawdown -41% (fails the -30% rule).
- 6-1 momentum: +8.3%. Trend rule + beats Nifty over 6m: +0.6%. Trend rule + Nifty>200d regime filter: -4.7% (drawdown -29%). 12-1 + regime filter: +4.0%.
- **Change made:** the scanner now ranks by 12-1 month momentum instead of risk-adjusted 3-month momentum. Not proven; watch the paper results.
- Not tested: results-date and delivery-% filters (need historical NSE data that Yahoo does not provide).

## Round 3: delivery percentage (2026-09-21, only ~3 years of data: 2023-09 to 2026-09)
Filter: 20-day average delivery % above the day's cross-sectional median. Compared with the same-window equal-weight basket (21.2% CAGR).
- Momentum 12-1: 30.3% CAGR, Sharpe 1.12, max DD -33%. **With the delivery filter: 35.3%, Sharpe 1.54, DD -27%** (excess +14.1%/yr; last year +18.2%).
- Trend rule: 19.5% -> 17.6% with the filter. No help.
- One short window, one bull-market regime, single test: promising, **not proven**, and NOT wired into the bot. Re-run `scripts/research_delivery.py` as the cache grows before using it.

## Intraday opening-range breakout (2026-09-21)
Rule (`src/intraday/strategy.py`, parameters fixed in advance, not tuned): long only; opening range = first 15 minutes; buy when a 5-minute bar closes above the range high, above VWAP, with volume >= 1.5x average, 09:30-14:00; stop = range low; target = 2x risk; everything closed by 15:15 IST; one trade per stock per day; risk 0.5% of equity per trade, at most 5 positions, 2% daily loss halt. Runs in its own paper account (`intraday.db`).
- **Backtest** (`scripts/backtest_intraday.py`, Nifty 100, Yahoo's last 58 trading days, costs + 0.05% slippage): 290 trades, win rate 36%, **-11.2%**, Sharpe -6.3, 28% profitable days, negative in both halves. Stops lost ₹189k, targets made ₹75k, fees ₹33k.
- Sample is one ~3-month regime, so this is not final, but there is no evidence of an edge. Not tuned afterwards on purpose (tuning on 58 days would just fit noise). The engine is paper-only to gather live evidence.

## Intraday: confidence-scaled sizing and shared budget (2026-09-22)
Same idea as the swing risk engine: the ORB rule has no AI confidence, so `Setup.strength` stands in for it (0.0 at the
minimum required volume surge of 1.5x average, 1.0 at 2x that or more). Position size scales linearly from
`MIN_POSITION_PCT` to `MAX_POSITION_PCT` of equity by that strength (`src/intraday/strategy.position_size`), replacing
the old fixed risk-per-share sizing. `IntradayEngine` now reads `max_open_positions`, `min_position_pct` and
`max_position_pct` from the same `.env` settings as the swing bot, so both accounts share the Rs 20,000 budget, 5-position
cap and 10%-25% sizing range set on 2026-09-22. Its own account (`intraday.db`) is separate and never mixed with swing.

## AI agent backtested on Indian history for the first time (2026-09-22)
Every prior result in this file used the plain mechanical rule as a stand-in for the AI. `scripts/backtest_ai_agent.py`
instead replays the REAL `TechnicalAgent` + `LLMClient` (real OpenRouter calls, the exact production prompt) against
point-in-time Indian daily data, monthly rebalance (not daily, to keep API calls bounded), no lookahead. Single runs,
not pre-registered like `scripts/research_signals.py`; today's Nifty 50/100 membership (survivorship bias applies).

| Run | AI-filtered CAGR | Mechanical-only CAGR | Hold-everything CAGR | AI vs mechanical | AI approval rate |
|---|---|---|---|---|---|
| Nifty 50, 1y, 163 calls | -5.4% (Sharpe -0.58) | -6.0% | **+9.3%** (Sharpe 0.73) | +0.6%/yr | 96% |
| Nifty 100, 2y, 251 calls | -8.0% (Sharpe -0.55) | -7.1% | **+20.4%** (Sharpe 1.36) | -0.9%/yr | 97% |

**Findings:**
1. The AI adds no value over the mechanical filter it sits on top of -- it approved 96-97% of everything shown to it,
   effectively rubber-stamping the rule rather than exercising independent judgment.
2. Both the AI-filtered and mechanical-only portfolios badly lagged simply holding every liquid stock equally, by
   ~15-28 points/yr in both runs. This is a materially worse result than the 10-year study's "roughly zero edge" (this
   file, above): over the last 1-2 years specifically, the entry filter (trend-consistency + 12-1 momentum, the current
   live combination) actively hurt versus doing nothing clever.
3. Caveat: this tests the CURRENT live filter combination, not pure 12-1 momentum alone (which showed a real edge over
   10 years in the Round 2 research above). The trend-consistency filter (price > MA50 > MA200, RSI < 70) may be the
   part hurting recent results, not the momentum ranking itself -- not yet isolated.

Reusable, cached results: `research_cache/ai_backtest_signals.json` (re-running the script does not re-spend API calls
on symbol/date pairs already scored). Raw logs: `logs/ai_backtest_full.log`.

**Decision:** the AI stays -- this project is an AI trading bot, not a rule-based one. The fix under consideration is a
richer prompt (inputs the mechanical filter doesn't already see, so the AI has a genuine reason to disagree) and/or
testing pure 12-1 momentum as the candidate list instead of the current trend+momentum combination.
