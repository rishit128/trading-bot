# Trading Strategy (as implemented)

Defaults chosen by the builder, NOT by the account owner: review them, then change via `.env` (see `.env.example`).

## Market and universe
- **India (default):** all NSE main-board EQ-series stocks (~2,300), scanned once per day. `MARKET=us` switches to US stocks.
- Filters: price >= Rs 100, average daily traded value >= Rs 10 crore, >= 200 days of history, daily volatility <= 4%,
  price > MA50 > MA200, RSI(14) < 70. Ranked by 63-day return / volatility; top 15 analysed, plus current holdings.
- `UNIVERSE=watchlist` uses a fixed `WATCHLIST` instead.

## Entry
- Technical agent (AI over price, MA50, MA200, RSI14, volume of the **last completed session**) proposes BUY / SELL / HOLD
  with a confidence; long only. US adds a news-sentiment advisor that can only confirm or veto.
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
