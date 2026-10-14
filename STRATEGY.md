# Trading Strategy (as implemented)

Defaults chosen by the builder, NOT by the account owner: review them, then change via `.env` (see `.env.example`).

The one-off research scripts named in the dated sections below were removed on 2026-10-10 once their verdicts were
recorded here; they remain in git history at commit 8284b54. The intraday audit and hypothesis scripts are kept: they
are meant to be re-run as the 5-minute archive grows.

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
- **Protective stop -15%** (broker-side; was -8% until 2026-09-25 and -2% originally, see the live-configuration section) and effectively **no profit target** (+100%; was +5%): winners run.
- **Deterministic trend exit** (`TREND_EXIT=true`): a held stock is sold when its last completed close is below its
  200-day average. Plain code, like the risk engine; it does not depend on the AI happening to say SELL.
- The AI can still say SELL. Re-buy cooldown 24h so a stop-out is not bought straight back.

## Risk limits (the risk engine is the only source of order size)
- 5% of equity per stock; 80% total exposure; at most 10 open positions.
- New buys halt at -2% on the day or -20% from peak equity. The -20% halt now lapses by itself after 30 calendar days (`DRAWDOWN_PAUSE_DAYS`; the peak is rebased to current equity), or at once with `/rebase`; 0 keeps the old permanent halt. `/pause` blocks all orders.

## Acceptance criteria (set before testing)
AgentSignal research (`scripts/research_signals.py`), all seven required: net CAGR > 10%; Sharpe > 0.5; held-out final 3 years
Sharpe > 0.5 and CAGR > 0; Sharpe above Nifty 50 buy-and-hold; profitable in >= 60% of calendar years; max drawdown better
than -30% (relaxed from the plan's -20% up front, since no long-only Indian equity strategy avoids that through 2020);
still profitable with costs doubled.

## Evidence

### 1. AgentSignal research: 10 years, Nifty 500, Indian costs (0.12%/side), trade one day after signal
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


## Live-configuration backtest (2026-09-25): the numbers that match the account actually running
Everything above was measured on a Rs 10 lakh account with 10 positions of 5%, and most of it on the plain trend rule.
The paper account is Rs 20,000 with 5 positions of 10-25%. `scripts/backtest_live_config.py` now runs the live pieces
themselves (the scanner's own `screen_symbol` filter and 12-1 momentum ranking, the affordability rule, the risk engine
with the live limits, 8% stop + MA200 trend exit, no time exit, the paper broker's fees and slippage) on Nifty 500 daily
bars, 2017-10 to 2026-09. The AI is a constant approval confidence of 0.75. Today's members only: survivorship bias
inflates every stock-picking row; the equal-weight basket of the same stocks carries the same bias.

**Two defects found and fixed by this exercise**
1. The -20% drawdown halt was permanent. The account first fell 20% on 2018-03-23, the halt engaged and it never bought
   again: the "9-year result" of -24.7% was six months of trading plus 8.5 years in cash. It now lapses after 30 days.
2. The scanner ranked and sent to the AI stocks the account could never buy (DIVISLAB, APOLLOHOSP at ~Rs 9,000 against a
   Rs 2,000-5,000 position budget); every live fill so far was a stock under Rs 1,600. It now keeps only stocks where
   one share fits in the smallest position (`ScreenConfig.affordable_pct`).
(Also fixed: the simulator filled buys in alphabetical order, not rank order, and cut the calendar to the shortest history.)

**Full period, 2017-10 .. 2026-09, halt with 30-day pause**

| Setup | Return | CAGR | Max DD | Trades | Win | Avg trade | Profit factor |
|---|---|---|---|---|---|---|---|
| **Live: Rs 20,000, 5 positions, 10-25%** | +171% | 12.0% | **-44.6%** | 184 | 22% | +5.2% | 1.43 |
| Live limits on Rs 10 lakh | +276% | 16.2% | -40.6% | 182 | 23% | +6.4% | 1.61 |
| Research setup: Rs 10 lakh, 10 x 5% | +194% | 13.0% | -23.4% | 379 | 26% | +6.3% | 1.80 |
| Nifty 50 buy and hold | +133% | 10.1% | -38.4% | | | | |
| Equal-weight basket, same stocks | +534% | 23.3% | -45.8% | | | | |

**Last 3 years (2023-09 .. 2026-09), with the delivery filter (the live default)**: live Rs 20,000 +64% (18.3%/yr, Sharpe
1.04, DD -17%); live limits on Rs 10 lakh +84%; research setup +44%; basket +77%; Nifty 50 +17%.

**Exit variants, live Rs 20,000 account, full period** (a diagnostic, not a tuned choice): 8% stop (default) +171%,
DD -44.6%, win 22%; 15% stop +255%, DD -36.8%, win 33%; no stop (trend exit only) +350%, DD -35.5%, win 39%, PF 2.06.
The 8% stop is too tight for the volatile momentum stocks the ranking picks; it converts many later winners into small
losses. Not changed automatically: one dataset, and a broker-side stop also guards against data outages and gaps.

**What this says**
- The live setup is not safe as configured: a -45% drawdown against a 20% halt (the halt stops new buys but cannot stop
  five 10-25% positions gapping down, and each pause rebases the peak). The 10 x 5% research sizing halves the drawdown
  (-23%) for a similar return. A Rs 20,000 account cannot hold 10 x 5% of most listed stocks (5% is Rs 1,000), which is
  the real cost of the small account.
- Fees are material: 3-8% of capital over the test on Rs 20,000 (the fixed Rs 15.93 charge per sale), against 1-2% on Rs 10 lakh.
- The entry ranking plus these exits beats the Nifty 50 but not the equal-weight basket of the same stocks in either window.
  Together with the earlier findings, there is still no evidence of an edge from the entry signal itself.
- Trade counts: about 20 a year on the live setup. Win-rate gates over a 4-week paper run cannot mean anything.

**Decision taken 2026-09-25 (after the sizing x stop grid below), applied to the live configuration**
- Sizing: 10 positions of a flat 5% (the code defaults; the `.env` overrides of 5 positions at 10-25% were removed).
  Concentration is the largest single driver of drawdown: 5 large positions lost more in every period at every stop
  width. 10 x 5% is the only setup with a drawdown near the 20% risk limit (about -23%) and it has the same return per
  unit of drawdown as the others.
- Stop: 15% (was 8%). Wider was better in nearly every setup and both halves; "no stop" scored highest but a stop is kept
  as a safety net against gaps and outages, and 15% captures most of the gain. The MA200 trend exit is the actual exit.
- Positions already open keep the 8% stop they were opened with.

| Full period, Rs 20,000 | 8% stop | 15% stop | no stop |
|---|---|---|---|
| A: 5 positions, 10-25% | +171% (DD -45%) | +255% (-37%) | +350% (-36%) |
| B: 8 positions, flat 10% | +331% (-37%) | +338% (-39%) | +361% (-34%) |
| C: 10 positions, flat 8% | +324% (-36%) | +351% (-35%) | +444% (-35%) |
| **D: 10 positions, flat 5% (chosen)** | +137% (-23%) | **+158% (-27%)** | +201% (-23%) |

Both halves (2017-10..2022-03 and 2022-03..2026-09) agree on the ordering by stop width; first-half returns are much lower
than second-half in every row (the 2018 small-cap bust and 2020 crash against the 2022-26 rally), so read them as a range.
Caveats: one dataset, survivorship bias, the AI is a constant stand-in, and the differences between neighbouring cells are
within what a different sample could reverse; the choice rests on the consistent direction, not on any single cell.

**Fee-drag rule (2026-09-25).** The first live cycle on the new sizing opened a one-share Rs 363 position: with the fixed
Rs 15.93 charge per sale its round trip costs ~4.6%. The risk engine now refuses a buy whose round-trip fees exceed
`MAX_FEE_DRAG_PCT` (default 4%, roughly a Rs 425 minimum position on Indian delivery costs). Backtest on the Rs 20,000
account (10 x 5%, 15% stop): no cap +158%, 5% cap +158%, **4% cap +158% (identical trades)**, 3% cap +142%, 2% cap -19%
(it starves the account of trades). So 4% costs nothing in the tested history and only blocks the degenerate case;
tighter caps do measurable harm.

## Learning from history and mistakes

The bot learns from **every** decision it analyses, not only the trades it took (a HOLD that then rallied is as informative as
a BUY that failed):

1. **Label outcomes** (`src/learning/outcomes.py`, once a day after a cycle, or `scripts/label_outcomes.py`). For each analysed
   stock-session: entry at the next open, exit `LEARNING_HORIZON` (60 since 2026-09-26; was 20) sessions later, plus the worst dip and whether the
   protective stop would have been hit. Only matured sessions are labelled; nothing looks ahead.
2. **Setup memory** (`src/learning/setups.py`, fed to the agent's learning step when `LLM_LEARNING=true`). "How did setups like
   this one (trend shape, RSI, ADX, volume) turn out, across all stocks?" Guardrails: silent below `LEARNING_MIN_SAMPLES`
   (30) independent observations; one observation per stock per week; results net of a 0.35% round-trip cost; it always
   shows the win rate of all comparable setups beside the matched one, so the market drifting up is not mistaken for an edge;
   it backs off from fine to coarse labels until enough data exists. The learning step can only nudge confidence by
   `LLM_MAX_ADJUST`; the risk engine is untouched.
3. **Mistake report** (`scripts/analyze_mistakes.py`): read-only. BUY vs HOLD forward returns with 95% intervals, confidence
   calibration, worst BUYs, biggest missed gains; groups too small to judge are marked insufficient.

Nothing here retunes settings automatically: that would fit noise. The memory stays silent until the first 20-session
outcomes mature and enough of them accumulate; whether it helps is then measurable with `scripts/ablate_phases.py`.

### Feature study on 10 years of Nifty 500 (`scripts/research_features.py`, report only)

Six indicators, five quintile buckets each, 20-session net excess return over the same-date universe average, edges from
2017-2021, tested on 2022-2026, pre-declared bar |t| >= 3 in both periods (30 buckets examined). Result: **2 of 30 pass**:
ADX above ~35 (+0.5% / +0.6% per 20 sessions, t 3.6 / 4.2) and 6-month momentum above ~37% (+0.9% / +0.7%, t 7.3 / 5.8).
RSI, volume ratio, distance from the 50-day average and ATR% do not pass. Caveats: today's index members only (survivorship
flatters momentum), and the effects are small next to the 15% stop. Momentum is already what the scanner ranks by; ADX is a
lead worth a proper backtest, not a rule change. Nothing was adopted.

### Intraday ORB: can it be fixed? (`scripts/research_intraday.py`, 2026-09-25)
Six pre-declared one-idea variants (entries only until 11:00, 1R target, narrow range, 3x volume, market-breadth gate, no
extended breakouts) on the cached last 59 days of Nifty 100 5-minute bars. Candidate bar: >= 100 trades, positive in BOTH
halves, expectancy > +0.10% per trade, still positive with slippage doubled. **None passed; every variant lost in both
halves** (about -0.25% per trade, -0.35% with doubled slippage). Before slippage the rule's gross edge is about -0.05% per
trade: there is nothing to filter, and costs (~0.11% fees + 0.10% slippage per round trip) turn zero into a steady loss.
Live paper so far agrees (19 trades, -1.7%). Conclusion: tuning this rule is not the answer.

## Does the entry rule beat holding the stocks? Two more pre-declared tests (2026-09-26)
Both use the live configuration (Rs 20,000, 10 positions, flat 5%, 15% stop, MA200 trend exit, delivery filter as in
the live default), Nifty 500 daily bars (today's members, so survivorship bias inflates every row equally with the
basket), a constant 0.75 AI stand-in confidence, and the same four gates fixed before running: beats the equal-weight
basket net of costs; positive in both halves; positive at 2x slippage; at least 40 trades. Scripts:
research_entry_filters.py and research_price_floor.py (removed 2026-09-26 after the tests failed; these tables are the record). Neither simulator models circuit limits.

**1. Extra entry gates (ADX >= 35, 12-1 momentum > 0, both).** 0 of 9 candidate x window combinations passed.

| Window | Basket | Baseline | ADX>=35 | Mom>0 | Both |
|---|---|---|---|---|---|
| Full, 2017-10..2026-09 | +534.7% | +156.6% | +178.5% | +153.8% | +153.3% |
| Last 3y | +77.4% | +65.6% | +13.0% | +65.6% | +13.2% |
| Last 3y + delivery | +77.4% | +48.9% | +18.8% | +47.8% | +19.1% |

ADX>=35 improves trade quality over the full period (win rate 33%->42%, profit factor 2.02->2.46) but is far worse over
the last 3 years and turns negative in the second half. The momentum floor is close to a no-op: the scanner already ranks
by that momentum, so nearly every candidate has positive 12-1 momentum. Conclusion: the entry rule is not the fix.

**2. Lower price floor (Rs 100 -> Rs 50 / Rs 30).** Prompted by liquid low-priced PSU stocks such as NBCC (Rs 82, Rs 37cr/day,
1.5% daily volatility) and IRFC (Rs 80, Rs 57cr/day, 1.0%), which pass every filter except the price floor. 0 of 6
combinations passed.

| Window | Basket | Rs 100 (live) | Rs 50 | Rs 30 |
|---|---|---|---|---|
| Full | +534.7% | +156.6% | +183.1% | +193.9% |
| Last 3y | +77.4% | +65.6% | +45.9% | +47.9% |
| Last 3y + delivery | +77.4% | +48.9% | +45.6% | +47.7% |

A lower floor helps over the full period and hurts over the last 3 years, so the effect is not stable. Left at Rs 100.
Sub-Rs 20 stocks were deliberately not tested: circuit-limit risk is unmodelled and grows as price falls.

**Reading both together.** The bot holds at most 10 names; the basket holds everything, so in a broad rally any
concentrated rule lags it whatever the filter. Filters change which trades are taken, not that structural gap. Further
tests should also report Sharpe and maximum drawdown next to "beats the basket".

## Sector strength and market breadth as entry gates (2026-09-26)
research_sector_breadth.py (removed 2026-09-26 after the test failed), same live configuration, universe and four gates as above, plus Sharpe and maximum
drawdown reported (not gated). Three variants fixed in advance: sector strong (its NSE "Industry" group's 6-month return
beats the universe's), sector strong + stock leads its group, and breadth >= 50% (at least half of stocks above their own
200-day average). Sectors are NSE's 20 coarse industries as labelled today. **0 of 9 combinations beat the basket.**

| Window / variant | Return | Sharpe | Max drawdown |
|---|---|---|---|
| Full: basket | +534.7% | 1.23 | -45.8% |
| Full: baseline / sector strong / +leader / breadth | +156.6% / +169.7% / +166.9% / +160.5% | 1.00 / 1.09 / 1.07 / 1.18 | -26.7% / -26.6% / -24.7% / -18.1% |
| Last 3y: basket | +77.4% | 1.19 | -21.2% |
| Last 3y: baseline / sector / +leader / breadth | +65.6% / +41.7% / +38.0% / +47.2% | 1.42 / 1.03 / 0.95 / 1.30 | -13.5% / -13.3% / -15.7% / -10.1% |
| Last 3y + delivery: baseline / sector / +leader / breadth | +48.9% / +50.6% / +39.1% / +47.2% | 1.31 / 1.41 / 1.16 / 1.39 | -13.8% / -8.0% / -10.7% / -10.7% |

Sector strength is inconsistent (slightly better over the full period, worse over the last 3 years without the delivery
filter, better with it). The breadth gate is a defensive filter: it kept the full-period return about the same while cutting
maximum drawdown from -26.7% to -18.1% and lifting Sharpe from 1.00 to 1.18, but over the last 3 years it gave up return
and Sharpe (1.42 -> 1.30) and sat out most of the second half (+0.0%). Read as a lead, not a result: the risk-adjusted bar
was not declared beforehand, so it cannot be adopted on this run. Note the baseline already beats the basket on Sharpe and
drawdown over the last 3 years (1.42 vs 1.19, -13.5% vs -21.2%) and only trails on raw return.

## Idle cash and the cash sweep (2026-09-26)
**Finding.** 10 positions x 5% caps the bot at ~50% invested. On the 9-year Nifty 500 run it was 45% invested on average
(all 10 slots full 84% of days; ~97 stocks pass the scanner on a typical day, so capacity, not candidates, is the limit),
and the simulator paid nothing on the rest. Fees were ~23% of starting capital over 9 years on Rs 20,000 (the fixed
Rs 15.93 charge per sale on Rs 1,000 positions). Exploratory exit/sizing variants (20% stop, no stop, no target, 16
positions, 8 x 10%) each helped one window and hurt the other; removing the +100% target hurt both. Exits left as they are.

**Change.** `CASH_YIELD_PCT` (default 0 = off): idle paper cash earns that annual rate, standing in for a liquid mutual
fund held outside demat (no DP charge or STT on redemption; instant redemption up to Rs 50,000). Implemented in the paper
broker (credited per elapsed day, before every cash movement, included in the ledger reconciliation), the backtest
simulator and the holdout reserve (so their drift stays like for like). `--report` now also shows the invested share,
interest earned, the max drawdown so far and the Nifty 50 over the same period.

**Test** (`scripts/research_cash_sweep.py`, pre-declared; judged at a deliberately low 3.5%): must beat the no-sweep run on
return, Sharpe and drawdown on every window and stay positive at 2x slippage. **Passed on both universes, every window.**

| Universe / window | No sweep | Sweep 3.5% | Basket |
|---|---|---|---|
| Nifty 500, full | +156.6%, Sh 1.00, DD -26.7% | +183.8%, 1.12, -23.5% | +534.7%, 1.23, -45.8% |
| Nifty 500, last 3y | +65.6%, 1.42, -13.5% | +73.9%, 1.57, -12.8% | +77.4%, 1.19, -21.2% |
| Nifty 500, last 3y + delivery | +48.9%, 1.31, -13.8% | +55.2%, 1.50, -11.5% | +77.4%, 1.19, -21.2% |
| All NSE, full | +159.2%, 0.98, -31.3% | +194.6%, 1.13, -27.4% | +474.7%, 1.15, -60.3% |
| All NSE, last 3y | +40.4%, 0.95, -19.8% | +48.4%, 1.09, -18.9% | +54.0%, 0.85, -30.7% |
| All NSE, last 3y + delivery | +50.4%, 1.28, -13.4% | **+69.8%, 1.64, -11.2%** | +54.0%, 0.85, -30.7% |

Not modelled: tax on fund gains, a varying rate, a one-day redemption delay. At 5% the full-period NSE return is almost
identical to 3.5% (sizing follows equity, so trades differ): the effect is robust, the exact number is not.

**Universe matters.** The live scanner covers all ~2,300 NSE stocks; every earlier test used today's Nifty 500. On the
full NSE list the basket is weaker (last 3y +54.0%, Sharpe 0.85, vs +77.4%, 1.19 on the Nifty 500) and the live setup
with the delivery filter and the sweep beats it outright, on raw return as well. That is the first window in which the
bot beats the basket on every measure; it is one window, and delisted stocks are still missing (survivorship remains).
Earlier "fails vs the basket" results were measured on the Nifty 500 and should be re-read with this in mind.

## The three entry-side tests re-run on all NSE stocks (2026-09-26)
Same scripts and pre-declared gates, `--universe nse` (2,317 stocks, the list the live scanner actually scans), no cash
sweep (so they compare with the Nifty 500 runs above). Basket: full +474.7%; last 3y +54.0% (Sharpe 0.85, DD -30.7%).

| Variant | Full 9y | Last 3y | Last 3y + delivery (live setup) |
|---|---|---|---|
| Baseline (live scanner) | +159.2% | +40.4% | +50.4% |
| ADX >= 35 | +110.1% | +23.8% | +18.3% |
| 12-1 momentum > 0 | +117.2% | +40.4% | **+59.0% (passes all 4)** |
| ADX >= 35 and momentum > 0 | +89.5% | +23.8% | +21.0% |
| Price floor Rs 50 | +100.5% | +45.3% | +40.1% |
| Price floor Rs 30 | +125.6% | +51.7% | +44.5% |
| Sector strong | **+213.1%** (Sh 1.20, DD -22.6%) | **+61.6% (passes all 4)** | +33.2% |
| Sector strong + stock leads | +128.8% | **+62.0% (passes all 4)** | +52.2% |
| Breadth >= 50% | +81.4% | +29.8% | +30.0% |

**Every variant fails the pre-declared rule** (pass on every window). Four single-window passes, each contradicted in
another window. The universe changes the answers: ADX >= 35 helped the full period on the Nifty 500 and hurts
everywhere on NSE; the lower price floor helped the full period on the Nifty 500 and hurts it on NSE; the breadth gate's
drawdown benefit on the Nifty 500 is gone on NSE (worse in every window). Sector variants are confounded: only 737 of
2,317 stocks carry an NSE industry label (Total Market list), so they also restrict the bot to larger, liquid stocks.
Nothing changed in the live configuration. Conclusion stands and is now firmer: price-derived entry filters are
unstable across universes and periods; further entry work needs genuinely new information (results calendar, insider
and corporate disclosures), not more transformations of price.

## Free-data inventory for Indian equities (Step 4, Phase 0; 2026-09-26)
Probed NSE's public JSON endpoints (www.nseindia.com/api/...) with a few dozen paced requests. They answered without
a session cookie; 5 of 5 repeated calls succeeded. The homepage returned 403, so NSE's terms of use could not be read
from here: check them before relying on these endpoints in the running bot.

| Data | Endpoint | Free? | History | Verdict |
|---|---|---|---|---|
| Board meetings (results dates) | corporate-board-meetings | yes | back to at least 2012 | **Best source.** Purpose field says "Results"/"Financial Results"; results meetings are announced a median 8-10 days ahead, 99-100% at least 2 days ahead, and the intimation timestamp is recorded, so "results within N days" can be backtested with no lookahead |
| Upcoming meetings | event-calendar | yes | forward-looking | For the live bot: which stocks report in the next days |
| Insider trades (SEBI PIT) | corporates-pit | yes | 2016 to Jan 2026 | **Gap:** Jun-Aug 2026 return 0 rows though Jan 2026 has 720. Fine for a historical event study, not yet reliable for live use |
| Announcements | corporate-announcements | yes | back to at least 2017 | ~13,000 a month, mostly routine ("Trading Window", meeting notices). Text for a later AI-reading experiment; noisy |
| Corporate actions | corporates-corporateActions | yes | back to at least 2017 | Dividends, splits, bonuses with ex-dates |
| Quarterly results figures | corporates-financial-results | inconsistent | - | 0 rows recently, 41 in Sep 2021, 1,642 in Sep 2017: unreliable, do not build on it |
| Bulk / block deals | historical/bulk-deals | no | - | Returns a web page, not data; a snapshot endpoint gives today only |

Next: the results calendar is the one to test first (point-in-time, deep history, forward-looking live feed).

**NSE terms of use (read 2026-09-26, www.nseindia.com/static/nse-terms-of-use). This overrides the "Next" line above.**
They state: "User is prohibited to conduct any systematic or automated data collection activities (including scraping,
data mining, data extraction and data harvesting)"; content may not be "stored ... in an electronic retrieval system
... without prior written permission of NSE"; and the data may not be used "for any gaming, virtual trading or
simulation activities under any circumstances whatsoever". So the results-calendar feature must NOT be built on NSE's
website API without NSE's permission. The same terms already bear on existing code that downloads from NSE websites
automatically: the daily delivery bhavcopy (src/data/delivery.py, nsearchives.nseindia.com) and the NSE equity list
(src/data/india.py, archives.nseindia.com); and the whole project is a paper-trading simulation. Index constituent
lists come from niftyindices.com (NSE Indices Ltd), whose terms were not checked. Not legal advice: the user decides.
Compliant routes: files the user downloads by hand from NSE and gives to the bot; a licensed data vendor or NSE's own
data products; a broker API whose terms allow this use; or written permission from NSE.
**Done (2026-09-26):** the bot and every research script now read NSE files that a person downloads in a browser into
`nse_files/` (NSE_FILES_DIR; git-ignored; mounted read-only in Docker): the stock list (EQUITY_L.csv, fallback
ind_nifty500list.csv), the index lists, and the daily bhavcopy files. No code downloads from nseindia.com or
niftyindices.com any more (a test enforces it). Startup refuses to run the whole-market scan without the stock list and
says what to download; the delivery filter switches off, with a warning, when its newest file is over 7 days old (it
used to keep ranking on whatever data it last had). Stored NSE content was deleted from research_cache. Cost: the
delivery filter, which helped the live setup in the NSE tests, now needs a file saved by hand each trading day. The
"virtual trading or simulation" clause still concerns the project as a whole; only a licensed source or NSE's
permission settles that. The results-calendar test is on hold for the same reason.
**Revised the same day:** a daily hand-saved file is not workable, so the stock list now comes from Yahoo Finance's NSE
screener automatically (same source as every price; checked on 2026-09-26 to cover all 2,317 NSE EQ-series stocks,
pre-cut to stocks trading at least half the liquidity floor). A hand-saved EQUITY_L.csv still takes priority; the
intraday universe is a saved Nifty 100 list or else the 100 most-traded stocks by value (Yahoo). The delivery filter has no permitted automatic
source and is now OFF by default (DELIVERY_FILTER=false). Yahoo's screener also gives a next-results date for ~82% of
liquid stocks: a possible live-only source for the results-calendar idea, with no history to backtest it on.

## Experiment 1: which live rule destroys the momentum edge? (2026-09-26)
`scripts/research_decomposition.py` walks from pure 12-1 momentum to the live setup one rule at a time, in the event
simulator with fees and slippage, on all NSE stocks. Benchmark: an equal-weight basket of the INVESTABLE universe (price
>= Rs 100, >= Rs 10 crore/day, known the day before): +318.5% full period (Sharpe 0.94), +56.9% last 3 years (0.89).
Earlier tests used every listed stock, including ones the bot can never trade (+474.7%), which overstated the bar.

Chain at Rs 10 lakh, 10 x 10% (fully invested). CAGR / Sharpe, full period | last 3y:
- S0 pure momentum, monthly top 10: 21.2% / 0.81 | 10.5% / 0.49 (max DD -48% | -44%)
- S3 + uptrend, RSI < 70, volatility <= 4% filters (monthly): 22.7% / 0.91 | 23.2% / 0.89
- S4 live holding rule (daily buys, MA200 exit): 18.0% / 0.79 | 15.4% / 0.66
- S7 + 15% stop, +100% target, halts = the live rules at full size: **21.1% / 0.95 | 24.0% / 0.97** (basket 17.6% | 16.4%)
- S8 the live account (Rs 20,000, 10 x 5%): 11.0% / 0.95 | 10.9% / 0.86 (max DD -31% | -20%)

**Reading.** The live rules, fully invested, beat the investable basket in both windows (+3.5% and +7.6% a year) at equal
or better Sharpe. The live account keeps the Sharpe but earns half, because it is half in cash (the cash sweep recovers
part of this); the small account's fixed per-sale charges and the affordability rule cost the rest.

**Leave-one-out and confirmation.** In the chain, dropping the uptrend filter or the RSI cap met the pre-declared
"harmful" rule. A stricter confirmation, declared before running (live account, both universes, both windows, both
halves, 2x slippage, drawdown within 5 points), **overturned both**: without the uptrend filter the live account trades
three times as often and CAGR falls 11.0% -> 3.1% (max DD -49%); dropping the RSI cap helps on NSE and on the Nifty 500
full period but hurts the Nifty 500 last 3 years (17.8% -> 10.1%, Sharpe 1.39 -> 0.83). **Nothing changed.** Lessons:
judge rule changes at the live account size and on both universes, never only at a research account; compare with the
investable basket.

## Learning memory horizon: 20 -> 60 sessions (2026-09-26)
The decision memory judged similar setups by their 20-session outcome, but live-setup trades are held far longer. On the
backtest's live-setup trades (Rs 20,000 account, 2017-2026): median hold 74 sessions (all NSE) and 65 (Nifty 500). How
well each forward window ranks the trades' real results (rank correlation): 5 sessions +0.30 / +0.44, 20 sessions
+0.53 / +0.62, **60 sessions +0.66 / +0.69**. LEARNING_HORIZON is now 60 (the outcome labels already record 5, 20 and
60; any other value is refused). Cost: 60-session outcomes take ~3 months to mature, so the memory stays silent longer.


## Intraday audit and Phase 0 repairs (2026-09-28, system version orb-v2)
A from-scratch review of the intraday engine (`scripts/audit_intraday.py`, `src/research/intraday_audit.py`; 5,800
stock-sessions of Nifty 100 5-minute bars, 2026-07-07..09-25). **The rule has no edge**, and the previous live and paper
evidence was not measuring what it claimed.

**Edge.** Live rule: 2,622 trades, gross -0.048% / net -0.254% per trade (±0.03%). Entering at a RANDOM time with the same
stop, target and exits: gross -0.066% / net -0.272%. The signal's edge over random is +0.018% per trade: noise. VWAP adds
nothing (identical trades), and the volume filter does not help (dropping it: gross -0.032%). No entry-time, volume,
gap, breadth or range bucket is profitable net of costs. Costs (~0.106% fees + 0.10% slippage per round trip) turn a
zero-edge rule into a steady loss. Today's low signal count is the market, not a fault: 6 to 11 breakouts against a
historical median of 47 a day.

**Defects found and repaired (orb-v1 -> orb-v2).**
1. *Paper stop/target replay dropped almost every bar's range.* Each check consumed the bar in progress, so a stop breached
   later in that bar was never seen (only each bar's first ~20 seconds were checked). Now every bar after the buy is
   replayed in full: the forming bar is examined but only marked checked once finished (`PaperBroker._settle_exits`; also
   the swing account's stops). Effect on the sample's average was small (-0.050% vs -0.048%), but the protection was
   weaker than documented.
2. *Sizing borrowed the swing 5%*, so a Rs 20,000 account could not buy any stock above Rs 985: 63% of the backtest's
   signals were silently skipped and burned the stock for the day. Intraday now has its own `INTRADAY_*` settings, sized
   by risk (0.5% of equity per trade) under a 20% position cap, cash only; a skip is explained and recorded.
3. *The traded universe was not the tested one*: live used Yahoo's top 100 by traded value, which shares 44 names with the
   Nifty 100 (it adds recently listed stocks). One selector (`src/intraday/universe.py`) now serves the engine and the
   scripts, records its source daily, and labels the fallback. The scripts had also lost their symbol loader.
4. *No record of skipped signals*: `intraday_signals` and `intraday_universe` now store every breakout, its outcome and
   its reason.
5. *No guard for a broken opening range* (a missing or late first bar builds the range from the wrong bars): such stocks are
   skipped. Bars from a previous session are never traded. A position left over from an earlier day is closed at the next
   session's first cycle.

Replayed on the last cached session through the real engine on a Rs 20,000 account: 6 of 7 signals entered (1 to 29
shares each), 1 skipped with its reason, flat at the close.

**What this does not change.** The rule still has no edge. Phase 0 makes the paper account measure honestly; results before
2026-09-28 belong to orb-v1 and should not be pooled with orb-v2.

### Intraday Phase 1: one simulation path, and history that grows (2026-09-28)
- **The backtest is the live engine.** `src/research/intraday_replay.py` runs `IntradayEngine.step()` every five minutes of every
  stored session against a real `PaperBroker`, so sizing, skips, stops, targets, the daily-loss halt, the square-off, fees and
  slippage are the live code paths (the old backtest had its own simulator and ignored account size). Assumptions a replay
  cannot avoid: the bar in progress shows only its open at each cycle; fills at that open plus 0.05%; stops/targets fill at their
  level; volume is final. **Result on all 58 archived sessions, Rs 20,000, live settings: -14.3%** (380 trades, win rate 34%,
  32 targets / 106 stops / 242 squared off, 121 signals skipped (in the last 8 sessions every skip was a share dearer than the
  position cap), Sharpe -8.5, max drawdown -14.3%,
  28% profitable days, -1,463 first half / -1,399 second half). It agrees with the per-trade audit (about -0.25% a trade).
- **The rule is written once.** `first_breakout` (numpy) is used by the live engine and every research script; a test pins it
  to a plain bar-by-bar loop on random sessions and to every boundary of the rule. Vectorising it, with a leaner replay feed, made a replayed session about 3x faster (12 s to 4 s).
- **History beyond Yahoo's 59 sessions.** The engine now saves each finished session's 5-minute bars (`intraday_bars`, one
  session a day, `INTRADAY_ARCHIVE`); `scripts/intraday_archive.py` shows, backfills and imports. Nothing was invented: it only
  starts growing from now, so the tests below get stronger with time (about 125 sessions in six months).

### Intraday Phase 2: do any other ideas work? (2026-09-28)
Pre-declared in `src/research/intraday_hypotheses.py` before any result was seen, run by `scripts/research_intraday.py`: five
fixed rules plus the live rule as a control, each against a random-timing or random-side null; a candidate needs >= 300 trades,
net t >= 2.6 (per-date clustered), both halves positive, still positive at double slippage, and to beat its null (t >= 2.6).

| idea | trades | gross | net | net t |
|---|---|---|---|---|
| control: the live ORB long | 2,622 | -0.048% | -0.254% | -7.4 |
| H1 ORB short | 2,807 | -0.005% | -0.211% | -6.2 |
| H2 gap-and-go | 92 | +0.069% | -0.137% | -1.7 |
| H3 gap fade | 157 | +0.054% | -0.152% | -1.6 |
| H4 first-hour momentum | 945 | +0.032% | -0.174% | -4.4 |
| H5 last-half-hour momentum | 2,070 | +0.003% | -0.203% | -13.6 |

**No candidates: none of these ideas beats the live rule.** Every idea earns about nothing before
costs (best +0.07%) and costs are ~0.2% a round trip. The harness is not blind: a test plants a 1.2% late-day continuation and it
is flagged as a candidate, and the same market without it is not. Limits, stated plainly: 58 sessions is one market regime; with
that many dates an edge smaller than ~0.35% net per trade would not be detectable (though any edge that small could not pay
0.2% of costs either); H2 and H3 had too few trades (92, 157) to judge. Re-run `scripts/research_intraday.py` as the archive
grows, and test new ideas the same way.

## Live-data review and fixes (2026-10-01)
A read of the production database (825 decisions on 50 stock-sessions, 2026-09-25..30) found three defects in how the AI's
calls reach orders. None was visible in tests; all are fixed.
- **Stale bars.** Yahoo posts NSE daily bars hours after the close. The snapshot took whatever bar existed at the first
  fetch and cached it for the whole session, so 269 of 825 decisions (33%) were made on a bar two sessions old and then
  filled at today's price. A snapshot whose bar is older than the last completed session is now refused
  (`BarNotReadyError`), never cached, and the stock shows as WAIT until the bar is published.
- **The same bar answered twice, differently.** 22 of 50 stock-sessions flipped between BUY and HOLD. The prompt cache
  worked (identical inputs got identical answers); the flips came from process restarts, after which re-downloaded data
  differing in the 12th decimal missed the cache and the model, sampled at its default temperature, answered again
  (e.g. HOLD 0.65, then BUY 0.70). With confidence near the 0.60 gate, that is buying on a re-roll. Now each stock is
  decided once per completed bar: later cycles and restarts reuse the recorded answer (`remembered_signals`, a LangGraph
  conditional edge from fetch_data straight to collect), a fail-safe HOLD is never reused, and requests are sent at
  temperature 0 (`LLM_TEMPERATURE`).
- **Corporate actions.** The paper broker ignored splits, bonuses and dividends: a 1:1 bonus halves the split-adjusted
  price and would fire the 15% stop as a fake loss. Positions are now adjusted on the ex-date (quantity times the ratio,
  stop and target divided by it, cost basis kept) and dividends credited; the ledger reconciliation includes them.

## Can the decision memory be seeded from history? Pre-declared walk-forward test (2026-10-01): FAILED
The memory needs 30 matured observations per setup label and 60 sessions to mature, so it would stay silent until about
January 2027. `scripts/seed_setup_memory.py` rebuilt the setups it would have learned from: on the first session of every
week, the live scanner's top 15 on all NSE stocks (today's list: survivorship bias), the snapshot the agent would have
seen, and the 60-session net outcome; 6,654 setups, 2017-12..2026-06. The walk-forward then asked the real `SetupMemory`
about every setup from 2022-06 on, using only outcomes finished by that day. Gate, fixed before running: setups it rated
above the market-wide baseline must beat those it rated at or below it, with the 95% interval above zero and positive in
both halves.

| Memory's rating | Setups | Mean net 60-session return | Win rate |
|---|---|---|---|
| above baseline | 1,538 | +6.48% (±1.31%) | 55% |
| at or below baseline | 1,590 | +7.69% (±1.34%) | 55% |

Above minus below: -1.21% (95% interval -3.08% to +0.67%); first half -0.40%, second half +0.50%. **Failed**: the
memory's labels (trend shape, RSI decade, ADX and volume bands) do not separate winners from losers among the scanner's
picks. `LEARNING_SEED` stays off. This also says the live memory (`LLM_LEARNING`) is unlikely to add value once it starts
speaking; that phase is left as it is (it is silent until then) and is a candidate to switch off, the owner's call.

## Earnings-date veto: not built, no usable free data (2026-10-07)
Idea: don't buy in the days before a quarterly results announcement (a gap the price data cannot see). Checked first:
Yahoo's past results dates are missing or stop years ago for the stocks this bot trades (none for UNIPARTS or CUPID,
STLTECH ends 2021, TFCILTD 2020; only large caps such as RELIANCE are covered), so it cannot be tested on history; and
Yahoo knew the upcoming results date for 1 of the 14 stocks the bot held or ranked that day. A veto blind to 13 of 14
stocks would look like protection without being any. NSE publishes the calendar, but its terms forbid automated
collection; a results file downloaded by hand each week would be the compliant route if it is ever wanted.

## Late daily bars completed from 5-minute data (2026-10-07)
After the stale-bar fix, 8-11 stocks a day (including held positions, whose trend exit then ran ~5 hours late) waited
until Yahoo posted the previous day's daily bar at about 14:20 IST. The previous session is now built from the 5-minute
bars the bot already downloads (open first, high max, low min, close last, volume summed), only when they cover the
session to at least 15:10. Checked against Yahoo's official daily bars on 2026-10-05/06: close identical or within
0.2%, highs/lows within ~0.1%, volume ~97%.

## Bigger positions (10 x 8%) vs the live 10 x 5%: pre-declared test FAILED (2026-10-07)
Rs 20,000, 15% stop, trend exit, costs, 3.5% cash sweep. Gate fixed before running: 8% must beat 5% on return in all four
cases, with Sharpe never lower by more than 0.05 and max drawdown better than -40%.

| Case | 10 x 5% (live) | 10 x 8% |
|---|---|---|
| All NSE, full 2017-2026 | +195%, Sharpe 1.13, DD -27% | +359%, Sharpe 1.06, DD -39% |
| All NSE, last 3y | +48%, Sharpe 1.09, DD -19% | +80%, Sharpe 1.11, DD -29% |
| Nifty 500, full | +184%, Sharpe 1.12, DD -24% | +487%, Sharpe 1.23, DD -36% |
| Nifty 500, last 3y | +74%, Sharpe 1.57, DD -13% | +68%, Sharpe 1.02, DD -26% |

Larger positions mostly buy more return with more risk, not better trading: Sharpe falls in two cases, and over the
most recent Nifty 500 window 8% loses on return and risk. Sizing stays 10 x 5%.

**Owner's decision 2026-10-10: live sizing changed to 10 x 8%** (`MIN_POSITION_PCT=MAX_POSITION_PCT=0.08` in `.env`),
so that about Rs 15,000-16,000 of the Rs 20,000 is working (the 80% exposure cap still applies) instead of ~43%. This is a
chosen trade of more risk for more return, not a better strategy: the table above is the expectation (drawdowns of
-26% to -39% instead of -13% to -27%, so the 20% drawdown pause will trigger more often). Positions opened at 5% keep
their size, and **held positions are never topped up**: the risk engine approves the extra shares, but the paper broker refuses a second buy of a held stock ("position already open"), so the order is logged as SKIPPED_OPEN_ORDER. Only slots freed by an exit are refilled at 8%, so the invested share rises only as old positions leave. (An earlier version of this note said held stocks would be topped up: that was wrong. It was found independently on 2026-10-13 and again on 2026-10-15, when the account was still 44% invested after four sessions.)

## The AI's job, redesigned (2026-10-09): from second-guessing the chart to reading what the chart cannot show
**Finding (live data, 134 stock-sessions, 2026-09-25..10-08).** The rule said BUY on 116; the AI bought 39. The stocks it
rejected and the ones it bought were indistinguishable on everything it was shown (median RSI 56.4 vs 57.2, Bollinger
position 0.59 vs 0.60, 6-month momentum +120% vs +121%, volume trend -19% vs -16%), and so were the results (5-session
net +3.07% for the rule's picks it passed, +3.12% for those it bought). Its usual reasons ("weak volume", "extended")
applied equally to both. Every backtest assumed the AI approves the rule's picks; live it was a coin flip that pushed the
account down the momentum ranking. Root cause: it re-judged the same price data with no information the rule lacked, was
never told why a stock was picked, and was framed as "cautious".

**Market-regime protection, pre-declared test: FAILED.** Live account, no new buys while the Nifty 50 closes below its
200-day average. Gate: drawdown at least 3 points better and Sharpe no lower, in all four cases.

| Case | Baseline | Regime gate |
|---|---|---|
| All NSE, full | +194.6%, Sharpe 1.13, DD -27.4% | +93.1%, Sharpe 0.77, DD -25.6% |
| All NSE, last 3y | +48.4%, Sharpe 1.09, DD -18.9% | +33.3%, Sharpe 0.89, DD -15.2% |
| Nifty 500, full | +183.8%, Sharpe 1.12, DD -23.5% | +194.8%, Sharpe 1.25, DD -20.5% |
| Nifty 500, last 3y | +73.9%, Sharpe 1.57, DD -12.8% | +63.0%, Sharpe 1.61, DD -8.4% |

On all NSE stocks it halved the full-period return; weak markets are where many later winners are bought. Not adopted as a
rule, and the AI is told a weak market alone is not a reason to skip a buy.

**What changed.** One informed analysis per stock (`PROMPT_VERSION` technical-v10, `FEATURE_VERSION` indicators-v4): the
chart and the rule's verdict, the stock's 12-1 momentum (why it was picked), the company's last quarters and ratios from
Yahoo (sales, profit, year-on-year growth, debt/equity, margin, promoter holding; checked to cover every stock the bot
held or ranked), and the market situation. It follows the rule unless the company data gives a concrete, numbered reason;
its company and market checks are recorded with the data they used, so replay rebuilds the exact prompt. The veto-only
critic stays. Removed, with their settings: the decision memory (failed its walk-forward test, 2026-10-01) and its
seeding tool, the separate market-context call (now part of the one analysis), the phase-ablation and walk-forward
harnesses and the historical AI backtest (`scripts/backtest_ai_agent.py`): an AI cannot be fairly replayed on past dates
(the model has seen what happened, and Yahoo's company data is today's, not what was known then). The AI is therefore
judged forward only: the weekly Telegram report compares the rule's picks the AI bought with those it passed.

## Quality + momentum: pre-declared test FAILED (2026-10-10)
Idea: buy only momentum stocks with healthy company results. No affordable, permitted source has 10 years of Indian
quarterly results (EODHD has no NSE/BSE at all; NSE's own data may not be collected automatically), so the test used
Yahoo's yearly results (March 2023-March 2026) over 2023-06-01..2026-09-25.
- Known-at-the-time rule: a year's results count only 62 days after the year ends (SEBI's 60-day deadline).
- Quality: profit > 0, return on equity > 12%, debt/equity < 1. Applied to new buys only; ranking unchanged.
- Live account: Rs 20,000, 10 x 8%, 15% stop, trend exit, costs, 3.5% cash sweep.
- Gate (fixed before running): higher return AND higher Sharpe AND max drawdown no more than 1 point worse, on both
  universes.

| Universe | Momentum (live) | Quality + momentum |
|---|---|---|
| All NSE | +120.2%, Sharpe 1.35, DD -35.8%, 93 trades | +118.2%, Sharpe 1.34, DD -25.9%, 88 trades |
| Nifty 500 | +134.9%, Sharpe 1.46, DD -28.9%, 81 trades | +114.2%, Sharpe 1.41, DD -26.0%, 91 trades |

About half of all candidate-days passed the quality rule. It lowered the worst fall (by 10 points on all NSE) but did
not raise return or Sharpe in either universe. **Not adopted.** The drawdown effect is noted, not acted on: it was not
the pre-declared goal, and 3 years with today's listed stocks only is a short, survivorship-biased sample.

## Intraday on 2 years of data: no rule beats its costs (2026-10-10)
A community dataset of 1-minute bars (214 F&O stocks, 2024-04-01..2026-04-30, 516 sessions; no licence file, kept local
and uncommitted in `research_cache/`) was checked against Yahoo: every regular session complete (375 bars), daily volume
99% of Yahoo's, closes within a median 0.14% (NSE's official close is a 30-minute average). Converted to 5-minute bars:
109,059 stock-sessions, 19x the archive. Same pre-declared criteria as Phase 2; detectable edge about 0.15-0.21% a trade.

| idea | trades | gross | net | net t | vs null t |
|---|---|---|---|---|---|
| control: the live ORB long | 37,390 | +0.028% | -0.178% | -13.2 | -1.5 |
| H1 ORB short | 38,167 | +0.021% | -0.185% | -13.3 | -4.1 |
| H2 gap-and-go | 2,648 | +0.157% | -0.049% | -2.2 | -0.1 |
| H3 gap fade | 6,111 | -0.002% | -0.208% | -1.2 | 3.2 |
| H4 first-hour momentum | 34,332 | +0.056% | -0.150% | -6.8 | 2.4 |
| H5 last-half-hour momentum | 54,397 | +0.029% | -0.177% | -26.3 | 3.3 |
| H6 ORB in 12-1 momentum leaders (new) | 6,713 | +0.078% | -0.128% | -8.7 | -0.7 |
| H7 ORB held to the close (new) | 37,390 | +0.033% | -0.173% | -12.8 | -1.5 |
| H8 one trade a day, strongest volume (new) | 513 | -0.027% | -0.233% | -4.2 | -4.1 |

H6-H8 were declared before they were run. **No candidates.** The 3-month result holds over 2 years and both halves: the
live rule's breakout is no better than a random entry time (null t -1.5), and every idea's gross edge (best: gap-and-go
+0.16%) is smaller than the ~0.21% a round trip costs (fees + 0.05% slippage a side). H6's better gross comes from the
leaders drifting up, not from the breakout (it does not beat its own null). The intraday engine keeps running and
recording; what would have to change is the cost per trade or a gross edge above it.

### Machine-learning filter for the intraday ORB rule: pre-declared test FAILED (2026-10-10)
Every live ORB long signal in the 2-year dataset (37,390) labelled with its net return under the live exits; 16 features
known at the entry bar (range width, volume ratio, entry time, gap, breadth, breakout strength, stop distance, VWAP
distance, day and market move so far, the stock's 1-day/5-day return, 12-1 momentum, 20-day volatility, Nifty vs its
200-day average, weekday). One fixed model (sklearn HistGradientBoostingRegressor, never tuned on the test year), trained
on 2024-04..2025-03, April 2025 left out, judged once on 2025-05..2026-04 (18,738 candidates). Rule: take a trade only when
the predicted net return is > 0.

Result: 2,164 trades selected, net -0.211% (all candidates -0.180%), both halves negative, paired t vs taking everything
0.3. The predictions carry no information out of sample: rank correlation with the actual result -0.02, and the decile the
model liked best (+0.13% predicted) made -0.20%. What it learned from the first year did not repeat in the second. **Not
adopted.** The features that describe a breakout do not separate good breakouts from bad ones here, which matches the
audit and the hypothesis tests.

### Searching for a better intraday signal: explore on year 1, confirm on year 2 (2026-10-10)
Pre-declared: 19 variants from five families with a stated reason (gap-and-go by gap size, leaders-only and exit; short-term
reversal; cross-sectional intraday momentum; market intraday momentum; momentum-leader drift), explored freely on
2024-04..2025-03; a fixed rule picked the one with the best date-clustered t among those with >= 300 trades and a positive
net; that one alone was run once on 2025-05..2026-04 with the Phase 2 criteria.

Year 1: reversal, cross-sectional momentum, market momentum and leader drift all lost more than the live rule (net -0.19% to
-0.35%). Only gap-and-go was positive after costs, and more so for bigger gaps (gap >= 1%: -0.08%, >= 2%: +0.03%, >= 3%:
+0.16%; leaders-only higher still but on 40-114 trades). Selected: **gap up >= 2%, first close above the opening range
in the first hour, stop at the range low, held to 15:15**.

Year 2 (once): 345 trades, gross +0.297%, **net +0.091%**, both halves positive (+0.06% / +0.11%), but date-clustered t
-0.2, -0.009% at doubled slippage, and no better than a random entry time on the same stocks (paired t -0.7). **Fails.**
It is the first idea to stay positive after costs in both years, but on 79 trading days an edge below ~0.55% a trade cannot
be told apart from luck, and the null test says the gain comes from big-gap stocks drifting up that day, not from the
breakout timing. A lead to watch forward, not a rule.

### Gap-and-go added as a second intraday PAPER strategy (2026-10-10, gapgo-v1)
On the owner's decision, the near miss above runs forward in paper, never with real money: gap up >= 2% over the previous
session's close, then the first 5-minute close above the opening range high between 09:30 and 10:30; stop at the range
low, no target, squared off at 15:15. Its own Rs 20,000 paper account (`GAPGO_DATABASE_URL`, `gapgo.db`), the same
universe and INTRADAY_* sizing as the breakout engine, stepped in the same process; Telegram `/gapgo`, `/gapgo_history`;
every signal recorded with system version `gapgo-v1`. The rule is one function (`strategy.first_gap_breakout`) used by
the live engine and the research test (re-run through it: identical trades and results). The breakout engine is unchanged.
Judge it on its own forward record: the 2-year test had ~1.4 signals a day across 214 stocks, so on the ~100-stock live
universe expect roughly one every one or two days; months are needed before the record means anything.

### Gap-only control vs gap-and-go: timing or selection? (2026-10-10)
Pre-declared control: the same gap filter (>= 2%), entry at the 09:30 close with no breakout needed, the same stop (range
low), held to the square-off. 2 years, 214 stocks:

| rule | trades | gross | net | net t | 2x slippage |
|---|---|---|---|---|---|
| gap-and-go (breakout by 10:30) | 750 | +0.294% | +0.087% | +0.6 | -0.013% |
| gap-only (enter 09:30) | 2,806 | +0.099% | -0.107% | -2.0 | -0.207% |

Per trade, the breakout condition looks like it matters: gap stocks that never broke out lost -0.53% a trade under
gap-only. But compared date by date on the 211 dates both traded, gap-only did better (-0.22% a day for gap-and-go,
paired t -2.0): on days with breakouts, buying every gap at 09:30 caught the same winners earlier and cheaper. So the
gap-and-go edge seems to come mostly from staying out on days when gaps fail, not from better entries on the days it
trades. The earlier sentence "the gain comes from big-gap stocks drifting up" was too simple: the evidence is mixed and
not conclusive either way, which is why the control is worth measuring forward.

### The live engine replayed on 2 years, Rs 20,000 account (2026-10-10)
`scripts/backtest_intraday.py --bars-file research_cache/intraday_5m_2y.pkl` (live engine + paper broker, INTRADAY_*
sizing, intraday fees, 0.05% slippage a side, 2% daily-loss halt, 5 positions):

| strategy | trades | win rate | fees | Rs 20,000 -> | max drawdown | 1st / 2nd half P&L |
|---|---|---|---|---|---|---|
| ORB (live rule) | 3,330 | 38% | Rs 6,743 | Rs 6,111 (-69%) | -71% | -9,454 / -4,436 |
| gap-and-go | 337 | 46% | Rs 1,076 | Rs 20,170 (+0.9%) | -8.4% | +367 / -197 |

Execution assumptions checked trade by trade on 3 months (432 ORB trades): never more than 5 positions open; no position
above 20% of equity or risking more than 0.5%; every entry exactly the next bar's open + 0.05% and every square-off the
15:15 open - 0.05%; fees equal the intraday schedule on all trades (0.106% of value on Rs ~3,200 positions); no trade
before a stock's first bar; 97 signals skipped because one share cost more than the 20% cap. The replay's per-trade loss
(-0.23%) is a little worse than the research estimate (-0.18%) because the live engine enters at the NEXT bar's open,
not the signal bar's close. Remaining bias: the 214 stocks are today's F&O list (survivorship), wider than the live
top-100 universe. At account level gap-and-go is about break-even (positive first year, negative second): the +0.09% a
trade of the research does not survive next-bar entries on a Rs 20,000 account. The forward paper comparison with
gap-only (below) is how it will be judged.

### Gap-only control added (2026-10-10, gaponly-v1)
Owner's decision, paper only: its own account (`GAPONLY_DATABASE_URL`, `gaponly.db`, Telegram `/gaponly`). Frozen rules
in `src/intraday/strategy.py`: a gap >= 2% bought on the close of the 09:25-09:30 bar on the 09:30 cycle only (never
chased later), stop at the opening-range low, no target, squared off at 15:15; skipped when the previous close is
unknown, the opening range is not the real 09:15-09:30 bars, or the entry is not above the stop. Same universe, sizing
and fee model as gap-and-go. Both gap accounts record the signals a full position limit made them skip. The breakout
engine and the swing bot are unchanged. Judged by `scripts/compare_gap.py`: gap-and-go is credited with a timing edge
only if it beats gap-only per trade AND date by date (paired t >= 2.6); the report also shows the same stock on the same
day, fees, daily return, drawdown and how much of each account was used.
