---
name: live-vs-backtest-drift-monitoring
description: 'Use this skill whenever a strategy is trading live or in paper mode, to build the actual monitoring that compares live performance against backtest expectations in real time. Also use it when a user asks "how do I know if my strategy is still working" or wants to catch a problem before it becomes a large loss. This skill exists because the gap between a backtest and a live implementation, whether from a genuine edge decaying, an implementation bug, or a data/environment mismatch, does not announce itself; without deliberate monitoring built to detect it, that gap is only discovered after it has already cost real money.'
---

# Live vs. Backtest Drift Monitoring

## Why this matters

A backtest predicts a range of expected outcomes; live trading produces one actual, real-time realization of results. Whether that live realization is behaving consistently with the backtest's predicted distribution is a question that needs an actual, running answer, not an occasional manual glance at a P&L chart. A strategy can drift away from its backtest expectations for several different reasons, a genuinely decaying edge, a subtle implementation bug that only shows up live, a live data feed quietly differing from the backtest's data source, and none of these show up as an obvious error; they show up as results that are gradually or suddenly worse than expected, and by the time that's obvious from casual observation, real money has usually already been lost.

## Do this

### Build an explicit statistical comparison between live results and the backtest's expected distribution, not a visual gut check

Use the resampled distribution from [monte-carlo-trade-resampling-and-drawdown-distribution](../monte-carlo-trade-resampling-and-drawdown-distribution/SKILL.md) as the actual reference: as live trades accumulate, check whether live performance is statistically consistent with that resampled distribution (is the live equity curve tracking within a normal range of the resampled paths, or has it fallen outside a reasonable percentile band), rather than relying on a visual impression of whether the live equity curve "looks okay."

### Track and alert on trade-by-trade divergence between the strategy's intended signal and what actually happened

For every live trade, log what the strategy's logic intended to do, the size and price it expected, against what actually happened (the fill price, size, and timing actually achieved), and alert when the divergence between intended and actual exceeds a defined threshold. This catches [execution-and-order-management-realism](../execution-and-order-management-realism/SKILL.md) failures (a position that should have closed didn't, a fill materially worse than expected) as a distinct signal from the strategy's underlying edge decaying, which matters because the appropriate response to each is different.

### Set explicit, predefined thresholds for what triggers investigation or an automatic pause, decided before going live

Decide in advance, not in the moment, what specific deviations from backtest expectations (a drawdown exceeding a specific percentile of the resampled distribution, a live Sharpe falling below a specific threshold over a defined rolling window, a specific number of consecutive losses beyond what the resampled distribution suggests is normal) trigger a required human review or an automatic reduction in position size or trading pause. A threshold decided calmly in advance, and enforced automatically, is far more reliable than a discretionary decision made while already in the middle of a live loss.

### Separately monitor the data and environment the live strategy runs on, not just its outputs

A live strategy can drift from backtest expectations because its live data feed differs subtly from its backtest data source (different vendor, different latency, different corporate-action handling), or because its live software environment differs from the one the backtest ran in (a library version difference that changes a calculation's result, a timezone handling difference). Monitor the live data feed and environment for consistency with what the backtest assumed, as a separate check from monitoring the strategy's trading outputs, since a data/environment mismatch is a different root cause than a genuinely decaying edge and needs a different fix.

### Distinguish a genuinely decaying edge from a temporarily unlucky stretch before reacting

Not every live drawdown or underperforming stretch means the edge is gone; [monte-carlo-trade-resampling-and-drawdown-distribution](../monte-carlo-trade-resampling-and-drawdown-distribution/SKILL.md)'s resampled distribution exists specifically to give a sense of what a normal unlucky stretch looks like versus a result that's genuinely outside plausible variation. Use that distribution, and an accumulating live track record, to distinguish the two rather than reacting to every drawdown as though it definitively proves the edge is gone, while still respecting the predefined thresholds that require action regardless of which explanation feels more likely in the moment.

## Never do this

- Never monitor live performance only by occasional visual inspection of an equity curve; build an explicit statistical comparison against the backtest's resampled expected distribution.
- Never treat a strategy's trading outputs as the only thing worth monitoring; separately track trade-by-trade divergence between intended and actual execution, and separately monitor the live data/environment for consistency with backtest assumptions.
- Never decide drawdown or underperformance thresholds for pausing or investigating a live strategy in the moment, while a real loss is happening; decide them in advance and enforce them automatically.
- Never treat every live drawdown as proof the edge has decayed, or conversely, dismiss every drawdown as normal variation; use the resampled distribution to judge which explanation the actual data supports.

## Verification checklist

- [ ] Live performance is compared against the backtest's resampled expected distribution through an explicit statistical check, not a visual gut check
- [ ] Trade-by-trade divergence between intended signal/size/price and actual fill outcome is logged and alerted on separately from overall P&L monitoring
- [ ] Explicit, predefined thresholds for required review or automatic pause/de-risk were decided before going live, not improvised during a live drawdown
- [ ] The live data feed and software environment are monitored for consistency with backtest assumptions, as a distinct check from monitoring trading outputs
- [ ] Drawdowns and underperforming stretches are evaluated against the resampled distribution to judge whether they're plausible normal variation or a genuine departure, rather than reacted to purely on instinct
