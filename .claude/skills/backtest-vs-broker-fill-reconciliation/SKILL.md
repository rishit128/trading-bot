---
name: backtest-vs-broker-fill-reconciliation
description: 'Use this skill whenever real historical fills from a broker or exchange account are available for the same instrument a backtest trades, to directly compare simulated fills against real ones rather than relying solely on theoretical fill-feasibility checks. Also use it during and after a strategy''s paper-trading or early live period to validate that the backtest''s fill assumptions actually matched reality. This skill exists because every other check in this repo''s fill-realism skills is still a theoretical model of what a fill should have looked like; comparing against actual real fills is the only way to check whether that model is correct, not just internally consistent.'
---

# Backtest vs. Broker Fill Reconciliation

## Why this matters

[fill-feasibility-and-impossible-fill-detection](../fill-feasibility-and-impossible-fill-detection/SKILL.md) and [order-book-and-intrabar-price-path-modeling](../order-book-and-intrabar-price-path-modeling/SKILL.md) both build better theoretical models of what a realistic fill should look like, but a theoretical model, however carefully built, is still a model; the only direct way to check whether it's actually correct is to compare it against real fills that genuinely happened. When real historical fill data exists, whether from the strategy's own paper/live trading or from a broker's historical execution data for the same instrument, reconciling the backtest's simulated fills against those real fills is a direct empirical check that no amount of purely theoretical fill modeling can substitute for.

## Do this

### Run the exact same historical period and orders through both the backtest's fill logic and real broker/execution data, and compare directly

Where real historical fills exist for the instrument being traded (from the strategy's own paper-trading period per [paper-trading-and-forward-test-protocol](../paper-trading-and-forward-test-protocol/SKILL.md), or from any other real execution history for the same instrument and similar order types), take the exact same orders and timestamps and compare the backtest's simulated fill price and size against what was actually achieved in reality. This is a direct empirical test of the backtest's fill assumptions, not an inference from theory about what should have happened.

### Quantify the systematic bias between simulated and real fills, not just whether they're "close enough" on average

Compute the actual distribution of differences between simulated and real fills (not just an average difference, which can hide a strategy where the backtest is unrealistically favorable specifically on the trades that matter most, like larger or more urgent orders). Check whether the bias is systematic in one direction (the backtest consistently assumes better fills than reality delivers) rather than randomly distributed around zero, since a systematic bias in either direction indicates the fill model needs correcting, not just accepting as noise.

### Break down reconciliation results by order type, size, and market condition, not just as one aggregate comparison

The backtest's fill model may be accurate for small, passive orders in calm conditions while being significantly wrong for large or urgent orders during volatile conditions; aggregate reconciliation statistics can hide this kind of conditional inaccuracy. Segment the comparison by order type, relative size, and market conditions at the time of the order, and specifically check whether the fill model's accuracy holds up in the conditions the strategy actually depends on most.

### Feed reconciliation findings back into the backtest's fill model and cost assumptions, then re-validate

When reconciliation reveals a systematic gap between simulated and real fills, that finding should directly update the backtest's fill logic and the cost assumptions in [transaction-cost-and-market-impact-modeling](../transaction-cost-and-market-impact-modeling/SKILL.md), and the strategy's core validation (per [overfitting-and-robustness-checks](../overfitting-and-robustness-checks/SKILL.md)) should be re-run with the corrected assumptions rather than leaving the original, now-known-to-be-inaccurate backtest results as the reference point for the strategy's expected performance.

### Repeat reconciliation periodically as market conditions and liquidity evolve, not just once

A fill model validated against real fills from one period and one set of market conditions can become stale as liquidity, volatility, or market structure for the instrument changes over time. Repeat the reconciliation process periodically (informed by [live-vs-backtest-drift-monitoring](../live-vs-backtest-drift-monitoring/SKILL.md)'s broader monitoring cadence) rather than treating a single historical reconciliation as a permanent validation of the fill model.

## Never do this

- Never rely solely on theoretical fill-feasibility modeling when real historical fills for the same instrument are available to check it against directly.
- Never evaluate reconciliation results using only an average difference between simulated and real fills; check the full distribution and look specifically for systematic, directional bias.
- Never accept an aggregate "close enough" reconciliation result without checking whether accuracy holds up conditionally, by order type, size, and market condition.
- Never leave a backtest's fill model and cost assumptions unchanged after reconciliation reveals a systematic gap against real fills; update them and re-validate the strategy.
- Never treat a single reconciliation exercise as a permanent validation; repeat it periodically as market conditions and liquidity evolve.

## Verification checklist

- [ ] The backtest's simulated fills have been directly compared against real historical fills for the same instrument and similar orders, not validated only through theoretical modeling
- [ ] The full distribution of differences between simulated and real fills has been examined for systematic, directional bias, not just an average difference
- [ ] Reconciliation results are broken down by order type, size, and market condition, checking specifically whether accuracy holds up under the conditions the strategy depends on most
- [ ] Any systematic gap found during reconciliation has been fed back into the backtest's fill model and cost assumptions, with the strategy re-validated under the corrected assumptions
- [ ] Reconciliation is repeated periodically as market conditions and liquidity evolve, not treated as a one-time, permanent validation
