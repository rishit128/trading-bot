---
name: paper-trading-and-forward-test-protocol
description: 'Use this skill after a strategy passes backtest validation and before any real capital is committed to it, to define how long and how rigorously it needs to run in paper/forward-test mode first. Also use it when a user asks "how long should I paper trade before going live" or is tempted to skip forward testing because the backtest already looked strong. This skill exists because a backtest, however rigorously validated, is still a test against historical data the strategy''s logic was built with knowledge of; a forward test on data that didn''t exist yet when the strategy was built is the first genuinely prospective evidence the strategy gets, and skipping it trades a real, avoidable check for impatience.'
---

# Paper Trading and Forward-Test Protocol

## Why this matters

Every check in [overfitting-and-robustness-checks](../overfitting-and-robustness-checks/SKILL.md), [statistical-significance-and-deflated-metrics](../statistical-significance-and-deflated-metrics/SKILL.md), and [monte-carlo-trade-resampling-and-drawdown-distribution](../monte-carlo-trade-resampling-and-drawdown-distribution/SKILL.md) still operates on historical data that existed and was knowable when the strategy was built. A forward test, generating real signals on data that genuinely didn't exist yet when the strategy's rules were finalized, is qualitatively different: it's the first test that can't have been influenced, even unconsciously, by hindsight. Skipping straight from a strong backtest to live capital skips the one validation step that's actually immune to every form of overfitting the earlier steps try to detect after the fact.

## Do this

### Fully finalize the strategy's rules before the forward test begins, and don't change them mid-test

The forward test's value comes specifically from the strategy's rules being fixed and unchanged while it runs; adjusting rules partway through a forward test based on how it's going reintroduces exactly the exploratory-bleeding-into-confirmatory problem [hypothesis-driven-strategy-design](../hypothesis-driven-strategy-design/SKILL.md) warns about, just at a later stage. If the strategy needs a rule change during the forward test, that change should restart the forward-test clock on the revised version, not be treated as a continuation of the same test.

### Set the forward-test duration based on getting a meaningful sample of trades, not a fixed calendar period regardless of strategy frequency

A strategy that trades daily needs a much shorter calendar duration to accumulate a meaningful trade sample than one that trades monthly; set the forward-test length based on accumulating enough trades to say something statistically meaningful (informed by the same trade-count-vs-parameter-count reasoning in [overfitting-and-robustness-checks](../overfitting-and-robustness-checks/SKILL.md)), not an arbitrary fixed number of weeks that might be far too short for a slow strategy or unnecessarily long for a fast one.

### Compare forward-test results against the backtest's resampled expected distribution, using the same statistical framework as live monitoring

Use [live-vs-backtest-drift-monitoring](../live-vs-backtest-drift-monitoring/SKILL.md)'s approach during the forward test itself: check whether forward-test performance falls within a plausible range of the backtest's resampled distribution, rather than judging it by whether it simply made money over the test period. A forward test that loses money but stays within the resampled distribution's normal range is a materially different (and less concerning) result than one that's statistically inconsistent with what the backtest predicted, even if the dollar outcome looks similar.

### Run the forward test with the same execution mechanics the live strategy will actually use

A forward/paper test that simulates fills at a theoretical price, ignoring the real execution mechanics from [execution-and-order-management-realism](../execution-and-order-management-realism/SKILL.md), isn't actually testing the thing that will eventually trade real capital. Where possible, run the forward test through the actual broker connection in paper mode (using a real sandbox/paper account rather than an internal simulation), so it captures real API behavior, real fill mechanics, and real data feed characteristics, not just the strategy's decision logic in isolation.

### Decide the go/no-go criteria before the forward test starts, not after seeing how it went

Define in advance what forward-test outcome constitutes "proceed to live capital," "extend the test," or "the strategy fails and doesn't go live," using the statistical-consistency framework above rather than a vague, after-the-fact judgment call. Deciding these criteria after already seeing a forward test's results allows exactly the same hindsight-driven rationalization that makes overfit backtests look convincing in the first place.

## Never do this

- Never change a strategy's rules mid-forward-test and treat the result as a continuous test of one strategy; a rule change restarts the clock.
- Never set forward-test duration as an arbitrary fixed calendar period without considering whether it will actually accumulate a meaningful number of trades for the strategy's actual trading frequency.
- Never judge a forward test purely by whether it made money over the period; compare it against the backtest's resampled expected distribution using the same framework as ongoing live monitoring.
- Never run a forward test using simplified or theoretical fill assumptions when a real paper/sandbox broker connection is available to test actual execution mechanics.
- Never decide the go/no-go criteria for moving to live capital after already seeing the forward test's results.

## Verification checklist

- [ ] The strategy's rules were fully finalized and unchanged for the entire duration of the forward test that's being used as evidence
- [ ] The forward-test duration was set based on accumulating a meaningful number of trades for this strategy's actual frequency, not an arbitrary fixed calendar period
- [ ] Forward-test results were evaluated against the backtest's resampled expected distribution, not just judged by whether the test period was profitable
- [ ] The forward test ran through real paper/sandbox broker execution mechanics where available, not only a theoretical fill simulation
- [ ] Go/no-go criteria for proceeding to live capital were decided and documented before the forward test began, not after seeing its results
