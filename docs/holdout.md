# Holdout reserve account

## Why it exists

Everything in this repo that *learns* or *tunes* — the learning phase, calibration, reconciliation,
the ablation and walk-forward harnesses — is judged against the paper account's own outcomes. If those
decisions are then graded on the same data, the validation is self-referential. The reserve is the
escape: **an account that is never tuned, never reads the paper account's order book, and only ever
answers one question — "did the recorded decisions themselves make money?"**

## What it is

A re-run of the stored **approved** `decisions` (BUY, risk-approved, `final_confidence` ≥
`MIN_CONFIDENCE`) through the same deterministic simulator every backtest uses
(`src.backtest.simulate`), in the exact D1/D2 conventions: signal on day T, fill at T+1 open with
adversarial slippage, bracket exit levels off the signal-day close, `india_delivery_fees` on both
sides. It seeds from the paper account's own starting cash so returns are directly comparable. The
decision list is point-in-time (each decision was made before the bars it trades); nothing is fitted
or read back from live broker state.

Each run appends one row to `holdout_marks` — a timestamped, independent equity history.

## The flag

`main.py --holdout` (with or without `--live`) replays the decisions after every cycle and prints the
reserve's equity, return, closed trades and open positions. The mark is written to `holdout_marks` in
the same database; the paper account is never modified.

Because a reserve that is tuned is no reserve at all:

- the reserve only ever reads `decisions` + daily bars; it never writes positions or trades, and
- no learning/tuning path consumes `holdout_marks`; it exists to be *read* after a phase changes.

## Rules of the road

| Do | Don't |
| --- | --- |
| Run `--holdout` every cycle while paper-trading | Tune or recalibrate *from* the reserve |
| Judge a model/phase change against the reserve first | Judge it against the paper account that trained it |
| Zero marks when nothing has been decided yet is fine | Let a symbol the reserve cannot price be guessed or dropped silently |
| Compare reserve equity to the paper account's `last_mark` for **drift** | Expect the two to be identical — the paper account reflects intraday stop settlement and the reserve reflects daily next-open fills |

## Reading a drift

The paper account settles tight 5-minute bars; the reserve only sees daily bars. A gap between the two
is expected and is itself information: when `paper < reserve` after a phase, the intraday execution
costs (or a bad intraday exit) ate the decision edge. When `reserve < paper`, the simulation is too
optimistic somewhere (a slippage/stop assumption to revisit). Trending drift, not a single cycle, is
the signal.

## Offline behaviour

The reserve needs daily bars per symbol, fetched on demand (`fetch_bars`). Offline (or a pure backtest
universe), hand it a synthetic/supplied frame per symbol, or any symbol without bars is simply "priced
out" of that mark — never guessed.