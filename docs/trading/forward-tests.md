# Forward-test protocol for the paper accounts

Fixed 2026-10-26, before the first frozen session (2026-10-28; the bots were restarted on the new versions mid-session on
10-27, so that day is history). Enforced by `scripts/forward_test_report.py`. Written
after a review against the skills `paper-trading-and-forward-test-protocol`, `live-vs-backtest-drift-monitoring`,
`backtest-vs-broker-fill-reconciliation`, `ml4t-deflated-sharpe` and `ml4t-meta-labels` (skills.sh: kryhr/26-quant-trading-skills, ml4t/skills; not kept in the repo).

## Why
Until now every account's rules changed during its paper run (orb-v3 on 10-15, the 500-stock universe on 10-19, entry
checks on 10-23 and 10-26; news v1 to v3 in five days; swing-v3, technical-v13 and new entry checks within ten days), and
some changes went live without a version bump. A paper result that mixes rules is not evidence for any of them.

## Rules
1. **Frozen versions.** orb-v4, gapgo-v2, gaponly-v2, news-v3, swing-v5, from 2026-10-28. Trades before that are history,
   not evidence.
2. **Any rule change** (entry, exit, sizing, universe, a new skip) bumps its version in `src/versions.py` in the same
   commit, and moves `FREEZE` in the report: the test restarts. The report refuses to judge a test that mixes versions.
   Data-source and logging changes are not rule changes, but are noted in STRATEGY.md with their date.
3. **Length is set by trades, not the calendar.** The test needs the number of trading dates that 80% power requires to
   see the target edge (+0.25% net a trade intraday, +3% a trade swing) from its own spread, and at least 30 dates.
4. **Verdicts** (decided now, not after the results):
   - GO: mean net > 0, date-clustered t >= 2, both halves > 0. Eligible for a small real-money trial, nothing more.
   - REDESIGN: enough dates and not GO. The frozen rule has no forward edge; design a new version and restart. The
     intraday paper accounts keep running in every case.
   - INVESTIGATE (any time): live worse than its backtest by z <= -2, or more than 10% of verified entries gone on the
     final bars. Run `scripts/reconcile_paper_replay.py` first: the result may be the plumbing's, not the rule's.
5. **Research results are read against every variant tried.** `docs/trading/research-trials.csv` lists the intraday
   variants tested so far (69 on 2026-10-26); add each new one before running it. A candidate's t is reported with its
   deflated probability (`src/research/significance.py`, `deflated_t`).

## Known limits no code fixes (data the project does not have)
- **Point in time:** results dates come from Yahoo's current calendar, not what was known on the day; 46% of its times fall
  on exact half-hours inside trading hours. The results-day lead rests on it. `news.db` now records exchange filings with
  their own timestamps going forward; history needs a licensed source.
- **Survivorship:** the swing backtests use today's listings; the intraday research used a top-500 list frozen at the end
  of the period. No point-in-time index membership or delisted-stock history is available for free.
- **Intrabar order:** 5-minute bars cannot show whether a stop or a target came first (the broker assumes the stop). It
  did not matter in the reconciliation (0 of 46 exits), so tick data is not worth buying for it.
- **Intraday history:** `research_bars.db` was deleted on 2026-10-21; tests beyond the archive's sessions need
  the Angel One downloader rebuilt and run again (hours; `scripts/angel_history.py` was removed with the data and is
  not in git).
