# Execution safety: what is guaranteed, and by what

**There is no live execution in this codebase.** Every order goes to `PaperBroker`, a simulator with its own ledger. The
flag that enables orders was called `--live` (it still works as an alias); it is now `--paper-orders`, because "live" meant
"place paper orders", never "trade real money". The only code that reads a real-money account is the Integrated portfolio
agent, which is read-only and walled off (below).

Each statement here names its enforcement. Where the enforcement is a test, that test fails if the rule is broken.

## The path from an AI answer to an order

```
AI call → AgentSignal (validated JSON) → combine_signals → trend-exit override → RiskEngine.evaluate → _place → PaperBroker
```

1. **The AI cannot place an order.** Its answer is a validated `AgentSignal`. A failed, malformed, truncated or unavailable
   answer is replaced by a `degraded` HOLD with confidence 0 (`fail_safe_hold`); a degraded signal is never reused.
   (`test_llm_resilience`, `test_agent_analysis`, `test_cycle_services`)
2. **The risk engine has the last word on whether and how much.** `RiskEngine.evaluate` refuses: HOLD or an unknown action,
   an invalid price, confidence below `MIN_CONFIDENCE` (0.60), a SELL with no position (no shorting), a halt (daily loss,
   drawdown), more than `MAX_OPEN_POSITIONS`, no room under the position size or exposure cap or cash, and a buy whose
   round-trip fees exceed `MAX_FEE_DRAG_PCT` of the position. (`test_risk_engine`, `test_characterization`)
3. **A deterministic exit overrides the AI.** A held stock whose last completed close is below its 200-day average is sold,
   whatever the AI said. (`test_characterization`)
4. **`_place` adds its own gates**, in this order: paused → re-buy cooldown (24 h) → dry run → market closed → an existing
   position or open order. Each outcome is recorded as an order row with its status. (`test_enums`, `test_characterization`)
5. **Research and learning cannot change what trades.** Production packages never import `research`, `ops` or `learning`
   (`test_layering`); research code reads history and never writes to the production databases (the holdout reserve, in `ops`, appends only to its own `holdout_marks` table). Settings change only through the server's
   `.env`, which a deploy does not overwrite (`test_deployment`).

## Invariants

| Invariant | Enforced by |
| --- | --- |
| Refactors do not silently change trading rules | `test_characterization` (nine full-cycle situations pinned: results, broker calls, audit rows, alerts); `test_prompt_versions` (every AI prompt and schema fingerprinted) |
| Live and paper stay separate | There is no live broker. The real-account agent imports nothing from the trading code and only `app/portfolio_agent.py` imports it (`test_layering`); a network guard aborts any order-like request from its browser (`test_portfolio_portal`) |
| A duplicate event cannot create a duplicate order | One order per decision and one signal per stock per session are **database** constraints (`ux_orders_decision`, `ux_intraday_signals_day_symbol`; `test_database_integrity`); one decision per stock per bar is reused across cycles and restarts (`test_llm_and_pipeline`); one position row per symbol, and a held stock cannot be bought again (`test_paper_broker`) |
| Every order and trade is traceable | order → `decision_id` → the decision row holds the prompt, features and strategy versions, the snapshot and the model that answered; trades keep their open and close times, reason and fees (`test_signal_details`, `test_prompt_versions`); `scripts/replay_decision.py` rebuilds the exact prompt a decision saw |
| The ledger ties out | `scripts/reconcile_paper.py` rebuilds cash from recorded fills and checks that every approved decision has an order and every fill landed in a position or trade; it exits non-zero on drift (`test_reconciliation`, `test_multi_day`) |
| Sizing respects capital and limits | The risk engine takes the smallest of position room, exposure room and cash; intraday sizing is the smaller of a risk budget (0.5% of equity at the stop) and a position cap (20%), and a share dearer than the cap is skipped, with the reason recorded (`test_risk_engine`, `test_intraday_strategy`) |
| Fees, slippage and P&L are consistent | One fee schedule per kind in `engine/costs.py` (delivery, intraday); the paper broker and every backtest charge the same schedule and 0.05% a side; the live engine replayed over history uses the live code path (`test_execution_convention`, `test_paper_broker`, `test_intraday_replay`) |
| Replay uses no future information | Indicators at date t use bars up to t only; a signal at the close fills at the next open (`test_no_lookahead`); the AI is not backtested on past dates at all, because it knows how those dates ended |
| Corporate actions do not corrupt positions | Splits, bonuses and dividends are applied to held positions once; a price-scale break is recognised as a split lookalike, not a crash (`test_paper_broker`, `test_data_honesty`) |
| Stale data is never traded | A snapshot older than the last completed session raises `BarNotReadyError`; the stock shows as WAIT (`test_hardening`) |

## If a real broker is ever added

This section is a requirement list, not a description. Nothing below exists today.

- It implements the `Broker` protocol in `engine/ports.py` and lives in its own package that production code reaches only
  through `app/wiring.py`.
- Order status must be confirmed with the broker before any retry (the old `ConfirmsFills` hook was removed because the
  simulator has nothing to confirm; a real broker brings it back). A timeout is an **unknown** state: reconcile, then decide.
- Idempotency keys on every submit, partial-fill handling, a protective stop that is verified to exist, and a kill switch
  that does not depend on the strategy process.
- Failure-mode tests (timeouts, duplicates, partial fills, ambiguous status, restart mid-order) **before** the first order,
  and an explicit, separately reviewed decision, never a flag flipped inside a cleanup.

## Known gaps

- Database failures during a cycle (disk full, locked file) are caught at the cycle boundary and alerted, without a dedicated test.
- The `decisions` table has no uniqueness rule (history holds repeated keys), so duplicate *decisions* are prevented in code
  only, not by the database.
- SSH access is controlled by a security-group allow list that must be edited whenever the owner's IP changes.
