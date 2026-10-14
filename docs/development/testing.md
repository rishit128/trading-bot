# Testing

About 900 tests, all offline (no network, no API keys), 93% line coverage (`pytest --cov`, gate 90%). The files are flat in
`tests/`; the table groups them by what they protect. Run one group with `pytest tests/test_x.py`.

## What each group protects

| Layer | Files | Protects |
| --- | --- | --- |
| **Unit: rules and money** | `test_rules`, `test_risk_engine`, `test_execution_convention`, `test_paper_broker`, `test_stop_protection`, `test_intraday_strategy`, `test_enums` | Position sizing, risk limits and halts, fees and slippage, stop and target fills, the buy/sell rules |
| **Unit: data and indicators** | `test_rich_indicators`, `test_data_and_strategy`, `test_data_honesty`, `test_india_data`, `test_universe`, `test_price_history`, `test_intraday_bars`, `test_delivery`, `test_architecture` | Indicators, the screener, the split/bonus lookalike guard, cached prices and archives, completed-session snapshots and India session logic, how agent signals combine by role, additive migrations |
| **Integration** | `test_database`, `test_database_integrity`, `test_workflow`, `test_cycle_services`, `test_llm_and_pipeline`, `test_multi_day`, `test_intraday_engine`, `test_gap_and_go`, `test_monitoring`, `test_reports`, `test_bootstrap`, `test_cli` | Real SQLite databases, the LangGraph cycle with a fake AI, a multi-day paper run, the engines against the real paper broker, Telegram commands, the wiring |
| **Regression (behaviour pinned)** | `test_characterization` (nine full-cycle situations: results, broker calls, audit rows, alerts), `test_prompt_versions` (fingerprint of every AI prompt and schema), `test_no_lookahead`, `test_intraday_replay`, `test_backtest`, `test_live_backtest` | A restructuring that changes a trading result, an AI prompt, or lets a backtest see the future fails here |
| **Failure modes** | `test_hardening`, `test_retries`, `test_llm_resilience`, `test_llm_upgrade_gaps`, `test_heartbeat`, `test_reconciliation`, `test_runner` | Timeouts and retries, AI failures and fallback, malformed AI output, restart recovery, a cycle that fails, ledger mismatch |
| **Architecture and operations** | `test_layering`, `test_ports`, `test_env_example`, `test_config`, `test_deployment`, `test_ops` | Import boundaries, the broker interfaces and the whole-application type check, `.env.example` matching the code, settings validation, the deploy script not overwriting the server's settings and the compose health checks, JSON logging, the CI gates |
| **Real-portfolio agent** | `test_portfolio_agent`, `test_portfolio_portal`, `test_portfolio_statements` | The read-only agent: it cannot send an order, it logs out |

## Failure modes: what is covered, what is not

| Failure | Covered by |
| --- | --- |
| AI timeout, rate limit, malformed or truncated answer, every model down | `test_llm_resilience`, `test_cot`, `test_agent_analysis`: becomes a `degraded` HOLD, never an order |
| Duplicate event or restart mid-day | `test_database_integrity` (unique indexes, restart), `test_gap_and_go`, `test_llm_and_pipeline` (one decision per bar) |
| Missing or stale market data | `test_hardening`, `test_data_and_strategy`: the stock waits, nothing is decided on an old bar |
| Trading outside market hours, paused, cooldown | `test_enums`, `test_monitoring`, `test_characterization` |
| Process restart and recovery | `test_paper_broker` (position survives), `test_intraday_engine`, `test_database_integrity` |
| Ledger drift | `test_reconciliation` |
| A loop that is stuck, or whose cycles all fail | `test_heartbeat` |
| **Not applicable** | Partial fills and ambiguous broker status: the paper broker fills exactly and instantly. If a real broker is ever added, it must come with those tests first (see [execution-safety](../trading/execution-safety.md)) |
| **Gap** | Database failures (disk full, locked file) during a cycle are handled by the cycle's catch-all but have no dedicated test |

## Writing tests here

- Fake the outside world (Yahoo, OpenRouter, Telegram, the clock); never skip a check to make a test pass.
- A test of a fix should **fail without the fix**: revert the fix, run it, restore. Several tests here were written that way
  (the deploy `.env` exclusion, the restart recovery).
- Pin behaviour you are about to restructure with a characterization test first, then change the structure.
- Tests must not write into the repository (use `tmp_path`; `tests/test_main_wiring.py` redirects the database paths).
- `tests/test_ports.py::test_the_application_type_checks_cleanly` runs mypy (about a minute) and is marked `typecheck`; CI
  deselects it because it runs mypy as its own step.
