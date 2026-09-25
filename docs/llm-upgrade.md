# LLM Upgrade — Implementation Status

Maps the "LLM Upgrade Implementation Review" (Phases 1–4 + cross-phase) to what now exists in this
repo. "Machinery PASS" means the code, schema, prompts and enforcement are implemented and unit-tested;
every **outcome** claim (win rate, calibration, effectiveness gap, Sharpe) needs real paper history and
is deliberately reported as *needs data* by the metrics scripts, never invented.

Status key: `PASS` = implemented + unit-tested · `needs data` = implemented, proven only by live metrics.

## Phase 1 — Reasoning transparency

| Review point | Status | Where |
|---|---|---|
| 5-step CoT stored per decision | PASS | `src/agents/agents.py` step chain → `reasoning_chain_json`, step tokens `[STEP 1–5]` |
| Each step gets 2–3 sub-questions and the risk list needs ≥3 severity-ranked risks | PASS | `TechnicalAgent.build_prompt` (steps 1–3: `(a)(b)(c)`; step 4: alignment; step 5: "at least three risks, most severe first") |
| Schema enforces ≥1 risk (resilience to model trimming) | PASS | `AdvancedSignal.risks: min_length=1` |
| Retry once on malformed/validation-failed JSON so a flaky model does not sink the cycle | PASS | `src/llm.py::LLMClient._ask` (retry-once per provider, appends a fix-it hint; dead model skipped, API error breaks; `LLMUnavailable` raised at the end) |
| Index for the per-stock decision query | PASS | `ix_decisions_confluence` (and `ix_decisions_pattern`) — `CREATE INDEX IF NOT EXISTS` in `make_session_factory` for pre-existing DBs |
| Representative risks/quality | needs data | judge from `analyze_calibration.py` once trades exist |

## Phase 2 — Decision memory (pattern learning)

| Review point | Status | Where |
|---|---|---|
| Only closed paper trades count | PASS | `find_similar_patterns` reads `PaperTradeRecord` |
| "Same setup" = trend shape + price ±5% + RSI(14) ±5 | PASS | `history.py` (`PRICE_TOL_FRAC=0.05`, `RSI_TOLERANCE=5.0`, `_trend_shape`) |
| Stable, human-readable pattern label | PASS | `_pattern_id` → e.g. `P>MA50+MA50>MA200|RSI60`; stored in `decisions.pattern_id` (indexed) |
| Profit factor = avg winner / avg loser (None when a bucket is empty) | PASS | `pattern_stats.profit_factor` |
| Sample-size-aware confidence in the memory (`n/30`, capped) | PASS | `PatternStats.confidence_in_pattern` |
| Reliability rendered as yes/maybe/no | PASS | `LearningSignal.pattern_reliability: Literal["yes","maybe","no"]`, schema v2, `min_pattern_sample=3` gate |
| Per-refinement confidence cap of ≤0.15 | PASS | `_clamp_adjustment` + `TechnicalAgent(max_adjustment=...)`, wired from `LLM_MAX_ADJUST` (default 0.15) |
| Store what was adjusted and why | PASS | `adjusted_signal_confidence`, `adjustment_reason`, `pattern_id` columns, written by the pipeline |
| Effectiveness over time | needs data | `scripts/analyze_learning_effectiveness.py` |

## Phase 3 — Market & sector context

| Review point | Status | Where |
|---|---|---|
| Regime label + 4 context questions in the prompt | PASS | `build_context_prompt` (regime, RSI, VIX, YTD + gap risk); 4 questions in `CONTEXT_SCHEMA` |
| Sector/earnings/beta inputs, honest n/a when no feed | PASS | `MarketContext` `beta_6m`, `sector_trend`, `earnings_days_until`, `correlation_with_index` default `None` = neutral; beta is `stock_mom/index_mom` proxy |
| Risk-off/euphoria must not inflate confidence | PASS+GATE | phases apply context *after* the cap; `extra(out, applied)` carries applied values |
| Store supports/risks/regime | PASS | `macro_support`, `sector_support`, `earnings_risk`, `diversification_score`, `context_regime`, `market_context_json` |
| Influence on decisions / regime calibration | needs data | `scripts/analyze_context_effectiveness.py` |

## Phase 4 — Reflective self-critique

| Review point | Status | Where |
|---|---|---|
| Six-stage chain incl. biggest risk + "what proves me wrong" + bias check | PASS | `ReflectionStage`/`ReflectionChain` + `REFLECT_SCHEMA` (`reflection_check_v2`) |
| Conviction only ever stays or falls, each stage | PASS | `_apply_reflection` (monotonic clamp, client-side enforced) |
| Deterministic humility discount then final confidence | PASS | `_reflect`: final = step-6 conviction − `HUMILITY` (0.10); reasoning marker `[reflection: … -> final confidence 0.50]` |
| Reflection only on non-HOLD calls | PASS | `analyze()` skips phase 4 on HOLD (and phase 3) |
| Store the chain + the concession fields | PASS | `conviction_adjustments` (JSON list of stages), `biggest_risk`, `what_proves_us_wrong`, `bias_check` |
| Contract adherence + bias/risk quality on real cycles | needs data | `scripts/analyze_phase4_calibration.py` |

## Cross-phase

| Review point | Status |
|---|---|
| Full pytest suite green | PASS (403 tests) |
| pyflakes clean on `src/ tests/ scripts/ main.py` | PASS |
| Coverage | run `python -m pytest --cov=src --cov-report=term-missing -q` |
| Benchmark prompt build + history lookup | `scripts/benchmark_performance.py` |
| Backtest unaffected | `backtest_ai_agent.py` uses only `build_prompt`/signal — verified |

## Known honest gaps
- Sector/earnings/correlation/beta feeds still None-neutral in the sandbox (no network); the phase is
  wired and exercised by unit tests, but live regime captures begin when the bot runs online.
- Every outcome metric is a *downstream ruler*: 0 closed paper trades in the live DB today, so all
  effectiveness/calibration numbers are "not yet measurable" by design, not absent.