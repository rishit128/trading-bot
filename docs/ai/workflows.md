# AI workflows

Where the AI is used, where it is not, and how its failures are contained.

## Where the AI is used

| Use | Where | Output | Consumed by |
| --- | --- | --- | --- |
| **The analyst**: one informed call per stock per session (chart + 12-1 momentum + company fundamentals + market situation), then a **veto-only** critic | `agents/technical`, prompt `technical-v10` | `AdvancedSignal` → `AgentSignal` (BUY/SELL/HOLD, confidence, one-line reasoning, which of the rule's inputs it disagreed with and why) | `combine_signals` → the risk engine |
| The portfolio note: a short plain-language note on your real holdings | `app/portfolio_agent.py` | one string | Telegram, **never an order** |
| Nothing else. The intraday strategies use **no AI**. | | | |

The analyst's job is narrow on purpose: follow the mechanical baseline (the trend rule) unless it can name a concrete, numbered
company-level reason not to. A weak market alone is not a reason. This came from measuring the earlier design: the AI had
rejected 77 of 116 rule BUYs with no difference in what then happened (STRATEGY.md, 2026-10-09). Whether the current design
adds value is judged forward, by the weekly AI-vs-rule report on Telegram.

## The workflow (LangGraph)

Three small graphs in `workflow.py`, built from whatever agents are registered (today: one):

```
cycle:     open_cycle → select_universe → [one parallel analysis per stock] → execute_cycle
analysis:  fetch_data → (this bar already decided? → collect) | (agent_technical → collect)
decision:  decide → (risk approved? → execute) | end
```

What LangGraph provides here: parallel analysis of many stocks (the slow, AI-bound part) with a bound on concurrency, a
conditional edge that skips the AI when this session's bar was already decided, and a retry on the data fetch. The decisions
and orders then run **one stock at a time in a fixed order** so risk limits and cash are evaluated against fresh state. The graph
knows the application only through the small `CycleServices` interface (`test_cycle_services`). It is more machinery than one
agent strictly needs; it is kept because it works, is tested, and the lead/advisor roles allow a second agent without graph
changes. If it is ever replaced by plain functions, `test_characterization` pins the behaviour to preserve.

## Model routing and failure handling (`llm/client.py`)

- **Models** are tried in order: `OPENROUTER_MODEL`, `OPENROUTER_FALLBACK_MODEL`, then the built-in list. All are free models.
- **Settings**: temperature 0 (the same bar must get the same answer), hidden "reasoning" off (it eats the token budget),
  requests paced at one every 3 s and at most 2 in flight (the free tier allows about 20 a minute).
- **Failures are classified**: `transient` (rate limit, timeout, 5xx: retried with backoff, honouring `Retry-After`),
  `format` (invalid JSON or schema: retried once), `truncated` (token limit: retried with a larger budget), `permanent` (model gone,
  auth: skip that model). A whole call has a 180 s budget.
- **Circuit breaker**: a model that fails 3 whole calls in a row is benched for 5 minutes so the others answer first.
- **Output is validated**: JSON schema on the way out (`response_format`), pydantic on the way in. A model cannot set the
  `degraded` flag; only our fail-safe does.
- **Observability**: the model that answered (`raw_model`) is stored on every decision; each cycle logs a health line per model
  (calls that worked, failures by kind, benched models, average latency). **Token and cost accounting is not recorded** (all
  models are free today); add it before a paid model is configured.

## Safety rules (each enforced by tests)

1. **The AI cannot bypass the risk engine.** Its output is one input to `combine_signals`; sizing and approval are plain code.
2. **Invalid, missing, contradictory or low-confidence output → an explicit safe response.** Failure → a `degraded` HOLD with
   confidence 0, never reused; a confidence below `MIN_CONFIDENCE` is refused by the risk engine.
3. **A model failure cannot create an order.** A degraded signal cannot be BUY. If every model fails in a cycle, the bot says
   `LLM UNAVAILABLE` and holds.
4. **The critic can only lower**: it may turn a BUY into a HOLD; it never raises confidence, and a fixed humility discount is
   recorded for audit only.
5. **Prompts, schemas and features are versioned** (`PROMPT_VERSION`, `FEATURE_VERSION`, `SCHEMA_VERSION` in `versions.py`) and
   stored with each decision; `test_prompt_versions` fingerprints every prompt, so changing wording without bumping the version
   fails. `scripts/replay_decision.py` rebuilds the exact prompt a stored decision saw.
6. **One answer per stock per bar**: the recorded decision is reused across cycles and restarts (a noisy model flipping
   HOLD → BUY on a repeated question once caused a buy on the flip).
7. **Research agents cannot change live settings**: nothing under `research/`, `ops/` or `learning/` is imported by production
   code, and a deploy never overwrites the server's `.env`.
8. **The AI is not backtested on past dates**: it knows how those dates ended, so the result would be meaningless. It is
   evaluated forward only.
