# Architecture overview

An India (NSE) **paper-trading** bot. Nothing in this repository places a real order: the only broker that exists is the
built-in simulator (`PaperBroker`). The one component that touches a real-money account, the Integrated portfolio agent,
is read-only and walled off from the trading code (see [execution-safety](../trading/execution-safety.md)).

It is a **modular monolith**: one Python codebase (about 86 modules, 10,000 lines), run as two processes. See
[decision 0001](decisions/0001-keep-the-package-layout.md) for why it is not split into `domain/application/infrastructure`
folders, and [dependency-map](dependency-map.md) for who imports whom.

## What runs

| Process (Docker service) | Command | What it does |
| --- | --- | --- |
| `trading-bot` | `main.py --loop 30 --paper-orders --holdout` | The **swing** bot: every 30 minutes while the market is open, scan the NSE, analyse the best candidates with the AI, decide, size, and place paper orders. Also serves the Telegram commands and the daily/weekly learning jobs. |
| `intraday-bot` | `main.py --intraday --paper-orders` | Three **intraday** paper strategies stepped together every 5 minutes, each in its own account: the opening-range breakout (ORB), gap-and-go, and gap-only (gap-and-go's control). No AI is involved. |

Each account has its own SQLite database on the shared `/data` volume: `trading.db`, `intraday.db`, `gapgo.db`,
`gaponly.db` ([database](../database.md)). Accounts never mix.

## The packages

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `engine` | The domain core: enums, the **risk engine**, the **paper broker**, costs and fees, the rules (uptrend, trend break), combining agent signals. A leaf: it imports no agent, LLM or data code. | `config`, `database` |
| `database` | SQLAlchemy models, session factory, additive migrations. | none |
| `data` | Market data (Yahoo), indicators, the stock screener, company fundamentals, market context, intraday bars. | `engine`, `database` |
| `llm` | The OpenRouter client: schema-enforced JSON, retries, model fallback, circuit breaker. Knows nothing about trading. | `engine` (the signal type) |
| `agents` | The AI analyst (`TechnicalAgent`): prompts, schemas, the veto-only critic. | `engine`, `data`, `llm` |
| `pipeline`, `workflow`, `results` | The swing cycle: three LangGraph graphs and the object that owns the safeguards around them. | `agents`, `data`, `engine`, `database` |
| `intraday` | The intraday rules (pure numpy functions), position sizing, the 5-minute engine, the universe selector. | `engine`, `data`, `database` |
| `learning` | Labelling decisions with what then happened; the weekly AI-vs-rule report. Reads the database only. | `control`, `engine`, `database` |
| `monitoring` | Telegram alerts and commands, the heartbeat. | `data`, `engine` |
| `app` | The wiring: `bootstrap` (assemble the swing bot), `wiring` (brokers, database paths), `intraday_cli`, `reports`, `portfolio_agent`. | everything it assembles |
| `ops` | Operational tools: ledger reconciliation, decision replay, the holdout reserve, startup checks. Never imported by production code. | production packages |
| `research` | Backtests and the intraday research harness. Never imported by production code. | production packages |
| `portfolio` | The read-only real-account agent (a browser session, a vault, a report). Imports nothing from the trading code. | none of the above |

## One swing cycle

```
open_cycle  → read the account, record equity, apply the drawdown pause, check halts
select      → the screener's best candidates plus every held position
analyse     → for each stock, in parallel: fetch the snapshot; reuse this bar's earlier answer if there is one;
              otherwise one AI call (+ the critic)                              [workflow.py, agents/]
decide      → one stock at a time, in a fixed order: the deterministic trend exit, else the combined agent
              signal → the RISK ENGINE sizes and approves or refuses → record the decision        [pipeline.py]
execute     → place the approved paper order (cooldown, open-order and market-hours checks), record the order
close_cycle → log AI health, alert on an outage or recovery
```

The AI proposes; **plain code decides and sizes**. A failed or malformed AI answer becomes a `degraded` HOLD and never
reaches the broker (see [AI workflows](../ai/workflows.md)).

## One intraday cycle

```
IntradayEngine.step()  (every 5 minutes)
  market closed → archive the finished session's bars (once)
  square-off time (15:15) → sell everything
  otherwise → download bars, run the strategy's rule on each stock not yet attempted today,
              size by risk under a position cap, place the paper order with its stop, record the signal
              (entered, skipped and why, dry run, or failed)
```

The rule is one pure function per strategy (`first_breakout`, `first_gap_breakout`, `gap_only_bar`), shared by the live
engine and every research script, so what is tested is what trades.

## Where state lives

- **Databases** (above): decisions, orders, positions, trades, equity history, signals, the bars archive.
- **Heartbeat files** (`/data/heartbeat-*`): each loop's latest cycle outcome, read by the container health check.
- **Cache** (`/data/cache`): the Nifty and VIX series for the AI's market context.
- **Settings**: environment variables read once into `config.Settings`; the `.env` is one folder above the project
  (`/home/rishit/trading bt/.env`) and is never committed.
