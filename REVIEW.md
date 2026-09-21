# AI Trading Bot - Technical & Strategic Review (filled in 2026-09-20; gaps fixed 2026-09-21)

> **Update 2026-09-21:** the engineering gaps this review found are fixed (marked **FIXED** below): stop-protection sweep with
> auto-repair, post-fill verification, staleness checks, timeouts, structured logging, decision replay, startup connectivity
> checks, a drawdown reset (`/rebase`), CI, 100% docstrings, troubleshooting and diagrams. 360 tests pass and every new
> safeguard was mutation-tested. The signal findings (Section 1) are unchanged: still no proven entry edge.

Method: answers come from the code, the test suite (360 passing, run in a clean venv), real runs against live data, and
the backtests in `STRATEGY.md`. Where something is untested or unverified it says so.

## PROJECT OVERVIEW
- **Name:** AI Trading Bot (India / NSE first, US optional)
- **Status:** Paper. Phase 5 (paper trading in the built-in simulator) started 2026-09-20 in Docker; **no market-hours cycle has run yet** (first is NSE open, Mon 09:15 IST). Real money: not built, not advised.
- **Language:** Python 3.13 (`requirements.txt` pinned)
- **Key tech:** OpenRouter free LLMs, **LangGraph**, SQLAlchemy/SQLite, Yahoo Finance + NSE lists (India), Alpaca (US), built-in paper broker, Docker Compose, Telegram
- **Backtest periods:** US AI agent 2025-09-22..2026-09-18 (1y); India event-simulator 2017-10..2026-09 (~9y); signal research 2016-09..2026-09 (10y, Nifty 500)
- **Live period:** none yet

---

## SECTION 1: SIGNAL QUALITY & STRATEGY

### 1.1 Backtest Performance (India, ~9y, current defaults: 8% protective stop + MA200 trend exit, Indian costs, rule-based stand-in for the AI)
- **Return vs benchmark**
  - Bot return: **+143.2%** (about 10.5%/yr), at most half invested (10 positions x 5%)
  - Buy-and-hold: Nifty 50 index **+128.3%** (about 9.7%/yr); the same 45 stocks **+377%** (survivor-biased)
  - Excess vs the index: +14.9 points over 9 years (~+0.8%/yr) with half the capital deployed; vs the same stocks: -234 points
  - Is alpha > 0? **Only vs the index, and that is survivorship-inflated. Vs an equal-weight basket of the same stocks the entry rule adds +0.2%/yr (information ratio -0.02): NO.**
  - Is alpha > transaction costs? Costs (~0.24% round trip) are already deducted; alpha over the basket is ~0, so nothing is left to cover them.
- **Risk-adjusted**
  - Sharpe **0.94**; max drawdown **-16.4%**; win rate **27%** (204 trades, average trade +11.4%)
  - Break-even win rate: **not meaningful for this exit** (asymmetric payoff: few big winners). It was meaningful for the old 2%/5% exit (28.6%), where the bot scored 28-34%.
  - Win rate > break-even? Old exits: no edge (28% vs 28.6%). New exits: n/a.
- **Acceptance criteria**
  - Return > 10% annualised? YES, marginally (~10.5%, survivor-biased)
  - Sharpe > 0.5? YES (0.94)
  - Max drawdown < 20%? YES (-16.4%)
  - Win rate > 40%? **NO (27%)** - a poor criterion for trend-following, but it fails as written
  - Better than baseline? vs index YES (inflated); vs same-universe basket **NO**
  - Pre-declared 7-rule research checklist: only low-volatility passed, and that pass is a survivorship artefact (it lags the basket by 7.3%/yr)

### 1.2 Signal Logic
- **Source:** Hybrid. AI technical analysis proposes entries; **deterministic rules** exit (MA200 trend break, 8% protective stop) and size (risk engine).
  - Inputs: last completed daily close, MA50, MA200, RSI(14), volume (US also headlines). Lookback 200 daily bars.
  - Frequency: signals from completed daily bars (change once a day); cycle every 30 min while the market is open.
- **Robustness**
  - Multiple regimes (2018 crash, 2020, 2021 bull, 2022 chop, 2023-26)? **YES** (10y research and 9y event study)
  - Walk-forward / hold-out? **PARTIAL.** Last 3 years held out and results split into halves, but parameters are fixed, not fitted, so there was nothing to retrain. The exit variants were partly added after seeing results (disclosed in `STRATEGY.md`).
  - Sensitivity (stop/target)? **YES** (five exit styles, both halves; US stop/target grid)
  - Out-of-sample vs in-sample: exit study first half +60.6% vs second half +36.8%; research held-out Sharpe below full-period for momentum (1.06 vs 1.32).
- **Verdict: Signal has edge?**
  - [ ] YES
  - [ ] NO
  - [x] **UNKNOWN - leans NO.** The exit fix is real (evidence in both halves, same-universe comparison). The *entry* signal is not shown to beat holding the universe, and **the AI itself has never been tested on Indian stocks.**

---

## SECTION 2: ARCHITECTURE & DESIGN

### 2.1 Multi-Agent Setup
- Agent 1: **technical** (LLM, role `lead`) - India and US (confidence: the model's own 0-1 estimate; min 0.6 to act)
- Agent 2: **sentiment** (LLM, role `advisor`, can only confirm/veto) - **US only**; disabled for India (no reliable free news source)
- Not agents but deterministic components: risk engine (sizing/limits), trend exit, cooldown
- **Orchestration:** [x] **LangGraph** (three graphs: cycle with `Send` map-reduce over stocks; per-stock analysis with N parallel agent nodes; per-stock decision)
- **Appropriate for agent count?** 1-2 agents: plain Python would be OK. 3-4: LangGraph needed - **already using**. 5+: manual wiring unsustainable - YES.

### 2.2 Agent Routing & Conflict Resolution
- **Routing:** [x] Parallel + merge (agents || -> deterministic combiner) [x] conditional branching (risk approved -> execute) [x] map-reduce across stocks
- **Conflict resolution:** [x] hardcoded rule by role: all leads must agree, advisors confirm or veto, missing/neutral advisors ignored. No voting, arbitrator or weights (deliberately simple and auditable).
- **Explicit and maintainable?** [x] **YES** - `python main.py --graph` prints Mermaid diagrams; behaviour is covered by tests (incl. 4-agent parallelism).

### 2.3 State Management
- [x] Framework-managed (LangGraph state) with **typed state** (`CycleState`, `AnalysisState`, `DecisionState` as TypedDicts, reducers for parallel writes) [x] database audit trail [x] in-memory caches (AI answers, universe scan)
- **Clear?** [x] **YES** - explicit types.

---

## SECTION 3: CODE QUALITY & MAINTAINABILITY

### 3.1 Architecture
- Isolated: LLM (`llm.py`, `agents/`), risk (`engine/risk_engine.py`), broker (`engine/broker.py`, `paper_broker.py`), data (`data/`), database (`database.py`), orchestration (`workflow.py`, `pipeline.py`), research (`research/`), monitoring (`monitoring/`)
- **Responsibilities clear?** [x] **YES.** One leak: `pipeline.py` holds order placement, cooldown and alert logic together (~200 lines).

### 3.2 Testing
- **Unit tests:** **360**; statement coverage of `src/` + `main.py` **about 92%** (`main.py` itself 57%: its wiring is mostly exercised by real runs; lowest others: `market_data` 75%, `research/data` 75%, `telegram` 87%)
  - Mocked: YES. Require keys: **NO**. Pass in CI: **FIXED - `.github/workflows/tests.yml` runs pyflakes + pytest** (verified locally with the exact same steps in a fresh venv; not yet run on GitHub because nothing is pushed).
  - Extra rigor: mutation checks by hand (re-breaking ~145 behaviours; tests caught all but the few gaps that were then closed).
- **Integration tests** (manual scripts, not in pytest)
  - Real Alpaca paper orders: **YES** - `scripts/test_alpaca_connection.py` passed 2026-09-20 (plain + bracket order accepted, cancelled, nothing left)
  - Real LLM calls: YES (`scripts/test_openrouter_api.py`, plus real dry-run cycles)
  - Database migrations: **YES** (old-schema upgrade test) - additive columns only, no Alembic
- **Backtest as test:** automated scripts YES; runs in CI NO; acceptance criteria YES (pre-declared in `research_signals.py`) but not enforced as a gate.

### 3.3 Debugging & Observability
- **Logging:** **FIXED.** [x] `LOG_FORMAT=json` (one object per line with symbol, action, confidence, risk verdict, order status and every agent's signal; docker uses JSON) [x] text format with key=value extras [x] `LOG_LEVEL` configurable and validated [x] noisy libraries silenced (also keeps the Telegram token out of logs).
- **DB audit:** every decision YES; every order YES (now with the broker's actual filled quantity and price); reasoning YES (plus every agent's signal as JSON); **replay: FIXED** - the indicator inputs are stored and `scripts/replay_decision.py` rebuilds the exact prompt and re-checks the rules (verified on real decisions).
- **Monitoring:** Telegram **configured today with a dedicated bot**; a direct message was delivered. **`/status` confirmed live (2026-09-21): correct reply, and the startup alert arrived.** `/positions`, `/pause`, `/resume`, `/rebase`: unit-tested against a mocked API, not yet confirmed live. Kill switch: YES (`/pause`, or stop the container).

---

## SECTION 4: RISK MANAGEMENT & SAFETY

### 4.1 Position Sizing
- Max position **5%** of equity; max exposure **80%**; max open positions **10**; daily loss halt **2%**; drawdown halt **20%** (does **not** reset by itself); re-buy cooldown **24h**; min confidence **0.6**
- **Enforced:** [x] at startup (typos refuse to start) [x] before each order (risk engine) [x] **after each fill: FIXED** (the broker is asked what really filled; quantity and price are stored; a partial fill alerts)
- **Who decides order size:** [x] **Risk engine only** (a test proves a hardcoded quantity is caught)

### 4.2 Broker Integration
- **Order type:** market entry + bracket (US: Alpaca, GTC; India: simulator)
- **Stop-loss 8%; take-profit 100% (effectively none); GTC (US)** - changed from 2%/5% after the exit study
- **Safeguards**
  - quantity > 0: YES (risk engine)
  - duplicate prevention: YES (open-order check + cooldown)
  - cancellation: YES (sell cancels legs first; also exercised for real on Alpaca)
  - **unprotected-position detection: FIXED.** Every cycle scans long positions for a working stop (bracket legs found with `nested=True`; a take-profit leg alone does not count), alerts once, re-checks after 3 s and places a protective stop (`AUTO_PROTECT`). A failed SELL triggers the scan immediately. Verified against the real Alpaca API for the read path; **`protect()` itself is unit-tested but not yet exercised on the real API (needs an open position).**

### 4.3 Edge Cases
- **Failure modes**
  - Market halted mid-order: **not specifically handled** (order fails -> `FAILED` + alert). India simulator has no circuit-limit model.
  - LLM down: [x] fallback chain -> HOLD + alert when all fail
  - Broker unreachable: [x] cycle-level catch, alert, retry next cycle (SELL retries once)
  - Database unreachable: **improved** - connections are pre-checked and SQLite waits up to 30 s for a lock; a hard outage still fails the cycle with an alert (no retry loop)
  - Network timeout: **FIXED** - LLM 60 s + 1 retry; Yahoo 30 s; Alpaca (its client had none) now 30 s via a session wrapper; Telegram 15-40 s; NSE list 30 s
- **Data validation**
  - NaN price: YES (latest close). Volume zero: excluded by the liquidity filter. Confidence in [0,1]: YES (Pydantic).
  - **Staleness check: FIXED.** A last bar more than 5 days before the expected session, or with zero volume, is refused (`StaleDataError`); the scanner drops stocks whose data stops early.

---

## SECTION 5: LLM INTEGRATION

### 5.1 Model Strategy
- **Primary:** `nvidia/nemotron-3-super-120b-a12b:free`; **fallbacks:** `dots-studio/dots-3-note-preview:free`, `nex-agi/nex-n2.5-pro:free`; chain length 4 configured, 3 working (your `.env` still lists a withdrawn DeepSeek model, skipped automatically; update it)
- Cost per call: **$0** (free tier). Monthly budget: $0.
- [x] Free tier (OpenRouter). Risk is availability, not cost: one model was withdrawn and 503 "overloaded" errors were observed.

### 5.2 Prompt Engineering
- **Schema:** JSON, **strict schema enforced by the API**, Pydantic validation YES; on failure it **falls to the next model** (no same-model retry); all fail -> HOLD flagged `degraded` + alert.
- **Prompt content:** [x] price/volume [x] indicators [x] headlines (US, marked untrusted) [ ] market conditions [ ] risk constraints
- **Prompt tested?** Few-shot: **NO**. Reasoning requested: one sentence. Confidence in output: YES. **Prompt quality has never been evaluated against outcomes.**

### 5.3 Robustness
- Rate limit: 1 SDK retry + fallback chain. Token limit: prompts are short; empty replies from reasoning models fall through. Model outage: YES. Malformed response: YES (fallback/HOLD).

---

## SECTION 6: DEPLOYMENT & OPERATIONS

### 6.1 Deployment Method
- [x] local (venv) [x] Docker image (non-root, no secrets) [x] Docker Compose (running now) [ ] **cloud-ready: not set up**
- **Startup checks:** [x] required keys present [x] **connectivity: FIXED** (`python main.py --check` and at every start: OpenRouter key, broker, Yahoo, Telegram; critical failures stop startup) [x] DB migrations [x] risk limits validated

### 6.2 Configuration Management
- Secrets all in `.env`: YES. `.env.example`: YES. Secrets not in code: YES. **Not in Docker image: YES (verified).**
- Configurable: watchlist/universe YES; loop interval YES; risk limits YES; models YES; dry-run/live YES.

### 6.3 Running the Bot
- `python main.py`, `--loop 30`, `--live`, `--report`, `--screen`, `--graph`, `docker compose up`, `pytest` - all exist and were run.
- **Market hours:** checks open YES (India: IST hours + NSE holidays, verified against 2026 real sessions; US: Alpaca clock); does not trade outside hours YES (`MARKET_CLOSED`, verified); waits for next open YES.

---

## SECTION 7: SCALABILITY & FUTURE-PROOFING

### 7.1 Multi-Agent Readiness
- Agents: 1 (India) / 2 (US). Orchestration: **LangGraph**.
- **Adding an agent:** write a class with `name`, `role`, `analyze(ctx)`, add it to the list in `main.py`; the graph gains a parallel node automatically (tested with 4 agents).
- Extensible: YES. Plain Python unmanageable at 3-5 agents: YES. Migrate to LangGraph: **ALREADY USING.**

### 7.2 Database Scalability
- SQLite. Paper trading: YES. Live trading: **NO** (single writer, no server) -> PostgreSQL. **Alembic: still not set up** (only an additive-column helper, tested). Deliberately left: not needed until a move to Postgres.

### 7.3 Performance
- **Cycle time:** first cycle **~84 s** (includes a 65 s full-market scan and 15 concurrent AI calls); later cycles **~1.5-2.2 s** (cached scan + cached AI answers). Max ~95 s.
- **LLM calls per cycle:** 15 on the first (30 for US with sentiment), **0 afterwards** (12 h cache; ~15-30/day)
- Fast enough for a 30-min loop: YES.
- **Cost:** per cycle $0; monthly $0; yearly $0. Sustainable: cost YES; **free-tier reliability MEDIUM.**

---

## SECTION 8: DOCUMENTATION & HANDOFF

### 8.1 Code Documentation
- Docstrings: **FIXED** - modules 28/28, classes 47/47, public functions 142/142 (was ~32%).
- README: quick start YES; setup YES; configuration YES; run local/Docker YES; CI YES; known limitations YES; **troubleshooting: FIXED** (symptom -> fix table); safeguards table added.

### 8.2 Architecture Documentation
- Flow diagram YES (README + `--graph`); agent dependency graph YES; **DB schema diagram: FIXED** (Mermaid ER); **risk decision tree: FIXED** (Mermaid flowchart).
- `STRATEGY.md` YES; backtest results explained YES; known issues YES.

---

## SECTION 9: CRITICAL ISSUES & BLOCKERS

### 9.1 Show-Stoppers
- **Signal has no proven edge:** status **UNKNOWN / leans FAIL.** Paper only.
- **Unprotected position risk:** **HANDLED** (retry, immediate scan, alert, automatic protective stop, per-cycle sweep); India simulator cannot have this failure. Remaining: real-API exercise of `protect()`.
- **Credential exposure**
  - Keys in `.env`: YES (git-ignored). Keys in git history: **NO** (no commits exist). Keys in Docker image: **NO** (verified).
  - **🚨 Two Telegram bot tokens were pasted into this chat.** The old `@myrishithome_bot` token is unused by this project; the trading bot's token (`@rishit_trading_alerts_bot`) also sits in the chat transcript. **Send `/revoke` to @BotFather for both and put the new trading-bot token in `.env`** once testing is done.

### 9.2 Major Issues
- Multi-agent scalability: **handled** (LangGraph). Planned agents: none yet.
- Walk-forward: partial (see 1.2). Overfitting risk: **MEDIUM** (few exit variants examined, one of them after seeing results; survivorship bias ~9 points/yr in the data).
- **Sentiment agent:** headlines available: US YES, India NO. Point-in-time enforced: **NO** (never backtested). Tested only on mocked data and a few real dry-runs: YES.
- **AI agent untested on Indian stocks** - all Indian evidence uses the plain rule as a stand-in.

### 9.3 Minor Issues
- [ ] No Alembic -> add if moving to Postgres
- [x] Structured logging with agent names: **done**
- [x] Telegram: alerts and `/status` confirmed live; `/positions` `/pause` `/resume` `/rebase` still to confirm
- [ ] Cloud deployment undocumented
- [ ] `.env` still lists a withdrawn fallback model
- [x] CI workflow, docstrings and troubleshooting: **done** (CI not yet run on GitHub)

---

## SECTION 10: OVERALL ASSESSMENT

### 10.1 Ratings (1-10)

| Category | Rating | Evidence |
|----------|--------|----------|
| **Code Quality** | 9 | 360 tests, ~92% statement coverage, isolated modules, lint clean, 100% docstrings, structured logs; `pipeline.py` still does a lot |
| **Architecture** | 8.5 | LangGraph with typed state, parallel agents, deterministic risk/exit layer, `--graph` diagrams; `pipeline.py` slightly overloaded |
| **Safety/Risk Mgmt** | 9 | Single order-sizer, startup validation and connectivity checks, halts with a human `/rebase`, cooldown, kill switch, stop-protection sweep with auto-repair, fill verification, staleness and timeout guards. Remaining: `protect()` unproven on the real API, no circuit-limit model, no DB-outage retry |
| **Signal Quality** | 2 | No proven entry edge; AI untested on India; survivorship-inflated evidence. (The exit fix is the one solid finding.) |
| **Documentation** | 9 | README (safeguards, troubleshooting, ER + risk diagrams), STRATEGY, this review, 100% docstrings |
| **Scalability** | 7 | Agents scale well; SQLite, in-memory caches, no Alembic limit live use |
| **DevOps/Deployment** | 8 | Docker/Compose, no secrets in image, market-hours aware, startup checks, JSON logs, CI workflow written; no cloud deploy, CI not yet run on GitHub |
| **OVERALL** | **7** | A hardened, well-documented platform (~8.5) carrying an unproven strategy (~2) |

### 10.2 Recommendation Matrix

| Decision | Recommendation | Why / condition |
|---|---|---|
| Keep Phase 5 paper trading running | **YES** | Tests plumbing, fills, exits and alerts with zero real risk. Do not read profits as evidence of edge. |
| Real money | **NO** | No proven entry edge, no Indian broker integration, regulated activity, survivorship-inflated evidence. |
| Next research step | Test the **AI agent itself** on Indian history, or drop it and keep only the rule; use survivorship-free data if obtainable; judge with pre-declared rules against the same-bias basket | The prompt has never been shown to add value over the plain rule. |
| Engineering fixes | **Done:** stop-protection sweep, staleness check, timeouts, stale fallback model in `.env`, post-fill check, replay, preflight, `/rebase`, CI. **Still yours:** revoke the exposed Telegram tokens; confirm `/positions` `/pause` `/resume` `/rebase` live in Telegram (`/status` is confirmed). | The failure modes are closed; the two remaining items need you. |
| Before any live-money discussion | PostgreSQL + Alembic, a real-API test of `protect()` with an open position, circuit-limit handling, cloud deploy doc, and above all a validated entry signal | Platform hardening is mostly done; the signal is not. |
