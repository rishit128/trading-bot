"""The tables: the audit log (decisions, orders, equity), control flags and the paper-trading account and reserve."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all tables."""

    pass


class DecisionRecord(Base):
    """One analysed stock in one cycle: every agent's signal, the final call, the risk verdict and the inputs."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    price: Mapped[float] = mapped_column(Float)
    technical_action: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    technical_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_action: Mapped[str] = mapped_column(String(8))
    final_confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text)
    risk_approved: Mapped[bool] = mapped_column(Boolean)
    risk_quantity: Mapped[int] = mapped_column(Integer)
    risk_reason: Mapped[str] = mapped_column(Text)
    snapshot_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the indicator inputs, for replay
    # Every agent's signal as JSON {name: {action, confidence, reasoning, degraded}}, so any number of agents is auditable.
    signals_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Chain-of-thought extras, denormalised for SQL queries; the full chain lives in signals_json.details.
    confluence_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-10, how aligned the signals are
    edge_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-1, how repeatable the pattern is
    risks_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of the identified risks
    reasoning_chain_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the step-by-step reasoning
    # The review step, taken from the lead agent's details (None when it did not run). Columns of removed phases (decision
    # memory, the separate market-context call, the sentiment agent) stay in older databases with their old values.
    base_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # confidence before the review
    adjusted_signal_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # after the review
    adjustment_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # what the review did
    market_context_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the market situation the model was shown
    # Review doc 4: the reflection chain's findings.
    biggest_risk: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    what_proves_us_wrong: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bias_check: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of biases the critic flagged
    conviction_adjustments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON of step-by-step conviction
    # AI accountability : which configured model answered, and whether it agreed with or overrode the
    # deterministic mechanical filter - the inputs to the rubber-stamp / approval-rate measurement.
    raw_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    rule_alignment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # agree / deviate / None
    # The mechanical entry rule's own verdict on this snapshot (BUY/HOLD), computed in code. Unlike `rule_alignment`
    # (the analyst's answer against the rule, before the critic) it does not move when the critic vetoes the call.
    # NULL on rows recorded before 2026-10-16: readers recompute it from `snapshot_json`.
    rule_action: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    falsification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the concrete thing that proves it wrong
    # Version stamps (src/versions.py), so a change of prompt, indicators or rules can be attributed per decision.
    prompt_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    strategy_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Who made the call: the AI agents, or a deterministic rule that overrode them (engine.enums.DecisionSource). None on
    # rows written before this column existed; replay then falls back to reading the reasoning text.
    decision_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # The session whose completed bar the analysis used (`snapshot.bar_date`): with the symbol it identifies WHICH market
    # situation this decision was about, so outcomes can be attached to it however many times it was re-decided that day.
    bar_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, index=True)
    # Per-decision universe snapshot: how many symbols were in the scanned universe when this call was made.
    universe_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Why the stock was analysed: "held", "screener", "research" (an open broker research call put it on the list) or
    # "screener+research". None on rows written before the research integration.
    candidate_source: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    # For a stock only a research call put on the list: what the screener's own filters made of it (UniverseScreener.
    # verdict: candidate / below the top N / unaffordable / filtered out). Recorded only; nothing is blocked on it. None
    # for other stocks, without a screener, or on rows written before 2026-10-16.
    screen_verdict: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    # The broker research call the AI was shown for this stock (src.data.house_calls.HouseCall as JSON); None = none shown.
    house_call_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The market pulse the AI was shown (src.data.market_pulse.HotStock as JSON): this stock's sector was surging; None
    # = none shown. A stock bought on it has candidate_source "pulse" and is sold on a close below its 50-day average.
    pulse_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class OrderRecord(Base):
    """An order attempt or skip, with the broker's reported fill."""

    __tablename__ = "orders"
    # One decision places (or records the skip of) at most one order: a repeated event can never double an order.
    __table_args__ = (Index("ux_orders_decision", "decision_id", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decision_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24))
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    filled_qty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # what the broker says was actually filled
    fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class EquityRecord(Base):
    """Account equity at the start of a cycle (the source of the peak for the drawdown limit)."""

    __tablename__ = "equity_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    equity: Mapped[float] = mapped_column(Float)


class ControlFlag(Base):
    """A named runtime switch stored in the database."""

    __tablename__ = "control_flags"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[str] = mapped_column(String(64))


class PaperAccountRecord(Base):
    """The simulated account's cash and daily-loss bookkeeping."""

    __tablename__ = "paper_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float)
    initial_cash: Mapped[float] = mapped_column(Float)
    day: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    day_start_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_mark: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Cash sweep (idle cash earning a liquid-fund rate): when interest was last credited, and the running total, so the
    # ledger reconciliation can still rebuild cash from the fills.
    interest_accrued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    interest_earned: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividends_received: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # credited on ex-dates, for the ledger


class PaperPositionRecord(Base):
    """An open simulated position with its stop and target levels."""

    __tablename__ = "paper_positions"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    qty: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_checked: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Start of the 5-minute bar whose latest close was the fill price (it may still have been forming); None when the
    # feed does not say, or on rows recorded before 2026-10-16.
    entry_price_bar: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Latest split/bonus/dividend ex-date already applied (YYYY-MM-DD); None = none since the position opened.
    actions_through: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)


class PaperTradeRecord(Base):
    """A closed simulated trade with fees and net P&L."""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    qty: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String(16))
    fees: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float)
    # Start of the 5-minute bar each price came from (see PaperPositionRecord.entry_price_bar); None when unknown.
    entry_price_bar: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price_bar: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class HoldoutMark(Base):
    """An independent, decision-only valuation of the reserve account (holdout).

    Written by HoldoutReserve.mark() and untouched by the live account: the reserve replays the stored
    approved decisions through the deterministic simulator so that any tuning or learning against the
    paper account's own outcomes can never contaminate this ruler. `recorded_at` timestamps the run the
    mark was made, so the table is a point-in-time equity history of the reserve."""

    __tablename__ = "holdout_marks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    equity: Mapped[float] = mapped_column(Float)
    return_pct: Mapped[float] = mapped_column(Float)
    decisions: Mapped[int] = mapped_column(Integer)
    closed_trades: Mapped[int] = mapped_column(Integer)
    open_positions: Mapped[int] = mapped_column(Integer)


class OutcomeRecord(Base):
    """What a stock actually did after the bot analysed it: the forward result of entering at the next session's open and
    holding `horizon` sessions. One row per (stock, analysed session, horizon), whether or not the bot traded it; this is
    what turns hundreds of decisions a week into lessons, instead of the ~20 closed trades a year a trend strategy makes."""

    __tablename__ = "decision_outcomes"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    bar_date: Mapped[str] = mapped_column(String(10), primary_key=True)  # the analysed session (YYYY-MM-DD)
    horizon: Mapped[int] = mapped_column(Integer, primary_key=True)  # sessions held after entry
    entry_date: Mapped[str] = mapped_column(String(10))  # the session whose open is the assumed entry
    exit_date: Mapped[str] = mapped_column(
        String(10), default=""
    )  # the session whose close ends the hold ("" only on legacy rows)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)  # the close `horizon` sessions after the analysed one
    gross_return: Mapped[float] = mapped_column(Float)  # exit / entry - 1, before costs
    worst_return: Mapped[float] = mapped_column(Float)  # the deepest intraday low over the hold, vs entry
    hit_stop: Mapped[bool] = mapped_column(Boolean)  # whether a stop `stop_pct` below entry would have been touched
    stop_pct: Mapped[float] = mapped_column(Float)
    labelled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class IntradaySignalRecord(Base):
    """Every opening-range breakout the intraday engine saw, and what it did with it (entered, skipped and why, dry run).
    Without this a skipped signal left no trace, and the engine could not be audited or its misses counted."""

    __tablename__ = "intraday_signals"
    # One attempt per stock per session (the engine's rule): a restart cannot record, or trade, the same signal twice.
    __table_args__ = (Index("ux_intraday_signals_day_symbol", "day", "symbol", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    day: Mapped[str] = mapped_column(String(10), index=True)  # the session (YYYY-MM-DD, IST)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    bar_time: Mapped[str] = mapped_column(String(32))  # start of the breakout bar, ISO with its IST offset
    signal_price: Mapped[float] = mapped_column(Float)  # the breakout bar's close
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    range_high: Mapped[float] = mapped_column(Float)
    range_low: Mapped[float] = mapped_column(Float)
    volume_ratio: Mapped[float] = mapped_column(Float)  # the breakout bar's volume over the average of the bars before it
    strength: Mapped[float] = mapped_column(Float)  # 0-1 rescaling of volume_ratio (informational: it no longer sizes anything)
    outcome: Mapped[str] = mapped_column(String(12))  # ENTERED | SKIPPED | DRY_RUN | FAILED
    qty: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # why it was skipped or failed
    system_version: Mapped[str] = mapped_column(String(16), default="")  # src.versions.INTRADAY_VERSION
    # The open broker research call on this stock when the signal fired ("Intraday buy", ...); None = none. A stock that is
    # not in the day's recorded universe (intraday_universe) was added to the scan by that call.
    house_call: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Data quality (all None on rows recorded before 2026-10-16). What the engine saw: when its cycle started and the
    # signal bar's volume (the close it saw is `signal_price`). What the final bars, downloaded after the close, say about
    # the same bar, and whether the rule run on them still signals at that bar (IntradayEngine._verify_signals).
    step_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    seen_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_signal: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)


class IntradayUniverseRecord(Base):
    """The stocks the intraday engine scanned on a day and where the list came from, so a result can always be tied to the
    universe it was earned on (a live list and a backtest list that differ make the two incomparable)."""

    __tablename__ = "intraday_universe"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    day: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(160))
    size: Mapped[int] = mapped_column(Integer)
    symbols_json: Mapped[str] = mapped_column(Text)
    system_version: Mapped[str] = mapped_column(String(16), default="")


class NewsFilingRecord(Base):
    """One exchange filing seen by the news-catalyst engine (src.news), traded or not: where it came from (headline and link
    only, never the filing's text), the AI's reading of it, whether the entry rule fired and why not, and how the price moved
    afterwards. Kept for every filing on a stock of the news universe, so two questions can be answered apart: does the AI
    read news correctly (label against the later move), and does the entry rule add anything (fired against not fired)."""

    __tablename__ = "news_filings"
    __table_args__ = (Index("ux_news_filings_key", "key", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)  # when the watcher first saw it
    key: Mapped[str] = mapped_column(String(200))  # the filing's link: one row per filing, whichever feed showed it first
    source: Mapped[str] = mapped_column(String(8))  # BSE | NSE
    symbol: Mapped[str] = mapped_column(String(16), index=True)  # NSE symbol
    company: Mapped[str] = mapped_column(String(120))
    subject: Mapped[str] = mapped_column(Text)  # the feed's one-line description
    link: Mapped[str] = mapped_column(Text)
    published: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # the feed's timestamp (UTC)
    session: Mapped[str] = mapped_column(String(10), index=True)  # the first session it can be traded in (YYYY-MM-DD, IST)
    # The AI's label: status ROUTINE (dropped by subject before any AI call) | LABELLED | FAILED (no answer, ignored)
    label_status: Mapped[str] = mapped_column(String(10), default="PENDING")
    event_type: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    direction: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # up | down | none
    surprise: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # large | moderate | none | unclear
    numbers_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the key numbers, as read from the filing
    ai_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_model: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    material: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    material_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The entry rule: WAITING (not decided yet) | FIRED | NO; why not; and the trade's outcome when it fired
    rule_status: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    rule_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    entry_kind: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)  # A (overnight) | B (during the session)
    # The move after the filing, measured after the session's close: from `ref_price` (the last price before the filing; the
    # previous close for a filing made outside market hours) to the price 5, 30 and 120 minutes later and at the close
    # (for an overnight filing those minutes are counted from the open).
    ref_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    move_5m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    move_30m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    move_120m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    move_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Angel One, record only: the live price when the watcher first saw the filing, and (for a rule that fired) the bid-ask
    # spread, circuit limits and F&O positioning at the entry, as JSON.
    seen_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    system_version: Mapped[str] = mapped_column(String(16), default="")


class IntradayBarRecord(Base):
    """One stored 5-minute bar. Yahoo serves only the last ~60 sessions of 5-minute data, so the intraday engine saves every
    finished session here: the archive grows a session a day and is the only way to test an intraday rule on more than
    two months of history. One row per (symbol, bar start); `day` is the IST session date, for cheap per-day queries."""

    __tablename__ = "intraday_bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)  # the bar's start time (UTC)
    day: Mapped[str] = mapped_column(String(10), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    source: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # "angel" | "yahoo"; empty on bars saved before it


class AngelBarRecord(Base):
    """A 5-minute bar built from Angel One's live stream (src.data.angel), kept beside the Yahoo archive (`intraday_bars`) so the
    two sources can be compared. `seen_at` is when the bar was complete here, to measure how soon after its close."""

    __tablename__ = "angel_bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)  # the bar's start time (UTC)
    day: Mapped[str] = mapped_column(String(10), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class HouseCallRecord(Base):
    """The broker's research calls as last read from the Integrated portal (src.data.house_calls): ONE snapshot, replaced
    whole on every read, never a history. The trading code reads it; a snapshot older than a few polls is ignored, so a
    day without the portal session trades exactly as before."""

    __tablename__ = "house_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # when the portal was read
    symbol: Mapped[str] = mapped_column(String(32), index=True)  # the NSE symbol (ScripCode)
    kind: Mapped[str] = mapped_column(String(32))  # Intraday | BTST | Swing Trade | Momentum | Stellar Pick | ...
    side: Mapped[str] = mapped_column(String(8))  # buy | sell
    entry_low: Mapped[float] = mapped_column(Float)
    entry_high: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    published: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # YYYY-MM-DD; None = not given
    is_open: Mapped[bool] = mapped_column(Boolean)
    exit_comment: Mapped[str] = mapped_column(String(64), default="")  # e.g. "Target Hit", "Stoploss Hit", "Exit"
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)  # None while open


class HouseCallEvent(Base):
    """The history of the broker's research calls (src.data.house_calls), beside the one-snapshot `house_calls` table the
    trading code reads. One row per distinct state of a call, keyed by (symbol, kind, side, published): a call read again
    unchanged only moves its row's `last_seen`; any change (closed, stop or target revised) appends a row. Rows are never
    deleted, so how each call began, changed and ended can be scored later."""

    __tablename__ = "house_call_events"
    __table_args__ = (Index("ix_house_call_events_call", "symbol", "kind", "side", "published"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    published: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # UTC
    entry_low: Mapped[float] = mapped_column(Float)
    entry_high: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    valid_to: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean)
    exit_comment: Mapped[str] = mapped_column(String(64), default="")
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)  # UTC
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # the read that first showed this state
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # the latest read that still showed it
