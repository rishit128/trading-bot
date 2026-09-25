"""The tables: the audit log (decisions, orders, equity), control flags and the paper-trading account and reserve."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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
    sentiment_action: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    sentiment_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
    # Phases 2-4 extras, taken from the lead agent's details (None when a phase did not run or was not enabled).
    base_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # confidence before any refinement
    context_regime: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # bull / bear / risk_off / None
    historical_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # similar closed trades (decision memory)
    historical_sample_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # and how many backed it
    # Review doc 2: the pattern the decision belongs to and what the learning/refinement phases did to the base call.
    pattern_id: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, index=True)  # e.g. P>MA50+MA50>MA200|RSI60
    adjusted_signal_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # after the phases moved it
    adjustment_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # why any phase changed the call
    # Review doc 3: the context-phase answers and the full context dataclass.
    macro_support: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # yes / neutral / no
    sector_support: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    earnings_risk: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    diversification_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-10
    market_context_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the regime the call faced
    # Review doc 4: the reflection chain's findings.
    biggest_risk: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    what_proves_us_wrong: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bias_check: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of biases the critic flagged
    conviction_adjustments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON of step-by-step conviction
    # AI accountability : which configured model answered, and whether it agreed with or overrode the
    # deterministic mechanical filter - the inputs to the rubber-stamp / approval-rate measurement.
    raw_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    rule_alignment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # agree / deviate / None
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


class OrderRecord(Base):
    """An order attempt or skip, with the broker's reported fill."""
    __tablename__ = "orders"

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
    exit_date: Mapped[str] = mapped_column(String(10), default="")  # the session whose close ends the hold ("" only on legacy rows)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)  # the close `horizon` sessions after the analysed one
    gross_return: Mapped[float] = mapped_column(Float)  # exit / entry - 1, before costs
    worst_return: Mapped[float] = mapped_column(Float)  # the deepest intraday low over the hold, vs entry
    hit_stop: Mapped[bool] = mapped_column(Boolean)  # whether a stop `stop_pct` below entry would have been touched
    stop_pct: Mapped[float] = mapped_column(Float)
    labelled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

