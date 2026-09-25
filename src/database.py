"""SQLAlchemy tables: the audit log (decisions, orders, equity), control flags and the paper-trading account."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# Extra indexes for the review-doc queries ("decisions with a strong confluence", "pattern x's track record") are
# created for every database, including ones that predate the columns, in `_ensure_indexes`.
EXTRA_INDEXES = {
    "ix_decisions_confluence": "CREATE INDEX IF NOT EXISTS ix_decisions_confluence ON decisions (confluence_score)",
    "ix_decisions_pattern": "CREATE INDEX IF NOT EXISTS ix_decisions_pattern ON decisions (pattern_id)",
    "ix_decisions_raw_model": "CREATE INDEX IF NOT EXISTS ix_decisions_raw_model ON decisions (raw_model)",
}


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
    # Phase 1 chain-of-thought extras, denormalised for SQL queries; the full chain lives in signals_json.details.
    confluence_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-10, how aligned the signals are
    edge_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-1, how repeatable the pattern is
    risks_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of the identified risks
    reasoning_chain_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the step-by-step reasoning
    # Phases 2-4 extras, taken from the lead agent's details (None when a phase did not run or was not enabled).
    base_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # confidence before any refinement
    context_regime: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # bull / bear / risk_off / None
    historical_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # similar closed trades, phase 2
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
    # AI accountability (phase 1): which configured model answered, and whether it agreed with or overrode the
    # deterministic mechanical filter - the inputs to the rubber-stamp / approval-rate measurement.
    raw_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    rule_alignment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # agree / deviate / None
    falsification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # the concrete thing that proves it wrong
    # Version stamps (src/versions.py), so a change of prompt, indicators or rules can be attributed per decision.
    prompt_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    strategy_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Per-decision universe snapshot (P0, A1b): how many symbols were in the scanned universe when this call was made.
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
    """An independent, decision-only valuation of the reserve account (P0, holdout).

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


def _add_missing_columns(engine) -> None:
    """create_all never alters existing tables; add any new nullable columns so an older database keeps working."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    if not column.nullable:
                        raise RuntimeError(f"{table.name}.{column.name} is new and NOT NULL; migrate the database manually")
                    ddl = column.type.compile(engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}'))


def _ensure_indexes(engine) -> None:
    """create_all does not add an index to a column that already exists; create the review-doc query indexes for
    databases that predate the columns."""
    with engine.begin() as conn:
        for statement in EXTRA_INDEXES.values():
            conn.execute(text(statement))


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    """Create the engine and tables, add any missing columns to an older database, and return a session factory."""
    kwargs = {"pool_pre_ping": True}  # a dropped connection is replaced instead of failing the cycle
    is_sqlite = database_url.startswith("sqlite")
    if is_sqlite:
        kwargs["connect_args"] = {"timeout": 30}  # wait for a lock instead of failing with "database is locked"
    engine = create_engine(database_url, **kwargs)
    if is_sqlite and ":memory:" not in database_url:
        # WAL: a reader (e.g. the swing bot's /intraday Telegram command) never blocks or is blocked by the writer
        # (the intraday engine's own process saving every 5 minutes) sharing the same file. The default journal mode
        # occasionally surfaced "attempt to write a readonly database" under that two-process access pattern.
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    _ensure_indexes(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
