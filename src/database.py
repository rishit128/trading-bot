"""SQLAlchemy tables: the audit log (decisions, orders, equity), control flags and the paper-trading account."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


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


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    """Create the engine and tables, add any missing columns to an older database, and return a session factory."""
    kwargs = {"pool_pre_ping": True}  # a dropped connection is replaced instead of failing the cycle
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"timeout": 30}  # wait for a lock instead of failing with "database is locked"
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
