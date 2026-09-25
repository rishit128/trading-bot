"""Shared scenario builders for the characterization (golden) tests: a fully deterministic trading cycle whose every
outcome is pinned, so restructuring the pipeline, the graphs or the agents cannot change behaviour unnoticed."""
from datetime import datetime, timezone

from sqlalchemy import select

from src.agents.base import LEAD
from src.config import RiskLimits, Settings
from src.data.indicators import Snapshot
from src.database import DecisionRecord, OrderRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.llm import AgentSignal
from src.pipeline import TradingPipeline
from tests.test_llm_and_pipeline import FakeBroker, StubAgent

# symbol -> (price, ma50, ma200, rsi, agent action, agent confidence)
UNIVERSE = {
    "NEWBUY": (100.0, 98.0, 95.0, 60.0, "BUY", 0.90),       # a clean approved entry
    "LOWCONF": (100.0, 98.0, 95.0, 60.0, "BUY", 0.50),      # below the 0.60 minimum confidence
    "HOLDER": (100.0, 98.0, 95.0, 60.0, "HOLD", 0.70),      # the agent stands aside
    "HELD1": (100.0, 98.0, 95.0, 60.0, "BUY", 0.90),        # already holds 20% of equity: no room to add
    "HELD2": (90.0, 92.0, 95.0, 55.0, "BUY", 0.90),         # price below MA200: the deterministic trend exit sells it
    "BROKEN": None,                                          # the data fetch raises
    "OPENORD": (100.0, 98.0, 95.0, 60.0, "BUY", 0.90),      # an order is already open at the broker
    "COOL": (100.0, 98.0, 95.0, 60.0, "BUY", 0.90),         # bought within the last 24 hours
    "SELLNOTHELD": (100.0, 98.0, 95.0, 60.0, "SELL", 0.90),  # a SELL for something we do not hold
}


def build(tmp_path, dry_run=False, risk=None):
    portfolio = Portfolio(cash=60_000.0, equity=100_000.0, positions={"HELD1": 20_000.0, "HELD2": 20_000.0},
                          position_qty={"HELD1": 200, "HELD2": 100}, start_of_day_equity=100_000.0)
    broker = FakeBroker(portfolio, open_orders={"OPENORD"})

    def snapshot(symbol):
        row = UNIVERSE[symbol]
        if row is None:
            raise ValueError(f"{symbol}: no market data returned")
        return Snapshot(symbol, row[0], row[1], row[2], row[3], 1_000_000, bar_date="2026-09-24")

    def decide(snap):
        _, _, _, _, action, confidence = UNIVERSE[snap.symbol]
        return AgentSignal(action=action, confidence=confidence, reasoning=f"{snap.symbol} scripted")

    sessions = make_session_factory(f"sqlite:///{tmp_path / 'golden.db'}")
    with sessions() as s:  # COOL was bought an hour ago
        s.add(OrderRecord(decision_id=0, symbol="COOL", side="BUY", quantity=5, status="filled",
                          created_at=datetime.now(timezone.utc)))
        s.commit()
    messages = []
    pipeline = TradingPipeline(
        Settings(watchlist=tuple(UNIVERSE), dry_run=dry_run, risk=risk or RiskLimits(), analysis_workers=3),
        [StubAgent(decide, "technical", LEAD)], broker, sessions, snapshot, lambda sym: [], notify=messages.append)
    return pipeline, broker, sessions, messages


def observe(pipeline, broker, sessions, messages):
    """Everything observable about one cycle, as plain data."""
    results = pipeline.run_once()
    with sessions() as s:
        decisions = [(d.symbol, d.final_action, round(d.final_confidence, 3), d.risk_approved, d.risk_quantity,
                      d.risk_reason.split(" (")[0][:60]) for d in s.scalars(select(DecisionRecord).order_by(DecisionRecord.id))]
        orders = [(o.symbol, o.side, o.quantity, o.status) for o in s.scalars(select(OrderRecord).order_by(OrderRecord.id))]
    return {
        "results": [(r.symbol, r.action, round(r.confidence, 3), r.risk.approved if r.risk else None,
                     r.risk.quantity if r.risk else None, r.order_status, r.error) for r in results],
        "broker_buys": broker.buys, "broker_sells": broker.sells, "decisions": decisions, "orders": orders,
        "alerts": messages,
    }
