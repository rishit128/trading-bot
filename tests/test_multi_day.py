"""End to end over several simulated days: the real pipeline, risk engine and paper broker (with its SQLite ledger) driven
by `run_loop`. Prices are scripted so that one position is stopped out, one is closed by the trend exit and one is held;
after every day the books must balance (`reconcile_paper`) and no risk limit may be exceeded.

Time note: the broker runs on simulated days, the pipeline's 24-hour re-buy cooldown reads the real clock, so within this
test every purchase stays in cooldown. That is what makes the day-5 re-entry attempts observable."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select

from src.agents.base import LEAD
from src.config import RiskLimits, Settings
from src.data.indicators import Snapshot
from src.database import EquityRecord, OrderRecord, PaperTradeRecord, make_session_factory
from src.engine.paper_broker import PaperBroker
from src.engine.risk_engine import Portfolio
from src.llm import AgentSignal
from src.pipeline import TradingPipeline
from src.ops.reconciliation import reconcile_paper
from src.runner import run_loop
from tests.test_llm_and_pipeline import StubAgent

DAY0 = datetime(2026, 9, 21, 4, 0, tzinfo=timezone.utc)
# symbol -> per day (price, ma50, ma200)
TAPE = {
    "AAA": [(100, 98, 90), (105, 99, 90), (108, 100, 90), (84, 99, 90), (86, 98, 90), (110, 99, 90)],  # trend breaks on day 3
    "BBB": [(200, 196, 180), (204, 197, 180), (168, 197, 180), (170, 195, 180), (176, 194, 180), (190, 194, 180)],
    "CCC": [(50, 49, 45), (51, 49, 45), (52, 50, 45), (53, 50, 45), (54, 51, 45), (55, 51, 45)],
}
DAYS = len(TAPE["AAA"])


class Sim:
    """A scripted market: the feed, the clock and the simulated time move together, one day per cycle."""

    def __init__(self):
        self.day = 0
        self.now = DAY0

    # feed
    def last_price(self, symbol):
        return float(TAPE[symbol][self.day][0])

    def bars_since(self, symbol, since):
        if symbol == "BBB" and self.day == 2:  # an intraday collapse through BBB's stop, then a partial recovery
            idx = pd.DatetimeIndex([self.now + timedelta(hours=1)])
            bars = pd.DataFrame({"Open": [195.0], "High": [196.0], "Low": [165.0], "Close": [168.0]}, index=idx)
            return bars[bars.index > pd.Timestamp(since)]
        return pd.DataFrame()

    # clock
    def is_open(self, now=None):
        return True

    def today_ist(self, now=None):
        return self.now.date().isoformat()

    def advance(self, _seconds):
        self.day += 1
        self.now = DAY0 + timedelta(days=self.day)


def build(tmp_path, limits=None):
    sim = Sim()
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'sim.db'}")
    broker = PaperBroker(sessions, sim, sim, initial_cash=1_000_000.0, now_fn=lambda: sim.now)

    def snapshot(symbol):
        price, ma50, ma200 = TAPE[symbol][sim.day]
        return Snapshot(symbol, float(price), float(ma50), float(ma200), 58.0, 1_000_000, bar_date=str(sim.now.date()))

    def decide(snap):  # a plain trend follower standing in for the AI
        up = snap.price > snap.ma50 > snap.ma200
        return AgentSignal(action="BUY" if up else "HOLD", confidence=0.9, reasoning="scripted trend follower")

    alerts = []
    pipeline = TradingPipeline(Settings(watchlist=tuple(TAPE), dry_run=False, risk=limits or RiskLimits()),
                               [StubAgent(decide, "technical", LEAD)], broker, sessions, snapshot, lambda s: [],
                               notify=alerts.append)
    return sim, broker, pipeline, sessions, alerts


def run(tmp_path, limits=None, per_day=None):
    sim, broker, pipeline, sessions, alerts = build(tmp_path, limits)
    lines = []

    def after_cycle():
        if per_day:
            per_day(sim, broker, sessions)

    run_loop(pipeline, 1, sleep=sim.advance, max_cycles=DAYS, after_cycle=after_cycle, out=lines.append)
    return sim, broker, pipeline, sessions, alerts, lines


def test_six_days_open_close_and_reconcile(tmp_path):
    sim, broker, pipeline, sessions, alerts, lines = run(tmp_path)
    assert not any("CYCLE FAILED" in a or "ERROR" in a for a in alerts)
    with sessions() as s:
        trades = {t.symbol: t for t in s.scalars(select(PaperTradeRecord))}
        equity_rows = len(list(s.scalars(select(EquityRecord))))
        orders = [(o.symbol, o.side, o.status) for o in s.scalars(select(OrderRecord).order_by(OrderRecord.id))]
    assert set(trades) == {"AAA", "BBB"}                              # CCC is still held
    assert trades["BBB"].reason == "STOP" and abs(trades["BBB"].exit_price - 170.0) < 1e-6  # the stop level, not the low
    assert trades["AAA"].reason == "SIGNAL"                            # closed by the deterministic trend exit on day 3
    assert set(broker.portfolio().positions) == {"CCC"}
    assert equity_rows == DAYS                                          # one equity mark per cycle
    assert orders[:3] == [("AAA", "BUY", "filled"), ("BBB", "BUY", "filled"), ("CCC", "BUY", "filled")]
    assert ("AAA", "SELL", "filled") in orders
    assert ("AAA", "BUY", "SKIPPED_COOLDOWN") in orders               # the day-5 re-entry was refused, not silently dropped
    report = reconcile_paper(sessions)
    assert report["ok"], report["findings"]                            # cash, positions, trades and orders all tie out


def test_risk_limits_hold_on_every_simulated_day_and_the_books_balance_throughout(tmp_path):
    limits = RiskLimits()
    seen = []

    def check(sim, broker, sessions):
        pf: Portfolio = broker.portfolio()
        exposure = sum(pf.positions.values())
        assert pf.cash >= -1e-6, f"day {sim.day}: negative cash"
        assert exposure <= pf.equity * limits.max_portfolio_exposure_pct + 1, f"day {sim.day}: exposure cap broken"
        assert len(pf.positions) <= limits.max_open_positions
        for symbol, value in pf.positions.items():
            assert value <= pf.equity * limits.max_position_pct * 1.25, f"day {sim.day}: {symbol} grew far past its cap"
        assert reconcile_paper(sessions)["ok"], f"day {sim.day}: the ledger stopped tying out"
        seen.append(sim.day)

    run(tmp_path, limits, per_day=check)
    assert seen == list(range(DAYS))  # the checks really ran every day


def test_a_disabled_trend_exit_keeps_the_position_the_default_would_have_closed(tmp_path):
    sim, broker, pipeline, sessions, alerts = build(tmp_path)
    pipeline.settings = Settings(watchlist=tuple(TAPE), dry_run=False, risk=RiskLimits(), trend_exit=False)
    run_loop(pipeline, 1, sleep=sim.advance, max_cycles=DAYS, out=lambda s: None)
    # with the exit off AAA is only sold if the agent says SELL; this trend follower says HOLD, so it is still held
    assert "AAA" in broker.portfolio().positions
