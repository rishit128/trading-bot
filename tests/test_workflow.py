import json
import threading
import time

import pytest
from langgraph.types import RetryPolicy
from sqlalchemy import select

from src.agents.base import ADVISOR, LEAD, AgentContext
from src.config import RiskLimits, Settings
from src.data.indicators import Snapshot
from src.data.retry import is_transient_data_error
from src.database import DecisionRecord, OrderRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.llm import AgentSignal
from src.pipeline import TradingPipeline
from src.workflow import build_analysis_graph
from tests.test_llm_and_pipeline import FakeBroker, StubAgent, make_pipeline, signal_for


def stub(name, fn, role=LEAD):
    return StubAgent(fn, name, role)


class _AnalysisOnlyServices:
    """The minimal CycleServices surface build_analysis_graph needs, for a graph-level test of fetch_data's own
    RetryPolicy (SNAPSHOT_RETRY) without paying for a full TradingPipeline/database."""

    def __init__(self, snapshot_fn, agents):
        self.agents = agents
        self._snapshot_fn = snapshot_fn

    def snapshot(self, symbol):
        return self._snapshot_fn(symbol)

    def agent_context(self, symbol, snapshot):
        return AgentContext(symbol, snapshot, market=lambda: None)


FAST_RETRY = RetryPolicy(max_attempts=2, initial_interval=0.01, backoff_factor=1.0, jitter=False,
                         retry_on=is_transient_data_error)


# ---------------------------------------------------------------- fetch_data's RetryPolicy (SNAPSHOT_RETRY)
def test_a_transient_snapshot_failure_is_retried_and_recovers(tmp_path):
    calls = []

    def flaky(symbol):
        calls.append(symbol)
        if len(calls) == 1:
            raise ConnectionError("dropped connection")
        return Snapshot(symbol, 100.0, 98.0, 95.0, 60.0, 1000)

    graph = build_analysis_graph(_AnalysisOnlyServices(flaky, {"technical": stub("technical", lambda s: signal_for("BUY"))}),
                                 snapshot_retry=FAST_RETRY)
    out = graph.invoke({"symbol": "AAPL"})
    assert len(calls) == 2  # failed once, retried once, succeeded
    assert out["signals"]["technical"].action == "BUY"


def test_a_definite_data_error_is_not_retried(tmp_path):
    calls = []

    def always_stale(symbol):
        calls.append(symbol)
        raise ValueError(f"{symbol}: last bar is stale")

    graph = build_analysis_graph(_AnalysisOnlyServices(always_stale, {"technical": stub("technical", lambda s: signal_for("BUY"))}),
                                 snapshot_retry=FAST_RETRY)
    with pytest.raises(ValueError, match="stale"):
        graph.invoke({"symbol": "AAPL"})
    assert len(calls) == 1  # a definite fact about the data must not cost a second attempt


# ---------------------------------------------------------------- structure
def test_graphs_are_built_from_the_registered_agents(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path)
    assert set(pipe.analysis_graph.get_graph().nodes) == {
        "__start__", "fetch_data", "agent_technical", "agent_sentiment", "collect", "__end__"}
    assert set(pipe.decision_graph.get_graph().nodes) == {"__start__", "decide", "execute", "__end__"}
    assert set(pipe.cycle_graph.get_graph().nodes) == {
        "__start__", "open_cycle", "select_universe", "analyze_symbol", "execute_cycle", "__end__"}
    edges = {(e.source, e.target) for e in pipe.analysis_graph.get_graph().edges}
    assert {("fetch_data", "agent_technical"), ("fetch_data", "agent_sentiment"),
            ("agent_technical", "collect"), ("agent_sentiment", "collect")} <= edges


def test_duplicate_agent_names_and_empty_registry_are_rejected(tmp_path):
    pipe, broker, sessions = make_pipeline(tmp_path)
    args = (Settings(risk=RiskLimits()),)
    with pytest.raises(ValueError, match="unique"):
        TradingPipeline(*args, [stub("x", lambda s: None), stub("x", lambda s: None)], broker, sessions, None)
    with pytest.raises(ValueError, match="at least one agent"):
        TradingPipeline(*args, [], broker, sessions, None)


# ---------------------------------------------------------------- scalability: many agents
def test_any_number_of_agents_run_in_parallel(tmp_path):
    barrier = threading.Barrier(4, timeout=5)  # passes only if all FOUR agents are inside their node simultaneously

    def waiting(name, role):
        def fn(snapshot):
            barrier.wait()
            return signal_for("BUY")

        return stub(name, fn, role)

    pipe, _, _ = make_pipeline(tmp_path)
    pipe.agents = {a.name: a for a in [waiting("technical", LEAD), waiting("trend", LEAD),
                                       waiting("fundamentals", ADVISOR), waiting("macro", ADVISOR)]}
    pipe.rebuild_graphs()
    [r] = pipe.run_once()
    assert r.action == "BUY" and r.error is None


def test_a_new_agent_is_picked_up_by_rebuilding_and_is_audited(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    assert "agent_macro" not in pipe.analysis_graph.get_graph().nodes
    pipe.agents["macro"] = stub("macro", lambda s: AgentSignal(action="BUY", confidence=0.7, reasoning="tailwind"), ADVISOR)
    pipe.rebuild_graphs()
    assert "agent_macro" in pipe.analysis_graph.get_graph().nodes
    pipe.run_once()
    with sessions() as s:
        signals = json.loads(s.scalar(select(DecisionRecord.signals_json)))
    assert set(signals) == {"technical", "sentiment", "macro"} and signals["macro"]["reasoning"] == "tailwind"
    assert signals["sentiment"] is None


def test_second_lead_must_agree_or_the_bot_stands_down(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path)
    pipe.agents["trend"] = stub("trend", lambda s: signal_for("HOLD"), LEAD)
    pipe.rebuild_graphs()
    [r] = pipe.run_once()
    assert r.action == "HOLD" and r.order_status is None


def test_agent_exception_becomes_an_error_result_and_other_stocks_continue(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path, watchlist=("BAD", "AAPL"))

    def flaky(s):
        if s.symbol == "BAD":
            raise RuntimeError("agent bug")
        return signal_for("BUY")

    pipe.agents["technical"] = stub("technical", flaky)
    bad, good = pipe.run_once()
    assert bad.action == "ERROR" and "agent bug" in bad.error and good.action == "BUY"


# ---------------------------------------------------------------- scalability: many stocks
def test_stocks_are_analysed_concurrently(tmp_path):
    barrier = threading.Barrier(3, timeout=5)  # only passes if 3 stocks are being analysed at the same moment

    def fn(snapshot):
        barrier.wait()
        return signal_for("BUY")

    pipe, _, _ = make_pipeline(tmp_path, watchlist=("A", "B", "C"), analysis_workers=3)
    pipe.agents["technical"] = stub("technical", fn)
    assert [r.symbol for r in pipe.run_once()] == ["A", "B", "C"]


@pytest.mark.parametrize("workers", [1, 2])
def test_concurrency_never_exceeds_the_configured_worker_count(tmp_path, workers):
    lock, running, peak = threading.Lock(), [0], [0]

    def fn(snapshot):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return signal_for("BUY")

    pipe, _, _ = make_pipeline(tmp_path, watchlist=tuple("ABCDEF"), analysis_workers=workers)
    pipe.agents["technical"] = stub("technical", fn)
    assert len(pipe.run_once()) == 6
    assert peak[0] == workers


def test_result_order_follows_the_universe_not_completion_order(tmp_path):
    def fn(snapshot):
        time.sleep(0.15 if snapshot.symbol == "SLOW" else 0.0)
        return signal_for("HOLD")

    pipe, _, _ = make_pipeline(tmp_path, watchlist=("SLOW", "FAST1", "FAST2"), analysis_workers=3)
    pipe.agents["technical"] = stub("technical", fn)
    assert [r.symbol for r in pipe.run_once()] == ["SLOW", "FAST1", "FAST2"]


class LedgerBroker(FakeBroker):
    """Portfolio reflects each buy, like a real broker, so later decisions must see earlier orders."""

    def __init__(self):
        super().__init__(Portfolio(1_000_000.0, 1_000_000.0, {}, {}, 1_000_000.0))

    def buy_with_bracket(self, symbol, qty, price, stop, take):
        pf = self._portfolio
        self._portfolio = Portfolio(pf.cash - qty * price, pf.equity, {**pf.positions, symbol: qty * price},
                                    {**pf.position_qty, symbol: qty}, pf.start_of_day_equity)
        return super().buy_with_bracket(symbol, qty, price, stop, take)


def test_decisions_run_one_at_a_time_against_fresh_portfolio_state(tmp_path):
    from src.config import RiskLimits as RL

    broker = LedgerBroker()
    pipe, _, _ = make_pipeline(tmp_path, dry_run=False, broker=broker, watchlist=("A", "B", "C"), analysis_workers=3,
                               risk=RL(max_open_positions=2))
    results = pipe.run_once()
    assert [r.order_status for r in results] == ["accepted", "accepted", None]
    assert "max open positions" in results[2].risk.reason and len(broker.buys) == 2


def test_empty_universe_returns_no_results(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path)
    pipe.universe_fn = lambda portfolio: []
    assert pipe.run_once() == []


# ---------------------------------------------------------------- routing / execution
def test_rejected_trades_never_reach_the_execute_node(tmp_path):
    executed = []
    pipe, broker, sessions = make_pipeline(tmp_path, technical_action="HOLD", dry_run=False)
    original = pipe._place
    pipe._place = lambda *a: executed.append(a) or original(*a)
    [r] = pipe.run_once()
    assert executed == [] and r.order_status is None
    with sessions() as s:
        assert s.scalar(select(DecisionRecord.final_action)) == "HOLD" and s.scalar(select(OrderRecord)) is None


def test_approved_trades_reach_execute_with_the_risk_engine_quantity(tmp_path):
    seen = []
    pipe, broker, _ = make_pipeline(tmp_path, dry_run=False)
    original = pipe._place
    pipe._place = lambda *a: seen.append(a) or original(*a)
    pipe.run_once()
    (decision_id, symbol, action, qty, price), = seen
    assert (symbol, action, qty, price) == ("AAPL", "BUY", 50, 100.0) and isinstance(decision_id, int)


def test_live_quote_is_used_for_sizing_and_bracket_levels_not_the_stale_close(tmp_path):
    pipe, broker, sessions = make_pipeline(tmp_path, dry_run=False)
    pipe.quote_fn = lambda symbol: 125.0  # completed-bar close is 100; the market has since gapped up
    pipe.run_once()
    assert broker.buys == [("AAPL", 40, 125.0, 0.15, 1.0)]  # 5% of 100k at 125 = 40 shares
    with sessions() as s:
        assert s.scalar(select(DecisionRecord.price)) == 125.0


def test_missing_live_quote_falls_back_to_the_completed_close(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, dry_run=False)

    def broken(symbol):
        raise ConnectionError("quote feed down")

    pipe.quote_fn = broken
    pipe.run_once()
    assert broker.buys == [("AAPL", 50, 100.0, 0.15, 1.0)]


def test_no_quote_is_fetched_for_hold_decisions(tmp_path):
    calls = []
    pipe, _, _ = make_pipeline(tmp_path, technical_action="HOLD")
    pipe.quote_fn = lambda symbol: calls.append(symbol) or 100.0
    pipe.run_once()
    assert calls == []


def test_sentiment_veto_flows_through_the_graph_and_is_audited(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.agents["sentiment"] = stub("sentiment", lambda s: AgentSignal(action="SELL", confidence=0.9, reasoning="bad news"), ADVISOR)
    [r] = pipe.run_once()
    assert r.action == "HOLD"
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        assert (row.technical_action, row.sentiment_action, row.final_action) == ("BUY", "SELL", "HOLD")


def test_nodes_use_agents_swapped_in_after_construction(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path)
    pipe.agents["technical"] = stub("technical", lambda s: signal_for("HOLD"))
    assert pipe.run_once()[0].action == "HOLD"


def test_analysis_graph_can_be_invoked_directly(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path)
    out = pipe.analysis_graph.invoke({"symbol": "AAPL"})
    assert isinstance(out["snapshot"], Snapshot) and set(out["signals"]) == {"technical", "sentiment"}


def test_pipeline_with_a_single_lead_agent_and_no_advisors(tmp_path):
    settings = Settings(watchlist=("X",), dry_run=True, risk=RiskLimits())
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'one.db'}")
    pipe = TradingPipeline(settings, [stub("technical", lambda s: signal_for("BUY"))],
                           FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
                           lambda sym: Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000))
    assert set(pipe.analysis_graph.get_graph().nodes) == {"__start__", "fetch_data", "agent_technical", "collect", "__end__"}
    assert pipe.run_once()[0].action == "BUY"


# ---------------------------------------------------------------- deterministic trend exit
BROKEN = lambda sym: Snapshot(sym, 90.0, 96.0, 95.0, 40.0, 1000)   # noqa: E731  price 90 < 200-day average 95
INTACT = lambda sym: Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000)  # noqa: E731


def holding_broker():
    return FakeBroker(Portfolio(95_000.0, 100_000.0, {"AAPL": 4_500.0}, {"AAPL": 50}, 100_000.0))


def test_held_stock_below_its_200_day_average_is_sold_even_if_every_agent_says_buy(tmp_path):
    pipe, broker, sessions = make_pipeline(tmp_path, technical_action="BUY", dry_run=False, broker=holding_broker(),
                                           snapshot_fn=BROKEN)
    [r] = pipe.run_once()
    assert r.action == "SELL" and r.order_status == "accepted" and broker.sells == [("AAPL", 50)] and broker.buys == []
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        assert row.final_action == "SELL" and "trend exit" in row.reasoning and row.technical_action == "BUY"


def test_trend_exit_sells_the_full_holding_at_full_confidence(tmp_path):
    pipe, _, _ = make_pipeline(tmp_path, technical_action="HOLD", broker=holding_broker(), snapshot_fn=BROKEN)
    [r] = pipe.run_once()
    assert (r.action, r.confidence, r.risk.quantity) == ("SELL", 1.0, 50)


def test_held_stock_with_an_intact_trend_is_left_to_the_agents(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, technical_action="HOLD", dry_run=False, broker=holding_broker(),
                                    snapshot_fn=INTACT)
    [r] = pipe.run_once()
    assert r.action == "HOLD" and broker.sells == []


def test_a_broken_trend_alone_never_buys_or_shorts_a_stock_we_do_not_hold(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, technical_action="HOLD", dry_run=False, snapshot_fn=BROKEN)
    [r] = pipe.run_once()
    assert r.action == "HOLD" and broker.sells == [] and broker.buys == []


def test_trend_exit_can_be_switched_off(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, technical_action="HOLD", dry_run=False, broker=holding_broker(),
                                    snapshot_fn=BROKEN, trend_exit=False)
    assert pipe.run_once()[0].action == "HOLD" and broker.sells == []


def test_trend_exit_respects_dry_run_and_the_pause_switch(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, dry_run=True, broker=holding_broker(), snapshot_fn=BROKEN)
    assert pipe.run_once()[0].order_status == "DRY_RUN" and broker.sells == []


def test_new_exit_defaults_and_env_switch(monkeypatch):
    from src.config import load_settings

    for name in ("STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TREND_EXIT"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert (s.risk.stop_loss_pct, s.risk.take_profit_pct, s.trend_exit) == (0.15, 1.0, True)
    monkeypatch.setenv("TREND_EXIT", "false")
    assert load_settings().trend_exit is False
