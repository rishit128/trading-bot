"""The workflow graphs depend on `CycleServices`, not on `TradingPipeline`. These tests run the real graphs against a small
fake, prove the pipeline satisfies the interface, and stop private access from creeping back in."""
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

from src.agents.base import ADVISOR, LEAD
from src.data.indicators import Snapshot
from src.engine.enums import Action
from src.engine.risk_engine import Portfolio, RiskDecision
from src.engine.agent_signal import AgentSignal
from src.engine.strategy import Decision
from src.pipeline import TradingPipeline
from src.results import Verdict
from tests.test_llm_and_pipeline import make_pipeline
from src.workflow import CycleServices, build_analysis_graph, build_cycle_graph, build_decision_graph


class FakeServices:
    """Records every call the graphs make. No broker, no database, no risk engine: only the interface."""

    def __init__(self, symbols, dry_run=False, approve=(), broken=(), degraded=()):
        self.calls, self.alerts = [], []
        self.symbols, self.approve, self.broken, self.degraded = list(symbols), set(approve), set(broken), set(degraded)
        self.settings = SimpleNamespace(dry_run=dry_run)
        agent = SimpleNamespace(name="technical", role=LEAD, analyze=self._analyze)
        advisor = SimpleNamespace(name="sentiment", role=ADVISOR, analyze=lambda ctx: None)
        self.agents = {"technical": agent, "sentiment": advisor}
        self.portfolio = Portfolio(cash=1_000.0, equity=1_000.0)

    def _analyze(self, ctx):
        self.calls.append(("analyze", ctx.symbol))
        return AgentSignal(action="BUY", confidence=0.9, reasoning="r", degraded=ctx.symbol in self.degraded)

    def snapshot(self, symbol):
        self.calls.append(("snapshot", symbol))
        if symbol in self.broken:
            raise ValueError(f"{symbol}: no data")
        return Snapshot(symbol, 100.0, 98.0, 95.0, 60.0, 1000)

    def agent_context(self, symbol, snapshot):
        return SimpleNamespace(symbol=symbol, snapshot=snapshot)

    def open_cycle(self):
        self.calls.append(("open_cycle",))
        return self.portfolio

    def select_symbols(self, portfolio):
        self.calls.append(("select_symbols",))
        return self.symbols

    def decide(self, symbol, portfolio, snapshot, signals, universe_size=None):
        self.calls.append(("decide", symbol, universe_size))
        ok = symbol in self.approve
        return Verdict(Decision(Action.BUY, 0.9, "r"), RiskDecision(ok, 5 if ok else 0, "ok" if ok else "no"), 100.0, len(self.calls))

    def place(self, symbol, verdict):
        self.calls.append(("place", symbol, verdict.risk.quantity))
        return "filled"

    def refresh_portfolio(self):
        self.calls.append(("refresh",))
        return self.portfolio

    def note_lead_health(self, degraded):
        self.calls.append(("lead_degraded", degraded))

    def close_cycle(self):
        self.calls.append(("close_cycle",))

    def notify(self, text):
        self.alerts.append(text)


def run_cycle(services):
    analysis, decision = build_analysis_graph(services), build_decision_graph(services)
    return build_cycle_graph(services, analysis, decision).invoke({"symbols": []}, config={"max_concurrency": 3})["results"]


def kinds(services, name):
    return [c[1:] for c in services.calls if c[0] == name]


def test_the_cycle_runs_on_a_fake_with_no_pipeline_at_all():
    services = FakeServices(["AAA", "BBB", "CCC"], approve={"AAA", "CCC"})
    results = run_cycle(services)
    assert [(r.symbol, r.action, r.order_status) for r in results] == [("AAA", "BUY", "filled"), ("BBB", "BUY", None), ("CCC", "BUY", "filled")]
    assert services.calls[0] == ("open_cycle",) and services.calls[1] == ("select_symbols",)
    assert services.calls[-1] == ("close_cycle",) and [c for c in services.calls if c[0] == "close_cycle"] == [("close_cycle",)]
    assert [c[1] for c in services.calls if c[0] == "decide"] == ["AAA", "BBB", "CCC"]  # decided one at a time, in universe order
    assert kinds(services, "place") == [("AAA", 5), ("CCC", 5)]  # only approved verdicts are placed
    assert sorted(kinds(services, "analyze")) == [("AAA",), ("BBB",), ("CCC",)]
    assert {c[2] for c in services.calls if c[0] == "decide"} == {3}  # every decision knows the universe size


def test_the_account_is_refreshed_after_orders_in_live_mode_but_not_in_dry_run():
    live = FakeServices(["AAA", "BBB"], approve={"AAA", "BBB"})
    run_cycle(live)
    assert len(kinds(live, "refresh")) == 2  # after each placed order, so the next stock sees fresh cash and positions
    dry = FakeServices(["AAA", "BBB"], approve={"AAA", "BBB"}, dry_run=True)
    run_cycle(dry)
    assert kinds(dry, "refresh") == []


def test_a_stock_that_fails_becomes_an_error_result_and_an_alert_without_stopping_the_others():
    services = FakeServices(["AAA", "BAD", "CCC"], approve={"AAA", "CCC"}, broken={"BAD"})
    results = run_cycle(services)
    assert [(r.symbol, r.action) for r in results] == [("AAA", "BUY"), ("BAD", "ERROR"), ("CCC", "BUY")]
    assert results[1].error == "ValueError: BAD: no data" and services.alerts == ["ERROR processing BAD: ValueError: BAD: no data"]
    assert "BAD" not in [c[1] for c in services.calls if c[0] == "decide"]  # nothing was decided for it


def test_a_failing_fail_safe_lead_is_reported_per_stock_for_outage_detection():
    services = FakeServices(["AAA", "BBB"], degraded={"BBB"})
    run_cycle(services)
    assert kinds(services, "lead_degraded") == [(False,), (True,)]


def test_an_empty_universe_still_opens_and_closes_the_cycle():
    services = FakeServices([])
    assert run_cycle(services) == []
    assert services.calls == [("open_cycle",), ("select_symbols",), ("close_cycle",)]


def test_the_trading_pipeline_implements_every_member_of_the_interface(tmp_path):
    members = {name for name, _ in inspect.getmembers(CycleServices) if not name.startswith("_")}
    assert {"agents", "settings", "snapshot", "agent_context", "open_cycle", "select_symbols", "decide", "place",
            "refresh_portfolio", "note_lead_health", "close_cycle", "notify"} <= members
    instance = make_pipeline(tmp_path)[0]
    for name in members:
        assert hasattr(instance, name), f"TradingPipeline is missing the public member {name}"
        declared = getattr(CycleServices, name)
        if inspect.isfunction(declared):
            wanted = list(inspect.signature(declared).parameters)
            actual = list(inspect.signature(getattr(TradingPipeline, name)).parameters)
            assert actual == wanted, f"{name}: the pipeline takes {actual}, the interface says {wanted}"


def test_the_workflow_never_reaches_into_private_members_or_imports_the_pipeline():
    source = (Path(__file__).resolve().parent.parent / "src" / "workflow.py").read_text()
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert not re.search(r"\bservices\._\w+", code), "the graphs must use only the public CycleServices interface"
    assert not re.search(r"\bp\._\w+", code)
    assert "from src.pipeline" not in source and "import src.pipeline" not in source
