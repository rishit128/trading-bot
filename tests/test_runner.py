from types import SimpleNamespace

from src.engine.risk_engine import RiskDecision
from src.pipeline import SymbolResult
from src.runner import format_result, run_cycle, run_loop


def make(market_open, results=(), raises=None):
    msgs = []

    def clock():
        if raises:
            raise raises
        return market_open

    pipeline = SimpleNamespace(
        broker=SimpleNamespace(is_market_open=clock),
        run_once=lambda: list(results),
        notify=msgs.append,
    )
    return pipeline, msgs


def test_loop_survives_a_failing_cycle_and_reports_it():
    pipeline, msgs = make(True, raises=ConnectionError("network down"))
    sleeps = []
    run_loop(pipeline, 5, sleep=sleeps.append, max_cycles=3, out=lambda s: None)
    assert len(msgs) == 3 and "network down" in msgs[0] and "CYCLE FAILED" in msgs[0]
    assert sleeps == [300, 300]


def test_loop_waits_when_market_closed_without_running_pipeline():
    pipeline, msgs = make(False, results=[object()])  # run_once output would break formatting if it ran
    out = []
    run_loop(pipeline, 1, sleep=lambda s: None, max_cycles=1, out=out.append)
    assert out == ["market closed; waiting"] and msgs == []


def test_loop_runs_cycle_when_market_open():
    r = SymbolResult("AAPL", "BUY", 0.8, RiskDecision(True, 10, "approved 10 shares"), "DRY_RUN")
    pipeline, _ = make(True, results=[r])
    out = []
    run_loop(pipeline, 1, sleep=lambda s: None, max_cycles=1, out=out.append)
    assert len(out) == 1 and "AAPL" in out[0] and "DRY_RUN" in out[0]


def test_format_result_handles_errors_and_missing_risk():
    text = format_result(SymbolResult("BAD", "ERROR", 0.0, None, None, "ValueError: no data"))
    assert "BAD" in text and "ValueError: no data" in text


def test_run_cycle_prints_every_result():
    rs = [SymbolResult(s, "HOLD", 0.5, None, None) for s in ("A", "B")]
    pipeline, _ = make(True, results=rs)
    out = []
    run_cycle(pipeline, out.append)
    assert len(out) == 2
