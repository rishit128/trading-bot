import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import select

from src.agents.base import ADVISOR, LEAD
from src.config import RiskLimits, Settings
from src.control import Control
from src.database import DecisionRecord, EquityRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.logging_setup import JsonFormatter, TextFormatter, configure_logging
from src.monitoring.telegram import HELP, handle_command
from src.preflight import (CheckResult, build_checks, broker_check, critical_failures, format_results, market_data_check,
                           openrouter_check, run_checks, telegram_check)
from src.replay import replay
from tests.test_llm_and_pipeline import FakeBroker, make_pipeline


# ---------------------------------------------------------------- logging
def capture(fmt, level="INFO"):
    stream = io.StringIO()
    configure_logging(level, fmt, stream=stream)
    return stream


def test_json_logs_are_one_object_per_line_with_structured_fields():
    stream = capture("json")
    logging.getLogger("t").info("hello %s", "world", extra={"symbol": "RELIANCE", "agents": {"technical": "BUY(0.78)"}})
    row = json.loads(stream.getvalue().strip())
    assert row["msg"] == "hello world" and row["level"] == "INFO" and row["logger"] == "t"
    assert row["symbol"] == "RELIANCE" and row["agents"] == {"technical": "BUY(0.78)"} and row["ts"].endswith("+00:00")


def test_json_logs_include_tracebacks():
    stream = capture("json")
    try:
        1 / 0
    except ZeroDivisionError:
        logging.getLogger("t").exception("boom")
    assert "ZeroDivisionError" in json.loads(stream.getvalue().strip())["exc"]


def test_text_logs_append_structured_fields_as_key_value():
    stream = capture("text")
    logging.getLogger("t").info("done", extra={"symbol": "TCS", "action": "BUY"})
    assert "done | symbol=TCS action=BUY" in stream.getvalue()
    stream2 = capture("text")
    logging.getLogger("t").info("plain")
    assert stream2.getvalue().strip().endswith("plain")


def test_log_level_filters_and_is_validated():
    stream = capture("text", level="WARNING")
    logging.getLogger("t").info("quiet")
    logging.getLogger("t").warning("loud")
    assert "quiet" not in stream.getvalue() and "loud" in stream.getvalue()
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        configure_logging("CHATTY")
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        configure_logging("INFO", "xml")


def test_reconfiguring_replaces_handlers_instead_of_duplicating_lines():
    capture("text")
    stream = capture("text")
    logging.getLogger("t").info("once")
    assert stream.getvalue().count("once") == 1 and len(logging.getLogger().handlers) == 1


def test_noisy_libraries_are_silenced_so_the_telegram_token_cannot_appear_in_logs():
    capture("text", level="DEBUG")
    assert logging.getLogger("httpx").level == logging.WARNING and logging.getLogger("openai").level == logging.WARNING


def test_each_decision_is_logged_with_every_agents_signal(tmp_path, caplog):
    pipe, _, _ = make_pipeline(tmp_path)
    with caplog.at_level(logging.INFO, logger="src.workflow"):
        pipe.run_once()
    rec = next(r for r in caplog.records if getattr(r, "symbol", None) == "AAPL")
    assert rec.action == "BUY" and rec.agents["technical"] == "BUY(0.90)" and rec.agents["sentiment"] == "none"
    assert "agents=technical:BUY(0.90)" in rec.getMessage()


# ---------------------------------------------------------------- preflight
def test_run_checks_collects_results_and_never_raises():
    def boom():
        raise ConnectionError("dns")

    results = run_checks([("a", True, lambda: "fine"), ("b", True, boom), ("c", False, boom)])
    assert [r.ok for r in results] == [True, False, False]
    assert results[1].detail.startswith("ConnectionError: dns")
    assert [r.name for r in critical_failures(results)] == ["b"]   # only the critical failure blocks startup


def test_results_are_formatted_as_ok_fail_or_warn():
    text = format_results([CheckResult("k", True, "ok", True), CheckResult("m", False, "x", True), CheckResult("d", False, "y", False)])
    assert "OK   k" in text and "FAIL m" in text and "WARN d" in text


def resp(status, payload=None):
    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")

    return SimpleNamespace(status_code=status, json=lambda: payload or {}, raise_for_status=raise_for_status)


def test_openrouter_check_accepts_a_good_key_and_rejects_a_bad_one_without_using_quota():
    seen = {}

    def get(url, **kw):
        seen.update(url=url, **kw)
        return resp(200)

    assert openrouter_check("sk-or-x", get) == "key accepted"
    assert seen["url"].endswith("/auth/key") and seen["headers"]["Authorization"] == "Bearer sk-or-x"
    with pytest.raises(RuntimeError, match="rejected the key"):
        openrouter_check("bad", lambda url, **kw: resp(401))


def test_market_data_check_needs_actual_rows():
    import pandas as pd

    assert "3 recent bars" in market_data_check(lambda *a, **k: pd.DataFrame({"Close": [1, 2, 3]}), "^NSEI")
    with pytest.raises(RuntimeError, match="no data"):
        market_data_check(lambda *a, **k: pd.DataFrame(), "^NSEI")


def test_broker_and_telegram_checks_report_what_they_found():
    broker = FakeBroker(Portfolio(90_000.0, 100_000.0, {"A": 1.0}, {"A": 1}, 100_000.0))
    assert "equity 100,000.00" in broker_check(broker, "paper") and "1 positions" in broker_check(broker, "paper")
    assert telegram_check("tok", lambda url, **kw: resp(200, {"result": {"username": "my_bot"}})) == "bot @my_bot"


def test_build_checks_marks_keys_and_broker_critical_and_covers_the_market():
    india = build_checks(Settings(market="india"), FakeBroker(Portfolio(1, 1)), {"OPENROUTER_API_KEY": "k"}, download=lambda *a, **k: None)
    assert [(n.split(" (")[0], crit) for n, crit, _ in india] == [("OpenRouter key", True), ("Broker", True), ("Market data", False)]
    with_tg = build_checks(Settings(market="india"), FakeBroker(Portfolio(1, 1)), {"TELEGRAM_BOT_TOKEN": "t"}, download=lambda *a, **k: None)
    assert with_tg[-1][0] == "Telegram bot" and with_tg[-1][1] is False


def test_the_market_data_probe_uses_the_nifty_index():
    seen = []
    checks = build_checks(Settings(market="india"), FakeBroker(Portfolio(1, 1)), {}, download=lambda t, **k: seen.append(t))
    run_checks([c for c in checks if c[0].startswith("Market data")])
    assert seen == ["^NSEI"]


# ---------------------------------------------------------------- drawdown baseline reset
def test_rebase_persists_and_reads_back_an_aware_timestamp(tmp_path):
    control = Control(make_session_factory(f"sqlite:///{tmp_path / 'c.db'}"))
    assert control.peak_since() is None
    stamp = datetime(2026, 9, 21, 4, 0, tzinfo=timezone.utc)
    control.rebase_peak(stamp)
    assert control.peak_since() == stamp


def test_rebase_clears_a_drawdown_halt_and_only_that_halt(tmp_path):
    broker = FakeBroker(Portfolio(75_000.0, 75_000.0, {}, {}, 75_000.0))
    pipe, _, sessions = make_pipeline(tmp_path, broker=broker)
    pipe.control = Control(sessions)
    messages = []
    pipe._notify = messages.append
    pipe._record_equity(100_000.0)                               # an old peak
    assert "drawdown" in pipe.run_once()[0].risk.reason           # halted
    pipe.control.rebase_peak(datetime.now(timezone.utc) + timedelta(seconds=1))
    with sessions() as s:                                          # equity recorded after the rebase point
        s.add(EquityRecord(equity=75_000.0, created_at=datetime.now(timezone.utc) + timedelta(seconds=2)))
        s.commit()
    result = pipe.run_once()[0]
    assert result.risk.approved and any("halt cleared" in m for m in messages)


def test_rebase_leaves_the_daily_loss_limit_in_force(tmp_path):
    broker = FakeBroker(Portfolio(97_000.0, 97_000.0, {}, {}, 100_000.0))  # -3% on the day
    pipe, _, sessions = make_pipeline(tmp_path, broker=broker)
    pipe.control = Control(sessions)
    pipe.control.rebase_peak()
    assert "daily loss" in pipe.run_once()[0].risk.reason


def test_rebase_command_replies_and_help_lists_it(tmp_path):
    broker = FakeBroker(Portfolio(75_000.0, 75_000.0, {}, {}, 75_000.0))
    control = Control(make_session_factory(f"sqlite:///{tmp_path / 'c.db'}"))
    settings = Settings(dry_run=True, risk=RiskLimits(), currency="₹")
    reply = handle_command("/rebase", control, broker, settings)
    assert "₹75,000.00" in reply and "daily-loss limit is unaffected" in reply and control.peak_since() is not None
    assert "/rebase" in HELP and handle_command("/help", control, broker, settings) == HELP


# ---------------------------------------------------------------- replay
def test_replay_reproduces_a_stored_decision_from_stored_inputs(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        result = replay(s.scalar(select(DecisionRecord)))
    assert result.inputs.symbol == "AAPL" and result.inputs.ma200 == 95.0
    assert "AAPL" in result.prompt and "RSI(14): 60.0" in result.prompt
    assert set(result.signals) == {"technical", "sentiment"} and result.signals["sentiment"] is None
    assert (result.stored_action, result.recomputed_action, result.reproduced) == ("BUY", "BUY", True)


def test_replay_flags_a_decision_the_rules_no_longer_reproduce(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        row.final_action = "SELL"
        assert replay(row).reproduced is False


def test_replay_handles_rows_recorded_before_inputs_were_stored(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        row.snapshot_json = None
        row.signals_json = None
        result = replay(row)
    assert result.inputs is None and result.prompt is None and result.reproduced is None


def test_replay_checks_a_trend_exit_against_the_stored_indicators(tmp_path):
    from tests.test_workflow import BROKEN, holding_broker

    pipe, _, sessions = make_pipeline(tmp_path, broker=holding_broker(), snapshot_fn=BROKEN)
    pipe.run_once()
    with sessions() as s:
        result = replay(s.scalar(select(DecisionRecord)))
    assert result.rule_based and result.recomputed_action == "SELL" and result.reproduced is True


def test_replay_roles_come_from_the_agent_registry_when_given(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        assert replay(row, {"technical": ADVISOR, "sentiment": ADVISOR}).recomputed_action == "HOLD"  # no lead -> no trade
        assert replay(row, {"technical": LEAD}).recomputed_action == "BUY"


# ---------------------------------------------------------------- continuous integration
def test_ci_workflow_runs_lint_and_tests_on_the_pinned_python():
    wf = yaml.safe_load((Path(__file__).resolve().parent.parent / ".github/workflows/tests.yml").read_text())
    steps = wf["jobs"]["test"]["steps"]
    runs = [s["run"] for s in steps if "run" in s]
    assert "python -m pytest -q" in runs and any("pyflakes" in r for r in runs)
    assert any(s.get("with", {}).get("python-version") == "3.13" for s in steps)
    assert "pyflakes" in (Path(__file__).resolve().parent.parent / "requirements-dev.txt").read_text()


def test_json_and_text_formatters_are_exported_for_reuse():
    assert JsonFormatter().format(logging.LogRecord("n", logging.INFO, "f", 1, "m", (), None)).startswith("{")
    assert "m" in TextFormatter().format(logging.LogRecord("n", logging.INFO, "f", 1, "m", (), None))


def test_drawdown_halt_ends_by_itself_after_the_pause_and_only_then(tmp_path):
    broker = FakeBroker(Portfolio(75_000.0, 75_000.0, {}, {}, 75_000.0))
    pipe, _, sessions = make_pipeline(tmp_path, broker=broker)
    pipe.control = Control(sessions)
    messages = []
    pipe._notify = messages.append
    pipe._record_equity(100_000.0)                                  # an old peak: equity is 25% below it
    assert "drawdown" in pipe.run_once()[0].risk.reason              # day 0: halted, the cool-off clock starts
    assert pipe.control.drawdown_since() is not None
    assert "drawdown" in pipe.run_once()[0].risk.reason              # still inside the 30 days
    pipe.control.set_drawdown_since(datetime.now(timezone.utc) - timedelta(days=31))
    result = pipe.run_once()[0]                                      # cool-off over: peak rebased, buying resumes
    assert result.risk.approved
    assert any("Drawdown pause over" in m for m in messages) and pipe.control.drawdown_since() is None
    assert pipe.control.peak_since() is not None


def test_a_zero_pause_leaves_the_drawdown_halt_permanent(tmp_path):
    import dataclasses
    broker = FakeBroker(Portfolio(75_000.0, 75_000.0, {}, {}, 75_000.0))
    pipe, _, sessions = make_pipeline(tmp_path, broker=broker)
    pipe.settings = dataclasses.replace(pipe.settings, risk=dataclasses.replace(pipe.settings.risk, drawdown_pause_days=0))
    pipe.control = Control(sessions)
    pipe._record_equity(100_000.0)
    pipe.control.set_drawdown_since(datetime.now(timezone.utc) - timedelta(days=400))
    assert "drawdown" in pipe.run_once()[0].risk.reason
