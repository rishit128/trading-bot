"""Assembling the bot (src/app/bootstrap.py): start-up refusals, Telegram wiring, the startup checks, the swing run and the
standalone portfolio run. Everything external is replaced; these tests check the wiring, not the parts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.app import bootstrap
from src.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEARTBEAT_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UNIVERSE", "watchlist")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'trading.db'}")
    monkeypatch.setenv("INTRADAY_DATABASE_URL", f"sqlite:///{tmp_path / 'intraday.db'}")
    monkeypatch.setenv("GAPGO_DATABASE_URL", f"sqlite:///{tmp_path / 'gapgo.db'}")
    monkeypatch.setenv("GAPONLY_DATABASE_URL", f"sqlite:///{tmp_path / 'gaponly.db'}")


# -- refusing to start --------------------------------------------------------------------------------------------------------
def test_an_invalid_configuration_is_refused_with_its_reason(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_PCT", "7")
    with pytest.raises(SystemExit, match="Invalid configuration"):
        bootstrap.build_pipeline(live=False)


def test_a_missing_model_or_key_is_refused_before_anything_is_built(monkeypatch):
    monkeypatch.setattr(bootstrap, "load_settings_or_exit", lambda: Settings(models=()))
    with pytest.raises(SystemExit, match="OPENROUTER_MODEL"):
        bootstrap.build_pipeline(live=False)
    monkeypatch.setattr(bootstrap, "load_settings_or_exit", lambda: Settings(models=("m",)))
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        bootstrap.build_pipeline(live=False)


def test_paper_orders_switch_off_the_dry_run(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert bootstrap.build_pipeline(live=False).settings.dry_run is True
    assert bootstrap.build_pipeline(live=True).settings.dry_run is False


# -- Telegram ------------------------------------------------------------------------------------------------------------------
def test_with_telegram_configured_the_listener_serves_every_account(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    started, handled = [], {}

    class FakeListener:
        def __init__(self, token, chat_id, respond, intercept=None):
            started.append((token, chat_id))
            handled["respond"] = respond

        def start(self):
            handled["started"] = True

    monkeypatch.setattr(bootstrap, "CommandListener", FakeListener)
    pipeline = bootstrap.build_pipeline(live=False)
    assert started == [("t", "1")] and handled["started"] and pipeline.notify is not None
    respond = handled["respond"]
    # the gap accounts are wired in, not only the swing and breakout ones
    assert "GAP-AND-GO" in respond("/gapgo") and "GAP-ONLY" in respond("/gaponly") and "INTRADAY" in respond("/intraday")


def test_without_telegram_the_bot_still_builds_and_says_so(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    bootstrap.build_pipeline(live=False)
    assert "Telegram not configured" in capsys.readouterr().out


def test_a_market_context_failure_means_no_context_not_a_failed_cycle(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    class Broken:
        def context(self, symbol, snapshot):
            raise ConnectionError("yahoo down")

    monkeypatch.setattr(bootstrap, "MarketContextProvider", Broken)
    pipeline = bootstrap.build_pipeline(live=False)
    assert pipeline.market_context_fn(SimpleNamespace(symbol="AAA")) is None


# -- the startup checks --------------------------------------------------------------------------------------------------------
def test_preflight_prints_every_check_and_returns_only_the_critical_failures(monkeypatch, capsys):
    pipeline = SimpleNamespace(settings=Settings(), broker="broker")
    monkeypatch.setattr(bootstrap, "build_checks", lambda settings, broker, env: ["a check"])
    monkeypatch.setattr(bootstrap, "run_checks", lambda checks: ["result"])
    monkeypatch.setattr(bootstrap, "format_results", lambda results: "  OK   x")
    monkeypatch.setattr(bootstrap, "critical_failures", lambda results: ["a critical one"])
    assert bootstrap.preflight(pipeline) == ["a critical one"]
    assert "Startup checks:\n  OK   x" in capsys.readouterr().out


def test_check_without_a_pipeline_builds_just_the_broker(monkeypatch):
    seen = {}
    monkeypatch.setattr(bootstrap, "build_checks", lambda settings, broker, env: seen.update(broker=broker) or [])
    monkeypatch.setattr(bootstrap, "run_checks", lambda checks: [])
    monkeypatch.setattr(bootstrap, "format_results", lambda results: "")
    assert bootstrap.preflight(None) == []
    assert seen["broker"] is not None


# -- the swing run ---------------------------------------------------------------------------------------------------------------
@pytest.fixture
def swing(monkeypatch):
    """A fake pipeline and recorders for the loop, the single cycle and the daily jobs."""
    calls = []
    pipeline = SimpleNamespace(
        settings=Settings(heartbeat_file="/data/hb"),
        broker_label="paper simulator",
        universe_fn=None,
        sessions=None,
        control=None,
        notify=lambda text: calls.append(("notify", text)),
    )

    def fake_loop(p, minutes, after_cycle, heartbeat):
        calls.append(("loop", minutes, heartbeat))
        after_cycle()

    monkeypatch.setattr(bootstrap, "build_pipeline", lambda live: calls.append(("build", live)) or pipeline)
    monkeypatch.setattr(bootstrap, "preflight", lambda p: [])
    monkeypatch.setattr(bootstrap, "build_outcome_labeller", lambda p: lambda: calls.append(("label",)))
    monkeypatch.setattr(bootstrap, "WeeklyLearningReport", lambda *a: lambda: calls.append(("weekly",)))
    monkeypatch.setattr(bootstrap, "build_holdout_callback", lambda p: lambda: calls.append(("holdout",)))
    monkeypatch.setattr(bootstrap, "run_cycle", lambda p: calls.append(("cycle",)))
    monkeypatch.setattr(bootstrap, "run_loop", fake_loop)
    return calls


def test_one_cycle_runs_the_daily_jobs_after_it(swing):
    bootstrap.run_trading_bot(live=True, holdout=False, loop_minutes=None)
    assert [c[0] for c in swing] == ["build", "cycle", "label", "weekly"] and swing[0] == ("build", True)


def test_the_loop_gets_the_heartbeat_and_the_holdout_runs_before_the_other_jobs(swing):
    bootstrap.run_trading_bot(live=False, holdout=True, loop_minutes=30)
    assert [c[0] for c in swing] == ["build", "notify", "loop", "holdout", "label", "weekly"]
    assert "every 30 min" in swing[1][1] and swing[2][1] == 30
    assert str(swing[2][2].path).replace("\\", "/") == "/data/hb"


def test_a_failed_critical_startup_check_stops_the_bot_before_any_cycle(swing, monkeypatch):
    monkeypatch.setattr(bootstrap, "preflight", lambda p: [SimpleNamespace(name="OpenRouter key", detail="rejected")])
    with pytest.raises(SystemExit, match=r"Startup checks failed: OpenRouter key \(rejected\)"):
        bootstrap.run_trading_bot(live=False, holdout=False, loop_minutes=30)
    assert [c[0] for c in swing] == ["build"]


# -- reports and the portfolio run -------------------------------------------------------------------------------------------------
def test_the_intraday_report_covers_each_account_on_its_own(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(
        bootstrap, "print_positions", lambda settings, url: shown.append(Path(url.removeprefix("sqlite:///")).name)
    )
    bootstrap.show_intraday_reports(Settings())
    assert shown == ["intraday.db", "gapgo.db", "gaponly.db"]
    out = capsys.readouterr().out
    assert "INTRADAY: opening-range breakout" in out and "GAP-ONLY (paper control for gap-and-go)" in out


@pytest.mark.parametrize("result,code", [({"run_dir": "/runs/1"}, 0), ({"error": "login failed"}, 1)])
def test_the_standalone_portfolio_run_exits_non_zero_on_error(monkeypatch, capsys, result, code):
    seen = {}

    def fake_run(token, chat_id, notify, explain, headless, otp_from, ask_terminal):
        seen.update(headless=headless, otp_from=otp_from, explain=explain)
        notify("<b>done</b>")
        return result

    monkeypatch.setattr(bootstrap, "run_standalone", fake_run)
    assert bootstrap.run_portfolio_agent(show=False, otp_from="terminal", use_ai=False) == code
    out = capsys.readouterr().out
    assert "done" in out and "<b>" not in out  # the terminal gets plain text
    assert seen == {"headless": True, "otp_from": "terminal", "explain": None}
    assert ("saved under /runs/1" in out) == ("run_dir" in result)
