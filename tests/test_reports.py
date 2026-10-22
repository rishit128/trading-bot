"""The read-only commands the user actually runs (`--report`, `--positions`, `--screen`, `--graph`, `--intraday`)."""

from types import SimpleNamespace

import pytest

from src.app import intraday_cli, reports, wiring
from src.config import Settings
from src.data.universe import Candidate
from src.database import make_session_factory
from tests.test_paper_broker import FakeClock, FakeFeed


@pytest.fixture
def account(tmp_path, monkeypatch):
    """A paper account on disk with one closed trade and one open position, priced by a scripted feed."""
    import src.data.india as india
    from src.engine.paper_broker import PaperBroker
    from tests.test_paper_broker import T0

    feed = FakeFeed({"WIN": 100.0, "OPEN": 200.0})
    url = f"sqlite:///{tmp_path / 'acct.db'}"
    monkeypatch.setattr(india, "IntradayFeed", lambda: feed)
    monkeypatch.setattr(india, "IndiaClock", lambda: FakeClock())
    broker = PaperBroker(
        make_session_factory(url),
        feed,
        FakeClock(),
        initial_cash=100_000.0,
        fees=lambda s, v: 1.0,
        slippage=0.0,
        now_fn=lambda: T0,
    )
    broker.buy_with_bracket("WIN", 100, 100.0, 0.15, 1.0)
    feed.prices["WIN"] = 110.0
    broker.sell("WIN", 100)  # a closed winner: +10 a share, less Rs 2 of fees
    broker.buy_with_bracket("OPEN", 50, 200.0, 0.15, 1.0)
    feed.prices["OPEN"] = 205.0  # an open position, up 5 a share
    return Settings(paper_initial_cash=100_000.0), url


def test_report_summarises_the_account(account, capsys):
    settings, url = account
    reports.print_report(settings, url)
    out = capsys.readouterr().out
    assert "Paper account (INDIA)" in out and "started ₹100,000" in out
    assert "open positions 1" in out and "closed trades 1 (win rate 100%)" in out and "exits:" in out


def test_report_on_an_empty_account_says_n_a_instead_of_dividing_by_zero(tmp_path, monkeypatch, capsys):
    import src.data.india as india

    monkeypatch.setattr(india, "IntradayFeed", lambda: FakeFeed())
    monkeypatch.setattr(india, "IndiaClock", lambda: FakeClock())
    reports.print_report(Settings(paper_initial_cash=50_000.0), f"sqlite:///{tmp_path / 'e.db'}")
    out = capsys.readouterr().out
    assert "win rate n/a" in out and "closed trades 0" in out and "+0.00%" in out


def test_positions_lists_open_and_closed_with_profit_and_loss(account, capsys):
    settings, url = account
    reports.print_positions(settings, url)
    out = capsys.readouterr().out
    assert "OPEN POSITIONS (1)" in out and "CLOSED TRADES (1)" in out and "OVERALL" in out
    assert "OPEN" in out and "WIN" in out
    assert "+250" in out and "+998" in out  # OPEN: 50 shares up 5 each; WIN: 100 shares up 10, less the Rs 2 fees
    assert "1 in profit, 0 in loss" in out


def test_screen_prints_one_line_per_candidate(monkeypatch, capsys):
    candidates = [Candidate("AAA", 250.0, 0.12, 58.0, 3e8, 0.021, 0.45), Candidate("BBB", 90.0, -0.02, 61.0, 1.5e8, 0.03, None)]
    monkeypatch.setattr(reports, "build_india_screener", lambda settings: SimpleNamespace(candidates=lambda: candidates))
    reports.print_candidates(Settings())
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("AAA") and "12-1m=+45%" in lines[0] and "12-1m=+0%" in lines[1]


def test_graph_command_draws_all_three_workflows(capsys):
    reports.print_graphs(Settings())
    out = capsys.readouterr().out
    for title in ("%% CYCLE", "%% ANALYSIS", "%% DECISION"):
        assert title in out
    for node in (
        "open_cycle",
        "select_universe",
        "analyze_symbol",
        "execute_cycle",
        "fetch_data",
        "agent_technical",
        "collect",
        "decide",
        "execute",
    ):
        assert node in out, f"workflow node {node} missing from the diagram"


def test_intraday_launcher_wires_the_engine_from_settings(monkeypatch, capsys):
    started, alerts = {}, []
    broker = SimpleNamespace(clock=FakeClock(), sessions=make_session_factory("sqlite://"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    urls = []
    monkeypatch.setattr(intraday_cli, "make_paper_broker", lambda settings, url, fees=None: urls.append(url) or broker)
    monkeypatch.setattr(
        intraday_cli, "select_universe", lambda explicit: (["AAA", "BBB", "CCC"], "Nifty 100 (hand-saved NSE file)")
    )
    monkeypatch.setenv("INTRADAY_RISK_PCT", "0.01")
    monkeypatch.setenv("INTRADAY_MAX_POSITION_PCT", "0.25")
    monkeypatch.setenv("INTRADAY_MAX_POSITIONS", "4")
    monkeypatch.setenv("INTRADAY_MAX_DAILY_LOSS_PCT", "0.03")
    monkeypatch.setattr(intraday_cli, "Notifier", lambda token, chat: SimpleNamespace(send=alerts.append))
    news = SimpleNamespace(label="NEWS")  # built and tested on its own (tests/test_news.py): no feeds or AI here
    monkeypatch.setattr(intraday_cli, "build_news_engine", lambda settings, universe, bars, notify, live, angel=None: news)
    monkeypatch.setattr(
        intraday_cli,
        "run_together",
        lambda engines, heartbeat=None: started.update(
            engine=engines[2], gapgo=engines[1], gaponly=engines[0], news=engines[3], count=len(engines), heartbeat=heartbeat
        ),
    )
    intraday_cli.run_intraday(live=True)
    engine = started["engine"]
    assert engine.live is True and engine.universe == ["AAA", "BBB", "CCC"] and engine.notify is not None
    # its own INTRADAY_* sizing, not the swing settings; and the universe's source is passed on so it is recorded daily
    assert (engine.risk_pct, engine.max_position_pct, engine.max_positions) == (0.01, 0.25, 4)
    assert engine.max_daily_loss_pct == 0.03  # the configurable intraday daily-loss limit reaches every engine
    assert engine.universe_source == "Nifty 100 (hand-saved NSE file)"
    assert engine.archive is not None and engine.archive.sessions is broker.sessions  # finished sessions are kept for research
    out = capsys.readouterr().out
    assert "3 stocks [Nifty 100 (hand-saved NSE file)]" in out and "PAPER ORDERS" in out and "no edge" in out
    assert "risk 1.00% of equity per trade" in out and "cap 25%" in out
    assert alerts == [
        f"[{name}] started: PAPER ORDERS (separate intraday accounts)" for name in ("GAP-ONLY", "GAP-GO", "INTRADAY", "NEWS")
    ]  # stepped in this order: gap-only trades on one step and must not wait behind the others; the news engine goes last
    # breakouts past the position limit are recorded too, for all three engines
    assert started["count"] == 4 and started["engine"].record_capped is True and started["news"] is news
    assert "news-catalyst (own account)" in out
    # gap-and-go: the same universe and sizing, its own rule, label, version and entry window, and no second archive
    from src.intraday.strategy import (
        GAP_ENTRY_UNTIL,
        GAP_ONLY_ENTRY_UNTIL,
        GAP_ONLY_LATEST_START,
        find_gap_only_setup,
        find_gap_setup,
    )
    from src.versions import GAPGO_VERSION, GAPONLY_VERSION

    gap = started["gapgo"]
    assert gap.universe == engine.universe and (gap.risk_pct, gap.max_position_pct, gap.max_positions) == (0.01, 0.25, 4)
    assert gap.find_setup is find_gap_setup and gap.entry_until == GAP_ENTRY_UNTIL and gap.label == "GAP-GO"
    assert gap.version == GAPGO_VERSION and gap.archive is None and gap.live is True and "gap-and-go" in out
    assert engine.fetch_bars is gap.fetch_bars is started["gaponly"].fetch_bars  # all three engines share one download per cycle
    assert gap.record_capped is True
    only = started["gaponly"]  # the control: the same everything except the rule, its window, label, version and account
    assert only.universe == engine.universe and (only.risk_pct, only.max_position_pct, only.max_positions) == (0.01, 0.25, 4)
    assert only.find_setup is find_gap_only_setup and only.entry_until == GAP_ONLY_ENTRY_UNTIL and only.label == "GAP-ONLY"
    assert only.version == GAPONLY_VERSION and only.archive is None and only.record_capped is True and "gap-only" in out
    assert only.step_deadline == GAP_ONLY_LATEST_START and gap.step_deadline is None and engine.step_deadline is None
    assert sorted(urls[:3]) == ["sqlite:///gapgo.db", "sqlite:///gaponly.db", "sqlite:///intraday.db"]  # three separate accounts
    intraday_cli.run_intraday(live=False)
    assert started["engine"].live is False and "DRY RUN" in capsys.readouterr().out
    monkeypatch.setenv("INTRADAY_ARCHIVE", "false")
    intraday_cli.run_intraday(live=True)
    assert started["engine"].archive is None


def test_intraday_launcher_refuses_an_invalid_configuration(monkeypatch):
    monkeypatch.setattr(wiring, "load_settings", lambda: (_ for _ in ()).throw(ValueError("bad MAX_POSITION_PCT")))
    with pytest.raises(SystemExit, match="Invalid configuration: bad MAX_POSITION_PCT"):
        intraday_cli.run_intraday(live=False)


def test_equity_risk_reports_the_worst_peak_to_trough_fall_and_the_start():
    from datetime import datetime

    from src.app.reports import equity_risk

    t = [datetime(2026, 9, d) for d in range(1, 6)]
    assert equity_risk([]) is None
    r = equity_risk(zip(t, [100.0, 120.0, 90.0, 130.0, 117.0]))
    assert r["since"] == t[0] and abs(r["max_drawdown"] - (90 / 120 - 1)) < 1e-12  # -25%, not the later -10%


def test_the_intraday_loop_gets_a_heartbeat_only_when_one_is_configured(monkeypatch):
    seen = {}
    broker = SimpleNamespace(clock=FakeClock(), sessions=make_session_factory("sqlite://"))
    monkeypatch.setattr(intraday_cli, "make_paper_broker", lambda settings, url, fees=None: broker)
    monkeypatch.setattr(intraday_cli, "select_universe", lambda explicit: (["AAA"], "a list"))
    monkeypatch.setattr(intraday_cli, "run_together", lambda engines, heartbeat=None: seen.update(heartbeat=heartbeat))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HEARTBEAT_FILE", raising=False)
    intraday_cli.run_intraday(live=True)
    assert seen["heartbeat"] is None
    monkeypatch.setenv("HEARTBEAT_FILE", "/data/heartbeat-intraday")
    intraday_cli.run_intraday(live=True)
    assert str(seen["heartbeat"].path).replace("\\", "/") == "/data/heartbeat-intraday"


def test_the_news_engine_is_not_started_without_an_ai_model_key(monkeypatch, capsys):
    from src.config import Settings

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert intraday_cli.build_news_engine(Settings(news_models=("m",)), [], None, None, True) is None
    assert "needs OPENROUTER_API_KEY" in capsys.readouterr().out
