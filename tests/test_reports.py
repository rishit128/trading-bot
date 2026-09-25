"""The read-only commands the user actually runs (`--report`, `--positions`, `--screen`, `--graph`, `--intraday`)."""
from types import SimpleNamespace

import pytest

from src.app import intraday_cli, reports
from src.config import Settings
from src.data.universe import Candidate
from tests.test_paper_broker import FakeClock, FakeFeed


@pytest.fixture
def account(tmp_path, monkeypatch):
    """A paper account on disk with one closed trade and one open position, priced by a scripted feed."""
    import src.data.india as india
    from src.database import make_session_factory
    from src.engine.paper_broker import PaperBroker
    from tests.test_paper_broker import T0

    feed = FakeFeed({"WIN": 100.0, "OPEN": 200.0})
    url = f"sqlite:///{tmp_path / 'acct.db'}"
    monkeypatch.setattr(india, "IntradayFeed", lambda: feed)
    monkeypatch.setattr(india, "IndiaClock", lambda: FakeClock())
    broker = PaperBroker(make_session_factory(url), feed, FakeClock(), initial_cash=100_000.0, fees=lambda s, v: 1.0,
                         slippage=0.0, now_fn=lambda: T0)
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
    for node in ("open_cycle", "select_universe", "analyze_symbol", "execute_cycle", "fetch_data", "agent_technical", "collect",
                 "decide", "execute"):
        assert node in out, f"workflow node {node} missing from the diagram"


def test_intraday_launcher_wires_the_engine_from_settings(monkeypatch, capsys):
    started, alerts = {}, []
    broker = SimpleNamespace(clock=FakeClock())
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setattr(intraday_cli, "make_paper_broker", lambda settings, url, fees=None: broker)
    monkeypatch.setattr(intraday_cli, "fetch_index_symbols", lambda index: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(intraday_cli, "Notifier", lambda token, chat: SimpleNamespace(send=alerts.append))
    monkeypatch.setattr(intraday_cli.IntradayEngine, "run_loop", lambda self: started.update(engine=self))
    intraday_cli.run_intraday(live=True)
    engine = started["engine"]
    assert engine.live is True and engine.universe == ["AAA", "BBB", "CCC"] and engine.notify is not None
    out = capsys.readouterr().out
    assert "3 Nifty 100 stocks" in out and "PAPER ORDERS" in out and "LOST money" in out
    assert alerts == ["[INTRADAY] started: PAPER ORDERS (separate intraday account)"]
    intraday_cli.run_intraday(live=False)
    assert started["engine"].live is False and "DRY RUN" in capsys.readouterr().out


def test_intraday_launcher_refuses_an_invalid_configuration(monkeypatch):
    monkeypatch.setattr(intraday_cli, "load_settings", lambda: (_ for _ in ()).throw(ValueError("bad MAX_POSITION_PCT")))
    with pytest.raises(SystemExit, match="Invalid configuration: bad MAX_POSITION_PCT"):
        intraday_cli.run_intraday(live=False)
