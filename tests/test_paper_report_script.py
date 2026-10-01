"""The combined report script runs end to end against two real (small, synthetic) paper accounts."""
import sys
from datetime import datetime, timedelta, timezone

from scripts import paper_report
from src.config import Settings
from src.database import make_session_factory
from src.engine.paper_broker import PaperBroker
from tests.test_paper_broker import FakeClock, FakeFeed


def seed(tmp_path, name, prices, cash=20_000.0):
    """A paper account with one closed trade and one open position."""
    feed, clock = FakeFeed(prices), FakeClock()
    sessions = make_session_factory(f"sqlite:///{tmp_path / name}")
    now = [datetime(2026, 9, 29, 4, 0, tzinfo=timezone.utc)]
    broker = PaperBroker(sessions, feed, clock, initial_cash=cash, fees=lambda s, v: 1.0, now_fn=lambda: now[0])
    broker.buy_with_bracket("WIN", 1, 100.0, 0.5, 0.5)
    now[0] += timedelta(hours=1)
    broker.sell("WIN", 1)
    now[0] += timedelta(days=1)
    broker.buy_with_bracket("OPEN", 1, 200.0, 0.5, 0.5)
    return f"sqlite:///{tmp_path / name}"


def run(monkeypatch, capsys, prices, *argv):
    """Run the script with network prices stubbed (a real IntradayFeed would otherwise hit Yahoo for open positions)."""
    def fake_broker(settings, database_url=None, fees=None):
        return PaperBroker(make_session_factory(database_url or settings.database_url), FakeFeed(prices), FakeClock(),
                           fees=fees or (lambda s, v: 1.0))

    monkeypatch.setattr(sys, "argv", ["paper_report.py", *argv])
    monkeypatch.setattr(paper_report, "load_settings", lambda: Settings())
    monkeypatch.setattr(paper_report, "make_paper_broker", fake_broker)
    paper_report.main()
    return capsys.readouterr().out


def test_the_report_runs_on_two_real_accounts_and_shows_every_section(tmp_path, monkeypatch, capsys):
    swing = seed(tmp_path, "s.db", {"WIN": 110.0, "OPEN": 205.0})
    intraday = seed(tmp_path, "i.db", {"WIN": 90.0, "OPEN": 195.0})
    out = run(monkeypatch, capsys, {"WIN": 110.0, "OPEN": 205.0}, "--database-url", swing, "--intraday-database-url", intraday)
    assert "SWING: 1 open, 1 closed" in out and "INTRADAY: 1 open, 1 closed" in out and "COMBINED (swing + intraday)" in out
    assert "WIN" in out and "OPEN" in out and "2026-09-29" in out and "2026-W40" in out
    assert out.count("every closed trade") == 2


def test_no_trades_hides_the_trade_by_trade_listing_but_keeps_the_pl_tables(tmp_path, monkeypatch, capsys):
    swing = seed(tmp_path, "s.db", {"WIN": 110.0, "OPEN": 205.0})
    intraday = seed(tmp_path, "i2.db", {"WIN": 90.0, "OPEN": 195.0})
    out = run(monkeypatch, capsys, {"WIN": 110.0, "OPEN": 205.0}, "--database-url", swing, "--intraday-database-url", intraday, "--no-trades")
    assert "every closed trade" not in out and "by day" in out and "by week" in out


def test_an_account_with_no_closed_trades_shows_n_a_win_rate_and_an_empty_pl_table(tmp_path, monkeypatch, capsys):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'empty.db'}")
    PaperBroker(sessions, FakeFeed(), FakeClock(), initial_cash=20_000.0)   # never trades
    swing = seed(tmp_path, "s.db", {"WIN": 110.0, "OPEN": 205.0})
    out = run(monkeypatch, capsys, {"WIN": 110.0, "OPEN": 205.0}, "--database-url", f"sqlite:///{tmp_path / 'empty.db'}",
             "--intraday-database-url", swing)
    assert "win rate n/a" in out
