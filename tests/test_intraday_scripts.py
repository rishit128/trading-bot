"""The intraday command-line scripts run end to end on a small synthetic archive. They import from src and from each other, so a
renamed function used to break them silently (the backtest script lost its symbol loader and nobody noticed): this is the
guard."""
import pickle
import sys

import pandas as pd
import pytest

from scripts import audit_intraday, backtest_intraday, intraday_archive, research_intraday
from src.data.intraday_bars import BarArchive
from src.database import make_session_factory
from tests.test_intraday_audit import scripted_session, session


@pytest.fixture
def archive(tmp_path):
    return BarArchive(make_session_factory(f"sqlite:///{tmp_path / 'archive.db'}"))


def filled(archive, days=3, stocks=6):
    """Random-walk sessions plus scripted breakouts, so every script has signals to work on."""
    bars = {f"S{s}": pd.concat([session(d, s * 100 + d, spike_prob=0.05) for d in range(days)]) for s in range(stocks)}
    bars["BRK"] = pd.concat([scripted_session(d) for d in range(days)])
    archive.save(bars)
    return archive


def run(module, monkeypatch, archive, *argv):
    monkeypatch.setattr(module, "open_bar_archive", lambda: archive)
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_the_archive_script_reports_an_empty_archive_and_then_what_it_holds(archive, monkeypatch, capsys):
    run(intraday_archive, monkeypatch, archive, "status")
    assert "the archive is empty" in capsys.readouterr().out
    filled(archive)
    run(intraday_archive, monkeypatch, archive, "status")
    assert "3 sessions, 2026-09-01 .. 2026-09-03; 7 stocks" in capsys.readouterr().out


def test_import_cache_accepts_both_pickle_layouts_and_is_idempotent(archive, tmp_path, monkeypatch, capsys):
    bars = {"AAA": scripted_session(0)}
    plain, labelled = tmp_path / "plain.pkl", tmp_path / "labelled.pkl"
    plain.write_bytes(pickle.dumps(bars))
    labelled.write_bytes(pickle.dumps({"label": "some universe", "data": bars}))
    run(intraday_archive, monkeypatch, archive, "import-cache", str(plain))
    assert "added 75 bars" in capsys.readouterr().out
    run(intraday_archive, monkeypatch, archive, "import-cache", str(labelled))
    assert "added 0 bars" in capsys.readouterr().out                                   # the same session: nothing new
    with pytest.raises(SystemExit):
        run(intraday_archive, monkeypatch, archive, "import-cache")                     # no file given


def test_refresh_downloads_the_universe_and_stores_the_sessions(archive, monkeypatch, capsys):
    monkeypatch.setattr(intraday_archive, "select_universe", lambda explicit, size=100: (["AAA"], "a test universe"))
    monkeypatch.setattr(intraday_archive, "fetch_5m_bars",
                        lambda symbols, period: {"AAA": scripted_session(0)} if period == "59d" else {})
    run(intraday_archive, monkeypatch, archive, "refresh")
    out = capsys.readouterr().out
    assert "downloading 59 sessions for 1 stocks [a test universe]" in out and "added 75 bars" in out and "1 sessions" in out


def test_the_backtest_script_replays_the_live_engine_and_prints_a_summary(archive, monkeypatch, capsys):
    filled(archive)
    run(backtest_intraday, monkeypatch, archive, "--capital", "20000")
    out = capsys.readouterr().out
    assert "replaying 7 stocks with ReplayConfig(capital=20000.0" in out and "3 sessions," in out
    assert "start Rs 20,000 -> end Rs" in out and "max drawdown" in out
    run(backtest_intraday, monkeypatch, archive, "--start", "2026-09-02", "--end", "2026-09-02")
    assert "1 sessions," in capsys.readouterr().out


def test_the_backtest_script_refuses_an_empty_archive_with_a_hint(archive, monkeypatch):
    with pytest.raises(SystemExit, match="intraday_archive.py refresh"):
        run(backtest_intraday, monkeypatch, archive)


def test_the_audit_script_prints_all_four_sections(archive, monkeypatch, capsys):
    filled(archive, days=4, stocks=10)
    run(audit_intraday, monkeypatch, archive, "--seeds", "1", "--capital", "20000")
    out = capsys.readouterr().out
    for heading in ("== 1. does each filter add anything?", "== 2. signal vs a RANDOM entry time", "== 3. paper-broker shortcut",
                    "== 4. slices of the live rule"):
        assert heading in out
    assert "the signal's net edge over random entry" in out and "too dear for the account" in out
    empty = BarArchive(make_session_factory("sqlite://"))
    with pytest.raises(SystemExit, match="archive is empty"):
        run(audit_intraday, monkeypatch, empty, "--seeds", "1")


def test_the_research_script_reports_every_hypothesis_and_a_verdict(archive, monkeypatch, capsys):
    filled(archive, days=4, stocks=10)
    run(research_intraday, monkeypatch, archive, "--seeds", "1")
    out = capsys.readouterr().out
    for name in ("C  ORB long (the live rule)", "H1 ORB short", "H2 gap-and-go", "H3 gap fade", "H4 first-hour momentum",
                 "H5 last-half-hour momentum"):
        assert name in out
    assert "CANDIDATES:" in out and "criteria: >= 300 trades" in out
    with pytest.raises(SystemExit, match="archive is empty"):
        run(research_intraday, monkeypatch, BarArchive(make_session_factory("sqlite://")), "--seeds", "1")
