"""Gap-only, the intraday paper account that buys a 2% gap on the 09:30 close: the shared gap-and-go entry test (still used by the
record-only S7 test), the live setup, the engine running it in its own account, both engines stepped together, its refusals, its
Telegram view and the Docker database path. (The gap-and-go account itself was removed on 2026-11-09.)"""

from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.app.intraday_cli import GAP_EXPERIMENTS
from src.intraday import strategy as st
from src.intraday.engine import run_together
from src.monitoring.telegram import HELP, handle_command
from src.versions import GAPONLY_VERSION
from tests.intraday_helpers import BREAKOUT, VOLS, make_bars
from tests.test_intraday_engine import at, engine, signals

ROOT = Path(__file__).resolve().parent.parent


def with_previous_session(today: pd.DataFrame, prev_close: float) -> pd.DataFrame:
    """Yesterday's three bars, all closing at `prev_close`, followed by today's session (what a 5-day download returns)."""
    yesterday = make_bars([prev_close] * 3)
    yesterday.index = yesterday.index - timedelta(days=1)
    return pd.concat([yesterday, today])


def arrays(df):
    idx = pd.DatetimeIndex(df.index)
    return df["High"].to_numpy(float), df["Close"].to_numpy(float), (idx.hour * 60 + idx.minute).to_numpy()


def gap_only_engine(tmp_path, bars, now, **kw):
    return engine(
        tmp_path,
        bars,
        now,
        find_setup=st.find_gap_only_setup,
        entry_until=st.GAP_ONLY_ENTRY_UNTIL,
        label="GAP-ONLY",
        version=GAPONLY_VERSION,
        record_capped=True,
        **kw,
    )


# -- the shared gap-and-go entry test (S7 uses it) ---------------------------------------------------------------------------
def test_the_gap_and_go_entry_needs_a_2pct_gap_and_a_close_above_the_range_in_the_first_hour():
    high, close, minutes = arrays(make_bars(BREAKOUT, volumes=VOLS))  # opens at 100, breaks the 100.5 range high at 09:35
    assert st.first_gap_breakout(100.0, 98.0, high, close, minutes) == 4  # gap +2.04%
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes) is None  # gap +1.83%: too small
    assert st.first_gap_breakout(100.0, None, high, close, minutes) is None  # no previous close known
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes, min_gap=0.01) == 4
    late = make_bars([100, 100.4, 100.2] + [100.3] * 12 + [101.5])  # the first close above the range at 10:30
    assert st.first_gap_breakout(100.0, 98.0, *arrays(late)) is None


# -- the breakout engine's last window bar ------------------------------------------------------------------------------------
def test_the_last_bar_of_the_breakout_window_is_still_traded_on_the_cycle_that_sees_it(tmp_path):
    """The 13:55 bar finishes at 14:00 and is first seen a few seconds later: closing the scan at 14:00 sharp would silently
    drop it."""
    closes = [100, 100.4, 100.2] + [100.3] * 53 + [101.5]  # breakout bar 13:55
    orb_bars = make_bars(closes, volumes=[1000] * 56 + [3000])
    eng, broker, *_ = engine(tmp_path, {"AAA": orb_bars}, at(14, 0), db="orb.db")
    assert "entered 1" in eng.step()
    eng, broker, *_ = engine(tmp_path, {"AAA": orb_bars}, at(14, 5), db="late.db")
    assert eng.step() == "entry window closed: managing open positions only" and broker.holdings() == []


def test_the_engines_are_stepped_every_cycle_and_one_failure_does_not_stop_the_other():
    class Stop(BaseException):
        pass

    steps = []

    def broken():
        steps.append("orb")
        raise ConnectionError("yahoo down")

    engines = [
        SimpleNamespace(label="INTRADAY", step=broken),
        SimpleNamespace(label="GAP-ONLY", step=lambda: steps.append("gap")),
    ]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise Stop()

    with pytest.raises(Stop):
        run_together(engines, sleep)
    assert steps == ["orb", "gap", "orb", "gap"] and all(s >= 30 for s in sleeps)


# == gap-only ==========================================================================================
def test_the_gap_only_rule_enters_on_the_0925_bar_close_with_the_range_low_stop_and_no_target():
    bars = make_bars(BREAKOUT, volumes=VOLS)  # 09:25 bar closes at 100.2; range 99.9..100.5
    setup = st.find_gap_only_setup("AAA", bars, prev_close=98.0)
    assert setup.bar_time.strftime("%H:%M") == "09:25" and setup.signal_price == pytest.approx(100.2)
    assert setup.stop == pytest.approx(99.9) and setup.target == pytest.approx(200.4)
    assert st.find_gap_only_setup("AAA", bars, prev_close=98.2) is None  # gap below 2%
    assert st.find_gap_only_setup("AAA", bars, prev_close=None) is None  # previous close unknown
    assert st.find_gap_only_setup("AAA", make_bars(BREAKOUT, start=(9, 20)), 98.0) is None  # no real opening range
    breached = make_bars([100, 100.4, 99.95], lows=[99.95, 100.0, 99.95])  # entry at the range low itself
    assert st.find_gap_only_setup("AAA", breached, prev_close=98.0) is None


def test_without_the_previous_session_there_is_no_gap_and_no_trade(tmp_path):
    eng, broker, *_ = gap_only_engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 30))
    eng.step()
    assert broker.holdings() == [] and signals(broker) == []


def test_gap_only_trades_on_the_0930_cycle_only_and_never_chases_a_late_bar(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, *_, messages = gap_only_engine(tmp_path, bars, at(9, 30))
    assert "entered 1" in eng.step() and broker.holdings()[0]["stop"] == pytest.approx(99.9)
    assert signals(broker)[0].system_version == GAPONLY_VERSION
    assert any(m.startswith("[GAP-ONLY] BUY AAA") and "held to the 15:15 square-off" in m for m in messages)
    eng, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 35), db="late.db")  # first seen at 09:35: too late
    assert eng.step() == "entry window closed: managing open positions only" and broker.holdings() == []
    two_bars = {"AAA": with_previous_session(make_bars([100, 100.4]), 98.0)}  # the 09:25 bar never arrived
    eng, broker, *_ = gap_only_engine(tmp_path, two_bars, at(9, 30), db="missing.db")
    eng.step()
    assert broker.holdings() == [] and signals(broker) == []


def test_gap_only_holds_to_the_square_off(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, feed, now, _ = gap_only_engine(tmp_path, bars, at(9, 30))
    eng.step()
    now[0] = at(15, 16).astimezone(timezone.utc)
    assert eng.step() == "square-off: closed 1" and broker.holdings() == []


def test_a_restart_never_buys_the_same_stock_twice_in_a_day(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    first, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30))
    first.step()
    again, broker2, *_ = gap_only_engine(tmp_path, bars, at(9, 30))  # a new process on the same database
    again.step()
    assert len(broker2.holdings()) == 1 and broker2.holdings()[0]["qty"] == broker.holdings()[0]["qty"]
    assert [r.outcome for r in signals(broker2)] == ["ENTERED"]  # no second order, no second record


def test_signals_the_position_limit_leaves_out_are_recorded_for_the_gap_account_only(tmp_path):
    bars = {s: with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0) for s in ("AAA", "BBB")}
    eng, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30), max_positions=1, prices={"AAA": 100.2, "BBB": 100.2})
    eng.universe = ["AAA", "BBB"]
    eng.step()
    rows = signals(broker)
    assert [(r.symbol, r.outcome) for r in rows] == [("AAA", "ENTERED"), ("BBB", "SKIPPED")]
    assert rows[1].reason == "position limit reached (1 open)"
    orb_bars = {s: make_bars(BREAKOUT, volumes=VOLS) for s in ("AAA", "BBB")}  # the breakout engine
    eng, broker, *_ = engine(tmp_path, orb_bars, at(9, 45), max_positions=1, prices={"AAA": 101.5, "BBB": 101.5}, db="orb.db")
    eng.universe = ["AAA", "BBB"]
    eng.step()
    assert [(r.symbol, r.outcome) for r in signals(broker)] == [("AAA", "ENTERED")]  # unchanged: nothing extra


def test_gap_only_in_dry_run_places_nothing(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30), live=False)
    eng.step()
    assert broker.holdings() == [] and [r.outcome for r in signals(broker)] == ["DRY_RUN"]


def test_telegram_shows_the_gap_only_account(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30))
    eng.step()
    settings = SimpleNamespace(currency="Rs ")
    view = handle_command("/gaponly", None, None, settings, gaponly=broker)
    assert "GAP-ONLY paper account" in view and "AAA" in view and "/gaponly - gap-only paper account" in HELP
    assert handle_command("/gaponly", None, None, settings) == HELP  # not configured: the help text, never a crash
    assert handle_command("/gapgo", None, None, settings, gaponly=broker) == HELP  # the removed account's command is unknown now


def test_the_docker_image_keeps_the_gap_only_account_on_the_data_volume():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV GAPONLY_DATABASE_URL=sqlite:////data/gaponly.db" in text and "GAPGO" not in text


# -- the refusals ------------------------------------------------------------------------------------------------------------
def test_a_refused_gap_is_recorded_not_bought_and_only_a_setup_pays_for_the_count(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30), refusals_fn=lambda scan, day: lambda symbol: "crowded gap day")
    assert "entered 0" in eng.step() and broker.holdings() == []
    row = signals(broker)[0]
    assert row.outcome == "SKIPPED" and row.reason == "crowded gap day" and row.qty == 0

    asked = []
    quiet = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 100.0)}  # no gap, so no setup
    eng, broker, *_ = gap_only_engine(tmp_path, quiet, at(9, 30), db="quiet.db", refusals_fn=lambda s, d: asked.append(d))
    eng.step()
    assert asked == []


def test_only_gap_only_is_a_gap_experiment_and_carries_the_refusals():
    wired = {row[0]: row[-1] for row in GAP_EXPERIMENTS}
    assert wired == {"GAP-ONLY": st.gap_only_refusals}
