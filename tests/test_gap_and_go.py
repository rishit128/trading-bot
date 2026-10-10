"""Gap-and-go, the second intraday paper strategy: the shared rule, the live setup, the engine running it in its own account,
both engines stepped together, its Telegram view, and the Docker database path."""
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.intraday import strategy as st
from src.intraday.engine import run_together
from src.monitoring.telegram import HELP, handle_command
from src.research import intraday_hypotheses as hyp
from src.research.intraday_audit import stock_days
from src.versions import GAPGO_VERSION
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


# -- the rule ----------------------------------------------------------------------------------------------------------------
def test_the_rule_needs_a_2pct_gap_and_a_close_above_the_range_in_the_first_hour():
    high, close, minutes = arrays(make_bars(BREAKOUT, volumes=VOLS))   # opens at 100, breaks the 100.5 range high at 09:35
    assert st.first_gap_breakout(100.0, 98.0, high, close, minutes) == 4         # gap +2.04%
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes) is None      # gap +1.83%: too small
    assert st.first_gap_breakout(100.0, None, high, close, minutes) is None      # no previous close known
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes, min_gap=0.01) == 4
    late = make_bars([100, 100.4, 100.2] + [100.3] * 12 + [101.5])                # the first close above the range at 10:30
    assert st.first_gap_breakout(100.0, 98.0, *arrays(late)) is None


def test_the_live_setup_stops_at_the_range_low_and_has_no_target():
    setup = st.find_gap_setup("AAA", make_bars(BREAKOUT, volumes=VOLS), prev_close=98.0)
    assert setup.signal_price == pytest.approx(101.5) and setup.stop == pytest.approx(99.9)
    assert setup.target == pytest.approx(203.0) and setup.range_high == pytest.approx(100.5)
    assert setup.bar_time.strftime("%H:%M") == "09:35" and setup.volume_ratio == pytest.approx(3.0)
    assert st.find_gap_setup("AAA", make_bars(BREAKOUT, volumes=VOLS), prev_close=None) is None
    assert st.find_gap_setup("AAA", make_bars(BREAKOUT, start=(9, 20)), prev_close=98.0) is None  # no real opening range


def test_research_and_live_pick_the_same_bar_and_stop():
    bars = with_previous_session(make_bars(BREAKOUT + [101.2] * 5, volumes=VOLS + [1000] * 5), 98.0)
    [day] = [d for d in stock_days({"AAA": bars}, min_bars=5) if d.prev_close]
    plan = hyp.gap_and_go(day)
    live = st.find_gap_setup("AAA", bars[bars.index.date == day.date], prev_close=day.prev_close)
    assert plan.index == 4 and day.c[plan.index] == pytest.approx(live.signal_price) and plan.stop == pytest.approx(live.stop)


# -- the engine running it ---------------------------------------------------------------------------------------------------
def gap_engine(tmp_path, bars, now, **kw):
    return engine(tmp_path, bars, now, find_setup=st.find_gap_setup, entry_until=st.GAP_ENTRY_UNTIL, label="GAP-GO",
                  version=GAPGO_VERSION, **kw)


def test_the_gap_engine_enters_records_its_version_and_holds_to_the_square_off(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, feed, now, messages = gap_engine(tmp_path, bars, at(9, 45))
    assert "entered 1" in eng.step()
    [h] = broker.holdings()
    assert h["stop"] == pytest.approx(99.9)
    assert any(m.startswith("[GAP-GO] BUY AAA") and "held to the 15:15 square-off" in m for m in messages)
    [row] = signals(broker)
    assert (row.outcome, row.system_version) == ("ENTERED", GAPGO_VERSION) and row.target == pytest.approx(203.0)
    now[0] = at(15, 16).astimezone(timezone.utc)
    assert eng.step() == "square-off: closed 1" and broker.holdings() == []


def test_without_the_previous_session_there_is_no_gap_and_no_trade(tmp_path):
    eng, broker, *_ = gap_engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.step()
    assert broker.holdings() == [] and signals(broker) == []


def test_the_last_bar_of_each_window_is_still_traded_on_the_cycle_that_sees_it(tmp_path):
    """The 10:25 bar (gap-and-go) and the 13:55 bar (breakout) finish at 10:30 / 14:00 and are first seen a few seconds
    later: closing the scan at 10:30 / 14:00 sharp would silently drop them."""
    gap_bars = with_previous_session(make_bars([100, 100.4, 100.2] + [100.3] * 11 + [101.5]), 98.0)   # breakout bar 10:25
    eng, broker, *_ = gap_engine(tmp_path, {"AAA": gap_bars}, at(10, 30))
    assert "entered 1" in eng.step()
    closes = [100, 100.4, 100.2] + [100.3] * 53 + [101.5]                                          # breakout bar 13:55
    orb_bars = make_bars(closes, volumes=[1000] * 56 + [3000])
    eng, broker, *_ = engine(tmp_path, {"AAA": orb_bars}, at(14, 0), db="orb.db")
    assert "entered 1" in eng.step()
    eng, broker, *_ = engine(tmp_path, {"AAA": orb_bars}, at(14, 5), db="late.db")
    assert eng.step() == "entry window closed: managing open positions only" and broker.holdings() == []


def test_after_its_window_the_gap_engine_stops_scanning_but_still_squares_off(tmp_path):
    calls = []
    eng, broker, *_ = gap_engine(tmp_path, {}, at(10, 35))
    eng.fetch_bars = lambda symbols: calls.append(symbols) or {}
    assert eng.step() == "entry window closed: managing open positions only" and calls == []


def test_both_engines_are_stepped_every_cycle_and_one_failure_does_not_stop_the_other():
    class Stop(BaseException):
        pass

    steps = []

    def broken():
        steps.append("orb")
        raise ConnectionError("yahoo down")

    engines = [SimpleNamespace(label="INTRADAY", step=broken), SimpleNamespace(label="GAP-GO", step=lambda: steps.append("gap"))]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise Stop()

    with pytest.raises(Stop):
        run_together(engines, sleep)
    assert steps == ["orb", "gap", "orb", "gap"] and all(s >= 30 for s in sleeps)


# -- seeing it: Telegram and the deployment ----------------------------------------------------------------------------------
def test_telegram_shows_the_gap_account_and_its_trades(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    eng, broker, *_ = gap_engine(tmp_path, bars, at(9, 45))
    eng.step()
    settings = SimpleNamespace(currency="Rs ")
    view = handle_command("/gapgo", None, None, settings, gapgo=broker)
    assert "GAP-AND-GO paper account" in view and "AAA" in view
    assert handle_command("/gapgo_history", None, None, settings, gapgo=broker) == "No closed trades yet."
    assert handle_command("/gapgo", None, None, settings) == HELP  # not configured: the help text, never a crash
    assert "/gapgo - gap-and-go paper account" in HELP


def test_both_docker_images_keep_the_gap_account_on_the_data_volume():
    for name in ("Dockerfile", "Dockerfile.aws"):
        assert "ENV GAPGO_DATABASE_URL=sqlite:////data/gapgo.db" in (ROOT / name).read_text(encoding="utf-8")