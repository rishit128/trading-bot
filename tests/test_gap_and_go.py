"""Gap-and-go, the second intraday paper strategy: the shared rule, the live setup, the engine running it in its own account,
both engines stepped together, its Telegram view, and the Docker database path."""

from datetime import datetime, timedelta, timezone
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
    high, close, minutes = arrays(make_bars(BREAKOUT, volumes=VOLS))  # opens at 100, breaks the 100.5 range high at 09:35
    assert st.first_gap_breakout(100.0, 98.0, high, close, minutes) == 4  # gap +2.04%
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes) is None  # gap +1.83%: too small
    assert st.first_gap_breakout(100.0, None, high, close, minutes) is None  # no previous close known
    assert st.first_gap_breakout(100.0, 98.2, high, close, minutes, min_gap=0.01) == 4
    late = make_bars([100, 100.4, 100.2] + [100.3] * 12 + [101.5])  # the first close above the range at 10:30
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
    return engine(
        tmp_path,
        bars,
        now,
        find_setup=st.find_gap_setup,
        entry_until=st.GAP_ENTRY_UNTIL,
        label="GAP-GO",
        version=GAPGO_VERSION,
        **kw,
    )


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
    gap_bars = with_previous_session(make_bars([100, 100.4, 100.2] + [100.3] * 11 + [101.5]), 98.0)  # breakout bar 10:25
    eng, broker, *_ = gap_engine(tmp_path, {"AAA": gap_bars}, at(10, 30))
    assert "entered 1" in eng.step()
    closes = [100, 100.4, 100.2] + [100.3] * 53 + [101.5]  # breakout bar 13:55
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


def test_the_docker_image_keep_the_gap_account_on_the_data_volume():
    for name in ("Dockerfile",):
        assert "ENV GAPGO_DATABASE_URL=sqlite:////data/gapgo.db" in (ROOT / name).read_text(encoding="utf-8")


# == gap-only: gap-and-go's control ==========================================================================================
from src.research.gap_compare import capped_signals, compare  # noqa: E402
from src.versions import GAPONLY_VERSION  # noqa: E402


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


def test_a_restart_never_buys_the_same_stock_twice_in_a_day(tmp_path):
    bars = {"AAA": with_previous_session(make_bars(BREAKOUT, volumes=VOLS), 98.0)}
    first, broker, *_ = gap_only_engine(tmp_path, bars, at(9, 30))
    first.step()
    again, broker2, *_ = gap_only_engine(tmp_path, bars, at(9, 30))  # a new process on the same database
    again.step()
    assert len(broker2.holdings()) == 1 and broker2.holdings()[0]["qty"] == broker.holdings()[0]["qty"]
    assert [r.outcome for r in signals(broker2)] == ["ENTERED"]  # no second order, no second record


def test_signals_the_position_limit_leaves_out_are_recorded_for_the_gap_accounts_only(tmp_path):
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
    view = handle_command("/gaponly", None, None, SimpleNamespace(currency="Rs "), gaponly=broker)
    assert "GAP-ONLY control account" in view and "AAA" in view and "/gaponly - gap-only control account" in HELP


def test_the_docker_image_keep_the_gap_only_account_on_the_data_volume():
    for name in ("Dockerfile",):
        assert "ENV GAPONLY_DATABASE_URL=sqlite:////data/gaponly.db" in (ROOT / name).read_text(encoding="utf-8")


# -- the comparison ----------------------------------------------------------------------------------------------------------
def trade(symbol, day, ret, value=4000.0, fees=5.0):
    opened = datetime(2026, 10, day, 4, tzinfo=timezone.utc)
    return {
        "symbol": symbol,
        "qty": 40,
        "entry_price": value / 40,
        "exit_price": value / 40 * (1 + ret),
        "opened_at": opened,
        "closed_at": opened + timedelta(hours=5),
        "reason": "SIGNAL",
        "fees": fees,
        "net_pnl": value * ret,
    }


def test_the_comparison_needs_both_per_trade_and_date_by_date_wins():
    days = range(1, 31)  # the minimum number of shared dates a verdict needs
    go = [trade("AAA", d, 0.01) for d in days]
    only = [trade("AAA", d, 0.002 + 0.001 * (d % 3)) for d in days] + [trade("BBB", d, -0.01) for d in days]
    c = compare(go, only, 20_000.0, 20_000.0)
    assert c.shared_dates == 30 and c.paired_diff > 0 and c.paired_t > 2.6 and c.same_stock_days == 30
    assert c.gapgo.trades == 30 and c.gapgo.net_per_trade == pytest.approx(0.01) and c.gapgo.win_rate == 1.0
    assert c.gaponly.utilisation == pytest.approx(8000 / 20_000, rel=0.05) and c.gaponly.fees == pytest.approx(300.0)
    assert c.verdict.startswith("gap-and-go wins per trade AND date by date")
    worse = compare([trade("AAA", d, -0.01) for d in days], only, 20_000.0, 20_000.0)
    assert worse.verdict.startswith("gap-and-go does not beat")
    assert compare([], [], 20_000.0, 20_000.0).verdict == "not enough trades yet to compare"


def test_a_handful_of_trades_never_produces_a_verdict():
    """The first live week had 2 and 3 trades: a paired t of -38 on two dates is an anecdote, not evidence."""
    few = compare(
        [trade("AAA", 1, -0.03), trade("AAA", 2, -0.03)], [trade("AAA", 1, -0.01), trade("AAA", 2, -0.02)], 20_000.0, 20_000.0
    )
    assert few.shared_dates == 2 and few.verdict.startswith("too early to judge: 2 of the 30 dates")
    assert "gap-and-go wins" not in few.verdict and "does not beat" not in few.verdict
    nearly = compare(
        [trade("AAA", d, 0.01) for d in range(1, 30)], [trade("AAA", d, 0.0) for d in range(1, 30)], 20_000.0, 20_000.0
    )
    assert nearly.verdict.startswith("too early to judge: 29 of the 30")  # one date short


def test_drawdown_comes_from_the_closed_trade_equity_curve():
    c = compare(
        [trade("AAA", 1, 0.05, value=4000), trade("AAA", 2, -0.10, value=4000)], [trade("AAA", 1, 0.0)], 20_000.0, 20_000.0
    )
    assert c.gapgo.max_drawdown == pytest.approx(-400 / 20_200)  # 20,000 -> 20,200 -> 19,800


def test_capped_signals_are_counted_from_the_recorded_reasons():
    rows = pd.DataFrame(
        {"reason": ["position limit reached (5 open)", None, "not enough cash", "position limit reached (5 open)"]}
    )
    assert capped_signals(rows) == 2 and capped_signals(pd.DataFrame()) == 0
