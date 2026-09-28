"""The replay drives the LIVE engine and paper broker over stored bars: the feed shows what Yahoo would have shown, and a
scripted market produces exactly the trades, sizes and prices that can be worked out by hand."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.india import IST
from src.research.intraday_replay import ReplayConfig, ReplayFeed, format_summary, replay, session_days, summarise
from tests.test_intraday_audit import DAY0, scripted_session

D1 = datetime(2026, 9, 21, tzinfo=IST)          # a Monday


def bars_for(day, opens, n=None):
    """5-minute bars from 09:15 on `day`; each bar's high/low/close are open+1 / open-1 / open+0.5, volume 1000."""
    idx = pd.DatetimeIndex([day.replace(hour=9, minute=15) + timedelta(minutes=5 * i) for i in range(len(opens))])
    return pd.DataFrame({"Open": opens, "High": [o + 1 for o in opens], "Low": [o - 1 for o in opens],
                         "Close": [o + 0.5 for o in opens], "Volume": [1000.0] * len(opens)}, index=idx)


def at(feed, day, hh, mm, ss=15):
    feed.now = day.replace(hour=hh, minute=mm, second=ss).astimezone(timezone.utc)
    return feed


# -- the feed -----------------------------------------------------------------------------------------------------------------
def test_at_a_cycle_finished_bars_are_complete_and_the_bar_in_progress_shows_only_its_open():
    feed = at(ReplayFeed({"A": bars_for(D1, [100, 110, 120, 130])}), D1, 9, 30)      # 09:30:15
    view = feed._upto("A")
    # bars start 09:15, 09:20, 09:25, 09:30: at 09:30:15 the first three have finished and the 09:30 bar has just begun
    assert list(view["Open"]) == [100, 110, 120, 130] and len(view) == 4
    assert list(view.iloc[2][["High", "Low", "Close"]]) == [121, 119, 120.5]         # finished: full range
    assert list(view.iloc[3][["High", "Low", "Close"]]) == [130, 130, 130]           # forming: only its first print


def test_a_bar_that_has_not_started_is_never_visible_and_nothing_is_visible_before_the_open():
    feed = at(ReplayFeed({"A": bars_for(D1, [100, 110, 120, 130])}), D1, 9, 22)      # 09:22:15
    # 09:15 finished, 09:20 forming, the rest not yet
    assert list(feed._upto("A")["Open"]) == [100, 110]
    early = at(ReplayFeed({"A": bars_for(D1, [100, 110])}), D1, 9, 10)
    assert early._upto("A").empty
    with pytest.raises(ValueError, match="no price"):
        early.last_price("A")
    with pytest.raises(ValueError, match="no price"):
        early.last_price("UNKNOWN")


def test_the_last_price_is_the_open_of_the_bar_in_progress():
    feed = at(ReplayFeed({"A": bars_for(D1, [100, 110, 120, 130])}), D1, 9, 22)
    assert feed.last_price("A") == 110


def test_bars_since_returns_only_bars_that_start_after_the_moment():
    feed = at(ReplayFeed({"A": bars_for(D1, [100, 110, 120, 130])}), D1, 9, 32)
    since = D1.replace(hour=9, minute=20).astimezone(timezone.utc)
    assert list(feed.bars_since("A", since)["Open"]) == [120, 130]                  # strictly after 09:20: 09:25 and 09:30
    assert feed.bars_since("UNKNOWN", since).empty


def test_today_returns_this_sessions_bars_only_for_symbols_that_have_them():
    yesterday = bars_for(D1 - timedelta(days=3), [50, 51, 52])                       # the previous Friday
    both = pd.concat([yesterday, bars_for(D1, [100, 110, 120])])
    feed = at(ReplayFeed({"A": both, "B": yesterday}), D1, 9, 32)
    out = feed.today(["A", "B", "C"])
    assert set(out) == {"A"} and list(out["A"]["Open"]) == [100, 110, 120]           # B has nothing today, C is unknown


# -- the replay ---------------------------------------------------------------------------------------------------------------
def scripted(days=3, price_scale=1.0):
    frames = []
    for d in range(days):
        frame = scripted_session(d)
        frames.append(frame * np.array([price_scale] * 4 + [1.0]))
    return {"AAA": pd.concat(frames)}


def test_a_scripted_breakout_is_traded_exactly_as_the_live_engine_would():
    result = replay(scripted(days=1), ReplayConfig(capital=20_000.0))
    [trade] = result.trades.to_dict("records")
    # entry: the bar-10 close of 101 filled at the next bar's open + 0.05% slippage;
    # size = the 20% cap (Rs 4,000 / 101 = 39 shares)
    assert trade["qty"] == 39 and trade["entry_price"] == pytest.approx(101 * 1.0005)
    assert trade["reason"] == "TARGET" and trade["exit_price"] == pytest.approx(104.0)      # target = 101 + 2 x (101 - 99.5)
    assert trade["net_pnl"] == pytest.approx((104.0 - 101 * 1.0005) * 39 - trade["fees"])
    [signal] = result.signals.to_dict("records")
    assert signal["outcome"] == "ENTERED" and signal["qty"] == 39 and signal["volume_ratio"] == pytest.approx(4.0)
    assert result.daily_equity.iloc[-1] == pytest.approx(20_000 + trade["net_pnl"])


def test_an_account_carries_over_from_one_session_to_the_next():
    result = replay(scripted(days=3), ReplayConfig(capital=20_000.0))
    assert len(result.trades) == 3 and set(result.trades["reason"]) == {"TARGET"}
    assert result.daily_equity.is_monotonic_increasing and len(result.days) == 3
    assert result.daily_equity.iloc[-1] == pytest.approx(20_000 + result.trades["net_pnl"].sum())


def test_a_stock_too_dear_for_the_account_is_skipped_with_its_reason_and_costs_nothing():
    result = replay(scripted(days=1, price_scale=40.0), ReplayConfig(capital=20_000.0))     # ~Rs 4,040 a share > the Rs 4,000 cap
    assert result.trades.empty and result.daily_equity.iloc[-1] == 20_000
    [signal] = result.signals.to_dict("records")
    assert signal["outcome"] == "SKIPPED" and "costs more than the largest position" in signal["reason"]


def test_the_replay_is_deterministic():
    a, b = replay(scripted(days=2)), replay(scripted(days=2))
    times = ["opened_at", "closed_at"]
    pd.testing.assert_frame_equal(a.trades.drop(columns=times), b.trades.drop(columns=times))
    pd.testing.assert_series_equal(a.daily_equity, b.daily_equity)


def test_the_date_range_and_universe_arguments_select_what_is_replayed():
    bars = scripted(days=3)
    assert len(replay(bars, start="2026-09-02", end="2026-09-02").days) == 1
    assert session_days(bars) == [DAY0.date() + timedelta(days=d) for d in range(3)]
    assert session_days(bars, start="2026-09-02") == session_days(bars)[1:]
    assert session_days(bars, end="2026-09-01") == session_days(bars)[:1]
    # nothing to trade in an empty universe
    assert replay(bars, universe=["ZZZ"]).trades.empty


def test_progress_is_reported_once_per_session():
    seen = []
    replay(scripted(days=2), progress=seen.append)
    assert seen == session_days(scripted(days=2))


def test_the_slippage_setting_reaches_the_broker():
    result = replay(scripted(days=1), ReplayConfig(slippage=0.0))
    assert result.trades["entry_price"].iloc[0] == pytest.approx(101.0)


def test_the_summary_reports_return_drawdown_win_rate_exits_and_skips():
    s = summarise(replay(scripted(days=3), ReplayConfig(capital=20_000.0)))
    assert (s.sessions, s.trades, s.win_rate, s.exits) == (3, 3, 1.0, {"TARGET": 3})
    assert s.total_return > 0 and s.end > s.start and s.max_drawdown == 0.0 and s.profitable_days == 1.0
    assert s.signals == {"ENTERED": 3} and s.fees > 0
    assert s.first_half_pnl + s.second_half_pnl == pytest.approx(s.end - s.start)
    empty = summarise(replay(scripted(days=1, price_scale=40.0), ReplayConfig(capital=20_000.0)))
    assert (empty.trades, empty.total_return, empty.win_rate, empty.signals) == (0, 0.0, 0.0, {"SKIPPED": 1})


def test_the_summary_prints_as_text():
    text = format_summary(summarise(replay(scripted(days=2), ReplayConfig(capital=20_000.0))))
    assert "2 sessions, 2 trades | win rate 100%" in text and "start Rs 20,000 -> end Rs" in text and "max drawdown 0.0%" in text


def test_exactly_on_a_bar_boundary_the_ending_bar_is_finished_and_the_new_one_has_begun():
    feed = at(ReplayFeed({"A": bars_for(D1, [100, 110, 120])}), D1, 9, 20, ss=0)      # 09:20:00 sharp
    view = feed._upto("A")
    assert list(view["Open"]) == [100, 110]
    assert list(view.iloc[0][["High", "Low", "Close"]]) == [101, 99, 100.5]           # 09:15 ended at 09:20:00: complete
    assert list(view.iloc[1][["High", "Low", "Close"]]) == [110, 110, 110]           # 09:20 has just started: only its open


def flat_after_entry():
    """The scripted breakout, but price then goes nowhere: no target, no stop, so the position is still open at 15:15."""
    frame = scripted_session(0)
    for column in ("Open", "High", "Low", "Close"):
        frame.iloc[11:, frame.columns.get_loc(column)] = 101.0
    return {"AAA": frame}


def test_a_position_still_open_at_1515_is_squared_off_and_the_account_ends_the_session_flat():
    result = replay(flat_after_entry(), ReplayConfig(capital=20_000.0, slippage=0.0))
    [trade] = result.trades.to_dict("records")
    assert trade["reason"] == "SIGNAL" and trade["exit_price"] == pytest.approx(101.0) and trade["qty"] == 39
    assert trade["closed_at"].astimezone(IST).time().isoformat() == "15:15:15"
    assert result.daily_equity.iloc[-1] == pytest.approx(20_000 + trade["net_pnl"])


def test_a_losing_session_shows_as_a_drawdown_and_trades_come_back_oldest_first():
    s = summarise(replay(flat_after_entry(), ReplayConfig(capital=20_000.0)))     # slippage + fees make a flat trade a small loss
    assert s.total_return < 0 and s.max_drawdown == pytest.approx(s.total_return) and s.profitable_days == 0.0
    trades = replay(scripted(days=3)).trades
    assert trades["opened_at"].is_monotonic_increasing


def test_the_default_configuration_is_the_settings_defaults_and_from_settings_reads_the_intraday_ones():
    from src.config import Settings

    default, settings = ReplayConfig(), Settings(intraday_risk_pct=0.01, intraday_max_position_pct=0.25, intraday_max_positions=4)
    assert (default.risk_pct, default.max_position_pct, default.max_positions) == (0.005, 0.20, 5)
    assert default.capital == Settings().paper_initial_cash and default.max_daily_loss_pct == Settings().risk.max_daily_loss_pct
    tuned = ReplayConfig.from_settings(settings, capital=20_000.0)
    assert (tuned.risk_pct, tuned.max_position_pct, tuned.max_positions, tuned.capital) == (0.01, 0.25, 4, 20_000.0)
    assert ReplayConfig.from_settings(settings).capital == settings.paper_initial_cash
