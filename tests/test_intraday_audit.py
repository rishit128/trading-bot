"""The intraday audit's machinery: it must agree with the rule that trades, find a planted edge, and not invent one in noise."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.india import IST
from src.intraday import strategy as st
from src.research.intraday_audit import (FEE_FRACTION, Entry, StockDay, evaluate, first_breakout, random_entry, simulate,
                                         stock_days)

DAY0 = datetime(2026, 9, 1, tzinfo=IST)


def minutes_of(times):
    return np.array([x.hour * 60 + x.minute for x in times])


def session(day_offset, seed, n=75, spike_prob=0.06, drift=0.0):
    """A synthetic 09:15-15:25 session of n 5-minute bars: a random walk with volume that occasionally spikes."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([DAY0 + timedelta(days=day_offset, hours=9, minutes=15 + 5 * i) for i in range(n)])
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
    vol = rng.integers(900, 1100, n).astype(float) * np.where(rng.random(n) < spike_prob, 4, 1)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001, "Low": np.minimum(open_, close) * 0.999,
                         "Close": close, "Volume": vol}, index=idx)


def universe(n_stocks=12, n_days=30, **kw):
    return {f"S{s}": pd.concat([session(d, s * 1000 + d, **kw) for d in range(n_days)]) for s in range(n_stocks)}


def scripted_session(day_offset, breakout_bar=10):
    """Flat 99.5-100.5 range, a volume-backed breakout close of 101 at `breakout_bar`, then a run to 104.5: every entry hits
    its target (entry 101, stop 99.5, target 104)."""
    n = 75
    idx = pd.DatetimeIndex([DAY0 + timedelta(days=day_offset, hours=9, minutes=15 + 5 * i) for i in range(n)])
    close = np.full(n, 100.0)
    close[breakout_bar:] = 101.0
    close[breakout_bar + 2:] = 104.5
    open_ = np.concatenate([[100.0], close[:-1]])
    high, low = np.maximum(open_, close) + 0.0, np.minimum(open_, close) - 0.0
    high[:3], low[:3] = 100.5, 99.5
    vol = np.full(n, 1000.0)
    vol[breakout_bar] = 4000.0
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_the_all_filters_on_variant_is_exactly_the_rule_that_trades():
    data = {"S": pd.concat([session(d, d) for d in range(60)])}
    compared = 0
    for day in stock_days(data):
        df = data["S"][data["S"].index.date == day.date]
        live, audit = st.find_setup("S", df), first_breakout(day)
        assert (live is None) == (audit is None)
        if live:
            compared += 1
            assert audit.index == list(df.index).index(live.bar_time) and audit.range_low == pytest.approx(live.stop)
            assert audit.volume_ratio == pytest.approx(live.volume_ratio)
    assert compared >= 5                                     # the comparison really ran on signals, not on empty days


def test_sessions_with_a_broken_opening_range_or_too_few_bars_are_left_out():
    good = session(0, 1)
    late_open = session(1, 2).iloc[1:]                       # first bar missing: not a real opening range
    short = session(2, 3, n=30)
    days = stock_days({"S": pd.concat([good, late_open, short])})
    assert [d.date for d in days] == [good.index[0].date()]


def test_simulate_follows_the_stop_target_and_square_off_rules():
    n = 75
    t = np.array([(DAY0 + timedelta(hours=9, minutes=15 + 5 * i)).time() for i in range(n)])
    flat = np.full(n, 100.0)

    def day(**over):
        arrays = dict(o=flat.copy(), h=flat.copy(), l=flat.copy(), c=flat.copy(), v=np.full(n, 1000.0))
        for k, v in over.items():
            arrays[k][v[0]] = v[1]
        return StockDay("S", DAY0.date(), t, minutes_of(t), arrays["o"], arrays["h"], arrays["l"], arrays["c"], arrays["v"], None)

    entry = Entry(10, 100.5, 99.0, 0.015, 3.0)                # entry at 100, stop 99, target 102
    # the low breaches the stop: filled at the stop
    assert simulate(day(l=(12, 98.5)), entry) == (99.0, "STOP")
    assert simulate(day(o=(12, 98.0)), entry) == (98.0, "STOP")                     # gaps through it: filled at the open
    assert simulate(day(h=(12, 102.5)), entry) == (102.0, "TARGET")
    assert simulate(day(l=(12, 98.5), h=(12, 102.5)), entry)[1] == "STOP"           # both in one bar: the stop is assumed first
    assert simulate(day(), entry) == (100.0, "EOD")                                 # nothing happens: flat at the 15:15 open
    # the paper broker's old shortcut: only opens are looked at, so a mid-bar breach that recovers is missed
    assert simulate(day(l=(12, 98.5)), entry, mode="open_only") == (100.0, "EOD")
    # ...but a bar OPENING above the target counts
    assert simulate(day(o=(12, 102.5)), entry, mode="open_only") == (102.5, "TARGET")
    # flat at the 15:15 bar's OPEN (bar 72), not the bar before it and not the one after
    assert simulate(day(o=(72, 100.7)), entry)[0] == 100.7 and simulate(day(o=(72, 100.7), h=(71, 100.4)), entry)[1] == "EOD"
    assert simulate(day(o=(73, 100.9)), entry) == (100.0, "EOD")                          # the bar after is never reached


def test_a_random_entry_is_only_ever_at_a_bar_that_leaves_the_stop_below_the_price():
    n = 75
    t = np.array([(DAY0 + timedelta(hours=9, minutes=15 + 5 * i)).time() for i in range(n)])
    close = np.full(n, 90.0)                                # everything after the opening range is below its 99.5 low...
    close[20] = 101.0                                       # ...except bar 20
    high, low = np.full(n, 100.5), np.full(n, 99.5)
    day = StockDay("S", DAY0.date(), t, minutes_of(t), close.copy(), high, low, close, np.full(n, 1000.0), None)
    picks = {random_entry(day, np.random.default_rng(seed)).index for seed in range(30)}
    assert picks == {20}


def test_evaluate_reports_exact_returns_on_a_scripted_market():
    days = stock_days({"S": pd.concat([scripted_session(d) for d in range(5)])})
    trades = evaluate(days, first_breakout, slippage=0.0)
    assert len(trades) == 5 and set(trades["why"]) == {"TARGET"}
    assert trades["gross"].round(6).eq(round(104 / 101 - 1, 6)).all()                # bought 101, sold at the 104 target
    assert trades["volume_ratio"].round(3).eq(4.0).all() and trades["price"].eq(101.0).all()
    assert trades["net"].round(9).eq((trades["gross"] - FEE_FRACTION).round(9)).all()


def test_in_pure_noise_the_signal_does_no_better_than_a_random_entry():
    days = stock_days(universe(n_days=40), min_bars=60)
    signal = evaluate(days, first_breakout)
    randoms = pd.concat([evaluate(days, lambda d, r=np.random.default_rng(s): random_entry(d, r)) for s in range(3)])
    assert len(signal) > 100 and len(randoms) > len(signal)
    band = 3 * signal["net"].std() / np.sqrt(len(signal))
    assert abs(signal["net"].mean() - randoms["net"].mean()) < band          # any gap is inside the signal's own uncertainty


def test_net_is_gross_less_slippage_and_fees():
    days = stock_days(universe(n_stocks=3, n_days=15), min_bars=60)
    frozen = evaluate(days, first_breakout, slippage=0.0)
    assert not frozen.empty
    assert (frozen["gross"] - frozen["net"]).round(9).eq(round(FEE_FRACTION, 9)).all()   # only the fees separate them
    slipped = evaluate(days, first_breakout, slippage=0.001)
    assert (slipped["net"] < frozen["net"]).all()
