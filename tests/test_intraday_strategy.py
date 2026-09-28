"""The intraday opening-range-breakout rule (pure functions): setup detection, the opening-range guard, risk-based sizing, fees,
and the numpy breakout core pinned to a plain loop and to its exact boundaries."""
import pandas as pd
import pytest

from src.intraday import strategy as st
from tests.intraday_helpers import BREAKOUT, VOLS, make_bars


def test_breakout_bar_above_range_vwap_and_volume_is_a_setup():
    s = st.find_setup("AAA", make_bars(BREAKOUT, volumes=VOLS))
    assert s and s.signal_price == 101.5 and s.stop == pytest.approx(99.9) and s.range_high == pytest.approx(100.5)
    assert s.target == pytest.approx(101.5 + 2 * (101.5 - 99.9))
    assert s.strength == pytest.approx(1.0)  # 3x average volume = double the 1.5x minimum -> full strength


def test_a_breakout_that_barely_clears_the_volume_bar_has_zero_strength():
    s = st.find_setup("AAA", make_bars(BREAKOUT, volumes=[1000, 1000, 1000, 1000, 1500]))  # exactly 1.5x average
    assert s and s.strength == pytest.approx(0.0)


def test_no_setup_without_a_volume_surge_or_with_a_too_narrow_range():
    assert st.find_setup("AAA", make_bars(BREAKOUT)) is None  # volume never surges
    flat = make_bars([100, 100.01, 100.0, 100.0, 100.6], highs=[100.02] * 5, lows=[99.99] * 5, volumes=VOLS)
    assert st.find_setup("AAA", flat) is None  # opening range far below 0.3%


def test_position_size_is_the_smaller_of_the_risk_budget_and_the_position_cap():
    # cap binds: 0.5% risk = Rs 5,000 over a Rs 1 stop distance = 5,000 shares, but 20% of equity is Rs 200,000 = 2,000 shares
    assert st.position_size(1_000_000, 100.0, 99.0, risk_pct=0.005, max_position_pct=0.20) == (2000, None)
    # risk binds: a Rs 10 stop distance -> Rs 5,000 / 10 = 500 shares
    assert st.position_size(1_000_000, 100.0, 90.0, risk_pct=0.005, max_position_pct=0.20) == (500, None)


def test_position_size_never_spends_more_cash_than_there_is_and_holds_back_a_little_for_costs():
    assert st.position_size(1_000_000, 100.0, 99.0, 0.005, 0.20, cash=10_000) == (99, None)   # 10,000 / (100 x 1.003)
    qty, why = st.position_size(1_000_000, 100.0, 99.0, 0.005, 0.20, cash=50)
    assert qty == 0 and "not enough cash" in why


def test_a_stock_too_dear_for_the_account_is_refused_with_a_reason_not_silently():
    qty, why = st.position_size(20_000, 4_500.0, 4_400.0, 0.005, 0.20)   # 20% of Rs 20,000 = Rs 4,000 < one share
    assert qty == 0 and "costs more than the largest position" in why and "4,000" in why
    assert st.position_size(20_000, 3_900.0, 3_800.0, 0.005, 0.20)[0] == 1   # one share fits under the cap


def test_a_risk_budget_smaller_than_one_share_is_refused_with_a_reason():
    qty, why = st.position_size(20_000, 1_000.0, 850.0, 0.005, 0.20)   # Rs 100 of risk, Rs 150 stop distance per share
    assert qty == 0 and "risking 0.50%" in why


def test_a_stop_at_or_above_the_entry_is_refused():
    for stop in (100.0, 105.0):
        qty, why = st.position_size(1_000_000, 100.0, stop, 0.005, 0.20)
        assert qty == 0 and "not below" in why
    assert st.position_size(1_000_000, 0.0, -1.0, 0.005, 0.20)[0] == 0


def test_the_setup_records_how_strong_the_volume_surge_was():
    assert st.find_setup("AAA", make_bars(BREAKOUT, volumes=VOLS)).volume_ratio == pytest.approx(3.0)
    assert st.find_setup("AAA", make_bars(BREAKOUT, volumes=[1000, 1000, 1000, 1000, 2000])).volume_ratio == pytest.approx(2.0)


# -- the opening range must really be 09:15-09:30 -------------------------------------------------------------------------
def test_the_opening_range_is_intact_only_when_the_first_bars_are_09_15_and_contiguous():
    assert st.opening_range_intact(make_bars(BREAKOUT))
    assert not st.opening_range_intact(make_bars(BREAKOUT, start=(9, 20)))          # the stock did not trade until 09:20
    assert not st.opening_range_intact(make_bars(BREAKOUT, start=(9, 10)))          # not the session's own first bar either
    gappy = make_bars(BREAKOUT, volumes=VOLS).drop(make_bars(BREAKOUT).index[1])     # the 09:20 bar is missing
    assert not st.opening_range_intact(gappy)
    assert not st.opening_range_intact(make_bars([100, 100.4]))                       # fewer bars than the range needs


def test_no_breakout_is_signalled_from_a_range_built_out_of_the_wrong_bars():
    late = make_bars(BREAKOUT + [101.6], volumes=VOLS + [3000], start=(9, 20))
    assert st.find_setup("AAA", late) is None
    gappy = make_bars(BREAKOUT + [101.6], volumes=VOLS + [3000]).drop(make_bars(BREAKOUT).index[1])
    assert st.find_setup("AAA", gappy) is None
    assert st.find_setup("AAA", make_bars(BREAKOUT, volumes=VOLS)) is not None      # the same data, intact, does signal


def test_intraday_fees_have_stt_on_sells_only_and_flat_brokerage():
    buy, sell = st.intraday_fees("BUY", 200_000), st.intraday_fees("SELL", 200_000)
    assert sell > buy and 20 < buy < 40 and 60 < sell < 90


# -- the vectorised breakout core equals a plain loop --------------------------------------------------------------------------
def reference_first_breakout(bars, use_range=True, use_vwap=True, use_volume=True):
    """The rule written the slow, obvious way (bar by bar), to pin the numpy core to."""
    rng = 3
    range_high, range_low = bars["High"].iloc[:rng].max(), bars["Low"].iloc[:rng].min()
    width = (range_high - range_low) / range_high
    if use_range and not st.MIN_RANGE <= width <= st.MAX_RANGE:
        return None
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    vwap = (typical * bars["Volume"]).cumsum() / bars["Volume"].cumsum()
    for i in range(rng, len(bars)):
        t = bars.index[i].time()
        if t < st.ENTRY_FROM or t >= st.ENTRY_UNTIL:
            continue
        average = bars["Volume"].iloc[:i].mean()
        ratio = bars["Volume"].iloc[i] / average if average > 0 else 0.0
        if bars["Close"].iloc[i] > range_high and (not use_vwap or bars["Close"].iloc[i] > vwap.iloc[i]) \
                and (not use_volume or ratio >= st.VOLUME_MULT):
            return (i, range_high, range_low, ratio) if bars["Close"].iloc[i] - range_low > 0 else None
    return None


@pytest.mark.parametrize("switches", [{}, {"use_vwap": False}, {"use_volume": False}, {"use_range": False},
                                      {"use_vwap": False, "use_volume": False, "use_range": False}])
def test_the_numpy_breakout_core_matches_the_obvious_loop_on_many_random_sessions(switches):
    import numpy as np

    compared = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(6, 75))
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        volume = rng.integers(0, 1500, n).astype(float) * np.where(rng.random(n) < 0.1, 4, 1)   # zeros and spikes included
        open_ = np.concatenate([[close[0]], close[:-1]])
        bars = make_bars(list(close), highs=list(np.maximum(open_, close) * 1.0015), lows=list(np.minimum(open_, close) * 0.9985),
                         volumes=list(volume))
        index = pd.DatetimeIndex(bars.index)
        found = st.first_breakout(bars["High"].to_numpy(), bars["Low"].to_numpy(), bars["Close"].to_numpy(),
                                  bars["Volume"].to_numpy(), (index.hour * 60 + index.minute).to_numpy(), **switches)
        expected = reference_first_breakout(bars, **switches)
        assert (found is None) == (expected is None), f"seed {seed}"
        if found:
            compared += 1
            assert found.index == expected[0] and found.range_high == pytest.approx(expected[1])
            assert found.range_low == pytest.approx(expected[2]) and found.volume_ratio == pytest.approx(expected[3])
    assert compared >= 8                                      # the comparison ran on real signals, not only on empty days


# -- the entry rule's exact boundaries, straight on the numpy core ------------------------------------------------------------
def core(n=12, first=(100.5, 99.5), closes=None, volumes=None, highs=None, lows=None, start_minute=9 * 60 + 15, **switches):
    """Call the breakout core on hand-built arrays: bars 0-2 are the range (high/low `first`), the rest quiet unless given."""
    import numpy as np

    high = np.full(n, first[0]) if highs is None else np.asarray(highs, float)
    low = np.full(n, first[1]) if lows is None else np.asarray(lows, float)
    close = np.full(n, (first[0] + first[1]) / 2) if closes is None else np.asarray(closes, float)
    volume = np.full(n, 1000.0) if volumes is None else np.asarray(volumes, float)
    return st.first_breakout(high, low, close, volume, start_minute + 5 * np.arange(n), **switches)


def surge_at(i, n=12, base=1000.0, surge=5000.0):
    return [surge if k == i else base for k in range(n)]


def close_at(i, price, n=12, quiet=100.0):
    return [price if k == i else quiet for k in range(n)]


def test_a_breakout_must_close_strictly_above_the_range_high():
    assert core(closes=close_at(5, 100.5), volumes=surge_at(5)) is None                 # equal to the high: not a breakout
    assert core(closes=close_at(5, 100.51), volumes=surge_at(5)).index == 5


def test_the_volume_bar_is_inclusive_at_exactly_one_and_a_half_times_the_prior_average():
    assert core(closes=close_at(5, 101), volumes=surge_at(5, surge=1500.0)).index == 5   # exactly 1.5x
    assert core(closes=close_at(5, 101), volumes=surge_at(5, surge=1499.0)) is None


def test_the_entry_window_opens_at_0930_and_closes_before_1400():
    n = 60
    # bar i starts at 09:15 + 5 min x i: 2 = 09:25 (still the range), 3 = 09:30 (the first eligible bar), 56 = 13:55, 57 = 14:00
    for i, expected in ((2, None), (3, 3), (56, 56), (57, None)):
        found = core(n=n, closes=close_at(i, 101, n), volumes=surge_at(i, n))
        assert (found.index if found else None) == expected, f"bar {i}"


def test_bars_inside_the_opening_range_can_never_be_the_signal():
    # times chosen so the range bars fall INSIDE the entry window (10:00 onward): only the explicit guard stops bar 1 signalling
    found = core(closes=close_at(1, 105), volumes=surge_at(1), start_minute=10 * 60)
    assert found is None or found.index >= 3


def test_the_range_width_limits_are_inclusive():
    closes, volumes = close_at(5, 1_001, 12, quiet=998.0), surge_at(5)
    assert core(first=(1000.0, 997.0), closes=closes, volumes=volumes).index == 5          # exactly 0.3% wide
    assert core(first=(1000.0, 997.1), closes=closes, volumes=volumes) is None             # 0.29%: too narrow
    assert core(first=(1000.0, 975.0), closes=closes, volumes=volumes).index == 5          # exactly 2.5% wide
    assert core(first=(1000.0, 974.0), closes=closes, volumes=volumes) is None             # 2.6%: too wide
    assert core(first=(1000.0, 997.1), closes=closes, volumes=volumes, use_range=False).index == 5   # the audit's switch


def test_a_breakout_below_vwap_is_refused_and_the_vwap_uses_the_typical_price():
    n = 12
    high = [100.5] * 3 + [140.0] * 6 + [113.0] * 3                # bars 3-8 spent the morning far above the breakout price
    low = [99.5] * 3 + [110.0] * 6 + [113.0] * 3
    close = [100.0] * 3 + [110.0] * 6 + [113.0] * 3
    volume = [1000.0] * 9 + [5000.0, 1000.0, 1000.0]
    # VWAP by typical price (H+L+C)/3 is ~113.2 at bar 9, just above its 113 close; by close alone it would be ~108.9
    assert core(n=n, highs=high, lows=low, closes=close, volumes=volume) is None
    assert core(n=n, highs=high, lows=low, closes=close, volumes=volume, use_vwap=False).index == 9


def test_a_zero_volume_history_gives_no_volume_signal_instead_of_an_infinite_ratio():
    volumes = [0.0] * 5 + [500.0] + [0.0] * 6
    assert core(closes=close_at(5, 101), volumes=volumes) is None                      # no prior volume to be a surge against
    assert core(closes=close_at(5, 101), volumes=volumes, use_volume=False).index == 5
