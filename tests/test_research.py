import numpy as np
import pandas as pd
import pytest

from src.research import engine, signals
from src.research.data import load_universe

DAYS = pd.bdate_range("2023-01-02", periods=400)


def frame(**cols):
    return pd.DataFrame(cols, index=DAYS[:len(next(iter(cols.values())))])


def walk(start, daily, n=300):
    return pd.Series(start * np.cumprod(np.full(n, 1 + daily)), index=DAYS[:n])


# ------------------------------------------------------------------ engine timing and costs
def small():
    idx = DAYS[:8]
    close = pd.DataFrame({"A": [100, 100, 100, 100, 110, 110, 110, 110.0]}, index=idx)
    weights = pd.DataFrame({"A": [0, 0, 1.0, 1, 1, 1, 1, 1]}, index=idx)  # decided at the close of day 2 (0-based)
    return close, weights


def test_position_is_taken_the_day_after_the_signal_and_earns_from_the_day_after_that():
    close, weights = small()
    res = engine.portfolio_returns(close, weights, cost_per_side=0.0)
    assert res["gross"].iloc[3] == 0 and res["gross"].iloc[4] == pytest.approx(0.10)  # jump on day 4 is captured
    assert res["exposure"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_signal_on_the_jump_day_itself_cannot_capture_the_jump():
    close, _ = small()
    w = pd.DataFrame({"A": [0, 0, 0, 0, 1.0, 1, 1, 1]}, index=close.index)  # decided AFTER seeing the +10% close
    assert engine.portfolio_returns(close, w, cost_per_side=0.0)["gross"].sum() == pytest.approx(0.0)


def test_costs_are_charged_on_turnover_on_the_execution_day():
    close, weights = small()
    res = engine.portfolio_returns(close, weights, cost_per_side=0.001)
    assert res["turnover"].iloc[3] == 1.0 and res["net"].iloc[3] == pytest.approx(-0.001)
    assert res["turnover"].sum() == pytest.approx(1.0)


def test_round_trip_pays_both_sides():
    idx = DAYS[:10]
    close = pd.DataFrame({"A": 100.0}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1, 1, 0, 0, 0, 0, 0, 0, 0]}, index=idx)
    res = engine.portfolio_returns(close, w, cost_per_side=0.002)
    assert res["turnover"].sum() == pytest.approx(2.0) and res["net"].sum() == pytest.approx(-0.004)


def test_earlier_returns_never_depend_on_later_prices():
    close, weights = small()
    altered = close.copy()
    altered.iloc[-1] = 500.0
    a = engine.portfolio_returns(close, weights)["net"].iloc[:-1]
    b = engine.portfolio_returns(altered, weights)["net"].iloc[:-1]
    pd.testing.assert_series_equal(a, b)


def test_cannot_hold_a_stock_before_it_has_prices():
    idx = DAYS[:6]
    close = pd.DataFrame({"A": [np.nan, np.nan, 100, 100, 200, 200.0]}, index=idx)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    res = engine.portfolio_returns(close, w, cost_per_side=0.0)
    assert res["exposure"].iloc[:3].sum() == 0 and res["gross"].iloc[4] == pytest.approx(1.0)


def test_exec_lag_zero_is_one_day_more_aggressive_than_default():
    close, weights = small()
    assert engine.portfolio_returns(close, weights, 0.0, exec_lag=0)["gross"].iloc[3] == 0
    assert engine.portfolio_returns(close, weights, 0.0, exec_lag=0)["gross"].iloc[4] == pytest.approx(0.10)
    early = engine.portfolio_returns(close, weights.shift(-1).fillna(1.0), 0.0, exec_lag=0)  # decided a day earlier
    assert early["gross"].iloc[4] == pytest.approx(0.10)


# ------------------------------------------------------------------ metrics
def test_cagr_sharpe_drawdown_and_years():
    r = pd.Series(0.001, index=DAYS[:252])
    assert engine.cagr(r) == pytest.approx(1.001 ** 252 - 1)
    assert engine.sharpe(r) == 0.0  # zero variance is reported as 0, never inf/NaN
    dd = pd.Series([0.1, -0.2, 0.05], index=DAYS[:3])
    assert engine.max_drawdown(dd) == pytest.approx(-0.2)
    two_years = pd.Series([0.1, 0.1], index=pd.DatetimeIndex(["2023-12-29", "2024-01-02"]))
    yearly = engine.yearly_returns(two_years)
    assert yearly.loc[2023] == pytest.approx(0.1) and yearly.loc[2024] == pytest.approx(0.1)


def test_sharpe_is_annualised():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
    assert engine.sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_summarize_and_slice_and_benchmark():
    close, weights = small()
    s = engine.summarize(engine.portfolio_returns(close, weights))
    assert set(s) == {"cagr", "sharpe", "max_drawdown", "turnover_per_year", "avg_exposure", "pct_years_positive", "years"}
    res = engine.benchmark_returns(close["A"])
    assert res["net"].sum() == pytest.approx(0.10) and len(engine.slice_period(res, start=DAYS[4])) == 4
    assert len(engine.slice_period(res, end=DAYS[1])) == 2


# ------------------------------------------------------------------ signal helpers
def test_month_end_dates_are_the_last_trading_day_of_each_month():
    ends = signals.month_end_dates(pd.bdate_range("2024-01-01", "2024-03-31"))
    assert [d.strftime("%Y-%m-%d") for d in ends] == ["2024-01-31", "2024-02-29", "2024-03-29"]


def test_stateful_enters_holds_and_exits():
    enter = pd.DataFrame({"A": [0, 1, 1, 0, 0, 1, 0]}, dtype=bool)
    exit_ = pd.DataFrame({"A": [1, 0, 0, 1, 0, 0, 1]}, dtype=bool)
    assert signals.stateful(enter, exit_)["A"].tolist() == [False, True, True, False, False, True, False]


def test_stateful_ignores_exit_when_flat_and_enter_when_held():
    enter = pd.DataFrame({"A": [1, 1, 1]}, dtype=bool)
    exit_ = pd.DataFrame({"A": [0, 0, 1]}, dtype=bool)
    assert signals.stateful(enter, exit_)["A"].tolist() == [True, True, False]


def test_equal_weight_is_capped_and_scales_down_when_more_are_held():
    few = pd.DataFrame({f"S{i}": [True] for i in range(3)})
    many = pd.DataFrame({f"S{i}": [True] for i in range(25)})
    assert signals._capped_equal_weight(few, 20).iloc[0].sum() == pytest.approx(3 / 20)
    assert signals._capped_equal_weight(many, 20).iloc[0].sum() == pytest.approx(1.0)


def test_liquidity_filter():
    close = pd.DataFrame({"BIG": 100.0, "TINY": 100.0}, index=DAYS[:100])
    volume = pd.DataFrame({"BIG": 1_000_000.0, "TINY": 100.0}, index=DAYS[:100])
    mask = signals.liquid_mask(close, volume, 1e7)
    assert mask["BIG"].iloc[-1] and not mask["TINY"].iloc[-1] and not mask["BIG"].iloc[10]  # needs a full 60-day window


def test_rsi_is_100_with_no_losses_and_nan_when_flat():
    up = pd.DataFrame({"A": np.arange(100.0, 130)})
    assert signals.rsi_series(up, 2)["A"].iloc[-1] == 100.0
    assert np.isnan(signals.rsi_series(pd.DataFrame({"A": [100.0] * 30}), 2)["A"].iloc[-1])


# ------------------------------------------------------------------ candidate signals
def market(n=320):
    close = pd.DataFrame({"UP": walk(100, 0.003, n), "FLAT": walk(100, 0.0, n), "DOWN": walk(100, -0.002, n)})
    volume = pd.DataFrame(5_000_000.0, index=close.index, columns=close.columns)
    return close, volume


def test_momentum_picks_the_strongest_stock_only_at_month_ends_and_only_with_history():
    close, volume = market()
    w = signals.xs_momentum(close, volume, 1e7, top_n=1)
    assert w.iloc[:252].abs().sum().sum() == 0                       # no rebalance until 252 days of history exist
    last = w.iloc[-1]
    assert last["UP"] == 1.0 and last["FLAT"] == 0 and last["DOWN"] == 0
    changes = w.diff().abs().sum(axis=1)
    month_ends = set(signals.month_end_dates(close.index))
    assert all(d in month_ends for d in changes[changes > 0].index)  # weights only move on rebalance dates


def test_momentum_score_skips_the_most_recent_month():
    close, volume = market()
    manual = close["UP"].iloc[-1 - 21] / close["UP"].iloc[-1 - 252] - 1
    score = close.shift(21) / close.shift(252) - 1
    assert score["UP"].iloc[-1] == pytest.approx(manual)


def test_momentum_needs_enough_liquid_names_to_fill_the_portfolio():
    close, volume = market()
    assert signals.xs_momentum(close, volume, 1e7, top_n=5).abs().sum().sum() == 0  # only 3 stocks exist


def test_illiquid_stocks_are_never_selected():
    close, volume = market()
    volume["UP"] = 10.0
    w = signals.xs_momentum(close, volume, 1e7, top_n=1)
    assert w["UP"].sum() == 0 and w["FLAT"].iloc[-1] == 1.0


def test_low_volatility_prefers_the_calmest_stock():
    rng = np.random.default_rng(1)
    idx = DAYS[:320]
    close = pd.DataFrame({"CALM": 100 * np.cumprod(1 + rng.normal(0, 0.002, 320)),
                          "WILD": 100 * np.cumprod(1 + rng.normal(0, 0.03, 320))}, index=idx)
    volume = pd.DataFrame(5e6, index=idx, columns=close.columns)
    assert signals.low_volatility(close, volume, 1e7, top_n=1).iloc[-1]["CALM"] == 1.0


def test_donchian_enters_on_a_new_high_and_exits_on_a_new_low():
    idx = DAYS[:200]
    prices = np.r_[np.full(80, 100.0), np.linspace(101, 130, 40), np.full(20, 130.0), np.linspace(129, 80, 60)]
    close = pd.DataFrame({"A": prices}, index=idx)
    volume = pd.DataFrame(5e6, index=idx, columns=["A"])
    held = signals.donchian_breakout(close, volume, 1e7, cap=1)["A"] > 0
    assert not held.iloc[:80].any() and held.iloc[85]                # in after the breakout
    assert not held.iloc[-1]                                          # out after the collapse
    assert held.iloc[60:80].sum() == 0                                # no trade during the flat base


def test_rsi2_buys_dips_only_inside_uptrends():
    idx = DAYS[:300]
    up = 100 * np.cumprod(np.full(300, 1.004))
    dip = up.copy()
    dip[250:253] *= np.array([0.97, 0.94, 0.92])
    dip[253:] = dip[252] * up[253:] / up[252]
    down = 100 * np.cumprod(np.full(300, 0.997))
    down[250:253] *= np.array([0.97, 0.94, 0.92])
    close = pd.DataFrame({"UPDIP": dip, "DOWNDIP": down}, index=idx)
    volume = pd.DataFrame(5e6, index=idx, columns=close.columns)
    w = signals.rsi2_mean_reversion(close, volume, 1e7)
    assert w["UPDIP"].iloc[250:260].sum() > 0 and w["DOWNDIP"].sum() == 0


def test_trend_rule_holds_uptrends_with_pullbacks_and_ignores_downtrends():
    close, volume = market()
    factors = np.where(np.arange(320) % 2 == 0, 1.01, 0.994)          # drifts up with regular dips: RSI ~62, not overbought
    close["PULLBACKS"] = pd.Series(50 * np.cumprod(factors), index=close.index)
    volume["PULLBACKS"] = 5_000_000.0
    w = signals.trend_rule(close, volume, 1e7)
    assert w["PULLBACKS"].iloc[-1] > 0 and w["DOWN"].sum() == 0 and w["FLAT"].sum() == 0
    assert w["UP"].sum() == 0  # a smooth straight-line climb has RSI 100 = overbought, so the rule never enters it


def test_equal_weight_benchmark_sums_to_one_at_rebalances():
    close, volume = market()
    w = signals.equal_weight_universe(close, volume, 1e7)
    ends = signals.month_end_dates(close.index)
    ends = [d for d in ends if d >= close.index[60]]
    assert all(w.loc[d].sum() == pytest.approx(1.0) for d in ends)


def test_index_timing_is_long_above_the_200_day_and_flat_below():
    up = walk(100, 0.002, 300)
    down = pd.concat([walk(100, 0.002, 250), walk(up.iloc[249], -0.01, 50).set_axis(DAYS[250:300])])
    assert signals.index_trend_timing(up)["INDEX"].iloc[-1] == 1.0
    assert signals.index_trend_timing(down)["INDEX"].iloc[-1] == 0.0
    assert signals.index_trend_timing(up)["INDEX"].iloc[:199].sum() == 0  # no signal before 200 days of history


# ------------------------------------------------------------------ data loader
def fake_download(calls, tickers_missing=()):
    def download(tickers, start, end):
        calls.append(list(tickers))
        idx = pd.bdate_range("2020-01-01", periods=30)
        cols = {}
        for t in tickers:
            if t in tickers_missing:
                continue
            for f, v in (("Close", 100.0), ("Volume", 1000.0)):
                cols[(t, f)] = [v] * 30
        return pd.DataFrame(cols, index=idx, columns=pd.MultiIndex.from_tuples(list(cols))) if cols else pd.DataFrame()

    return download


def test_loader_builds_aligned_frames_skips_unresolvable_tickers_and_chunks(tmp_path):
    calls = []
    close, volume, nifty = load_universe(50, 1, cache_dir=tmp_path, download=fake_download(calls, {"C.NS"}),
                                         symbols=["A", "B", "C", "D", "E"], chunk=2)
    assert list(close.columns) == ["A", "B", "D", "E"] and close.shape == volume.shape
    assert close.index.equals(nifty.index) and len(nifty) == 30
    assert [len(c) for c in calls] == [2, 2, 1, 1] and calls[-1] == ["^NSEI"]


def test_loader_uses_the_disk_cache_until_refreshed(tmp_path):
    calls = []
    kw = dict(index=50, years=1, cache_dir=tmp_path, symbols=["A"], chunk=10)
    load_universe(download=fake_download(calls), **kw)
    n = len(calls)
    load_universe(download=fake_download(calls), **kw)
    assert len(calls) == n
    load_universe(download=fake_download(calls), refresh=True, **kw)
    assert len(calls) == 2 * n


def test_index_constituents_come_from_niftyindices():
    from types import SimpleNamespace

    from src.data.india import fetch_index_symbols

    seen = {}

    def get(url, **kw):
        seen["url"] = url
        return SimpleNamespace(text="Company Name,Industry,Symbol,Series,ISIN Code\nA,X,ABB,EQ,1\nB,Y,TCS,EQ,2\n",
                               raise_for_status=lambda: None)

    assert fetch_index_symbols(50, get) == ["ABB", "TCS"] and "nifty50list" in seen["url"]


# ------------------------------------------------------------------ gaps found by mutation testing
def test_a_position_executed_on_the_very_first_day_pays_its_entry_cost():
    idx = DAYS[:5]
    close = pd.DataFrame({"A": 100.0}, index=idx)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    res = engine.portfolio_returns(close, w, cost_per_side=0.001, exec_lag=0)
    assert res["turnover"].iloc[0] == 1.0 and res["net"].iloc[0] == pytest.approx(-0.001)


def test_max_drawdown_is_measured_from_the_running_peak_not_from_the_start():
    r = pd.Series([-0.1, 0.5, -0.3], index=DAYS[:3])  # equity 0.9 -> 1.35 -> 0.945: peak is day 2
    assert engine.max_drawdown(r) == pytest.approx(0.945 / 1.35 - 1)


def test_momentum_ignores_a_collapse_in_the_latest_month_but_plain_12_month_return_does_not():
    ends = signals.month_end_dates(DAYS)
    n = DAYS.get_loc(ends[ends <= DAYS[340]][-1]) + 1              # end the data exactly on a month end
    idx = DAYS[:n]
    steady_then_crash = 100 * np.cumprod(np.r_[np.full(n - 21, 1.004), np.full(21, 0.97)])
    slow_and_steady = 100 * np.cumprod(np.full(n, 1.0018))
    close = pd.DataFrame({"CRASH": steady_then_crash, "STEADY": slow_and_steady}, index=idx)
    volume = pd.DataFrame(5e6, index=idx, columns=close.columns)

    with_skip = signals.xs_momentum(close, volume, 1e7, top_n=1).iloc[-1]
    no_skip = signals.xs_momentum(close, volume, 1e7, top_n=1, skip=0).iloc[-1]
    assert with_skip["CRASH"] == 1.0                                # 12-1 looks past the recent crash
    assert no_skip["STEADY"] == 1.0                                 # a plain 12-month return is dominated by it


def test_index_cache_expires_and_falls_back_to_the_stale_copy(tmp_path):
    import os, time
    from datetime import date
    import pandas as pd
    from src.research.data import load_index_close

    def frame(last):
        idx = pd.bdate_range(end=last, periods=5)
        return pd.DataFrame({"Close": range(5)}, index=idx)

    calls = []

    def fresh(tickers, start, end):
        calls.append(1)
        return frame("2026-09-25")

    old = load_index_close("^X", 1, tmp_path, download=lambda *a: frame("2026-09-23"), today=lambda: date(2026, 9, 23))
    assert old.index[-1] == pd.Timestamp("2026-09-23")
    path = tmp_path / "index_X_1y.pkl"
    # a fresh cache is reused, and a research caller without max_age_hours never refreshes
    assert load_index_close("^X", 1, tmp_path, download=fresh, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-23")
    os.utime(path, (time.time() - 48 * 3600,) * 2)
    assert load_index_close("^X", 1, tmp_path, download=fresh).index[-1] == pd.Timestamp("2026-09-23")
    assert not calls
    # an expired cache is re-downloaded
    assert load_index_close("^X", 1, tmp_path, download=fresh, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-25")
    # ...and if that fails the stale copy is used rather than crashing the cycle
    os.utime(path, (time.time() - 48 * 3600,) * 2)

    def broken(*a):
        raise RuntimeError("yahoo down")

    assert load_index_close("^X", 1, tmp_path, download=broken, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-25")


def test_ai_backtest_snapshot_carries_the_enriched_production_indicators():
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.backtest_ai_agent import point_in_time_snapshot

    idx = pd.bdate_range("2022-01-03", periods=320)
    close = pd.Series(np.linspace(100, 160, 320) + np.sin(np.arange(320)), index=idx)
    volume = pd.Series(1e6, index=idx)
    snap = point_in_time_snapshot("A", close, volume, idx[300])
    assert snap.bar_date == str(idx[300].date())
    assert None not in (snap.macd_histogram, snap.bb_position, snap.momentum, snap.volume_trend, snap.avg_volume)
    assert snap.atr is None and snap.adx is None  # no High/Low in the research frames
    future = close.copy()
    future.iloc[301:] = 1.0  # data after as_of must not leak in
    assert point_in_time_snapshot("A", future, volume, idx[300]) == snap
