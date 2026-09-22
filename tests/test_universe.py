from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.universe import (Candidate, ScreenConfig, UniverseScreener, delivery_filter, rank, screen_bars)
from src.engine.risk_engine import Portfolio

CFG = ScreenConfig(max_candidates=10)


def series(factors, start, n=260):
    """Price path from repeating daily factors, e.g. [1.01, 0.994] = up 1%, down 0.6%."""
    return start * np.cumprod([factors[i % len(factors)] for i in range(n)])


def frame(symbol, close, volume=1_000_000):
    idx = pd.MultiIndex.from_product([[symbol], pd.date_range("2025-08-01", periods=len(close), freq="B")],
                                     names=["symbol", "timestamp"])
    return pd.DataFrame({"close": close, "volume": float(volume)}, index=idx)


def market():
    return pd.concat([
        frame("GOOD", series([1.01, 0.994], 50)),                     # smooth uptrend, RSI ~62, liquid
        frame("PENNY", series([1.01, 0.994], 2)),                     # price < $10
        frame("ILLIQ", series([1.01, 0.994], 50), volume=1_000),      # tiny dollar volume
        frame("DOWN", series([0.99, 1.004], 100)),                    # downtrend
        frame("HOT", series([1.01], 20)),                             # only rises -> RSI 100 (overbought)
        frame("SHORT", series([1.01, 0.994], 50, n=100)),             # < 200 bars of history
        frame("WILD", series([1.09, 0.95], 20)),                      # uptrend but ~7% daily volatility
    ])


def test_screen_keeps_only_liquid_smooth_uptrends_not_overbought():
    found = {c.symbol for c in screen_bars(market(), CFG)}
    assert found == {"GOOD"}


def test_screen_handles_empty_input():
    assert screen_bars(pd.DataFrame(), CFG) == []


def test_screen_accepts_capitalised_columns_too():
    df = market().rename(columns={"close": "Close", "volume": "Volume"})
    assert {c.symbol for c in screen_bars(df, CFG)} == {"GOOD"}


def test_thresholds_are_configurable():
    lax = ScreenConfig(min_price=1.0, min_traded_value=1_000_000.0)  # PENNY ~$3 x 1M sh; ILLIQ ~$80k/day still out
    assert {c.symbol for c in screen_bars(market(), lax)} == {"GOOD", "PENNY"}


def cand(symbol, mom, vol):
    return Candidate(symbol, 50.0, mom, 60.0, 50e6, vol)


def test_ranking_prefers_smooth_trend_over_spiky_and_truncates():
    smooth, spiky, weak = cand("SMOOTH", 0.30, 0.01), cand("SPIKY", 0.90, 0.04), cand("WEAK", 0.05, 0.02)
    assert [c.symbol for c in rank([spiky, weak, smooth], 10)] == ["SMOOTH", "SPIKY", "WEAK"]
    assert [c.symbol for c in rank([spiky, weak, smooth], 1)] == ["SMOOTH"]


def test_score_is_safe_with_zero_volatility():
    assert cand("X", 0.2, 0.0).score == 0.0


def make_screener(tmp_days, chunk=500, fetch=None, listing=None, sleep=lambda s: None):
    calls = {"list": 0, "fetch": []}

    def default_listing():
        calls["list"] += 1
        return ["GOOD", "PENNY", "ILLIQ", "DOWN", "HOT"]

    def default_fetch(symbols):
        calls["fetch"].append(list(symbols))
        df = market()
        return df[df.index.get_level_values(0).isin(symbols)]

    s = UniverseScreener(listing or default_listing, fetch or default_fetch, CFG, today=lambda: tmp_days[0],
                         chunk=chunk, sleep=sleep)
    return s, calls


def test_screen_runs_once_per_day_then_again_next_day():
    day = [date(2026, 9, 21)]
    s, calls = make_screener(day)
    assert [c.symbol for c in s.candidates()] == ["GOOD"]
    s.candidates()
    assert calls["list"] == 1
    day[0] = date(2026, 9, 22)
    s.candidates()
    assert calls["list"] == 2


def test_symbols_are_fetched_in_chunks():
    s, calls = make_screener([date(2026, 9, 21)], chunk=2)
    s.candidates()
    assert [len(c) for c in calls["fetch"]] == [2, 2, 1]


def test_chunk_fetch_is_retried_then_succeeds():
    attempts, sleeps = [], []

    def flaky(symbols):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("blip")
        return market()

    s, _ = make_screener([date(2026, 9, 21)], fetch=flaky, sleep=sleeps.append)
    assert [c.symbol for c in s.candidates()] == ["GOOD"] and len(attempts) == 3 and sleeps == [3, 6]


def test_persistent_fetch_failure_raises_and_is_not_cached():
    def broken(symbols):
        raise ConnectionError("down")

    s, _ = make_screener([date(2026, 9, 21)], fetch=broken)
    with pytest.raises(ConnectionError):
        s.candidates()
    assert s._day is None


def test_symbols_for_puts_holdings_first_without_duplicates():
    s, _ = make_screener([date(2026, 9, 21)])
    held = Portfolio(cash=0, equity=1, positions={"GOOD": 500.0, "OLD": 100.0}, position_qty={"GOOD": 5, "OLD": 1})
    assert s.symbols_for(held) == ["GOOD", "OLD"]
    assert s.symbols_for(Portfolio(cash=0, equity=1)) == ["GOOD"]


def test_price_filter_works_independently_of_liquidity():
    cheap_but_liquid = frame("CHEAP", series([1.01, 0.994], 3), volume=10_000_000)  # ~$5 x 10M sh = $50M/day
    assert screen_bars(cheap_but_liquid, CFG) == []
    assert [c.symbol for c in screen_bars(cheap_but_liquid, ScreenConfig(min_price=1.0))] == ["CHEAP"]


def test_score_uses_12_1_momentum_when_known_and_the_old_score_otherwise():
    with_12_1 = Candidate("A", 50.0, 0.1, 60.0, 50e6, 0.02, momentum_12_1=0.8)
    assert with_12_1.score == 0.8
    assert cand("B", 0.2, 0.02).score == pytest.approx(0.2 / (0.02 * 63 ** 0.5))


def test_stocks_are_ranked_by_12_1_momentum_not_by_recent_smoothness():
    smooth_recent = Candidate("SMOOTH", 50.0, 0.30, 60.0, 50e6, 0.005, momentum_12_1=0.10)
    year_winner = Candidate("WINNER", 50.0, 0.05, 60.0, 50e6, 0.03, momentum_12_1=0.90)
    assert [c.symbol for c in rank([smooth_recent, year_winner], 2)] == ["WINNER", "SMOOTH"]


def test_screen_needs_a_full_year_of_bars_for_12_1_momentum():
    days = pd.bdate_range(end="2026-09-18", periods=252)
    bars = pd.DataFrame({"Close": np.linspace(100, 200, 252), "Volume": 5_000_000.0},
                        index=pd.MultiIndex.from_product([["SHORT"], days]))
    assert screen_bars(bars, CFG) == []


# ---------------------------------------------------------------- delivery filter
def delivery_frame(dates, values):
    """dates x symbols delivery percent, as src.research.delivery.load_delivery returns."""
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates))


def test_delivery_filter_keeps_only_symbols_above_the_days_median():
    dates = pd.bdate_range("2026-08-01", periods=20)
    deliv = delivery_frame(dates, {"HIGH": [80.0] * 20, "LOW": [20.0] * 20, "MID": [50.0] * 20})
    ok = delivery_filter(date(2026, 8, 28), load_delivery=lambda years: deliv)
    assert ok == {"HIGH"}  # only above the median of {80, 20, 50} = 50


def test_delivery_filter_returns_none_when_unavailable_or_empty():
    assert delivery_filter(date(2026, 8, 28), load_delivery=lambda years: (_ for _ in ()).throw(RuntimeError("down"))) is None
    assert delivery_filter(date(2026, 8, 28), load_delivery=lambda years: pd.DataFrame()) is None


def test_delivery_filter_uses_the_latest_available_day_on_or_before_as_of():
    dates = pd.bdate_range("2026-08-01", periods=15)  # >= min_days (10) so the rolling average is defined
    deliv = delivery_frame(dates, {"HIGH": [80.0] * 15, "LOW": [20.0] * 15})
    ok = delivery_filter(date(2026, 12, 1), load_delivery=lambda years: deliv)  # far after the data ends
    assert ok == {"HIGH"}


def test_screener_applies_the_delivery_filter_when_enabled():
    def listing():
        return ["GOOD", "GOOD2", "PENNY"]

    def fetch(symbols):
        df = pd.concat([market(), frame("GOOD2", series([1.012, 0.995], 60))])
        return df[df.index.get_level_values(0).isin(symbols)]

    # GOOD2 passes the trend/liquidity screen too, but only GOOD clears the delivery filter (above the day's median).
    dates = pd.bdate_range("2026-08-01", periods=20)
    deliv = delivery_frame(dates, {"GOOD": [80.0] * 20, "GOOD2": [20.0] * 20})
    s = UniverseScreener(listing, fetch, CFG, today=lambda: date(2026, 9, 21), use_delivery_filter=True,
                         delivery_loader=lambda years: deliv)
    assert [c.symbol for c in s.candidates()] == ["GOOD"]


def test_screener_ignores_the_delivery_filter_by_default_and_when_data_is_unavailable():
    s, _ = make_screener([date(2026, 9, 21)])  # use_delivery_filter defaults to False
    assert [c.symbol for c in s.candidates()] == ["GOOD"]

    def boom(years):
        raise RuntimeError("NSE unreachable")

    s2 = UniverseScreener(lambda: ["GOOD"], lambda syms: market()[market().index.get_level_values(0).isin(syms)],
                          CFG, today=lambda: date(2026, 9, 21), use_delivery_filter=True, delivery_loader=boom)
    assert [c.symbol for c in s2.candidates()] == ["GOOD"]  # fails open: screening still works
