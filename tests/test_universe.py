from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.universe import (Candidate, ScreenConfig, UniverseScreener, alpaca_bar_fetcher, list_tradable_stocks,
                               rank, screen_bars)
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


def asset(symbol, exchange="NASDAQ", name="Acme Corporation Common Stock", tradable=True):
    return SimpleNamespace(symbol=symbol, exchange=SimpleNamespace(value=exchange), name=name, tradable=tradable)


def test_tradable_stock_filter():
    assets = [
        asset("AAPL", name="Apple Inc. Common Stock"),
        asset("IBM", "NYSE", "International Business Machines Corporation Common Stock"),
        asset("SPY", "ARCA", "SPDR S&P 500 ETF Trust"),            # wrong exchange
        asset("QQQX", "NASDAQ", "Some Growth ETF"),                 # ETF by name
        asset("BKLN", "NASDAQ", "Invesco Senior Loan ETF"),
        asset("FTRA.WS", "NYSE", "Foo Warrants"),                   # non-alpha symbol
        asset("PENN", "OTC", "Penny Corp"),                         # OTC
        asset("HALT", "NYSE", "Halted Inc", tradable=False),
        asset("SPOT", "NYSE", "Spotify Technology S.A. Ordinary Shares"),  # foreign common stock stays
    ]
    assert list_tradable_stocks(assets) == ["AAPL", "IBM", "SPOT"]


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


def test_alpaca_fetcher_requests_completed_sessions_with_configured_feed_and_renames_columns():
    seen = {}

    def get_stock_bars(request):
        seen["req"] = request
        return SimpleNamespace(df=market())

    fetch = alpaca_bar_fetcher(SimpleNamespace(get_stock_bars=get_stock_bars), feed="iex")
    df = fetch(["GOOD"])
    req = seen["req"]
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    assert req.end.replace(tzinfo=None) == today and req.feed.value == "iex" and req.symbol_or_symbols == ["GOOD"]
    assert "Close" in df.columns and "Volume" in df.columns


def test_price_filter_works_independently_of_liquidity():
    cheap_but_liquid = frame("CHEAP", series([1.01, 0.994], 3), volume=10_000_000)  # ~$5 x 10M sh = $50M/day
    assert screen_bars(cheap_but_liquid, CFG) == []
    assert [c.symbol for c in screen_bars(cheap_but_liquid, ScreenConfig(min_price=1.0))] == ["CHEAP"]
