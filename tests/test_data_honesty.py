"""P0 corporate-action guard: a split/bonus restates a symbol's price scale while a real crash expands the
intraday range; only a bar with a large overnight gap AND an ordinary range is a split lookalike."""
import pandas as pd
import pytest

from src.data.data_honesty import check_bars_honesty, split_like_gaps


def _bars(closes):
    idx = pd.date_range("2020-01-02", periods=len(closes), freq="B")
    high = [c * 1.02 for c in closes]
    low = [c * 0.98 for c in closes]
    return pd.DataFrame({"Open": closes, "High": high, "Low": low, "Close": closes, "Volume": 10000}, index=idx)


def test_a_halving_bar_with_an_ordinary_range_is_flagged_as_a_split():
    closes = [100.0] * 300 + [50.0]
    df = _bars(closes)
    df.loc[df.index[-1], "High"] = 51.5
    df.loc[df.index[-1], "Low"] = 49.5
    flag = split_like_gaps(df)
    assert flag["is_split_like"].iloc[-1]
    assert flag["overnight_ret"].iloc[-1] == pytest.approx(-0.5)


def test_a_gap_down_with_a_widening_range_is_not_flagged():
    closes = [100.0] * 300 + [45.0]
    df = _bars(closes)
    df.loc[df.index[-1], "High"] = 58.0
    df.loc[df.index[-1], "Low"] = 40.0
    flag = split_like_gaps(df)
    assert not flag["is_split_like"].iloc[-1]


def test_small_returns_are_never_flagged():
    closes = [100.0 * (1 + 0.001 * i) for i in range(260)]
    df = _bars(closes)
    assert not split_like_gaps(df)["is_split_like"].any()


def test_check_bars_honesty_reports_the_split_and_data_quality():
    closes = [100.0] * 300 + [50.0]
    df = _bars(closes)
    df.loc[df.index[-1], "High"] = 51.5
    df.loc[df.index[-1], "Low"] = 49.5
    df.loc[df.index[0], "Volume"] = 0
    findings = check_bars_honesty(df)
    assert any("split-like" in f for f in findings)
    assert any("zero or negative volume" in f for f in findings)


def test_check_bars_honesty_reports_misordered_and_duplicate_bars():
    closes = [100.0] * 10
    df = _bars(closes)
    df.index = df.index[::-1]  # newest first, like an unsorted frame
    df = pd.concat([df, df.iloc[[0]]])
    findings = check_bars_honesty(df)
    assert any("not monotonic" in f for f in findings)
    assert any("duplicate timestamps" in f for f in findings)


def test_yf_fetcher_wiring_logs_a_split_finding(caplog):
    import logging

    from src.data.india import yf_bar_fetcher

    def fake_download(tickers, **kw):
        ns = lambda s: pd.DataFrame(  # noqa: E731 - compact OHLCV for one symbol
            {"Close": [100.0] * 300 + [50.0], "High": [102.0] * 300 + [51.5],
             "Low": [98.0] * 300 + [49.5], "Volume": [10000] * 301},
            index=pd.date_range("2020-01-02", periods=301, freq="B"))
        wide = pd.concat({t: ns(t) for t in ["A.NS"]}, axis=1)
        return wide

    fetch = yf_bar_fetcher(lookback_days=400, download=fake_download)
    with caplog.at_level(logging.WARNING):
        fetch(["A"])
    assert any("split-like" in m for m in caplog.messages)