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

def test_the_fetcher_summarises_data_findings_instead_of_logging_every_stock(caplog):
    """Regression: a warning per zero-volume stock per scan made up 83% of a three-day log."""
    import logging
    import numpy as np
    from src.data.india import yf_bar_fetcher
    from datetime import datetime

    idx = pd.date_range("2025-06-02", periods=300, freq="B")

    def frame(close, volume=10_000):
        close = np.asarray(close, dtype=float)
        return pd.DataFrame({"Close": close, "High": close * 1.02, "Low": close * 0.98, "Volume": volume}, index=idx)

    thin = [frame([100.0 + i * 0.1 for i in range(300)], volume=[0 if i % 7 == 0 else 5000 for i in range(300)]) for _ in range(3)]
    split = frame([100.0] * 299 + [50.0])
    split.loc[idx[-1], ["High", "Low"]] = [51.5, 49.5]
    good = frame([100.0 + i * 0.1 for i in range(300)])
    names = ["THIN1", "THIN2", "THIN3", "SPLIT", "GOOD"]
    wide = pd.concat({f"{n}.NS": f for n, f in zip(names, thin + [split, good])}, axis=1)
    fetch = yf_bar_fetcher(download=lambda tickers, **kw: wide, today=lambda: datetime(2026, 9, 25))
    with caplog.at_level(logging.INFO, logger="src.data.india"):
        out = fetch(names)
    assert set(out.index.get_level_values(0)) == set(names)  # nothing is dropped: findings are reported, not acted on
    messages = [r.getMessage() for r in caplog.records]
    assert sum("zero-volume" in m for m in messages) == 1 and any("3 of 5" in m for m in messages)  # one line, a count
    assert not any(m.startswith("THIN") for m in messages)  # no per-symbol zero-volume warning
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "SPLIT" in warned[0].getMessage() and "split-like" in warned[0].getMessage()


def test_honesty_check_is_fast_enough_to_run_on_the_whole_market():
    import time
    import numpy as np

    rng = np.random.default_rng(0)
    frames = [_bars(list(100 + np.cumsum(rng.normal(0, 1, 260)))) for _ in range(200)]
    start = time.perf_counter()
    for df in frames:
        check_bars_honesty(df)
    assert (time.perf_counter() - start) / 200 < 0.004  # was ~12 ms per stock (an iterrows over every bar)
