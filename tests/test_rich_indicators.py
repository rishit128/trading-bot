"""Phase-5 signal enrichment: MACD, Bollinger position, volume trend, ATR and ADX on
the snapshot, the Close+Volume-only fallback path, and that the CoT prompt renders
them (with an n/a fallback when data is missing).
"""
import pandas as pd
import pytest

from src.agents.agents import TechnicalAgent
from src.data.indicators import (Snapshot, _bollinger_position, _volume_trend,
                                 build_snapshot, compute_adx, compute_atr)


def bars(closes, volume=10_000, drift=0.3):
    """Daily OHLCV with High/Low bracketing the close; a gentle uptrend keeps ADX/ATR defined."""
    closes = [float(c) for c in closes]
    high = [c * 1.02 for c in closes]
    low = [c * 0.98 for c in closes]
    vols = [volume * (1 + drift * (i % 3)) for i in range(len(closes))]
    return pd.DataFrame({"Close": closes, "High": high, "Low": low, "Volume": vols},
                        index=pd.date_range("2025-09-01", periods=len(closes), freq="B"))


def ohclv_only(closes):
    """Pre-Phase-5 bar format: Close + Volume only, as the old bulk fetch provided."""
    return pd.DataFrame({"Close": [float(c) for c in closes], "Volume": [10_000] * len(closes)},
                        index=pd.date_range("2025-09-01", periods=len(closes), freq="B"))


def test_build_snapshot_populates_the_phase5_indicators():
    series = [100.0 * (1 + 0.003 * i) for i in range(250)]  # a firm, steady uptrend
    s = build_snapshot("X", bars(series))
    assert isinstance(s, Snapshot)
    assert s.macd is not None and s.macd_signal is not None and s.macd_histogram is not None
    assert s.macd_histogram == pytest.approx(s.macd - s.macd_signal, abs=1e-9)
    assert 0.0 <= s.bb_position <= 1.0
    assert s.volume_trend is not None
    assert s.atr is not None and s.atr > 0
    assert 0.0 <= s.adx <= 100.0


def test_snapshot_without_high_low_leaves_atr_and_adx_none_but_keeps_the_rest():
    s = build_snapshot("X", ohclv_only([100.0 * (1 + 0.003 * i) for i in range(250)]))
    assert s.atr is None and s.adx is None
    assert s.bb_position is not None and s.volume_trend is not None and s.macd is not None


def test_flat_market_renders_neutral_indicators():
    s = build_snapshot("X", bars([100.0] * 250))
    assert s.adx is None  # zero true range leaves ADX undefined, not a fake 0 or 50
    assert s.volume_trend is not None  # flat volume still measurable
    assert s.bb_position == pytest.approx(0.5, abs=0.15)  # no drift -> price sits mid-band


def test_helpers_return_none_for_thin_input():
    close = pd.Series([1.0, 2.0, 3.0])
    assert _volume_trend(close, 20) is None  # shorter than 2 periods
    short = pd.Series([1.0] * 10)
    assert _bollinger_position(short, 20) is None
    assert compute_atr(None, None, close) is None
    assert compute_adx(pd.Series([1.0] * 10), pd.Series([1.0] * 10), pd.Series([1.0] * 10)) is None


def test_prompt_renders_rich_indicators_and_degrades_to_n_a():
    s = build_snapshot("X", bars([100.0 * (1 + 0.003 * i) for i in range(250)]))
    p = TechnicalAgent.build_prompt(s)
    assert "MACD histogram" in p and "Bollinger range" in p
    assert "volume trend" in p and "ATR" in p and "ADX" in p
    assert "n/a" not in p

    plain = build_snapshot("X", ohclv_only([100.0 * (1 + 0.003 * i) for i in range(250)]))
    p2 = TechnicalAgent.build_prompt(plain)
    assert "trend strength n/a ADX" in p2  # high/low-dependent values fall back to n/a
    assert "volatility ATR n/a" in p2 and "n/a" in p2


def test_build_snapshot_still_supports_close_and_volume_only_bars():
    s = build_snapshot("X", ohclv_only(list(range(1, 251))))
    assert s.ma50 > 0 and s.ma200 > 0 and s.rsi > 0  # pre-Phase-5 path unchanged behaviour
