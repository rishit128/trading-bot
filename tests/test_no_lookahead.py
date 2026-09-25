"""No-lookahead property tests: a feature decided from data up to day t must not depend on data after t.

Three layers are pinned:
  1. every scalar indicator returned by `build_snapshot` at date t equals the same indicator recomputed
     causally (rolling/ewm references) on the full series and read at position t;
  2. the portfolio backtest (`simulate`) fills BUY at the next day's open, never at the decision-day price,
     so a gap between the two is not borrowed from the future;
  3. the vectorised research engine holds a weight decided at day t only from day t + 1 + exec_lag, so the
     return of the decision day is never earned on a weight that did not exist yet."""
import numpy as np
import pandas as pd
import pytest

from src.research.backtest import rule_signals, simulate
from src.config import RiskLimits
from src.data.indicators import build_snapshot
from src.research.engine import portfolio_returns


def _ohlcv(closes, start="2020-01-02"):
    """Daily OHLCV (Close+Volume required, High/Low for ATR/ADX), oldest first."""
    idx = pd.date_range(start, periods=len(closes), freq="B")
    closes = pd.Series([float(c) for c in closes], index=idx)
    high, low = closes * 1.02, closes * 0.98
    volume = pd.Series(10000 + (np.arange(len(closes)) % 3) * 3000, index=idx)
    return pd.DataFrame({"Open": closes, "High": high, "Low": low, "Close": closes, "Volume": volume})


def _rsi_vector(close, period=14):
    """Reference Wilder RSI as a full series (same seed/recursion as `compute_rsi`, aligned to the index)."""
    delta = close.diff().dropna()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gains.iloc[:period].mean(), losses.iloc[:period].mean()
    out = pd.Series(np.nan, index=close.index)
    for i in range(period, len(delta)):
        gain, loss = gains.iloc[i], losses.iloc[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out.iloc[i + 1] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def _causal_references(df):
    """Full-length causal vectors for every feature, keyed by snapshot field name."""
    close, volume = df["Close"], df["Volume"]
    prev = close.shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return {
        "price": close,
        "ma50": close.rolling(50).mean(),
        "ma200": close.rolling(200).mean(),
        "rsi": _rsi_vector(close),
        "momentum": close / close.shift(13) - 1,  # return over the last 14 sessions (14 closes, 13 steps)
        "momentum_6m": close / close.shift(126) - 1,
        "macd": close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean(),
        "macd_histogram": lambda m_e: m_e - m_e.ewm(span=9, adjust=False).mean(),
        "atr": tr.ewm(alpha=1 / 14, adjust=False).mean(),
        "volume_trend": volume.rolling(20).mean() / volume.shift(20).rolling(20).mean() - 1,
    }


def _band_position_vector(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = sma + 2 * std, sma - 2 * std
    band = upper - lower
    pos = (close - lower) / band
    pos = pos.where(band > 1e-12, 0.5)
    return (pos.clip(lower=0.0, upper=1.0)).where(std.notna())


def _adx_vector(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    denom = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / denom.where(denom > 0)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def test_every_snapshot_feature_is_causal_across_horizons():
    df = _ohlcv(100.0 * (1 + 0.004 * np.arange(260)))
    ref = _causal_references(df)
    ref["macd_signal"] = ref["macd"].ewm(span=9, adjust=False).mean()
    ref["macd_histogram"] = ref["macd"] - ref["macd_signal"]
    ref["bb_position"] = _band_position_vector(df["Close"])
    ref["adx"] = _adx_vector(df)
    positions = {d: df.index.get_loc(d) for d in (df.index[210], df.index[235], df.index[259])}
    for d, pos in positions.items():
        snap = build_snapshot("X", df.loc[:d])
        assert snap.price == pytest.approx(ref["price"].iloc[pos])
        assert snap.ma50 == pytest.approx(ref["ma50"].iloc[pos], rel=1e-6)
        assert snap.ma200 == pytest.approx(ref["ma200"].iloc[pos], rel=1e-6)
        assert snap.rsi == pytest.approx(ref["rsi"].iloc[pos], abs=1e-6)
        assert snap.momentum == pytest.approx(ref["momentum"].iloc[pos], rel=1e-6)
        assert snap.momentum_6m == pytest.approx(ref["momentum_6m"].iloc[pos], rel=1e-6)
        assert snap.macd == pytest.approx(ref["macd"].iloc[pos], rel=1e-6)
        assert snap.macd_signal == pytest.approx(ref["macd_signal"].iloc[pos], rel=1e-6)
        assert snap.macd_histogram == pytest.approx(ref["macd_histogram"].iloc[pos], rel=1e-6)
        assert snap.bb_position == pytest.approx(ref["bb_position"].iloc[pos], abs=1e-6)
        assert snap.atr == pytest.approx(ref["atr"].iloc[pos], rel=1e-6)
        assert snap.adx == pytest.approx(ref["adx"].iloc[pos], abs=1e-6)
        assert snap.volume_trend == pytest.approx(ref["volume_trend"].iloc[pos], rel=1e-6)


def test_snapshot_bar_date_is_the_last_used_bar_not_the_fetch_date():
    df = _ohlcv([100.0 + i for i in range(230)])
    d = df.index[200]
    snap = build_snapshot("X", df.loc[:d])
    assert snap.bar_date == str(d.date())


def test_backtest_fills_never_use_the_decision_day_price():
    closes = [100.0 + i for i in range(5)]
    df = _ohlcv(closes)
    d0, d1, d2 = df.index[0], df.index[1], df.index[2]
    df.loc[d1, "Open"] = closes[0] * 1.5  # a big gap between decision close and next open
    df.loc[d2, "Open"] = closes[1] * 1.3
    signals = {"A": {d0: ("BUY", 0.9), d1: ("SELL", 0.9)}}
    result = simulate({"A": df}, signals, RiskLimits())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == d1 and trade.entry_price == pytest.approx(df.loc[d1, "Open"])
    assert trade.entry_price != pytest.approx(df.loc[d0, "Close"])  # would be the lookahead fill
    assert trade.exit_date == d2 and trade.exit_price == pytest.approx(df.loc[d2, "Open"])


def test_rule_signals_use_only_data_up_to_the_decision_date():
    n = 230
    # A noisy long uptrend (so RSI stays mid-range instead of pinning at 100) followed by a crash in the final bar.
    closes = [100.0 * (1 + 0.004 * i + 0.008 * ((i % 7) - 3)) for i in range(n)]
    df = _ohlcv(closes)
    last = df.index[-1]
    df.loc[last, "Close"] = df.loc[df.index[-2], "Close"] * 0.5
    dates = [df.index[200], last]
    out = rule_signals({"A": df}, dates)
    assert out["A"][dates[0]][0] == "BUY"  # decided while the trend was still up: no future knowledge
    assert out["A"][last][0] == "SELL"  # only the bar <= that date can flip it


def test_research_engine_earns_a_weight_only_after_the_execution_lag():
    idx = pd.date_range("2021-01-04", periods=6, freq="B")
    returns_a = [np.nan, 0.01, 0.02, 0.03, 0.04, 0.05]
    returns_b = [np.nan, 0.005, 0.005, 0.005, 0.005, 0.005]
    close = pd.DataFrame({
        "A": (1 + pd.Series(returns_a, index=idx)).cumprod() * 100,
        "B": (1 + pd.Series(returns_b, index=idx)).cumprod() * 100,
    })
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[3], "A"] = 1.0  # decided at the close of day 3
    result = portfolio_returns(close, weights)
    assert float(result["gross"].iloc[3]) == pytest.approx(0.0)  # not owned yet
    assert float(result["gross"].iloc[4]) == pytest.approx(0.0)  # executed at day-4 close, no same-day return
    assert float(result["gross"].iloc[5]) == pytest.approx(0.05)  # first day the weight earns a return
    assert float(result["turnover"].iloc[4]) == pytest.approx(1.0)  # the weight's execution, charged that day