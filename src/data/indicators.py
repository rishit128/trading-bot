"""Technical indicators (moving averages, Wilder RSI) and the per-stock snapshot the agents analyse."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

MIN_BARS = 200
MAX_STALE_DAYS = 5  # calendar days a last bar may lag the expected session (covers long weekends)


class StaleDataError(ValueError):
    """The latest bar is too old, or shows no trading: refuse to analyse rather than act on a dead price."""


@dataclass(frozen=True)
class Snapshot:
    """Indicator values for one stock as of its last completed bar."""
    symbol: str
    price: float
    ma50: float
    ma200: float
    rsi: float
    volume: int
    bar_date: Optional[str] = None  # date of the last bar used, for audit/replay and staleness checks
    avg_volume: Optional[int] = None  # 20-day average volume, gives the volume/context step a baseline (None pre-migration)
    momentum_6m: Optional[float] = None  # return over the last 6 months (126 sessions), for market-relative context
    momentum: Optional[float] = None  # 14-day return, short-horizon momentum for the step-2 context
    # Enriched indicators (all optional so pre-existing construction sites and Close+Volume-only bars keep working).
    macd: Optional[float] = None  # 12-26 EMA difference
    macd_signal: Optional[float] = None  # 9-EMA of the MACD line
    macd_histogram: Optional[float] = None  # macd - signal, positive is bullish
    bb_upper: Optional[float] = None  # upper 2-sigma 20-day Bollinger band, price level
    bb_middle: Optional[float] = None  # the 20-day middle band, price level
    bb_lower: Optional[float] = None  # lower 2-sigma band, price level
    bb_position: Optional[float] = None  # 0-1 inside the 20-day band; 0 = lower, 1 = upper
    volume_trend: Optional[float] = None  # recent 20-day mean volume vs the prior 20, fractional change
    atr: Optional[float] = None  # 14-day average true range (absolute, use with price for a %)
    adx: Optional[float] = None  # 14-day ADX, 0-100 trend strength


def _macd(close: pd.Series):
    """MACD line, signal line and histogram from a Close series (returns floats or None when data is too thin)."""
    if len(close) < 40:
        return None, None, None
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(macd.iloc[-1] - signal.iloc[-1])


def compute_momentum(close: pd.Series, period: int = 14) -> Optional[float]:
    """Short-horizon return over the last `period` sessions; None when the series is too short."""
    if len(close) < period:
        return None
    base = float(close.iloc[-period])
    if base == 0:
        return None
    return float(close.iloc[-1] / base - 1)


def _bollinger(close: pd.Series, period: int = 20):
    """Full Bollinger output: (upper, middle, lower, position), or None when the series is short."""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = sma + 2 * std, sma - 2 * std
    upper_v, mid_v, lower_v = float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])
    band = upper_v - lower_v
    if band == 0 or not pd.notna(band):
        pos = 0.5
    else:
        pos = min(1.0, max(0.0, float((close.iloc[-1] - lower_v) / band)))
    return upper_v, mid_v, lower_v, pos


def _volume_trend(volume: pd.Series, period: int = 20) -> Optional[float]:
    """How the last `period` bars' average volume compares with the `period` before: fractional change, or None if too short."""
    if len(volume) < 2 * period:
        return None
    recent = float(volume.iloc[-period:].mean())
    older = float(volume.iloc[-2 * period:-period].mean())
    if older <= 0:
        return None
    return float(recent / older - 1)


def compute_atr(high: Optional[pd.Series], low: Optional[pd.Series], close: pd.Series, period: int = 14) -> Optional[float]:
    """Wilder-smoothed average true range (absolute price volatility), or None when High/Low are missing."""
    if high is None or low is None or len(high) < period:
        return None
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    value = atr.iloc[-1]
    return float(value) if pd.notna(value) else None


def compute_adx(high: Optional[pd.Series], low: Optional[pd.Series], close: pd.Series, period: int = 14) -> Optional[float]:
    """Wilder ADX (0-100), higher = stronger trend in either direction; None without High/Low or a long enough series."""
    if high is None or low is None or len(high) < 2 * period:
        return None
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
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    value = adx.iloc[-1]
    return float(value) if pd.notna(value) else None


def rsi_from_array(values, period: int = 14) -> float:
    """Wilder's RSI of a numeric array: SMA-seeded first average, then recursive smoothing. Plain floats in the loop:
    the pandas version spent ~2 ms per call on Series overhead, and the scanner runs this for thousands of stocks."""
    delta = np.diff(np.asarray(values, dtype=float))
    delta = delta[~np.isnan(delta)]
    if len(delta) < period:
        raise ValueError("not enough data for RSI")
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    decay = period - 1
    for gain, loss in zip(gains[period:].tolist(), losses[period:].tolist()):
        avg_gain = (avg_gain * decay + gain) / period
        avg_loss = (avg_loss * decay + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI of a close series."""
    return rsi_from_array(close.to_numpy(dtype=float), period)


def build_snapshot(symbol: str, bars: pd.DataFrame) -> Snapshot:
    """bars: daily OHLCV with single-level columns (Close and Volume required; High/Low optional, used by ATR/ADX),
    oldest first."""
    if len(bars) < MIN_BARS:
        raise ValueError(f"{symbol}: need {MIN_BARS} bars for MA200, got {len(bars)}")
    close = bars["Close"].astype(float)
    if close.isna().iloc[-1]:
        raise ValueError(f"{symbol}: latest close is NaN")
    last = bars.index[-1]
    high = bars["High"].astype(float) if "High" in bars.columns else None
    low = bars["Low"].astype(float) if "Low" in bars.columns else None
    volume = bars["Volume"].astype(float)
    macd, macd_signal, macd_histogram = _macd(close)
    bollinger = _bollinger(close)
    return Snapshot(
        bar_date=str(last.date()) if isinstance(bars.index, pd.DatetimeIndex) else None,
        symbol=symbol,
        price=float(close.iloc[-1]),
        ma50=float(close.rolling(50).mean().iloc[-1]),
        ma200=float(close.rolling(200).mean().iloc[-1]),
        rsi=compute_rsi(close),
        volume=int(bars["Volume"].iloc[-1]),
        avg_volume=int(float(volume.tail(20).mean())),
        momentum_6m=float(close.iloc[-1] / close.iloc[-127] - 1) if len(close) > 126 else None,
        momentum=compute_momentum(close),
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bollinger[0] if bollinger else None,
        bb_middle=bollinger[1] if bollinger else None,
        bb_lower=bollinger[2] if bollinger else None,
        bb_position=bollinger[3] if bollinger else None,
        volume_trend=_volume_trend(volume),
        atr=compute_atr(high, low, close),
        adx=compute_adx(high, low, close),
    )
