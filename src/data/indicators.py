"""Technical indicators (moving averages, Wilder RSI) and the per-stock snapshot the agents analyse."""
from dataclasses import dataclass
from typing import Optional

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


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI: SMA-seeded first average, then recursive smoothing."""
    delta = close.diff().dropna()
    if len(delta) < period:
        raise ValueError("not enough data for RSI")
    gains = delta.clip(lower=0).to_numpy()
    losses = (-delta.clip(upper=0)).to_numpy()
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def build_snapshot(symbol: str, bars: pd.DataFrame) -> Snapshot:
    """bars: daily OHLCV with single-level columns Close and Volume, oldest first."""
    if len(bars) < MIN_BARS:
        raise ValueError(f"{symbol}: need {MIN_BARS} bars for MA200, got {len(bars)}")
    close = bars["Close"].astype(float)
    if close.isna().iloc[-1]:
        raise ValueError(f"{symbol}: latest close is NaN")
    last = bars.index[-1]
    return Snapshot(
        bar_date=str(last.date()) if isinstance(bars.index, pd.DatetimeIndex) else None,
        symbol=symbol,
        price=float(close.iloc[-1]),
        ma50=float(close.rolling(50).mean().iloc[-1]),
        ma200=float(close.rolling(200).mean().iloc[-1]),
        rsi=compute_rsi(close),
        volume=int(bars["Volume"].iloc[-1]),
    )
