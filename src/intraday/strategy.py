"""Intraday opening-range-breakout (ORB), long only, on completed 5-minute bars. Pure functions, no I/O.

Rules (literature-default parameters, fixed before any backtest and NOT tuned):
  * Opening range = high/low of the first 3 five-minute bars (09:15-09:30 IST).
  * AgentSignal: a 5-minute bar closes above the range high, above the day's VWAP, with volume >= 1.5x the average volume of the
    bars so far, between 09:30 and 14:00, and the range is 0.3%-2.5% of price (too narrow = noise, too wide = poor risk).
  * Entry: the next bar's open (the caller supplies the live price). Stop: the range low. Target: 2 x the risk.
  * One trade per stock per day. Every position is closed at 15:15 IST at the latest."""
from dataclasses import dataclass
from datetime import time as dtime
from typing import Optional

import pandas as pd

from src.engine.enums import Action

RANGE_BARS = 3
ENTRY_FROM, ENTRY_UNTIL, SQUARE_OFF = dtime(9, 30), dtime(14, 0), dtime(15, 15)
MIN_RANGE, MAX_RANGE = 0.003, 0.025
VOLUME_MULT = 1.5
TARGET_R = 2.0


@dataclass(frozen=True)
class Setup:
    """A breakout signal: where to stop and where to take profit, for an entry at `signal_price`.

    `strength` stands in for an AI confidence score, since this rule has none: 0.0 at the minimum required volume
    surge (1.5x average), 1.0 at 2x that (3x average) or more. Used to size the position like a confidence-scaled
    swing signal -- a barely-qualifying breakout gets the floor size, a much stronger one gets closer to the ceiling."""
    symbol: str
    signal_price: float
    stop: float
    target: float
    range_high: float
    range_low: float
    bar_time: pd.Timestamp
    strength: float = 1.0


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative volume-weighted average price of the day's bars."""
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    return (typical * bars["Volume"]).cumsum() / bars["Volume"].cumsum()


def find_setup(symbol: str, day_bars: pd.DataFrame, start_index: int = RANGE_BARS) -> Optional[Setup]:
    """First breakout of the day in `day_bars` (one session's completed 5-minute bars, IST index), or None.

    `start_index` lets a backtest continue scanning after a stopped-out trade is not wanted: one trade per stock per day
    means only the first signal counts, so callers normally leave it at the default."""
    if len(day_bars) <= RANGE_BARS:
        return None
    opening = day_bars.iloc[:RANGE_BARS]
    range_high, range_low = float(opening["High"].max()), float(opening["Low"].min())
    width = (range_high - range_low) / range_high
    if not MIN_RANGE <= width <= MAX_RANGE:
        return None
    vw = vwap(day_bars)
    for i in range(start_index, len(day_bars)):
        bar = day_bars.iloc[i]
        t = day_bars.index[i].time()
        if t < ENTRY_FROM or t >= ENTRY_UNTIL:
            continue
        avg_volume = day_bars["Volume"].iloc[:i].mean()
        volume_ratio = bar["Volume"] / avg_volume if avg_volume > 0 else 0.0
        if bar["Close"] > range_high and bar["Close"] > vw.iloc[i] and volume_ratio >= VOLUME_MULT:
            entry = float(bar["Close"])
            risk = entry - range_low
            if risk <= 0:
                return None
            strength = min(1.0, max(0.0, (volume_ratio - VOLUME_MULT) / VOLUME_MULT))
            return Setup(symbol, entry, range_low, entry + TARGET_R * risk, range_high, range_low,
                        day_bars.index[i], strength)
    return None


def position_size(equity: float, entry: float, stop: float, strength: float = 1.0,
                  min_pct: float = 0.02, max_pct: float = 0.05) -> int:
    """Shares to buy: position value scaled linearly with signal strength, from min_pct of equity (a barely-qualifying
    breakout) to max_pct (a much stronger one) -- same scaling logic as the swing risk engine's confidence sizing."""
    if entry <= 0 or stop >= entry:
        return 0
    pct = min_pct + min(1.0, max(0.0, strength)) * (max_pct - min_pct)
    return max(0, int(equity * pct / entry))


def intraday_fees(side: str, value: float) -> float:
    """Approximate NSE intraday (MIS) equity costs with a flat-fee discount broker: brokerage min(Rs 20, 0.03%) per side,
    STT 0.025% on sells, stamp 0.003% on buys, exchange 0.00297%, SEBI 0.0001%, 18% GST on brokerage+exchange+SEBI."""
    brokerage = min(20.0, 0.0003 * value)
    exchange, sebi = 0.0000297 * value, 0.000001 * value
    stt = 0.00025 * value if side == Action.SELL else 0.0
    stamp = 0.00003 * value if side == Action.BUY else 0.0
    return brokerage + exchange + sebi + stt + stamp + 0.18 * (brokerage + exchange + sebi)
