"""Intraday opening-range-breakout (ORB), long only, on completed 5-minute bars. Pure functions, no I/O.

Rules (literature-default parameters, fixed before any backtest and NOT tuned):
  * Opening range = high/low of the first 3 five-minute bars (09:15-09:30 IST).
  * Signal: a 5-minute bar closes above the range high, above the day's VWAP, with volume >= 1.5x the average volume of the
    bars so far, between 09:30 and 14:00, and the range is 0.3%-2.5% of price (too narrow = noise, too wide = poor risk).
  * Entry: the next bar's open (the caller supplies the live price). Stop: the range low. Target: 2 x the risk.
  * One trade per stock per day. Every position is closed at 15:15 IST at the latest.
  * The opening range must really be the 09:15-09:30 bars: a stock whose first bar is missing or late (thin trading, a
    data gap) has no valid range and is skipped rather than given one built from whichever bars happen to exist.

Sizing is by risk under a position cap (`position_size`). The rule itself is one numpy function, `first_breakout`, shared by the
live engine and every research script, so nothing that is tested differs from what trades.

An audit of this rule on 58 sessions of Nifty 100 bars (STRATEGY.md, 2026-09-28) found no edge: it does no better than
entering at a random time with the same stop and target, and neither the VWAP nor the volume filter adds anything."""
from dataclasses import dataclass
from datetime import time as dtime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.engine.enums import Action
from src.engine.ports import BAR

RANGE_BARS = 3
MARKET_OPEN = dtime(9, 15)
ENTRY_FROM, ENTRY_UNTIL, SQUARE_OFF = dtime(9, 30), dtime(14, 0), dtime(15, 15)
MIN_RANGE, MAX_RANGE = 0.003, 0.025
VOLUME_MULT = 1.5
TARGET_R = 2.0
CASH_HEADROOM = 1.003  # cash kept back per share bought: slippage (0.05%) and both fees fit inside 0.3%


@dataclass(frozen=True)
class Setup:
    """A breakout signal: where to stop and where to take profit, for an entry at `signal_price`.

    `volume_ratio` is the breakout bar's volume over the average of the bars before it; `strength` rescales it to 0-1
    (0.0 at the 1.5x minimum, 1.0 at 3x or more). It is recorded for analysis and no longer sizes the position: the audit
    found the strongest-volume signals did no better than the weakest."""
    symbol: str
    signal_price: float
    stop: float
    target: float
    range_high: float
    range_low: float
    bar_time: pd.Timestamp
    strength: float = 1.0
    volume_ratio: float = 0.0


@dataclass(frozen=True)
class Breakout:
    """The first breakout bar found in a session's arrays (see `first_breakout`)."""
    index: int
    range_high: float
    range_low: float
    width: float           # the opening range as a fraction of its high
    volume_ratio: float    # the bar's volume over the average of the bars before it


def minute_of_day(t: dtime) -> int:
    """Minutes since midnight, the unit `first_breakout` compares bar times in."""
    return t.hour * 60 + t.minute


def session_vwap_and_volume_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                  volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per bar: the session's VWAP up to and including that bar (typical price H+L+C over 3, volume-weighted), and the bar's
    volume over the mean volume of the bars BEFORE it (0 when there was none). Both use only data up to that bar."""
    running_volume = np.cumsum(volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum((high + low + close) / 3 * volume) / running_volume
        prior_average = np.concatenate([[np.nan], running_volume[:-1] / np.arange(1, len(close))])   # mean of bars 0..i-1
        ratio = np.where(prior_average > 0, volume / prior_average, 0.0)
    return vwap, ratio


def first_breakout(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, minutes: np.ndarray,
                   use_range: bool = True, use_vwap: bool = True, use_volume: bool = True) -> Optional[Breakout]:
    """The first bar after the opening range that closes above its high, above VWAP, on a volume surge, inside the entry
    window; or None. Works on whole-session numpy arrays (`minutes` = minutes since midnight per bar) and is the one place
    the rule is written down: the live engine reaches it through `find_setup`, the audit through switches that disable a
    filter to measure what it adds. Every value at bar i uses bars up to and including i only."""
    n = len(close)
    if n <= RANGE_BARS:
        return None
    range_high, range_low = float(high[:RANGE_BARS].max()), float(low[:RANGE_BARS].min())
    width = (range_high - range_low) / range_high
    if use_range and not MIN_RANGE <= width <= MAX_RANGE:
        return None
    vwap, ratio = session_vwap_and_volume_ratio(high, low, close, volume)
    ok = (close > range_high) & (minutes >= minute_of_day(ENTRY_FROM)) & (minutes < minute_of_day(ENTRY_UNTIL))
    ok[:RANGE_BARS] = False
    if use_vwap:
        ok &= close > vwap
    if use_volume:
        ok &= ratio >= VOLUME_MULT
    hits = np.flatnonzero(ok)
    if len(hits) == 0:
        return None
    i = int(hits[0])
    return Breakout(i, range_high, range_low, width, float(ratio[i]))


def opening_range_intact(day_bars: pd.DataFrame) -> bool:
    """True when the first RANGE_BARS bars are the real opening bars: the first starts at 09:15 and they follow each other
    without a gap. Yahoo drops a bar a stock did not trade in, so an illiquid stock's "first three bars" can be 09:25-09:35
    -- a range of the wrong period, which would fire a breakout signal on a range that never existed."""
    if len(day_bars) < RANGE_BARS:
        return False
    idx = day_bars.index[:RANGE_BARS]
    if idx[0].time() != MARKET_OPEN:
        return False
    return all(idx[i + 1] - idx[i] == BAR for i in range(RANGE_BARS - 1))


def find_setup(symbol: str, day_bars: pd.DataFrame) -> Optional[Setup]:
    """The day's first breakout in `day_bars` (one session's completed 5-minute bars, IST index), or None. One trade per
    stock per day means only the first signal counts."""
    if not opening_range_intact(day_bars):
        return None
    index = pd.DatetimeIndex(day_bars.index)
    found = first_breakout(day_bars["High"].to_numpy(float), day_bars["Low"].to_numpy(float), day_bars["Close"].to_numpy(float),
                           day_bars["Volume"].to_numpy(float), (index.hour * 60 + index.minute).to_numpy())
    if found is None:
        return None
    entry = float(day_bars["Close"].iloc[found.index])
    risk = entry - found.range_low
    strength = min(1.0, max(0.0, (found.volume_ratio - VOLUME_MULT) / VOLUME_MULT))
    return Setup(symbol, entry, found.range_low, entry + TARGET_R * risk, found.range_high, found.range_low,
                 day_bars.index[found.index], strength, found.volume_ratio)


def position_size(equity: float, entry: float, stop: float, risk_pct: float, max_position_pct: float,
                  cash: Optional[float] = None) -> Tuple[int, Optional[str]]:
    """Shares to buy and, when that is zero, the reason (so a skipped signal is explained, never silent).

    Sized by risk: the loss if the stop is hit costs at most `risk_pct` of equity. The position is also capped at
    `max_position_pct` of equity and at the cash available (cash only, no leverage; a little is held back for slippage and
    fees). A single share dearer than the cap cannot be bought at all."""
    if entry <= 0 or stop >= entry:
        return 0, "the stop is not below the entry price"
    cap_value = equity * max_position_pct
    by_cap = int(cap_value / entry)
    if by_cap < 1:
        return 0, f"one share (Rs {entry:,.2f}) costs more than the largest position allowed (Rs {cap_value:,.0f})"
    by_risk = int(equity * risk_pct / (entry - stop))
    if by_risk < 1:
        return 0, f"risking {risk_pct:.2%} of equity buys less than one share at a Rs {entry - stop:,.2f} stop distance"
    qty = min(by_cap, by_risk)
    if cash is not None:
        by_cash = int(cash / (entry * CASH_HEADROOM))
        if by_cash < 1:
            return 0, f"not enough cash (Rs {cash:,.0f}) for one share at Rs {entry:,.2f}"
        qty = min(qty, by_cash)
    return qty, None


def intraday_fees(side: str, value: float) -> float:
    """Approximate NSE intraday (MIS) equity costs with a flat-fee discount broker: brokerage min(Rs 20, 0.03%) per side,
    STT 0.025% on sells, stamp 0.003% on buys, exchange 0.00297%, SEBI 0.0001%, 18% GST on brokerage+exchange+SEBI."""
    brokerage = min(20.0, 0.0003 * value)
    exchange, sebi = 0.0000297 * value, 0.000001 * value
    stt = 0.00025 * value if side == Action.SELL else 0.0
    stamp = 0.00003 * value if side == Action.BUY else 0.0
    return brokerage + exchange + sebi + stt + stamp + 0.18 * (brokerage + exchange + sebi)
