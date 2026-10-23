"""News-Catalyst v1 entry rules (long only), on completed 5-minute bars. Pure functions, no I/O. Fixed before any test and
NOT tuned; the plan is written up in STRATEGY.md ("News-catalyst intraday", 2026-10-21).

  A. News filed overnight (after the previous session's close, before 09:15): the stock opens flat or up, at most 8%; buy on the first
     5-minute close above the 09:15-09:30 high, from 09:30 to 10:30, while above VWAP. Stop at the 09:15-09:30 low; skip the
     trade when that stop is more than 3% or less than 0.6% below the entry.
  B. News filed during the session (09:15 to 14:30): the first 5-minute bar that closes after the filing must close at least
     0.75% above the price at filing time (the last close before it), on at least 3x that stock's normal volume for that
     time of day (its average at the same bar time on the earlier sessions in hand). Stop at the price at filing time.
  C. Market filter: no long trades when fewer than 40% of the universe trade above their previous close at 09:30.
  D. (v2) Live order book at the entry: skip when the bid-ask spread is above 0.5% or sellers dominate the book (balance
     below -0.3). Applied only when Angel One's quote is available.
Every position: stop to the entry once the trade is up by its original risk (1R); no target; closed at 15:15.

The plan also skips stocks with a 2% or 5% price band and stocks within 1% of the upper circuit. Free data does not give a
stock's band; the news universe (the most liquid stocks, mostly F&O names without bands) rarely has one, so the rule here is
an approximation: no entry 9% or more above the previous close (within 1% of a 10% band)."""

from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.india import IST
from src.engine.ports import BAR
from src.intraday import strategy as st

GAP_MIN, GAP_MAX = 0.0, 0.08  # v2: v1 needed +2%; the breakout and VWAP below now carry the price confirmation
SPREAD_MAX = 0.5  # percent: the live bid-ask spread above which the entry costs too much
BOOK_MIN = -0.3  # order book balance (+1 all buyers): below it sellers dominate and the entry is skipped
A_FROM, A_UNTIL = dtime(9, 30), dtime(10, 30)  # A's signal bars start in [A_FROM, A_UNTIL)
STOP_MIN, STOP_MAX = 0.006, 0.03  # A's stop distance as a fraction of the entry
B_FROM, B_UNTIL = dtime(9, 15), dtime(14, 30)  # B's filings, inclusive
B_MOVE, B_VOLUME = 0.0075, 3.0
BREADTH_MIN = 0.40
BREADTH_BAR = dtime(9, 25)  # the bar whose close is the price "at 09:30"
NEAR_CIRCUIT = 0.09

WAITING, FIRED, NO = "WAITING", "FIRED", "NO"


@dataclass(frozen=True)
class RuleCheck:
    """What a rule says about one filing now: FIRED (with the setup), WAITING (not decided yet) or NO (with the reason)."""

    status: str
    reason: str = ""
    setup: Optional[st.Setup] = None
    ref_price: Optional[float] = None


def _minutes(index: pd.Index) -> np.ndarray:
    stamps = pd.DatetimeIndex(index).tz_convert(IST)
    return (stamps.hour * 60 + stamps.minute).to_numpy()


def _near_circuit(price: float, prev_close: Optional[float]) -> bool:
    return prev_close is not None and prev_close > 0 and price >= prev_close * (1 + NEAR_CIRCUIT)


def _setup(symbol: str, day: pd.DataFrame, i: int, stop: float, range_high: float, range_low: float, ratio: float) -> st.Setup:
    entry = float(day["Close"].iloc[i])
    return st.Setup(symbol, entry, stop, entry * (1 + st.NO_TARGET), range_high, range_low, day.index[i], volume_ratio=ratio)


def rule_a(symbol: str, day: pd.DataFrame, prev_close: Optional[float]) -> RuleCheck:
    """Overnight news: gap 2-8%, then the first close above the opening-range high and VWAP from 09:30 to 10:30."""
    if prev_close is None:
        return RuleCheck(NO, "no previous close")
    if len(day) < st.RANGE_BARS:
        return RuleCheck(WAITING, "opening range not finished")
    if not st.opening_range_intact(day):
        return RuleCheck(NO, "opening range incomplete (missing or late first bars)")
    gap = float(day["Open"].iloc[0]) / prev_close - 1
    if not GAP_MIN <= gap <= GAP_MAX:
        return RuleCheck(NO, f"gap {gap:+.1%} (needs {GAP_MIN:+.0%} to {GAP_MAX:+.0%})", ref_price=prev_close)
    high, low = day["High"].to_numpy(float), day["Low"].to_numpy(float)
    close, volume = day["Close"].to_numpy(float), day["Volume"].to_numpy(float)
    range_high, range_low = float(high[: st.RANGE_BARS].max()), float(low[: st.RANGE_BARS].min())
    vwap, ratio = st.session_vwap_and_volume_ratio(high, low, close, volume)
    minutes = _minutes(day.index)
    ok = (close > range_high) & (close > vwap)
    ok &= (minutes >= st.minute_of_day(A_FROM)) & (minutes < st.minute_of_day(A_UNTIL))
    ok[: st.RANGE_BARS] = False
    hits = np.flatnonzero(ok)
    if len(hits) == 0:
        if minutes[-1] < st.minute_of_day(A_UNTIL) - 5:
            return RuleCheck(WAITING, "no close above the 09:15-09:30 high and VWAP yet", ref_price=prev_close)
        return RuleCheck(NO, "no close above the 09:15-09:30 high and VWAP by 10:30", ref_price=prev_close)
    i = int(hits[0])
    entry = float(close[i])
    distance = (entry - range_low) / entry
    if not STOP_MIN <= distance <= STOP_MAX:
        return RuleCheck(
            NO, f"stop {distance:.2%} below the entry (needs {STOP_MIN:.1%} to {STOP_MAX:.0%})", ref_price=prev_close
        )
    if _near_circuit(entry, prev_close):
        return RuleCheck(NO, "within 1% of a 10% price band", ref_price=prev_close)
    setup = _setup(symbol, day, i, range_low, range_high, range_low, float(ratio[i]))
    return RuleCheck(FIRED, f"gap {gap:+.1%}, close above {range_high:,.2f} and VWAP", setup, prev_close)


def book_veto(note: Dict[str, object]) -> Optional[str]:
    """Rule D: why Angel One's live quote refuses the entry (a wide spread or a seller-heavy book), else None."""
    spread, balance = note.get("spread_pct"), note.get("book_imbalance")
    if isinstance(spread, (int, float)) and spread > SPREAD_MAX:
        return f"spread {spread:.2f}% (max {SPREAD_MAX}%)"
    if isinstance(balance, (int, float)) and balance < BOOK_MIN:
        return f"order book {balance:+.2f}: sellers dominate (needs {BOOK_MIN:+.1f} or better)"
    return None


def normal_volume(history: pd.DataFrame, bar_start: pd.Timestamp, today) -> Optional[float]:
    """The stock's average volume in the bar starting at the same time of day, over the earlier sessions in `history`."""
    stamps = pd.DatetimeIndex(history.index).tz_convert(IST)
    at = pd.Timestamp(bar_start).tz_convert(IST)
    same = history[(stamps.date < today) & (stamps.hour == at.hour) & (stamps.minute == at.minute)]
    return float(same["Volume"].mean()) if len(same) else None


def rule_b(symbol: str, day: pd.DataFrame, filed_at: datetime, prev_close: Optional[float], usual: Optional[float]) -> RuleCheck:
    """News during the session: the first bar closing after the filing closes 0.75%+ above the price at filing, on 3x the
    normal volume for that time of day. `usual` is that normal volume (see `normal_volume`), for the signal bar."""
    if not len(day):
        return RuleCheck(WAITING, "no bars yet")
    starts = pd.DatetimeIndex(day.index)
    ends = starts + BAR
    filed = pd.Timestamp(filed_at)
    before = np.flatnonzero(ends <= filed)
    ref = float(day["Close"].iloc[before[-1]]) if len(before) else float(day["Open"].iloc[0])
    after = np.flatnonzero(ends > filed)
    if len(after) == 0:
        return RuleCheck(WAITING, "the first bar after the filing has not closed yet", ref_price=ref)
    i = int(after[0])
    close, volume = float(day["Close"].iloc[i]), float(day["Volume"].iloc[i])
    move = close / ref - 1
    if move < B_MOVE:
        return RuleCheck(
            NO, f"first bar after the filing closed {move:+.2%} from {ref:,.2f} (needs +{B_MOVE:.2%})", ref_price=ref
        )
    if not usual:
        return RuleCheck(NO, "no volume history for that time of day", ref_price=ref)
    ratio = volume / usual
    if ratio < B_VOLUME:
        return RuleCheck(NO, f"volume {ratio:.1f}x normal for that time (needs {B_VOLUME:.0f}x)", ref_price=ref)
    if _near_circuit(close, prev_close):
        return RuleCheck(NO, "within 1% of a 10% price band", ref_price=ref)
    setup = _setup(symbol, day, i, ref, ref, ref, ratio)
    return RuleCheck(FIRED, f"{move:+.2%} after the filing on {ratio:.1f}x volume", setup, ref)


def breadth(sessions: Dict[str, pd.DataFrame], prev_closes: Dict[str, float]) -> Optional[float]:
    """Share of stocks whose 09:25-09:30 bar closed above the previous close; None when under half of them have that bar."""
    up = seen = 0
    for symbol, day in sessions.items():
        prev = prev_closes.get(symbol)
        if not prev or not len(day):
            continue
        at = np.flatnonzero(_minutes(day.index) == st.minute_of_day(BREADTH_BAR))
        if len(at):
            seen += 1
            up += float(day["Close"].iloc[at[0]]) > prev
    return up / seen if seen and seen >= 0.5 * len(sessions) else None


def breakeven_due(entry: float, stop: float, high_since_entry: float) -> bool:
    """The stop moves to the entry once a bar has reached the entry plus the original risk (1R)."""
    return stop < entry and high_since_entry >= entry + (entry - stop)
