"""Scoring the broker's research calls by what the price did, not by what the portal says. The portal's closed list shows only
"BOOK PROFIT" exits (no stop-loss ever appears), so its own record is flattering; and a call first read already closed is
skipped here because only the winners are left in that list. Each call that was read while still open is followed on the
5-minute bars: bought (or, for a sell call, sold short) at the open of the first bar that starts after it was published, and left at its stop,
its target, its time limit or the broker's own "BOOK PROFIT" exit, whichever comes first (a follower of the call would
have been told to book then). If one bar touches the stop and either the target or the booking, the stop counts (the
unfavourable reading). The round-trip cost is the intraday figure for every kind, an approximation."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dtime
from typing import Optional

import pandas as pd

from src.data.india import IST
from src.engine.ports import BAR

COST = 0.0021  # round trip, as a fraction (the intraday account's measured figure)
SQUARE_OFF = dtime(15, 15)
SWING_KINDS = ("Swing Trade", "Momentum", "Stellar Pick")
SWING_MAX_SESSIONS = 20


@dataclass(frozen=True)
class Call:
    symbol: str
    kind: str
    side: str  # buy | sell
    published: datetime  # UTC
    entry_low: float
    entry_high: float
    stop: float
    target: float
    valid_to: Optional[date]
    booked_at: Optional[datetime] = None  # UTC: when the broker closed the call with "BOOK PROFIT @price"
    booked_price: Optional[float] = None


def booked_price(comment: str) -> Optional[float]:
    """The price in the portal's exit comment ("BOOK PROFIT @7420"), or None for any other comment."""
    found = re.search(r"BOOK PROFIT\s*@\s*([0-9]+(?:\.[0-9]+)?)", comment or "", re.IGNORECASE)
    return float(found.group(1)) if found else None


@dataclass(frozen=True)
class Outcome:
    result: str  # TARGET | STOP | BOOKED (the broker's exit) | TIME | OPEN (the limit has not been reached yet) | NO_DATA
    entry: Optional[float] = None
    exit: Optional[float] = None
    gross: Optional[float] = None  # return on the entry price, the sign flipped for a sell call
    net: Optional[float] = None
    drift: Optional[float] = (
        None  # the signed return from the entry to the time limit (or the latest bar), stop and target ignored
    )
    in_zone: Optional[bool] = None  # whether the entry price was inside the call's own entry range (a follower could act)


def _limit(call: Call, sessions: list) -> Optional[pd.Timestamp]:
    """The last moment a position may be held, given the sessions (dates) on the bars; None when that is not reached yet."""
    if call.kind == "Intraday":
        return pd.Timestamp(datetime.combine(sessions[0], SQUARE_OFF, IST))
    if call.kind == "BTST":
        return pd.Timestamp(datetime.combine(sessions[1], SQUARE_OFF, IST)) if len(sessions) > 1 else None
    end = min(call.valid_to, sessions[min(SWING_MAX_SESSIONS, len(sessions)) - 1]) if call.valid_to else sessions[-1]
    reached = call.valid_to is not None and sessions[-1] >= call.valid_to or len(sessions) >= SWING_MAX_SESSIONS
    return pd.Timestamp(datetime.combine(end, SQUARE_OFF, IST)) if reached else None


def score_call(call: Call, bars: pd.DataFrame) -> Outcome:
    """The outcome of one call on a stock's 5-minute bars (IST index, Open High Low Close)."""
    stamps = pd.DatetimeIndex(bars.index).tz_convert(IST)
    later = bars[stamps >= pd.Timestamp(call.published).tz_convert(IST)]
    if later.empty:
        return Outcome("NO_DATA")
    sign = 1.0 if call.side == "buy" else -1.0
    entry = float(later["Open"].iloc[0])
    in_zone = call.entry_low * 0.995 <= entry <= call.entry_high * 1.005
    days = sorted(set(pd.DatetimeIndex(later.index).tz_convert(IST).date))
    limit = _limit(call, days)
    exit_price, result = float(later["Close"].iloc[-1]), "OPEN"
    booked = pd.Timestamp(call.booked_at).tz_convert(IST) if call.booked_at and call.booked_price else None
    held = later if limit is None else later[(pd.DatetimeIndex(later.index) + BAR) <= limit]
    drift = sign * (float(held["Close"].iloc[-1]) / entry - 1)
    for ts, row in later.iterrows():
        hit_stop = row["Low"] <= call.stop if sign > 0 else row["High"] >= call.stop
        hit_target = row["High"] >= call.target if sign > 0 else row["Low"] <= call.target
        if hit_stop:  # the stop wins a bar that touches both
            exit_price, result = call.stop, "STOP"
        elif hit_target:
            exit_price, result = call.target, "TARGET"
        elif booked is not None and booked < pd.Timestamp(str(ts)) + BAR:
            exit_price, result = float(call.booked_price), "BOOKED"  # type: ignore[arg-type]
        elif limit is not None and pd.Timestamp(str(ts)) + BAR >= limit:
            exit_price, result = float(row["Close"]), "TIME"
        if result != "OPEN":
            break
    gross = sign * (exit_price / entry - 1)
    return Outcome(result, entry, exit_price, gross, gross - COST, drift, in_zone)
