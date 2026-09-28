"""Audit of the intraday opening-range breakout on stored 5-minute bars: does the signal beat entering at a random time, does
each filter add anything, and how much do the live simulator's shortcuts change the result. Pure functions on numpy arrays.

The rule is not re-implemented here: `first_breakout` calls the same numpy core the live engine uses
(`src.intraday.strategy.first_breakout`), with switches that disable a filter so its contribution can be measured.

Reading the result: `gross` is the trade's return before slippage and fees; `net` is after both. A rule has an edge only if
its net beats a RANDOM entry with the same stop, target and exits by more than the noise."""
from dataclasses import dataclass
from datetime import date as Date, time as dtime
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from src.intraday import strategy as st

MAX_SESSION_GAP_DAYS = 6  # a weekend plus a long holiday; further apart, the "previous close" is not the previous session's
FEE_NOTIONAL = 50_000.0  # a fixed trade value, so fees are a constant fraction and variants compare on expectancy alone
FEE_FRACTION = (st.intraday_fees("BUY", FEE_NOTIONAL) + st.intraday_fees("SELL", FEE_NOTIONAL)) / FEE_NOTIONAL


@dataclass(frozen=True)
class StockDay:
    """One stock's regular session as arrays, with the previous session's close (None for the first day seen)."""
    symbol: str
    date: Date
    t: np.ndarray            # each bar's start time (datetime.time)
    minutes: np.ndarray      # the same, as minutes since midnight
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    prev_close: Optional[float]


@dataclass(frozen=True)
class Entry:
    """A breakout entry of the live rule: the bar it happens on and the range it broke."""
    index: int          # the bar whose close is the entry price
    range_high: float
    range_low: float    # also the stop
    width: float        # opening range as a fraction of its high
    volume_ratio: float


def stock_days(data: Dict[str, pd.DataFrame], min_bars: int = 60) -> List[StockDay]:
    """Every usable stock-session. Sessions with fewer than `min_bars` bars are dropped, as in the backtest (a filter that
    uses end-of-day information), and so are sessions whose opening range is not the real 09:15-09:30 (as live)."""
    out: List[StockDay] = []
    for symbol, df in data.items():
        prev: Optional[float] = None
        prev_day: Optional[Date] = None
        for day, g in df.groupby(pd.DatetimeIndex(df.index).date):
            adjacent = prev_day is not None and (cast(Date, day) - prev_day).days <= MAX_SESSION_GAP_DAYS
            if len(g) >= min_bars and st.opening_range_intact(g):
                stamps = pd.DatetimeIndex(g.index)
                out.append(StockDay(symbol, cast(Date, day), np.array([x.time() for x in g.index]),
                                    (stamps.hour * 60 + stamps.minute).to_numpy(), g["Open"].to_numpy(float),
                                    g["High"].to_numpy(float), g["Low"].to_numpy(float), g["Close"].to_numpy(float),
                                    g["Volume"].to_numpy(float), prev if adjacent else None))
            prev, prev_day = float(g["Close"].iloc[-1]), cast(Date, day)
    return out


def first_breakout(d: StockDay, use_vwap: bool = True, use_volume: bool = True, use_range: bool = True) -> Optional[Entry]:
    """The strategy's first breakout of the day, with each filter switchable. All on = the rule that trades."""
    found = st.first_breakout(d.h, d.l, d.c, d.v, d.minutes, use_range=use_range, use_vwap=use_vwap, use_volume=use_volume)
    return None if found is None else Entry(found.index, found.range_high, found.range_low, found.width, found.volume_ratio)


def random_entry(d: StockDay, rng: np.random.Generator) -> Optional[Entry]:
    """The null hypothesis: a random entry bar in the same window, anywhere still above the range low (so the stop is valid)."""
    range_high, range_low = float(d.h[:st.RANGE_BARS].max()), float(d.l[:st.RANGE_BARS].min())
    width = (range_high - range_low) / range_high
    if not st.MIN_RANGE <= width <= st.MAX_RANGE:
        return None
    in_window = (d.minutes >= st.minute_of_day(st.ENTRY_FROM)) & (d.minutes < st.minute_of_day(st.ENTRY_UNTIL))
    hits = np.flatnonzero(in_window & (d.c > range_low))    # (the window opens at 09:30, after the range bars)
    if len(hits) == 0:
        return None
    return Entry(int(rng.choice(hits)), range_high, range_low, width, 0.0)


@dataclass(frozen=True)
class Plan:
    """A trade to simulate: enter at the close of bar `index`, long (+1) or short (-1), with an optional stop and target."""
    index: int
    side: int = 1
    stop: Optional[float] = None
    target: Optional[float] = None


def simulate_plan(d: StockDay, plan: Plan, mode: str = "full") -> Tuple[float, str]:
    """(raw exit price, why). Every later bar is checked in order: the 15:15 square-off at that bar's open; a stop (the bar
    opening through it fills at the open, otherwise it fills at the level); a target. A bar touching both stop and target
    counts as the stop. mode "full" examines each bar's whole range (the backtest); mode "open_only" sees only each bar's
    open (what the paper broker did before 2026-09-28, when a bar's range was skipped after its first ~20 seconds)."""
    side, stop, target = plan.side, plan.stop, plan.target
    for j in range(plan.index + 1, len(d.c)):
        if d.t[j] >= st.SQUARE_OFF:
            return float(d.o[j]), "EOD"
        adverse, favourable = (d.l[j], d.h[j]) if side > 0 else (d.h[j], d.l[j])
        if stop is not None:
            if side * (d.o[j] - stop) <= 0:
                return float(d.o[j]), "STOP"
            if mode == "full" and side * (adverse - stop) <= 0:
                return stop, "STOP"
        if target is not None:
            if mode == "full" and side * (favourable - target) >= 0:
                return target, "TARGET"
            if mode != "full" and side * (d.o[j] - target) >= 0:
                return float(d.o[j]), "TARGET"
    return float(d.c[-1]), "EOD"


def trade_returns(entry: float, exit_price: float, side: int, slippage: float) -> Tuple[float, float]:
    """(gross, net) return on the entry value. Gross ignores slippage and fees; net pays slippage on both fills (a long
    buys higher and sells lower; a short sells lower and buys back higher) and the round-trip fee."""
    if side > 0:
        return exit_price / entry - 1, exit_price * (1 - slippage) / (entry * (1 + slippage)) - 1 - FEE_FRACTION
    return 1 - exit_price / entry, (1 - slippage) - exit_price / entry * (1 + slippage) - FEE_FRACTION


def orb_plan(d: StockDay, e: Entry) -> Plan:
    """The live rule's trade: long at the breakout close, stop at the range low, target 2 x the risk above."""
    entry = float(d.c[e.index])
    return Plan(e.index, 1, e.range_low, entry + st.TARGET_R * (entry - e.range_low))


def simulate(d: StockDay, e: Entry, mode: str = "full") -> Tuple[float, str]:
    """The live rule's exit for a breakout entry (see `simulate_plan`)."""
    return simulate_plan(d, orb_plan(d, e), mode)


def _trade_row(d: StockDay, index: int, side: int, exit_price: float, why: str, slippage: float,
               breadth: Dict[Date, float]) -> dict:
    entry = float(d.c[index])
    gross, net = trade_returns(entry, exit_price, side, slippage)
    return {"symbol": d.symbol, "date": d.date, "t": d.t[index], "side": side, "why": why, "price": entry, "exit": exit_price,
            "gross": gross, "net": net, "gap": (d.o[0] / d.prev_close - 1) if d.prev_close else np.nan,
            "breadth": breadth.get(d.date, np.nan)}


def evaluate(days: List[StockDay], pick: Callable[[StockDay], Optional[Entry]], mode: str = "full",
             slippage: float = 0.0005) -> pd.DataFrame:
    """One row per trade of the live rule (or a variant of it): returns, exit reason, entry time and price, the breakout's
    width and volume surge, and the day's context."""
    breadth = _breadth(days)
    rows = []
    for d in days:
        e = pick(d)
        if e is not None:
            exit_price, why = simulate(d, e, mode)
            row = _trade_row(d, e.index, 1, exit_price, why, slippage, breadth)
            rows.append({**row, "width": e.width, "volume_ratio": e.volume_ratio})
    return pd.DataFrame(rows)


def evaluate_plans(days: List[StockDay], pick: Callable[[StockDay], Optional[Plan]], mode: str = "full",
                   slippage: float = 0.0005) -> pd.DataFrame:
    """One row per trade for any long/short plan (see `simulate_plan`)."""
    breadth = _breadth(days)
    rows = []
    for d in days:
        plan = pick(d)
        if plan is not None:
            exit_price, why = simulate_plan(d, plan, mode)
            rows.append(_trade_row(d, plan.index, plan.side, exit_price, why, slippage, breadth))
    return pd.DataFrame(rows)


def _breadth(days: List[StockDay]) -> Dict[Date, float]:
    """Per date, the share of stocks that opened above the previous close: the market's mood, known at 09:15."""
    up: Dict[Date, List[bool]] = {}
    for d in days:
        if d.prev_close:
            up.setdefault(d.date, []).append(bool(d.o[0] > d.prev_close))
    return {k: float(np.mean(v)) for k, v in up.items()}


def describe(name: str, trades: pd.DataFrame) -> str:
    """One report line: trades, win rate, mean gross and net return with a 95% interval, share of stop exits."""
    if trades.empty:
        return f"{name:<44} no trades"
    interval = 1.96 * trades["net"].std() / np.sqrt(len(trades))
    return (f"{name:<44}{len(trades):>7}{(trades['net'] > 0).mean():>6.0%}{trades['gross'].mean():>+9.3%}"
            f"{trades['net'].mean():>+9.3%} (±{interval:.2%}){(trades['why'] == 'STOP').mean():>6.0%}")


HEADER = f"{'':<44}{'trades':>7}{'win':>6}{'gross':>9}{'net':>9}{'':>10}{'stops':>6}"


def time_bucket(x: dtime) -> str:
    return ("09:35-10:00" if x < dtime(10, 0) else "10:00-11:00" if x < dtime(11, 0)
            else "11:00-12:30" if x < dtime(12, 30) else "12:30-14:00")
