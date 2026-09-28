"""Phase 2: do any of a few intraday ideas beat their costs? PRE-DECLARED before any result was seen.

Five hypotheses are examined (plus the live rule as a control). Each is a fixed rule with no fitted parameter: thresholds
are round numbers chosen in advance, never tuned on the data. For each, on every stock-session of the archive:

  * SIGNAL: the trades the rule takes, simulated with the live costs (fees + 0.05% slippage a side) and full-bar exits;
  * NULL:   the same days, sides and exits but a random entry time (rules that enter on a breakout) or a random side (rules
            that enter at a fixed time), repeated over several seeds. This is what "no information in the signal" looks like.

A hypothesis is a CANDIDATE only if ALL of these hold (five comparisons, so |t| >= 2.6 rather than 2):
  1. at least 300 trades;
  2. mean net return per trade > 0 with a per-DATE clustered t >= 2.6 (stocks on one day move together, so the day, not the
     trade, is the independent observation);
  3. the mean net is positive in BOTH the first and the second half of the dates;
  4. it is still positive with slippage doubled;
  5. it beats its NULL: the paired per-date difference (signal minus null) has t >= 2.6.

KILL CRITERION: if no hypothesis is a candidate, the intraday engine has no rule worth running and should be retired: more
indicators or a wider universe were tested and do not change that. A candidate is a lead for forward paper trading, never proof:
one archive is one market regime, and its length limits what any test can detect (reported as the minimum detectable edge)."""
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.intraday import strategy as st
from src.research.intraday_audit import Plan, StockDay, evaluate_plans, first_breakout, orb_plan, random_entry, trade_returns

T_BAR = 2.6
MIN_TRADES = 300
NULL_SEEDS = 5
GAP_GO, GAP_FADE, MOMENTUM, LAST_HOUR = 0.01, 0.015, 0.01, 0.01   # fixed round-number thresholds, chosen in advance
MOMENTUM_STOP = 0.015

# Time windows, as minutes since midnight [start, end): the live entry window, and the first hour after the range.
ORB_WINDOW: Tuple[int, int] = (st.minute_of_day(st.ENTRY_FROM), st.minute_of_day(st.ENTRY_UNTIL))
FIRST_HOUR: Tuple[int, int] = (st.minute_of_day(st.ENTRY_FROM), st.minute_of_day(dtime(10, 30)))

Picker = Callable[[StockDay], Optional[Plan]]
NullPicker = Callable[[StockDay, np.random.Generator], Optional[Plan]]


@dataclass(frozen=True)
class Hypothesis:
    """A rule to test: how it picks its trade on a session, and the null that picks a comparable trade without its signal."""
    name: str
    idea: str
    pick: Picker
    null: NullPicker


# -- helpers ----------------------------------------------------------------------------------------------------------------
def _range(d: StockDay) -> Tuple[float, float]:
    """The opening range: (high, low) of the first bars."""
    return float(d.h[:st.RANGE_BARS].max()), float(d.l[:st.RANGE_BARS].min())


def _range_ok(d: StockDay) -> bool:
    high, low = _range(d)
    return st.MIN_RANGE <= (high - low) / high <= st.MAX_RANGE


def _window(d: StockDay, window: Tuple[int, int]) -> np.ndarray:
    """Bars starting inside `window` (minutes since midnight). Every window here opens at 09:30, after the opening range."""
    return (d.minutes >= window[0]) & (d.minutes < window[1])


def _first(mask: np.ndarray) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _gap(d: StockDay) -> Optional[float]:
    return d.o[0] / d.prev_close - 1 if d.prev_close else None


def _bar_at(d: StockDay, hour: int, minute: int) -> Optional[int]:
    return _first(d.minutes == hour * 60 + minute)


def _random_bar(mask: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(rng.choice(hits)) if len(hits) else None


# -- control: the live rule ---------------------------------------------------------------------------------------------------
def orb_long(d: StockDay) -> Optional[Plan]:
    e = first_breakout(d)
    return None if e is None else orb_plan(d, e)


def orb_long_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    entry = random_entry(d, rng)
    return None if entry is None else orb_plan(d, entry)


# -- H1: the opening-range breakdown, short ---------------------------------------------------------------------------------
def orb_short(d: StockDay) -> Optional[Plan]:
    if not _range_ok(d):
        return None
    high, low = _range(d)
    vwap, ratio = st.session_vwap_and_volume_ratio(d.h, d.l, d.c, d.v)
    i = _first((d.c < low) & (d.c < vwap) & (ratio >= st.VOLUME_MULT) & _window(d, ORB_WINDOW))
    if i is None:
        return None
    entry = float(d.c[i])
    return Plan(i, -1, high, entry - st.TARGET_R * (high - entry))


def orb_short_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    if not _range_ok(d):
        return None
    high, _ = _range(d)
    i = _random_bar(_window(d, ORB_WINDOW) & (d.c < high), rng)
    return None if i is None else Plan(i, -1, high, float(d.c[i]) - st.TARGET_R * (high - float(d.c[i])))


# -- H2: gap up, then the opening range breaks upward: hold the day ---------------------------------------------------------
def gap_and_go(d: StockDay) -> Optional[Plan]:
    gap = _gap(d)
    if gap is None or gap < GAP_GO:
        return None
    high, low = _range(d)
    i = _first((d.c > high) & _window(d, FIRST_HOUR))      # (the first close above the range high is necessarily above VWAP)
    return None if i is None else Plan(i, 1, low)


def gap_and_go_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    gap = _gap(d)
    if gap is None or gap < GAP_GO:
        return None
    _, low = _range(d)
    i = _random_bar(_window(d, FIRST_HOUR) & (d.c > low), rng)
    return None if i is None else Plan(i, 1, low)


# -- H3: a big gap that starts to close: fade it ----------------------------------------------------------------------------
def _fade_side(d: StockDay) -> int:
    gap = _gap(d)
    return 0 if gap is None else -1 if gap >= GAP_FADE else 1 if gap <= -GAP_FADE else 0


def gap_fade(d: StockDay) -> Optional[Plan]:
    side = _fade_side(d)
    if side == 0:
        return None
    high, low = _range(d)
    started = d.c < d.o[0] if side < 0 else d.c > d.o[0]          # price has moved back toward the previous close
    i = _first(started & _window(d, FIRST_HOUR))
    return None if i is None else Plan(i, side, high if side < 0 else low)


def gap_fade_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    side = _fade_side(d)
    if side == 0:
        return None
    high, low = _range(d)
    valid = d.c < high if side < 0 else d.c > low
    i = _random_bar(valid & _window(d, FIRST_HOUR), rng)
    return None if i is None else Plan(i, side, high if side < 0 else low)


# -- H4: the first hour's direction continues to the close ------------------------------------------------------------------
def _fixed_time_plan(d: StockDay, hour: int, minute: int, threshold: float, need_vwap: bool, stop_pct: Optional[float],
                     flip: Optional[np.random.Generator] = None) -> Optional[Plan]:
    """Enter at the close of the bar starting hour:minute, in the direction the stock has moved since the open, if it moved at
    least `threshold` (and, if asked, is on the same side of VWAP). `flip` replaces the direction with a coin toss (the null)."""
    i = _bar_at(d, hour, minute)
    if i is None:
        return None
    move = d.c[i] / d.o[0] - 1
    side = 1 if move >= threshold else -1 if move <= -threshold else 0
    if side == 0:
        return None
    if need_vwap:
        vwap, _ = st.session_vwap_and_volume_ratio(d.h, d.l, d.c, d.v)
        if side * (d.c[i] - vwap[i]) <= 0:
            return None
    if flip is not None:
        side = 1 if flip.random() < 0.5 else -1
    return Plan(i, side, None if stop_pct is None else float(d.c[i]) * (1 - side * stop_pct))


def first_hour_momentum(d: StockDay) -> Optional[Plan]:
    return _fixed_time_plan(d, 10, 10, MOMENTUM, True, MOMENTUM_STOP)


def first_hour_momentum_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    return _fixed_time_plan(d, 10, 10, MOMENTUM, True, MOMENTUM_STOP, flip=rng)


# -- H5: the day's direction continues through the last half hour -----------------------------------------------------------
def last_half_hour(d: StockDay) -> Optional[Plan]:
    return _fixed_time_plan(d, 14, 40, LAST_HOUR, False, None)


def last_half_hour_null(d: StockDay, rng: np.random.Generator) -> Optional[Plan]:
    return _fixed_time_plan(d, 14, 40, LAST_HOUR, False, None, flip=rng)


CONTROL = Hypothesis("C  ORB long (the live rule)", "long a breakout above the opening range, stop at its low, target 2R",
                     orb_long, orb_long_null)
HYPOTHESES = [
    Hypothesis("H1 ORB short", "short a breakdown below the opening range, stop at its high, target 2R",
               orb_short, orb_short_null),
    Hypothesis("H2 gap-and-go", "gap up >= 1% and the range breaks up: long, stop at the range low, hold to 15:15",
               gap_and_go, gap_and_go_null),
    Hypothesis("H3 gap fade", "gap >= 1.5% that starts to close: trade back toward the previous close, stop at the range extreme",
               gap_fade, gap_fade_null),
    Hypothesis("H4 first-hour momentum", "up/down >= 1% and beyond VWAP at 10:15: hold that direction to 15:15, 1.5% stop",
               first_hour_momentum, first_hour_momentum_null),
    Hypothesis("H5 last-half-hour momentum", "up/down >= 1% at 14:45: hold that direction to 15:15",
               last_half_hour, last_half_hour_null),
]


# -- statistics -------------------------------------------------------------------------------------------------------------
def cluster_t(per_date: pd.Series) -> float:
    """t-statistic of the mean of per-date averages (the day is the independent observation)."""
    if len(per_date) < 2 or per_date.std() == 0:
        return 0.0
    return float(per_date.mean() / (per_date.std() / np.sqrt(len(per_date))))


@dataclass
class Outcome:
    name: str
    idea: str
    trades: int = 0
    dates: int = 0
    mean_gross: float = 0.0
    mean_net: float = 0.0
    t_net: float = 0.0
    first_half: float = 0.0
    second_half: float = 0.0
    net_double_slippage: float = 0.0
    null_net: float = 0.0
    paired_t: float = 0.0
    detectable: float = 0.0          # the smallest mean net per trade this many days could show at t = T_BAR
    failed: List[str] = field(default_factory=list)

    @property
    def candidate(self) -> bool:
        return self.trades > 0 and not self.failed


def judge(name: str, idea: str, signal: pd.DataFrame, null: pd.DataFrame, slippage: float) -> Outcome:
    """Apply the pre-declared criteria to a hypothesis' signal trades against its null trades."""
    out = Outcome(name, idea)
    if signal.empty:
        out.failed.append("no trades")
        return out
    per_date = signal.groupby("date")["net"].mean()
    out.trades, out.dates = len(signal), len(per_date)
    out.mean_gross, out.mean_net, out.t_net = float(signal["gross"].mean()), float(signal["net"].mean()), cluster_t(per_date)
    ordered = sorted(per_date.index)
    early = signal[signal["date"].isin(ordered[:len(ordered) // 2])]
    late = signal[signal["date"].isin(ordered[len(ordered) // 2:])]
    out.first_half = float(early["net"].mean()) if len(early) else 0.0
    out.second_half = float(late["net"].mean()) if len(late) else 0.0
    doubled = [trade_returns(float(p), float(x), int(side), 2 * slippage)[1]
               for p, x, side in zip(signal["price"], signal["exit"], signal["side"])]
    out.net_double_slippage = float(np.mean(doubled))
    out.detectable = T_BAR * float(per_date.std() / np.sqrt(len(per_date))) if len(per_date) > 1 else 0.0
    if len(null):
        null_per_date = null.groupby("date")["net"].mean()
        out.null_net = float(null["net"].mean())
        shared = per_date.index.intersection(null_per_date.index)
        out.paired_t = cluster_t(per_date[shared] - null_per_date[shared])
    checks = [(out.trades >= MIN_TRADES, f"only {out.trades} trades (need {MIN_TRADES})"),
              (out.mean_net > 0 and out.t_net >= T_BAR,
               f"net {out.mean_net:+.3%}, t {out.t_net:.1f} (need > 0 and t >= {T_BAR})"),
              (out.first_half > 0 and out.second_half > 0,
               f"halves {out.first_half:+.3%} / {out.second_half:+.3%} (both must be > 0)"),
              (out.net_double_slippage > 0, f"net {out.net_double_slippage:+.3%} with doubled slippage"),
              (out.paired_t >= T_BAR, f"beats the null by t {out.paired_t:.1f} (need >= {T_BAR})")]
    out.failed = [why for ok, why in checks if not ok]
    return out


def _null_trades(h: Hypothesis, days: List[StockDay], seed: int, slippage: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return evaluate_plans(days, lambda d: h.null(d, rng), slippage=slippage)


def run(days: List[StockDay], hypotheses: List[Hypothesis], slippage: float = 0.0005, seeds: int = NULL_SEEDS) -> List[Outcome]:
    """Evaluate each hypothesis against its null over all the stock-sessions."""
    outcomes = []
    for h in hypotheses:
        nulls = [_null_trades(h, days, seed, slippage) for seed in range(seeds)]
        outcomes.append(judge(h.name, h.idea, evaluate_plans(days, h.pick, slippage=slippage),
                              pd.concat(nulls) if nulls else pd.DataFrame(), slippage))
    return outcomes


def report(outcomes: List[Outcome]) -> str:
    header = (f"{'hypothesis':<30}{'trades':>7}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}"
              f"{'2x slip':>9}{'null':>9}{'vs null t':>10}")
    lines = [header, "-" * len(header)]
    for o in outcomes:
        lines.append(f"{o.name:<30}{o.trades:>7}{o.mean_gross:>+9.3%}{o.mean_net:>+9.3%}{o.t_net:>6.1f}{o.first_half:>+10.3%}"
                     f"{o.second_half:>+10.3%}{o.net_double_slippage:>+9.3%}{o.null_net:>+9.3%}{o.paired_t:>10.1f}"
                     f"{'  <-- CANDIDATE' if o.candidate else ''}")
    for o in outcomes:
        if o.failed:
            lines.append(f"  {o.name}: fails - " + "; ".join(o.failed))
    dates = max((o.dates for o in outcomes), default=0)
    if dates:
        lines.append(f"\nWith {dates} trading dates the smallest edge that could be detected (t = {T_BAR}) is about "
                     f"{max(o.detectable for o in outcomes):.3%} net per trade: a smaller true edge would not show.")
    passed = [o for o in outcomes if o.candidate]
    lines.append("\nCANDIDATES: " + (", ".join(o.name for o in passed) if passed else
                                     "none -> the kill criterion applies: no rule tested is worth running intraday"))
    return "\n".join(lines)
