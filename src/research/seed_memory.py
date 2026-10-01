"""Setups rebuilt from price history, to seed the decision memory, and the walk-forward check of whether that memory
predicts anything at all.

`rebuild`: on the first session of every week, the stocks the live scanner would have ranked top (`ranked_candidates`:
the same filters and 12-1 momentum ranking), the snapshot the agent would have been shown (two years of bars up to that
day, as live), and what holding each from the next open for `horizon` sessions returned (`forward_outcome`, the same
arithmetic as the live labels).

`walk_forward`: for every rebuilt setup in a test window, ask the real `SetupMemory` as of that day (only outcomes that
had finished by then) and compare what it said with what then happened. The memory is worth switching on only if the
gate below passes; it was fixed before any run:
  setups the memory rated ABOVE the market-wide baseline must return more, net of costs, than setups it rated AT OR
  BELOW it, with the 95% interval of the difference above zero, and the difference must be positive in both halves of
  the test window.

Known biases: today's NSE list only (delisted stocks are missing, which flatters every row); the top-N cut ignores the
live affordability rule (a setup's outcome does not depend on the account that could afford it)."""
import dataclasses
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from src.data.indicators import Snapshot, build_snapshot
from src.data.universe import ScreenConfig
from src.learning.outcomes import forward_outcome
from src.learning.setups import ROUND_TRIP_COST, SetupMemory
from src.research.live_backtest import ranked_candidates

SNAPSHOT_WINDOW_DAYS = 730  # the live snapshot is built from two years of daily bars
GROUPS = ("above baseline", "at or below baseline", "memory silent")


@dataclass(frozen=True)
class Observation:
    """One rebuilt setup and its forward result."""
    symbol: str
    bar_date: str
    horizon: int
    exit_date: str
    snapshot: dict
    gross_return: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), default=float)

    @staticmethod
    def from_json(line: str) -> "Observation":
        return Observation(**json.loads(line))


def weekly_dates(index: pd.DatetimeIndex, warmup: int, horizon: int) -> List[pd.Timestamp]:
    """The first session of each week that has `warmup` sessions of history before it and `horizon` sessions after."""
    usable = index[warmup:len(index) - horizon - 1]
    if len(usable) == 0:
        return []
    iso = usable.isocalendar()
    return list(pd.Series(usable, index=usable).groupby([iso.year.to_numpy(), iso.week.to_numpy()]).first())


def rebuild(bars: Dict[str, pd.DataFrame], cfg: ScreenConfig, horizon: int, stop_pct: float, warmup: int = 300,
            progress: Callable[[str], None] = lambda line: None) -> List[Observation]:
    """Every weekly scanner pick in `bars` (symbol -> daily OHLCV) with its snapshot and `horizon`-session outcome."""
    close = pd.DataFrame({s: b["Close"] for s, b in bars.items()}).sort_index()
    volume = pd.DataFrame({s: b["Volume"] for s, b in bars.items()}).reindex(close.index).fillna(0.0)
    dates = weekly_dates(pd.DatetimeIndex(close.index), warmup, horizon)
    ranked = ranked_candidates(close, volume, dates, cfg)
    out: List[Observation] = []
    for i, d in enumerate(dates):
        for c in ranked[d][:cfg.max_candidates]:
            b = bars[c.symbol]
            try:
                snap = build_snapshot(c.symbol, b.loc[d - pd.Timedelta(days=SNAPSHOT_WINDOW_DAYS):d])
            except ValueError:
                continue
            outcome = forward_outcome(c.symbol, b, snap.bar_date or "", horizon, stop_pct)
            if outcome is not None:
                out.append(Observation(c.symbol, outcome.bar_date, horizon, outcome.exit_date, dataclasses.asdict(snap),
                                       outcome.gross_return))
        if i % 50 == 0:
            progress(f"{d.date()}: {len(out)} setups so far ({i + 1}/{len(dates)} weeks)")
    return out


def store(sessions, observations: Sequence[Observation]) -> int:
    """Write rebuilt setups to the `setup_observations` table (re-running replaces the same rows). Returns the count."""
    from src.database import SetupObservationRecord

    with sessions() as s:
        for o in observations:
            s.merge(SetupObservationRecord(symbol=o.symbol, bar_date=o.bar_date, horizon=o.horizon, exit_date=o.exit_date,
                                           snapshot_json=json.dumps(o.snapshot, default=float), gross_return=o.gross_return,
                                           source="history"))
        s.commit()
    return len(observations)


def walk_forward(sessions, observations: Sequence[Observation], horizon: int, min_samples: int, test_from: str,
                 cost: float = ROUND_TRIP_COST) -> Dict[str, List[Tuple[str, float]]]:
    """Group every setup from `test_from` on by what the memory said about it as of its own day (only outcomes finished
    by then), keeping (bar_date, net return). `sessions` must already hold `observations` (see `store`)."""
    groups: Dict[str, List[Tuple[str, float]]] = {g: [] for g in GROUPS}
    by_day: Dict[str, List[Observation]] = {}
    for o in observations:
        if o.bar_date >= test_from:
            by_day.setdefault(o.bar_date, []).append(o)
    for day in sorted(by_day):
        memory = SetupMemory(sessions, horizon=horizon, min_samples=min_samples, use_history=True, ttl_seconds=math.inf)
        for o in by_day[day]:
            stats = memory.stats(Snapshot(**o.snapshot), as_of=date.fromisoformat(day))
            if stats is None:
                key = "memory silent"
            elif stats.baseline_win_rate is not None and stats.win_rate > stats.baseline_win_rate:
                key = "above baseline"
            else:
                key = "at or below baseline"
            groups[key].append((day, o.gross_return - cost))
    return groups


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and the half-width of its 95% interval (0 below two values)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, 1.96 * sd / math.sqrt(n)


def gate(groups: Dict[str, List[Tuple[str, float]]]) -> Tuple[bool, List[str]]:
    """Apply the pre-declared gate (module docstring). Returns (passed, report lines)."""
    lines = []
    for g in GROUPS:
        values = [v for _, v in groups[g]]
        if values:
            mean, ci = mean_ci(values)
            win = sum(v > 0 for v in values) / len(values)
            lines.append(f"{g:<22} n={len(values):<6} mean net {mean:+.2%} (±{ci:.2%})  win {win:.0%}")
        else:
            lines.append(f"{g:<22} n=0")
    above, below = groups["above baseline"], groups["at or below baseline"]
    if len(above) < 2 or len(below) < 2:
        return False, lines + ["GATE: not enough rated setups to judge (memory mostly silent)"]

    def diff(a, b):
        (ma, ca), (mb, cb) = mean_ci([v for _, v in a]), mean_ci([v for _, v in b])
        return ma - mb, math.sqrt(ca ** 2 + cb ** 2)

    d, ci = diff(above, below)
    days = sorted({day for day, _ in above + below})
    middle = days[len(days) // 2]
    halves = []
    for first in (True, False):
        a = [x for x in above if (x[0] < middle) == first]
        b = [x for x in below if (x[0] < middle) == first]
        halves.append(diff(a, b)[0] if len(a) >= 2 and len(b) >= 2 else float("nan"))
    passed = d - ci > 0 and all(h > 0 for h in halves)
    lines.append(f"above minus below: {d:+.2%} (95% interval {d - ci:+.2%} to {d + ci:+.2%}); "
                 f"first half {halves[0]:+.2%}, second half {halves[1]:+.2%} (split at {middle})")
    lines.append("GATE: PASSED - the memory's ratings separated winners from losers out of sample" if passed else
                 "GATE: FAILED - the memory's ratings did not reliably separate winners from losers; keep LEARNING_SEED off")
    return passed, lines
