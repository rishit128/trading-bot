"""Mistake analysis: with real outcomes attached to every decision, where was the bot right and where was it wrong?

Read-only. It reports facts with their uncertainty and never changes a setting: a group smaller than `min_samples` is
listed as insufficient rather than judged, and every mean carries a 95% interval so noise is not read as a lesson."""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy import func, select

from src.database import DecisionRecord, OutcomeRecord
from src.learning.setups import ROUND_TRIP_COST


@dataclass(frozen=True)
class Judged:
    """One analysed stock-session: what the bot decided and what the market then did (net of estimated costs)."""
    symbol: str
    bar_date: str
    action: str
    confidence: float
    source: Optional[str]
    reasoning: str
    net_return: float
    hit_stop: bool
    model: Optional[str] = None
    rule_alignment: Optional[str] = None

    @property
    def vetoed(self) -> bool:
        """The reflection critic turned this call into a HOLD/SELL (the veto is recorded only in the reasoning text)."""
        return "[reflection vetoed" in self.reasoning


@dataclass(frozen=True)
class GroupStats:
    label: str
    n: int
    mean_net: Optional[float]   # None when insufficient
    ci95: Optional[float]       # half-width of the 95% interval of the mean
    win_rate: Optional[float]
    stop_rate: Optional[float]
    sufficient: bool


def load_judged(sessions, horizon: int = 20, cost: float = ROUND_TRIP_COST) -> List[Judged]:
    """The latest decision per stock-session joined to its outcome at `horizon`."""
    latest = (select(func.max(DecisionRecord.id)).where(DecisionRecord.bar_date.isnot(None))
              .group_by(DecisionRecord.symbol, DecisionRecord.bar_date))
    query = (select(DecisionRecord.symbol, DecisionRecord.bar_date, DecisionRecord.final_action, DecisionRecord.final_confidence,
                    DecisionRecord.decision_source, DecisionRecord.reasoning, OutcomeRecord.gross_return, OutcomeRecord.hit_stop,
                    DecisionRecord.raw_model, DecisionRecord.rule_alignment)
             .join(OutcomeRecord, (OutcomeRecord.symbol == DecisionRecord.symbol) & (OutcomeRecord.bar_date == DecisionRecord.bar_date))
             .where(DecisionRecord.id.in_(latest), OutcomeRecord.horizon == horizon)
             .order_by(DecisionRecord.bar_date, DecisionRecord.symbol))
    with sessions() as s:
        return [Judged(sym, bd, act, conf, src, why, gross - cost, bool(stop), model, align)
                for sym, bd, act, conf, src, why, gross, stop, model, align in s.execute(query)]


def group_stats(label: str, judged: Sequence[Judged], min_samples: int) -> GroupStats:
    n = len(judged)
    if n < max(min_samples, 2):  # an interval needs at least two observations
        return GroupStats(label, n, None, None, None, None, False)
    returns = [j.net_return for j in judged]
    mean = sum(returns) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
    return GroupStats(label, n, mean, 1.96 * sd / math.sqrt(n), sum(r > 0 for r in returns) / n,
                      sum(j.hit_stop for j in judged) / n, True)


def breakdown(judged: Sequence[Judged], key: Callable[[Judged], str], min_samples: int) -> List[GroupStats]:
    groups: Dict[str, List[Judged]] = {}
    for j in judged:
        groups.setdefault(key(j), []).append(j)
    return [group_stats(label, groups[label], min_samples) for label in sorted(groups)]


def confidence_bucket(j: Judged) -> str:
    return "<0.70" if j.confidence < 0.70 else "0.70-0.80" if j.confidence < 0.80 else ">=0.80"


def worst_calls(judged: Sequence[Judged], action: str, count: int = 10) -> List[Judged]:
    """The calls that hurt most: the lowest-return BUYs (mistakes of commission)."""
    return sorted((j for j in judged if j.action == action), key=lambda j: j.net_return)[:count]


def missed_gains(judged: Sequence[Judged], count: int = 10) -> List[Judged]:
    """The biggest gains among calls that did not buy (mistakes of omission)."""
    return sorted((j for j in judged if j.action != "BUY"), key=lambda j: -j.net_return)[:count]


def fmt(g: GroupStats) -> str:
    if not g.sufficient:
        return f"{g.label:<22} n={g.n:<5} insufficient data"
    return (f"{g.label:<22} n={g.n:<5} mean {g.mean_net:+.2%} (±{g.ci95:.2%})  win {g.win_rate:.0%}  stop hit {g.stop_rate:.0%}")
