"""Mistake analysis: with real outcomes attached to every decision, where was the bot right and where was it wrong?

Read-only. It reports facts with their uncertainty and never changes a setting: a group smaller than `min_samples` is
listed as insufficient rather than judged, and every mean carries a 95% interval so noise is not read as a lesson.

The intervals respect how the outcomes overlap. A stock analysed on consecutive sessions has 20-session outcomes that share
19 of their 20 days, so treating every row as independent made the intervals far too narrow. Each group therefore keeps
at most one observation per stock per `horizon`-session block, averages those per date, and computes the interval over
the dates (whose count is reported next to it)."""

import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy import func, select

from src.database import DecisionRecord, OutcomeRecord
from src.engine.costs import ROUND_TRIP_COST
from src.engine.rules import rule_verdict


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
    rule_action: Optional[str] = None  # the mechanical rule's own verdict (BUY/HOLD); None when it cannot be known

    @property
    def vetoed(self) -> bool:
        """The reflection critic turned this call into a HOLD/SELL (the veto is recorded only in the reasoning text)."""
        return "[reflection vetoed" in self.reasoning


@dataclass(frozen=True)
class GroupStats:
    label: str
    n: int  # stock-sessions in the group
    mean_net: Optional[float]  # None when insufficient
    ci95: Optional[float]  # half-width of the 95% interval of the mean
    win_rate: Optional[float]
    stop_rate: Optional[float]
    sufficient: bool
    used: int = 0  # observations left after keeping one per stock per horizon block
    dates: int = 0  # dates those observations fall on: the interval's sample size


def rule_action_from_snapshot(snapshot_json: Optional[str]) -> Optional[str]:
    """The rule's verdict recomputed from a stored snapshot (rows recorded before `rule_action` existed). The rule is
    deterministic, so this is what the bot's rule said that day; None when the snapshot is missing or incomplete."""
    try:
        s = json.loads(snapshot_json or "")
        return str(rule_verdict(float(s["price"]), float(s["ma50"]), float(s["ma200"]), float(s["rsi"])))
    except (ValueError, KeyError, TypeError):
        return None


def load_judged(sessions, horizon: int = 20, cost: float = ROUND_TRIP_COST) -> List[Judged]:
    """The latest decision per stock-session joined to its outcome at `horizon`."""
    latest = (
        select(func.max(DecisionRecord.id))
        .where(DecisionRecord.bar_date.isnot(None))
        .group_by(DecisionRecord.symbol, DecisionRecord.bar_date)
    )
    query = (
        select(
            DecisionRecord.symbol,
            DecisionRecord.bar_date,
            DecisionRecord.final_action,
            DecisionRecord.final_confidence,
            DecisionRecord.decision_source,
            DecisionRecord.reasoning,
            OutcomeRecord.gross_return,
            OutcomeRecord.hit_stop,
            DecisionRecord.raw_model,
            DecisionRecord.rule_alignment,
            DecisionRecord.rule_action,
            DecisionRecord.snapshot_json,
        )
        .join(
            OutcomeRecord, (OutcomeRecord.symbol == DecisionRecord.symbol) & (OutcomeRecord.bar_date == DecisionRecord.bar_date)
        )
        .where(DecisionRecord.id.in_(latest), OutcomeRecord.horizon == horizon)
        .order_by(DecisionRecord.bar_date, DecisionRecord.symbol)
    )
    with sessions() as s:
        return [
            Judged(
                sym,
                bd,
                act,
                conf,
                src,
                why,
                gross - cost,
                bool(stop),
                model,
                align,
                rule or rule_action_from_snapshot(snapshot),
            )  # fmt: skip
            for sym, bd, act, conf, src, why, gross, stop, model, align, rule, snapshot in s.execute(query)
        ]


def independent_sample(judged: Sequence[Judged], horizon: int) -> List[Judged]:
    """At most one observation per stock per `horizon`-session block (a stock's next one is kept only once `horizon`
    sessions have passed), so overlapping outcomes are not counted as separate evidence. Sessions are counted on the
    calendar of dates present in `judged`."""
    calendar = {d: i for i, d in enumerate(sorted({j.bar_date for j in judged}))}
    kept: List[Judged] = []
    last: Dict[str, int] = {}
    for j in sorted(judged, key=lambda j: (j.bar_date, j.symbol)):
        i = calendar[j.bar_date]
        if j.symbol not in last or i - last[j.symbol] >= horizon:
            kept.append(j)
            last[j.symbol] = i
    return kept


def group_stats(label: str, judged: Sequence[Judged], min_samples: int, horizon: int = 20) -> GroupStats:
    """Mean net return with a 95% interval over dates (see the module docstring), win and stop rates."""
    kept = independent_sample(judged, horizon)
    by_date: Dict[str, List[float]] = {}
    for j in kept:
        by_date.setdefault(j.bar_date, []).append(j.net_return)
    if len(kept) < max(min_samples, 2) or len(by_date) < 2:  # an interval needs at least two dates
        return GroupStats(label, len(judged), None, None, None, None, False, len(kept), len(by_date))
    means = [sum(v) / len(v) for v in by_date.values()]
    mean = sum(means) / len(means)
    sd = math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1))
    return GroupStats(
        label,
        len(judged),
        mean,
        1.96 * sd / math.sqrt(len(means)),
        sum(j.net_return > 0 for j in kept) / len(kept),
        sum(j.hit_stop for j in kept) / len(kept),
        True,
        len(kept),
        len(by_date),
    )


def breakdown(judged: Sequence[Judged], key: Callable[[Judged], str], min_samples: int, horizon: int = 20) -> List[GroupStats]:
    groups: Dict[str, List[Judged]] = {}
    for j in judged:
        groups.setdefault(key(j), []).append(j)
    return [group_stats(label, groups[label], min_samples, horizon) for label in sorted(groups)]


def confidence_bucket(j: Judged) -> str:
    return "<0.70" if j.confidence < 0.70 else "0.70-0.80" if j.confidence < 0.80 else ">=0.80"


def ai_vs_rule(j: Judged) -> str:
    """Where the bot's final call sits against the mechanical entry rule's own verdict. The AI adds value only if, among
    stocks the rule would BUY, the ones it bought beat the ones it passed on. A BUY the critic vetoed is a pass on a rule
    BUY (`rule_buy_passes` separates it from the analyst's own passes). Rule-based exits and rows whose rule verdict is
    unknown are 'n/a'."""
    if j.source not in (None, "agents") or j.rule_action not in ("BUY", "HOLD"):
        return "n/a"
    if j.rule_action == "BUY":
        return "rule BUY, AI bought" if j.action == "BUY" else "rule BUY, AI passed"
    return "rule passed, AI bought" if j.action == "BUY" else "both passed"


def rule_buy_passes(j: Judged) -> str:
    """Within 'rule BUY, AI passed': who passed. The analyst itself, or the critic overruling the analyst's BUY."""
    return "critic vetoed the BUY" if j.vetoed else "analyst passed"


def worst_calls(judged: Sequence[Judged], action: str, count: int = 10) -> List[Judged]:
    """The calls that hurt most: the lowest-return BUYs (mistakes of commission)."""
    return sorted((j for j in judged if j.action == action), key=lambda j: j.net_return)[:count]


def missed_gains(judged: Sequence[Judged], count: int = 10) -> List[Judged]:
    """The biggest gains among calls that did not buy (mistakes of omission)."""
    return sorted((j for j in judged if j.action != "BUY"), key=lambda j: -j.net_return)[:count]


def fmt(g: GroupStats) -> str:
    if not g.sufficient:
        return f"{g.label:<22} n={g.n:<5} insufficient data ({g.used} independent observations on {g.dates} dates)"
    return (
        f"{g.label:<22} n={g.n:<5} mean {g.mean_net:+.2%} (±{g.ci95:.2%}, {g.used} independent observations on "
        f"{g.dates} dates)  win {g.win_rate:.0%}  stop hit {g.stop_rate:.0%}"
    )
