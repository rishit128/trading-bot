"""Setup memory: how did setups LIKE this one actually turn out, across every stock the bot has analysed?

This feeds the agent's decision-memory step. The rules that keep it honest (a memory that adapts to noise makes the bot
worse, not better):
  * evidence only from labelled outcomes (src/learning/outcomes.py): what really happened after each analysed session,
    entered at the next open, net of an estimated round-trip cost;
  * a minimum number of independent observations, or it stays silent and the agent decides as if it had no memory;
  * overlapping observations are thinned to one per stock per calendar week (a stock analysed every day for a month is one
    story, not twenty);
  * "similar" is as specific as the data allows: it starts fine (trend shape, RSI band, ADX band, volume band) and backs off
    to coarser labels until a label has enough observations;
  * it always reports the win rate of ALL comparable setups beside the matched one, so the agent can tell an edge from the
    market simply drifting up;
  * point-in-time: asked "as of" a date, it only uses outcomes that had finished by then."""
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

from sqlalchemy import func, select

from src.agents.history import PatternStats, pattern_id
from src.data.indicators import Snapshot
from src.database import DecisionRecord, OutcomeRecord

log = logging.getLogger(__name__)

ROUND_TRIP_COST = 0.0035  # estimated fees + slippage for a round trip on a small delivery account (see STRATEGY.md)
LEVELS = ("full", "no_volume", "trend_rsi")  # from the most specific label to the coarsest


def adx_band(adx: Optional[float]) -> str:
    """Trend strength in three bands (the ADX the agent is shown), or "na" when it was not available."""
    if adx is None:
        return "na"
    return "weak" if adx < 20 else "mid" if adx < 30 else "strong"


def volume_band(volume: float, avg_volume: Optional[float]) -> str:
    """Today's volume against its 20-day average, in three bands, or "na" without a baseline."""
    if not avg_volume or avg_volume <= 0:
        return "na"
    ratio = volume / avg_volume
    return "low" if ratio < 0.8 else "normal" if ratio <= 1.25 else "high"


def setup_labels(s: Snapshot) -> Dict[str, str]:
    """The labels for one setup, finest first: trend shape + RSI decade, then ADX band, then volume band."""
    base = pattern_id(s)
    with_adx = f"{base}|ADX:{adx_band(s.adx)}"
    return {"full": f"{with_adx}|VOL:{volume_band(s.volume, s.avg_volume)}", "no_volume": with_adx, "trend_rsi": base}


@dataclass(frozen=True)
class _Row:
    symbol: str
    bar_date: str
    labels: Dict[str, str]
    net_return: float


def summarise(returns: List[float], horizon: int, min_samples: int, scope_label: str,
              baseline_win_rate: Optional[float]) -> PatternStats:
    """PatternStats (percent units, as the agent's prompt expects) from a list of net forward returns."""
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    avg_win = sum(wins) / len(wins) * 100 if wins else 0.0
    avg_loss = sum(losses) / len(losses) * 100 if losses else 0.0
    return PatternStats(
        sample_size=len(returns), win_rate=len(wins) / len(returns), avg_win_pct=avg_win, avg_loss_pct=avg_loss,
        profit_factor=(avg_win / abs(avg_loss)) if wins and losses and avg_loss < 0 else None,
        best_holding_days=horizon, worst_holding_days=horizon,
        confidence_in_pattern=min(1.0, len(returns) / (2 * min_samples)),
        scope="market", matched_on=scope_label, baseline_win_rate=baseline_win_rate)


class SetupMemory:
    """Answers "how did setups like this one turn out?" from the database. `stats(snapshot)` is the agent's history hook."""

    def __init__(self, sessions, horizon: int = 20, min_samples: int = 30, round_trip_cost: float = ROUND_TRIP_COST,
                 ttl_seconds: float = 600.0, clock: Callable[[], float] = time.monotonic):
        if min_samples < 1 or horizon < 1:
            raise ValueError("horizon and min_samples must be >= 1")
        self.sessions, self.horizon, self.min_samples = sessions, horizon, min_samples
        self.cost, self.ttl, self.clock = round_trip_cost, ttl_seconds, clock
        self._cache: Dict[Optional[str], Tuple[float, List[_Row]]] = {}

    def rows(self, as_of: Optional[date] = None) -> List[_Row]:
        """Every usable labelled observation (thinned), cached for `ttl_seconds` so one cycle reads the database once."""
        key = as_of.isoformat() if as_of else None
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < self.ttl:
            return hit[1]
        rows = self._load(as_of)
        self._cache[key] = (self.clock(), rows)
        return rows

    def _load(self, as_of: Optional[date]) -> List[_Row]:
        latest = (select(func.max(DecisionRecord.id)).where(DecisionRecord.bar_date.isnot(None), DecisionRecord.snapshot_json.isnot(None))
                  .group_by(DecisionRecord.symbol, DecisionRecord.bar_date))  # one decision per stock-session
        query = (select(DecisionRecord.symbol, DecisionRecord.bar_date, DecisionRecord.snapshot_json, OutcomeRecord.gross_return)
                 .join(OutcomeRecord, (OutcomeRecord.symbol == DecisionRecord.symbol) & (OutcomeRecord.bar_date == DecisionRecord.bar_date))
                 .where(DecisionRecord.id.in_(latest), OutcomeRecord.horizon == self.horizon)
                 .order_by(DecisionRecord.bar_date, DecisionRecord.symbol))
        if as_of is not None:  # point-in-time: only outcomes that had finished by then
            query = query.where(OutcomeRecord.exit_date != "", OutcomeRecord.exit_date <= as_of.isoformat())
        rows: List[_Row] = []
        seen = set()
        with self.sessions() as s:
            for symbol, bar_date, snapshot_json, gross in s.execute(query):
                week = date.fromisoformat(bar_date).isocalendar()[:2]
                if (symbol, week) in seen:
                    continue  # a stock analysed again the same week is the same story
                try:
                    labels = setup_labels(Snapshot(**json.loads(snapshot_json)))
                except (ValueError, TypeError):
                    continue  # a stored snapshot from an older schema that no longer loads: skip it, never guess
                seen.add((symbol, week))
                rows.append(_Row(symbol, bar_date, labels, gross - self.cost))
        return rows

    def stats(self, snapshot: Snapshot, as_of: Optional[date] = None) -> Optional[PatternStats]:
        """Stats for setups like `snapshot`, or None when even the coarsest label has fewer than `min_samples` observations."""
        rows = self.rows(as_of)
        if len(rows) < self.min_samples:
            return None
        target = setup_labels(snapshot)
        baseline = sum(1 for r in rows if r.net_return > 0) / len(rows)
        for level in LEVELS:
            matched = [r.net_return for r in rows if r.labels[level] == target[level]]
            if len(matched) >= self.min_samples:
                return summarise(matched, self.horizon, self.min_samples, f"{target[level]} (level: {level})", baseline)
        return None
