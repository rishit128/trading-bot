"""Decision memory: what actually happened on setups similar to the current one.

The bot's edge hypothesis is that certain setups repeat. The chain-of-thought step records the reasoning chain for every decision so this step can look back: for the symbol being analysed, find closed paper trades whose opening BUY decision had the
same trend shape, a price within 5% and an RSI(14) within 5 points, and summarise their win rate, average win/loss,
profit factor and holding period. The agent then adjusts its confidence (or flips the call) from that record.

API (used by the agent and by the metrics scripts):
  * `pattern_id(snapshot)` - a stable label for a setup, keyed to the trend shape + RSI bucket.
  * `find_similar_patterns(...)` - the prior closed trades (and their opening decisions) that match a pattern; [] when
    there is nothing to learn from yet.
  * `analyze_pattern(trades)` - turns those matches into a PatternStats summary.

Honesty notes:
  * Only closed paper trades count as outcomes; a position still open says nothing yet.
  * A trading day without any closed trade yields None (the agent keeps its base signal). With 0 closed trades in the
    live account today, this phase is inert until paper history accumulates - which is the point of measuring first.
  * The estimate is only as good as its sample; `confidence_in_pattern` scales with n/30 so a 3-trade record is
    treated as a weak prior, not a fact."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

from sqlalchemy import select

from src.data.indicators import Snapshot
from src.database import DecisionRecord, PaperTradeRecord
from src.engine.enums import Action
from src.engine.rules import above_ma50, ma_stack_bullish

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 365
RSI_TOLERANCE = 5.0  # points of RSI(14) a past setup may differ from the current one and still count as "similar"
PRICE_TOL_FRAC = 0.05  # a past setup's opening price must be within ±5% of the current price to count as "similar"
MAX_MATCHES = 25  # cap the pattern sample so one symbol's boom period can never dominate
MIN_OPEN_GAP = timedelta(days=2)  # an opening decision and its paper fill land within the same day
HUMILITY = 0.10  # reflection's deterministic humility discount (review doc 4.2)


@dataclass(frozen=True)
class PatternStats:
    """A summary of the closed trades most similar to the current technical setup."""
    sample_size: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: Optional[float]  # average winner / average loser size, None when there are no losses (or wins)
    best_holding_days: int
    worst_holding_days: int
    confidence_in_pattern: float  # 0-1, how much data the estimate is based on (n/30, capped)
    scope: str = "symbol"  # "symbol": this stock's own closed trades; "market": labelled outcomes across every analysed stock
    matched_on: Optional[str] = None  # what "similar" meant for a market-scope result (the setup label that had enough samples)
    baseline_win_rate: Optional[float] = None  # market scope: the win rate of ALL comparable setups, to tell an edge from the market's drift


@dataclass(frozen=True)
class PatternMatch:
    """One prior closed trade on a similar setup, tagged with the pattern it belonged to."""
    pattern_id: str
    symbol: str
    opened_at: datetime
    entry_price: float
    exit_price: float
    held_days: int
    return_pct: float  # exit/entry - 1, as a percentage
    quality_score: float  # how similar the opening setup was; lower is closer (rsi gap + price gap, normalised)


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def pattern_id(snap: Snapshot, rsi_bucket: int = 10) -> str:
    """A stable, human-readable pattern label: the trend shape + the RSI(14) decade, e.g. 'P>MA50+MA50>MA200|RSI60'."""
    shape = "+".join((
        "P>MA50" if above_ma50(snap.price, snap.ma50) else "P<=MA50",
        "MA50>MA200" if ma_stack_bullish(snap.ma50, snap.ma200) else "MA50<=MA200",
    ))
    rungs = int(snap.rsi // rsi_bucket) * rsi_bucket
    return f"{shape}|RSI{rungs}"


def pattern_stats(outcomes: Iterable[Tuple[float, int]]) -> Optional[PatternStats]:
    """outcomes: (return_pct, holding_days) per closed trade. None when there are none."""
    outcomes = list(outcomes)
    if not outcomes:
        return None
    returns = [r for r, _ in outcomes]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    holding = [h for _, h in outcomes]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return PatternStats(
        sample_size=len(outcomes),
        win_rate=len(wins) / len(outcomes),
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        # Average-winner vs average-loser (the review doc's definition); None when either bucket is empty.
        profit_factor=(avg_win / abs(avg_loss)) if losses and wins else None,
        best_holding_days=max(holding),
        worst_holding_days=min(holding),
        confidence_in_pattern=min(1.0, len(outcomes) / 30.0),
    )


def _trend_shape(snap: Snapshot) -> Tuple[bool, bool]:
    return above_ma50(snap.price, snap.ma50), ma_stack_bullish(snap.ma50, snap.ma200)


def _past_setups(decisions) -> List[Tuple[datetime, Snapshot, Tuple[bool, bool]]]:
    """(decision time, stored snapshot, trend shape) for BUY decisions whose exact inputs were stored."""
    setups = []
    for d in decisions:
        if not d.snapshot_json:
            continue
        try:
            past = Snapshot(**json.loads(d.snapshot_json))
            setups.append((_aware(d.created_at), past, _trend_shape(past)))
        except Exception as e:
            log.debug("unusable stored snapshot for decision %s: %s", d.id, e)
    return setups


def find_similar_patterns(sessions, symbol: str, price_range: Tuple[float, float], trend: Tuple[bool, bool],
                          rsi_range: Tuple[float, float], lookback_days: int = LOOKBACK_DAYS,
                          limit: int = MAX_MATCHES, now: Optional[datetime] = None) -> List[PatternMatch]:
    """Closed paper trades opened on setups similar to `(price_range, trend, rsi_range)` on `symbol`.

    A past trade counts when its opening BUY decision has the same trend shape, a price inside `price_range` and an
    RSI(14) inside `rsi_range`, is the BUY decision nearest in time to that trade's `opened_at`, and is no older than
    `lookback_days`. Returns [] when there are no matches - the agent keeps its base call."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    with sessions() as s:
        trades = list(s.scalars(select(PaperTradeRecord).where(
            PaperTradeRecord.symbol == symbol, PaperTradeRecord.opened_at >= cutoff)))
        # Ordered so the "closest in time" match is deterministic instead of database row order.
        decisions = list(s.scalars(select(DecisionRecord).where(
            DecisionRecord.symbol == symbol, DecisionRecord.final_action == Action.BUY,
            DecisionRecord.created_at >= cutoff - MIN_OPEN_GAP)
            .order_by(DecisionRecord.created_at.asc(), DecisionRecord.id.asc())))
    setups = _past_setups(decisions)
    if not trades or not setups:
        return []
    pmin, pmax = price_range
    rmin, rmax = rsi_range
    p_mid, r_mid = (pmin + pmax) / 2.0, (rmin + rmax) / 2.0
    matches: List[Tuple[float, float, PatternMatch]] = []
    for t in trades:
        opened = _aware(t.opened_at)
        if t.closed_at is None or _aware(t.closed_at) > now:
            continue  # still open, or closed after the analysis date: never learn from the future
        best, best_gap = None, MIN_OPEN_GAP
        for opened_at, past, shape in setups:
            gap = abs(opened_at - opened)
            if gap <= best_gap:
                best, best_gap = (past, shape), gap
        if best is None or not t.entry_price:
            continue
        past, shape = best
        if shape != trend or not (pmin <= past.price <= pmax) or not (rmin <= past.rsi <= rmax):
            continue
        return_pct = (t.exit_price / t.entry_price - 1) * 100
        held = max(1, (_aware(t.closed_at) - opened).days)
        quality = abs(past.rsi - r_mid) + abs(past.price - p_mid) / max(p_mid, 1e-9)
        matches.append((quality, -opened.timestamp(), PatternMatch(
            pattern_id=pattern_id(past), symbol=symbol, opened_at=opened, entry_price=t.entry_price,
            exit_price=t.exit_price, held_days=held, return_pct=return_pct, quality_score=quality)))
    matches.sort()
    return [m for _, _, m in matches[:limit]]


def analyze_pattern(trades: Iterable[PatternMatch]) -> Optional[PatternStats]:
    """Summarise prior similar trades into the stats the agent and the metrics scripts use. None when there are none."""
    return pattern_stats((t.return_pct, t.held_days) for t in trades)


def history_stats(sessions, snapshot: Snapshot, lookback_days: int = LOOKBACK_DAYS, rsi_tol: float = RSI_TOLERANCE,
                  limit: int = MAX_MATCHES, now: Optional[datetime] = None) -> Optional[PatternStats]:
    """Closed-trade stats for setups most similar to `snapshot`, or None when there is nothing to learn from yet.

    Same trend shape, price within 5% and RSI(14) within `rsi_tol` (5 points by default, per the review doc)."""
    price = snapshot.price
    matches = find_similar_patterns(
        sessions, snapshot.symbol,
        price_range=(price * (1 - PRICE_TOL_FRAC), price * (1 + PRICE_TOL_FRAC)),
        trend=_trend_shape(snapshot),
        rsi_range=(snapshot.rsi - rsi_tol, snapshot.rsi + rsi_tol),
        lookback_days=lookback_days, limit=limit, now=now,
    )
    return analyze_pattern(matches)