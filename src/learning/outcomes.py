"""Outcome labelling: what did each stock do after the bot analysed it?

The bot analyses ~20 stocks a day and buys almost none of them, so its own closed trades (a trend strategy closes about
20 a year) are far too few to learn from. But every analysed stock, bought or not, has an outcome: enter at the next
session's open (exactly how the bot trades: it decides after a close and fills the next session), hold `horizon` sessions,
and see what happened. Labelling all of them gives hundreds of lessons a week, including the two kinds of mistake that
closed trades hide: buys that lost, and stocks it passed on (or vetoed) that then rose.

`forward_outcome` is a pure function of a price table, so the arithmetic is testable on its own; `label_outcomes` applies it
to the decisions in the database, is idempotent, and only labels a session once `horizon` further sessions have closed."""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from sqlalchemy import select

from src.database import DecisionRecord, OutcomeRecord

log = logging.getLogger(__name__)

HORIZONS = (5, 20, 60)  # sessions held after entry: a week, a month, a quarter


@dataclass(frozen=True)
class Outcome:
    """One stock, one analysed session, one holding period."""
    symbol: str
    bar_date: str
    horizon: int
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    gross_return: float
    worst_return: float
    hit_stop: bool
    stop_pct: float


def forward_outcome(symbol: str, bars: pd.DataFrame, bar_date: str, horizon: int, stop_pct: float) -> Optional[Outcome]:
    """The forward result of the call made on `bar_date`, or None when it cannot be computed yet or at all.

    `bars`: daily Open/High/Low/Close, oldest first. Entry is the OPEN of the session after `bar_date`; the exit is the
    CLOSE `horizon` sessions after `bar_date`. None when `bar_date` is not a session in `bars` (missing data, never
    guessed) or when fewer than `horizon` sessions have closed since (not yet matured)."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize() if pd.DatetimeIndex(bars.index).tz is not None \
        else pd.DatetimeIndex(bars.index).normalize()
    day = pd.Timestamp(bar_date).normalize()
    pos = int(index.searchsorted(day))
    if pos >= len(index) or index[pos] != day or pos + horizon >= len(index):
        return None
    window = slice(pos + 1, pos + horizon + 1)
    entry = float(bars["Open"].iloc[pos + 1])
    if not entry > 0:
        return None
    exit_price = float(bars["Close"].iloc[pos + horizon])
    lows = bars["Low"].iloc[window].astype(float)
    return Outcome(symbol, bar_date, horizon, str(index[pos + 1].date()), str(index[pos + horizon].date()), entry, exit_price,
                   exit_price / entry - 1,
                   float(lows.min()) / entry - 1, bool((lows <= entry * (1 - stop_pct)).any()), stop_pct)


def backfill_bar_dates(sessions) -> int:
    """Give older decisions their `bar_date` (read from the stored snapshot). Returns how many rows were filled."""
    import json

    filled = 0
    with sessions() as s:
        for d in s.scalars(select(DecisionRecord).where(DecisionRecord.bar_date.is_(None), DecisionRecord.snapshot_json.isnot(None))):
            try:
                d.bar_date = json.loads(d.snapshot_json).get("bar_date")
            except (ValueError, TypeError):
                continue
            filled += d.bar_date is not None
        s.commit()
    return filled


@dataclass(frozen=True)
class LabelSummary:
    """What a labelling run did."""
    labelled: int = 0
    already_done: int = 0
    not_mature: int = 0
    no_data: int = 0
    symbols_without_bars: Tuple[str, ...] = ()


def _matured(bar_date: str, horizon: int, today: date) -> bool:
    """A cheap calendar test before fetching anything: `horizon` business days after `bar_date` must be behind us."""
    return (pd.Timestamp(bar_date) + pd.offsets.BDay(horizon)).date() < today


def label_outcomes(sessions, fetch_bars: Callable[[str], Optional[pd.DataFrame]], horizons: Sequence[int] = HORIZONS,
                   stop_pct: float = 0.15, today: Optional[date] = None) -> LabelSummary:
    """Label every analysed (stock, session) that has matured and is not labelled yet. Idempotent and safe to interrupt.

    `fetch_bars(symbol)` returns that stock's daily bars or None. Only stocks with something to label are fetched."""
    today = today or datetime.now(timezone.utc).date()
    backfill_bar_dates(sessions)
    with sessions() as s:
        analysed: Set[Tuple[str, str]] = {(sym, bd) for sym, bd in s.execute(
            select(DecisionRecord.symbol, DecisionRecord.bar_date).where(DecisionRecord.bar_date.isnot(None)).distinct())}
        done: Set[Tuple[str, str, int]] = {(o.symbol, o.bar_date, o.horizon) for o in s.scalars(select(OutcomeRecord))}
    wanted: Dict[str, List[Tuple[str, int]]] = {}
    already = not_mature = 0
    for symbol, bar_date in sorted(analysed):
        for h in horizons:
            if (symbol, bar_date, h) in done:
                already += 1
            elif not _matured(bar_date, h, today):
                not_mature += 1
            else:
                wanted.setdefault(symbol, []).append((bar_date, h))
    labelled = no_data = 0
    missing: List[str] = []
    for symbol, todo in wanted.items():
        try:
            bars = fetch_bars(symbol)
        except Exception as e:
            log.warning("outcome labelling: no bars for %s (%s: %s)", symbol, type(e).__name__, str(e)[:80])
            bars = None
        if bars is None or len(bars) == 0:
            missing.append(symbol)
            no_data += len(todo)
            continue
        with sessions() as s:
            for bar_date, h in todo:
                o = forward_outcome(symbol, bars, bar_date, h, stop_pct)
                if o is None:
                    no_data += 1
                    continue
                s.add(OutcomeRecord(symbol=symbol, bar_date=bar_date, horizon=h, entry_date=o.entry_date, exit_date=o.exit_date,
                                    entry_price=o.entry_price, exit_price=o.exit_price, gross_return=o.gross_return,
                                    worst_return=o.worst_return, hit_stop=o.hit_stop, stop_pct=stop_pct))
                labelled += 1
            s.commit()
    return LabelSummary(labelled, already, not_mature, no_data, tuple(sorted(missing)))
