"""Results dates for the swing bot, from the exchange feeds the news watcher already reads.

A BSE board-meeting notice carries the meeting date ("... is scheduled on 27/10/2026, inter alia, to consider and approve
Unaudited Financial Results"). The watcher keeps every such notice and every results filing, for all stocks, and the swing
bot's entry record then says how close the stock's results are, so a later forward test can ask whether entries just
before results do worse. Free history of results dates does not exist (STRATEGY.md, 2026-10-07), so this builds it."""

import re
from datetime import date, datetime
from typing import Callable, Dict, Iterable, Optional, Tuple

from sqlalchemy import select

from src.data.angel_quotes import EntryContext
from src.data.filings import Filing
from src.database import FilingNoticeRecord

MEETING = re.compile(r"scheduled on\s*(\d{1,2})/(\d{1,2})/(\d{4})", re.IGNORECASE)
RESULTS = re.compile(r"financial result|outcome of board meeting", re.IGNORECASE)
LOOK_BACK_DAYS = 45
BLOCK_DAYS = 3  # no new swing buy when the stock's results are due within this many days (CASTROLIND, 2026-10-26)


def classify(subject: str) -> Tuple[Optional[str], Optional[str]]:
    """(kind, meeting date) of a filing's subject: MEETING with its ISO date, RESULT, or (None, None)."""
    meeting = MEETING.search(subject)
    if meeting and "result" in subject.lower():
        day, month, year = (int(g) for g in meeting.groups())
        try:
            return "MEETING", date(year, month, day).isoformat()
        except ValueError:
            return None, None
    if RESULTS.search(subject):
        return "RESULT", None
    return None, None


def record_notices(session, filings: Iterable[Filing]) -> int:
    """Store the results-related filings not seen before (one row per link). `session` is an open SQLAlchemy session."""
    added = 0
    for f in filings:
        kind, meeting_date = classify(f.subject)
        if kind is None or session.scalar(select(FilingNoticeRecord.id).where(FilingNoticeRecord.key == f.key)):
            continue
        session.add(FilingNoticeRecord(key=f.key, symbol=f.symbol, kind=kind, published=f.published, meeting_date=meeting_date))
        session.flush()
        added += 1
    return added


def results_note(sessions, symbol: str, today: date) -> Dict[str, object]:
    """What the feeds say about this stock's results: `results_in_days` (the nearest announced meeting, today or later) and
    `results_filed_days_ago` (its latest results filing within 45 days). Empty when the feeds said nothing."""
    note: Dict[str, object] = {}
    with sessions() as s:
        rows = list(s.scalars(select(FilingNoticeRecord).where(FilingNoticeRecord.symbol == symbol)))
    meetings = sorted(r.meeting_date for r in rows if r.kind == "MEETING" and r.meeting_date and r.meeting_date >= today.isoformat())
    if meetings:
        note["results_in_days"] = (date.fromisoformat(meetings[0]) - today).days
    filed = [r.published for r in rows if r.kind == "RESULT"]
    if filed:
        age = (datetime.combine(today, datetime.min.time()) - max(filed).replace(tzinfo=None)).days
        if 0 <= age <= LOOK_BACK_DAYS:
            note["results_filed_days_ago"] = age
    return note


def with_results(context_fn: Callable[[str, float], EntryContext], sessions, today_fn: Callable[[], date]):
    """`context_fn` with the stock's results timing added to the entry record; a buy within BLOCK_DAYS of an announced results
    meeting is refused (a results gap goes straight through the 15% stop and the price data cannot see it coming)."""

    def wrapped(symbol: str, price: float) -> EntryContext:
        context = context_fn(symbol, price)
        try:
            extra = results_note(sessions, symbol, today_fn())
        except Exception:
            extra = {}
        due = extra.get("results_in_days")
        why = context.skip or (
            f"results due in {due} day(s): no new buy until they are out" if isinstance(due, int) and due <= BLOCK_DAYS else None
        )
        return EntryContext(why, {**context.note, **extra})

    return wrapped
