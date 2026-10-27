"""The filing watcher: reads the exchange feeds, keeps the filings of the news universe's stocks, and has the AI label each
non-routine one, in a background thread so a slow model never delays the engines' 5-minute steps.

Polls every 60 seconds from 08:30 to 15:00 IST on weekdays (the plan's window) and every 5 minutes at other times, because
the BSE feed starts afresh each night: results filed in the evening must be caught that evening to be traded the next
morning (rule A). Each filing is stored once (keyed by its link; the same filing from the other exchange's feed within two
minutes is dropped as a duplicate)."""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Callable, Optional, Set

from sqlalchemy import select

from src.data.filings import FeedReader, Filing
from src.data.india import IST, SUFFIX
from src.database import NewsFilingRecord
from src.news.labels import as_utc, ask_label, first_session, label_fields, materiality, routine_subject
from src.news.results_calendar import record_notices
from src.versions import NEWS_VERSION

log = logging.getLogger(__name__)

FAST_FROM, FAST_UNTIL = dtime(8, 30), dtime(15, 0)
FAST, SLOW = 60.0, 300.0  # seconds between polls inside and outside the plan's window
DUPLICATE_WITHIN = timedelta(minutes=2)
LABELS_PER_POLL = 12  # a backlog (a restart in the evening) is worked through over several polls


def yahoo_market_cap_crore(symbol: str) -> Optional[float]:
    """The company's market value in Rs crore (Yahoo); None when unknown."""
    import yfinance as yf

    try:
        cap = yf.Ticker(symbol + SUFFIX).fast_info["marketCap"]
        return float(cap) / 1e7 if cap else None
    except Exception as e:  # Yahoo gives no value for many symbols
        log.debug("no market cap for %s: %s: %s", symbol, type(e).__name__, e)
        return None


class FilingWatcher:
    def __init__(
        self,
        sessions,
        reader: FeedReader,
        llm,
        universe: Callable[[], Set[str]],
        is_session_day: Callable[[datetime], bool],
        market_cap_crore: Callable[[str], Optional[float]] = yahoo_market_cap_crore,
        notify: Optional[Callable[[str], object]] = None,
        price_fn: Callable[[str], Optional[float]] = lambda symbol: None,
        held_fn: Callable[[], Set[str]] = lambda: set(),
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.sessions, self.reader, self.llm, self.universe = sessions, reader, llm, universe
        self.is_session_day, self.market_cap_crore, self.notify = is_session_day, market_cap_crore, notify
        self.price_fn, self.held_fn, self.now_fn = price_fn, held_fn, now_fn
        self.failures = 0  # feed reads failed in a row

    def poll_once(self) -> int:
        """Read the feed, store new filings of universe stocks, label up to LABELS_PER_POLL pending ones. Returns the number of
        new filings stored."""
        added = self._store(self.reader.read())
        self._label_pending()
        return added

    def _store(self, filings) -> int:
        wanted = self.universe() | self.held_fn()  # the swing account's holdings are read too, to warn about them
        added = 0
        with self.sessions() as s:
            record_notices(s, filings)  # results dates of every stock, for the swing bot's entry record
            for f in filings:
                if f.symbol not in wanted or s.scalar(select(NewsFilingRecord.id).where(NewsFilingRecord.key == f.key)):
                    continue
                twin = s.scalar(
                    select(NewsFilingRecord.id).where(
                        NewsFilingRecord.symbol == f.symbol,
                        NewsFilingRecord.source != f.source,
                        NewsFilingRecord.published.between(f.published - DUPLICATE_WITHIN, f.published + DUPLICATE_WITHIN),
                    )
                )
                if twin:
                    continue
                session = first_session(f.published, self.is_session_day)
                marker = routine_subject(f.subject)
                s.add(
                    NewsFilingRecord(
                        key=f.key,
                        source=f.source,
                        symbol=f.symbol,
                        company=f.company,
                        subject=f.subject[:2000],
                        link=f.link,
                        published=f.published,
                        seen_price=self.price_fn(f.symbol),  # Angel One's live price now, if streaming
                        session=session,
                        label_status="ROUTINE" if marker else "PENDING",
                        material=False if marker else None,
                        material_reason=f"routine by subject ({marker})" if marker else None,
                        system_version=NEWS_VERSION,
                    )
                )
                added += 1
            s.commit()
        return added

    def _label_pending(self) -> None:
        with self.sessions() as s:
            pending = list(
                s.scalars(
                    select(NewsFilingRecord)
                    .where(NewsFilingRecord.label_status == "PENDING")
                    .order_by(NewsFilingRecord.published.desc())  # the newest first: they are the ones still tradable
                    .limit(LABELS_PER_POLL)
                )
            )
        for row in pending:
            self._label(row)

    def _label(self, row: NewsFilingRecord) -> None:
        filing = Filing(row.source, row.symbol, row.company, row.subject, row.link, as_utc(row.published))
        try:
            label = ask_label(self.llm, filing, self.reader.attachment_text(row.link))
            verdict = materiality(label, lambda: self.market_cap_crore(row.symbol))
            fields = label_fields(label, verdict)
        except Exception as e:
            log.warning("could not label %s %s (%s: %s)", row.symbol, row.link, type(e).__name__, str(e)[:200])
            fields = dict(label_status="FAILED", material=False, material_reason=f"no AI label ({type(e).__name__})")
        with self.sessions() as s:
            stored = s.get(NewsFilingRecord, row.id)
            for k, v in fields.items():
                setattr(stored, k, v)
            s.commit()
        if fields.get("material") and self.notify:
            held = " ⚠️ YOU HOLD THIS (swing account)" if row.symbol in self.held_fn() else ""
            self.notify(
                f"[NEWS] material filing{held}: {row.symbol} {fields['event_type']} ({fields['direction']}, surprise "
                f"{fields['surprise']}): {fields['material_reason']}. {row.subject[:120]}"
            )

    # ------------------------------------------------------------------ the thread
    def interval(self) -> float:
        local = self.now_fn().astimezone(IST)
        return FAST if local.weekday() < 5 and FAST_FROM <= local.time() < FAST_UNTIL else SLOW

    def run(self, sleep: Callable[[float], None] = time.sleep) -> None:
        while True:
            try:
                self.poll_once()
                self.failures = 0
            except Exception as e:
                self.failures += 1
                log.warning("filing feed read failed (%s: %s)", type(e).__name__, e)
                if self.failures == 5 and self.notify:
                    self.notify("[NEWS] ⚠️ ALERT: the filing feeds (BSE and NSE) failed 5 times in a row: no news is being read")
            sleep(self.interval())

    def start(self) -> None:
        threading.Thread(target=self.run, name="filing-watcher", daemon=True).start()
