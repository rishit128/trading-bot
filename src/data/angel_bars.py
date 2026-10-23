"""Angel One's bars for the engines, with Yahoo as the automatic backup, and the check that decides when Angel One is trusted.

`AngelBars` has the signature of the engines' `fetch_bars` (symbols -> 5-minute frames, IST index, columns Open High Low Close
Volume Source). When Angel One is in use it serves today's bars and the earlier sessions saved in `angel_bars` (so a bar and
its time-of-day volume average come from the same source); a stock it has no bars for, or the whole call when the stream is
down, is read from Yahoo instead. Every row says which source it came from (`Source`), and the archive stores it.

Angel One is used when ANGEL_BARS is "on", or "auto" (the default) once at least MIN_CLEAN_DAYS sessions agreed with Yahoo."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy import select

from src.data.angel import BAR, HISTORY_DAYS, AngelStream
from src.data.india import IST
from src.database import AngelBarRecord, IntradayBarRecord

log = logging.getLogger(__name__)

MIN_CLEAN_DAYS = 5
COVERAGE_MIN = 0.95  # of Yahoo's bars, the share Angel One also has
CLOSE_TOLERANCE, CLOSE_SHARE_MIN = 0.001, 0.95  # closes within 0.1% on at least this share of the shared bars
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def agreement(angel: pd.DataFrame, yahoo: pd.DataFrame) -> Dict[str, float]:
    """How well one session's Angel One bars match Yahoo's (frames with columns symbol, ts, close): the share of Yahoo's bars
    Angel One also has, and the share of those whose closes agree within CLOSE_TOLERANCE."""
    both = angel.merge(yahoo, on=["symbol", "ts"], suffixes=("_a", "_y"))
    if yahoo.empty or both.empty:
        return {"bars_in_both": 0, "coverage": 0.0, "close_agrees": 0.0}
    gap = (both["close_a"] / both["close_y"] - 1).abs()
    return {"bars_in_both": len(both), "coverage": len(both) / len(yahoo), "close_agrees": float((gap < CLOSE_TOLERANCE).mean())}


def session_frames(sessions, day: str) -> Dict[str, pd.DataFrame]:
    """One session's Angel One bars and Yahoo bars (as saved before Angel One was in use) from the database."""
    with sessions() as s:
        a = s.execute(
            select(AngelBarRecord.symbol, AngelBarRecord.ts, AngelBarRecord.close).where(AngelBarRecord.day == day)
        ).all()
        y = s.execute(
            select(IntradayBarRecord.symbol, IntradayBarRecord.ts, IntradayBarRecord.close).where(
                IntradayBarRecord.day == day, IntradayBarRecord.source.is_distinct_from("angel")
            )
        ).all()
    return {
        "angel": pd.DataFrame(a, columns=["symbol", "ts", "close"]),
        "yahoo": pd.DataFrame(y, columns=["symbol", "ts", "close"]),
    }


def clean_days(sessions) -> List[str]:
    """The sessions on which Angel One's bars agreed with Yahoo's well enough (see the constants above)."""
    with sessions() as s:
        days = sorted(
            set(s.scalars(select(AngelBarRecord.day).distinct())) & set(s.scalars(select(IntradayBarRecord.day).distinct()))
        )
    clean = []
    for day in days:
        frames = session_frames(sessions, day)
        stats = agreement(frames["angel"], frames["yahoo"])
        if stats["coverage"] >= COVERAGE_MIN and stats["close_agrees"] >= CLOSE_SHARE_MIN:
            clean.append(day)
    return clean


class AngelTrust:
    """Whether Angel One may be used: ANGEL_BARS "on", or "auto" once enough sessions agreed with Yahoo (checked once a day)."""

    def __init__(self, sessions, mode: str = "auto", now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.sessions, self.mode, self.now_fn = sessions, mode, now_fn
        self._trusted = False
        self._checked_on: Optional[str] = None

    def __call__(self) -> bool:
        if self.mode in ("on", "off"):
            return self.mode == "on"
        today = self.now_fn().astimezone(IST).date().isoformat()
        if self._checked_on != today:
            clean = clean_days(self.sessions)
            self._trusted, self._checked_on = len(clean) >= MIN_CLEAN_DAYS, today
            log.info(
                "Angel One bars: %d clean sessions of %d needed -> %s",
                len(clean),
                MIN_CLEAN_DAYS,
                "in use" if self._trusted else "Yahoo",
            )
        return self._trusted


class AngelBars:
    def __init__(
        self,
        sessions,
        stream: AngelStream,
        yahoo: Callable[[List[str]], Dict[str, pd.DataFrame]],
        mode: str = "auto",
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.sessions, self.stream, self.yahoo, self.now_fn = sessions, stream, yahoo, now_fn
        self.trusted = AngelTrust(sessions, mode, now_fn)

    def __call__(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        if not (self.trusted() and self.stream.ready()):
            return self._yahoo(symbols)
        frames = self._angel(symbols)
        missing = [s for s in symbols if s not in frames]
        if missing:
            frames.update(self._yahoo(missing))
        return {s: frames[s] for s in symbols if s in frames}

    def _yahoo(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        out = self.yahoo(symbols)
        for frame in out.values():
            frame["Source"] = "yahoo"
        return out

    def _angel(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """The stocks with bars today and on an earlier session, from the database."""
        now = self.now_fn().astimezone(IST)
        columns = [
            AngelBarRecord.symbol,
            AngelBarRecord.ts,
            AngelBarRecord.open,
            AngelBarRecord.high,
            AngelBarRecord.low,
            AngelBarRecord.close,
            AngelBarRecord.volume,
        ]
        with self.sessions() as s:
            rows = s.execute(
                select(*columns)
                .where(
                    AngelBarRecord.symbol.in_(symbols),
                    AngelBarRecord.day >= (now.date() - timedelta(days=HISTORY_DAYS)).isoformat(),
                )
                .order_by(AngelBarRecord.ts)
            ).all()
        wide = pd.DataFrame(rows, columns=["symbol", "ts"] + COLUMNS)
        wide["ts"] = pd.DatetimeIndex(pd.to_datetime(wide["ts"], utc=True)).tz_convert(IST)
        out = {}
        for symbol, frame in wide.groupby("symbol"):
            frame = frame.set_index("ts")[COLUMNS]
            days = pd.DatetimeIndex(frame.index).date
            if (days == now.date()).any() and (days < now.date()).any():
                frame["Source"] = "angel"
                out[str(symbol)] = frame
        return out


class AngelPriceFeed:
    """The paper brokers' price feed (fills, valuation, and the replay of stops and targets) on Angel One's bars, with the
    Yahoo feed as the backup. Angel One's latest bar is used while it is recent; the bar still forming is added when this
    process runs the stream, so a stop is seen as soon as a real print crosses it, as with Yahoo. Any stock or moment Angel One
    does not cover is read from Yahoo."""

    FRESH = timedelta(minutes=10)  # a latest bar older than this (before the open, after a gap) is not "the price now"

    def __init__(self, sessions, yahoo, trusted: Callable[[], bool], stream: Optional[AngelStream] = None, now_fn=None):
        self.sessions, self.yahoo, self.trusted, self.stream = sessions, yahoo, trusted, stream
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def _latest(self, symbol: str) -> Optional[tuple]:
        """(bar start, close) of the newest finished Angel One bar if it is recent, else None."""
        now = self.now_fn()
        with self.sessions() as s:
            row = s.execute(
                select(AngelBarRecord.ts, AngelBarRecord.close)
                .where(AngelBarRecord.symbol == symbol)
                .order_by(AngelBarRecord.ts.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        start = row[0].replace(tzinfo=timezone.utc) if row[0].tzinfo is None else row[0]
        return (start, float(row[1])) if now - (start + BAR) < self.FRESH else None

    def last_price(self, symbol: str) -> float:
        hit = self._latest(symbol) if self.trusted() else None
        return hit[1] if hit else self.yahoo.last_price(symbol)

    def last_quote(self, symbol: str):
        hit = self._latest(symbol) if self.trusted() else None
        return (hit[1], hit[0]) if hit else self.yahoo.last_quote(symbol)

    def bars_since(self, symbol: str, since: datetime) -> pd.DataFrame:
        """Today's bars after `since`; anything reaching back before today's open, or with no Angel One bar yet, is Yahoo's."""
        now = self.now_fn()
        if not self.trusted() or since.astimezone(IST).date() != now.astimezone(IST).date():
            return self.yahoo.bars_since(symbol, since)
        cutoff = since.astimezone(timezone.utc).replace(tzinfo=None)  # SQLite holds naive UTC
        with self.sessions() as s:
            rows = s.execute(
                select(AngelBarRecord.ts, *[getattr(AngelBarRecord, c.lower()) for c in COLUMNS])
                .where(AngelBarRecord.symbol == symbol, AngelBarRecord.ts > cutoff)
                .order_by(AngelBarRecord.ts)
            ).all()
        forming = self.stream.forming(symbol) if self.stream else None
        if forming and forming["ts"] > since:
            rows.append((forming["ts"], forming["open"], forming["high"], forming["low"], forming["close"], 0.0))
        if not rows:
            return self.yahoo.bars_since(symbol, since)
        frame = pd.DataFrame(rows, columns=["ts"] + COLUMNS)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("ts"), utc=True)).tz_convert(IST)
        return frame
