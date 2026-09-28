"""5-minute bars: fetching them from Yahoo, and an archive that keeps every finished session.

Yahoo serves only the last ~60 sessions of 5-minute data, which is too short to judge an intraday rule (one market regime,
a few dozen independent days). The intraday engine therefore saves each finished session into `intraday_bars`, so the
history grows by one session a day and a rule can eventually be tested on months. `BarArchive.save` is idempotent, so a
restart, a backfill from Yahoo's window or an import of an older cache never duplicates a bar."""
import logging
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Set

import pandas as pd
from sqlalchemy import insert, select

from src.data.india import IST, SUFFIX
from src.database import IntradayBarRecord

log = logging.getLogger(__name__)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _yahoo_download(tickers: List[str], **kw) -> pd.DataFrame:
    """Bulk yfinance download (one frame, one column level per ticker) with a timeout."""
    import yfinance as yf

    return yf.download(tickers, group_by="ticker", auto_adjust=True, progress=False, threads=True, timeout=60, **kw)


def fetch_5m_bars(symbols: List[str], period: str = "1d", download: Optional[Callable[..., Optional[pd.DataFrame]]] = None,
                  batch: int = 100) -> Dict[str, pd.DataFrame]:
    """5-minute bars (IST index) per symbol for a Yahoo `period` ("1d" = the latest session, "59d" = the whole window).
    Symbols Yahoo has no data for are omitted."""
    download = download or _yahoo_download
    out: Dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), batch):
        chunk = symbols[start:start + batch]
        wide = download([s + SUFFIX for s in chunk], period=period, interval="5m")
        if wide is None or wide.empty:
            continue
        top = set(wide.columns.get_level_values(0))
        for s in chunk:
            if s + SUFFIX in top:
                df = pd.DataFrame(wide[s + SUFFIX]).dropna(subset=["Close"])
                if len(df):
                    df.index = pd.DatetimeIndex(df.index).tz_convert(IST)
                    out[s] = df
    return out


def session_rows(df: pd.DataFrame, day: date) -> pd.DataFrame:
    """The rows of a 5-minute frame that fall on the IST session date `day` (Yahoo can return the previous session too)."""
    return df[pd.DatetimeIndex(df.index).tz_convert(IST).date == day]


def fetch_today_bars(symbols: List[str], download: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
    """The latest session's 5-minute bars per symbol, in one bulk call."""
    return fetch_5m_bars(symbols, "1d", download)


class BarArchive:
    """Saves and loads 5-minute bars in the `intraday_bars` table of the given database."""

    def __init__(self, sessions):
        self.sessions = sessions

    def days(self) -> List[str]:
        """Every session date stored, oldest first."""
        with self.sessions() as s:
            return list(s.scalars(select(IntradayBarRecord.day).distinct().order_by(IntradayBarRecord.day)))

    def _stored(self, days: Iterable[str]) -> Set[tuple]:
        with self.sessions() as s:
            query = (select(IntradayBarRecord.symbol, IntradayBarRecord.day)
                     .where(IntradayBarRecord.day.in_(list(days))).distinct())
            return {(symbol, day) for symbol, day in s.execute(query)}

    def save(self, bars: Dict[str, pd.DataFrame]) -> int:
        """Store the given bars (a frame may span several sessions), skipping any (symbol, session) already stored.
        Returns the number of bars added."""
        wanted = {}
        for symbol, df in bars.items():
            for day in sorted(set(pd.DatetimeIndex(df.index).tz_convert(IST).date)):
                wanted[(symbol, day.isoformat())] = session_rows(df, day)
        if not wanted:
            return 0
        stored = self._stored({day for _, day in wanted})
        rows = []
        for (symbol, day), frame in wanted.items():
            if (symbol, day) in stored:
                continue
            stamps = pd.DatetimeIndex(frame.index).tz_convert("UTC").to_pydatetime()
            prices = zip(stamps, frame["Open"], frame["High"], frame["Low"], frame["Close"], frame["Volume"])
            rows += [{"symbol": symbol, "ts": ts, "day": day, "open": float(o), "high": float(h), "low": float(low),
                      "close": float(c), "volume": float(v)} for ts, o, h, low, c, v in prices]
        if rows:
            with self.sessions() as s:
                s.execute(insert(IntradayBarRecord), rows)
                s.commit()
        return len(rows)

    def load(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Stored bars per symbol (IST index), for sessions from `start` to `end` inclusive (ISO dates; open-ended if omitted)."""
        query = select(IntradayBarRecord).order_by(IntradayBarRecord.symbol, IntradayBarRecord.ts)
        if start:
            query = query.where(IntradayBarRecord.day >= start)
        if end:
            query = query.where(IntradayBarRecord.day <= end)
        rows: Dict[str, list] = {}
        with self.sessions() as s:
            for r in s.scalars(query):
                rows.setdefault(r.symbol, []).append((r.ts, r.open, r.high, r.low, r.close, r.volume))
        out = {}
        for symbol, data in rows.items():
            frame = pd.DataFrame(data, columns=["ts"] + COLUMNS)
            stamps = pd.DatetimeIndex(pd.to_datetime(frame.pop("ts"), utc=True))  # SQLite hands back naive UTC
            frame.index = stamps.tz_convert(IST).rename(None)
            out[symbol] = frame
        return out
