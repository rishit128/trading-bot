"""India (NSE) data plumbing: official stock list, bulk daily bars, market clock, intraday feed. All via public sources."""
import io
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .data_honesty import check_bars_honesty

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
SUFFIX = ".NS"
NSE_EQUITY_LIST = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY500_LIST = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def _parse_symbols(csv_text: str) -> List[str]:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]  # skips trade-to-trade (BE/BZ) segments
    return sorted(df["SYMBOL"].astype(str).str.strip().unique())


def fetch_nse_symbols(get: Callable = httpx.get) -> List[str]:
    """Every NSE main-board EQ-series stock (~2,300). Falls back to the Nifty 500 if NSE's archive is unreachable."""
    for name, url in (("NSE equity list", NSE_EQUITY_LIST), ("Nifty 500 list", NIFTY500_LIST)):
        try:
            r = get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
            r.raise_for_status()
            symbols = _parse_symbols(r.text)
            if len(symbols) > 100:
                log.info("%s: %d symbols", name, len(symbols))
                return symbols
        except Exception as e:
            log.warning("%s unavailable (%s: %s)", name, type(e).__name__, e)
    raise RuntimeError("could not load an NSE stock list from NSE or niftyindices.com")


NIFTY_LISTS = {
    50: "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    100: "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    500: NIFTY500_LIST,
}


def fetch_index_symbols(index: int = 500, get: Callable = httpx.get) -> List[str]:
    """Constituents of a Nifty index as NSE symbols (today's membership, so historical tests carry survivorship bias)."""
    r = get(NIFTY_LISTS[index], headers=_HEADERS, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return _parse_symbols(r.text)


def yf_bar_fetcher(lookback_days: int = 400, download: Optional[Callable] = None,
                   today: Callable[[], datetime] = lambda: datetime.now(IST)):
    """Bulk completed daily bars in the same long format the screener uses. Symbols Yahoo cannot resolve are skipped."""
    if download is None:
        import yfinance as yf

        def download(tickers, **kw):
            """Default bulk yfinance download, with a timeout."""
            return yf.download(tickers, group_by="ticker", auto_adjust=True, progress=False, threads=True, timeout=30, **kw)

    def fetch(symbols: List[str]) -> pd.DataFrame:
        """Download completed daily bars for a batch of NSE symbols in the screener's long format."""
        end = today().date()  # yfinance's end is exclusive, so today's unfinished session is never included
        start = end - timedelta(days=lookback_days)
        wide = download([s + SUFFIX for s in symbols], start=start.isoformat(), end=end.isoformat(), interval="1d")
        if wide is None or wide.empty:
            return pd.DataFrame()
        frames = []
        top = set(wide.columns.get_level_values(0))
        for s in symbols:
            ticker = s + SUFFIX
            if ticker not in top:
                continue
            sub = wide[ticker][["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
            if sub.empty:
                continue
            for finding in check_bars_honesty(sub):
                log.warning("%s: %s", s, finding)
            sub.index = pd.MultiIndex.from_product([[s], sub.index], names=["symbol", "timestamp"])
            frames.append(sub)
        return pd.concat(frames) if frames else pd.DataFrame()

    return fetch


class IndiaClock:
    """NSE/BSE sessions (09:15-15:30 IST) with exchange holidays from the BSE calendar, which matched Yahoo's
    actual 2026 Nifty trading days exactly. Past the calendar's last covered date it falls back to weekdays + hours."""

    def __init__(self, calendar=None):
        if calendar is None:
            import exchange_calendars as xc

            calendar = xc.get_calendar("XBOM")
        self.cal = calendar

    def is_open(self, now: Optional[datetime] = None) -> bool:
        """True when NSE is trading at this moment (holiday-aware)."""
        now = now or datetime.now(timezone.utc)
        ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        local = ts.tz_convert(IST)
        if local.date() > self.cal.last_session.date():
            log.warning("holiday calendar ends %s; assuming weekdays are trading days", self.cal.last_session.date())
            minutes = local.hour * 60 + local.minute
            return local.weekday() < 5 and 9 * 60 + 15 <= minutes < 15 * 60 + 30
        return bool(self.cal.is_trading_minute(ts.floor("min")))

    def today_ist(self, now: Optional[datetime] = None) -> str:
        """Today's date in India."""
        return (now or datetime.now(timezone.utc)).astimezone(IST).date().isoformat()

    def last_completed_session(self, now: Optional[datetime] = None) -> date:
        """Date of the most recent session that has fully closed: yesterday's during market hours, today's after the
        close, Friday's on a weekend, and the previous trading day across holidays."""
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(IST)
        if local.date() > self.cal.last_session.date():  # past the holiday calendar: weekdays + 15:30 close
            day = local.date() if local.hour * 60 + local.minute >= 15 * 60 + 30 else local.date() - timedelta(days=1)
            while day.weekday() >= 5:
                day -= timedelta(days=1)
            return day
        ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        return self.cal.previous_close(ts.floor("min")).tz_convert(IST).date()


class IntradayFeed:
    """5-minute Yahoo bars: latest price for paper fills and high/low history for simulated stop/target checks."""

    def __init__(self, download: Optional[Callable] = None, ttl_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        # Settling exits, valuing positions and filling an order all need the same bars within seconds of each other.
        self._cache, self._lock, self._ttl, self._clock = {}, threading.Lock(), ttl_seconds, clock
        if download is None:
            import yfinance as yf

            def download(ticker, **kw):
                """Default single-symbol 5-minute yfinance download, with a timeout."""
                df = yf.download(ticker, progress=False, auto_adjust=True, timeout=30, **kw)
                if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df

        self._download = download

    def _bars(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            hit = self._cache.get(symbol)
            if hit and self._clock() - hit[0] < self._ttl:
                return hit[1]
        df = self._download(symbol + SUFFIX, period="5d", interval="5m")
        bars = df.dropna(subset=["Close"]) if df is not None and not df.empty else pd.DataFrame()
        with self._lock:
            self._cache[symbol] = (self._clock(), bars)
        return bars

    def last_price(self, symbol: str) -> float:
        """Latest 5-minute close for an NSE symbol."""
        bars = self._bars(symbol)
        if bars.empty:
            raise ValueError(f"{symbol}: no price data available")
        return float(bars["Close"].iloc[-1])

    def bars_since(self, symbol: str, since: datetime) -> pd.DataFrame:
        """5-minute bars after a point in time (used to replay stops and targets)."""
        bars = self._bars(symbol)
        if bars.empty:
            return bars
        idx = bars.index.tz_convert("UTC") if bars.index.tz else bars.index.tz_localize(IST).tz_convert("UTC")
        cutoff = pd.Timestamp(since)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return bars[idx > cutoff]
