"""India (NSE) data plumbing: the stock and index lists, bulk daily bars, market clock, intraday feed.

The stock and index lists are NSE files a person downloads in a browser: the terms of use of nseindia.com and
niftyindices.com (read 2026-09-26) prohibit "systematic or automated data collection" without written consent, so the
bot never downloads from those sites itself. It reads the files from NSE_FILES_DIR (default ./nse_files). Prices come
from Yahoo Finance."""
import io
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from .retry import retry_call
from .data_honesty import ZERO_VOLUME, check_bars_honesty

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
SUFFIX = ".NS"
EQUITY_LIST_FILE = "EQUITY_L.csv"  # nseindia.com: the equity segment's list of securities available for trading
INDEX_LIST_FILES: Dict[int, str] = {  # niftyindices.com: each index's page, "download constituents"
    50: "ind_nifty50list.csv", 100: "ind_nifty100list.csv", 500: "ind_nifty500list.csv"}
STALE_LIST_DAYS = 60  # an older stock list misses new listings; warn, but still use it


class NseFileMissing(RuntimeError):
    """A file the user downloads by hand from NSE is not in the folder (or is unusable)."""


def nse_files_dir() -> Path:
    """Where the hand-downloaded NSE files live (NSE_FILES_DIR, default ./nse_files)."""
    return Path(os.getenv("NSE_FILES_DIR", "nse_files"))


def read_nse_file(name: str, directory: Optional[Path] = None) -> str:
    """The text of one hand-downloaded NSE file, or NseFileMissing saying exactly what to download and where."""
    path = (directory or nse_files_dir()) / name
    if not path.exists():
        raise NseFileMissing(f"{path} not found: download {name} from NSE's website in a browser and save it there "
                             "(the bot never downloads from NSE sites itself; see README, 'Files you download')")
    age = (time.time() - path.stat().st_mtime) / 86400
    if age > STALE_LIST_DAYS:
        log.warning("%s is %.0f days old: download a fresh copy so new listings are included", path, age)
    return path.read_text(encoding="utf-8-sig")


def _parse_symbols(csv_text: str) -> List[str]:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = pd.Index([c.strip().upper() for c in df.columns])
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]  # skips trade-to-trade (BE/BZ) segments
    return sorted(df["SYMBOL"].astype(str).str.strip().unique())


# Yahoo lists other NSE segments under the same exchange code: SME platform ("-SM"), InvITs ("-IV") and a few series
# tickers. Real main-board symbols can contain hyphens too (BAJAJ-AUTO, NAM-INDIA, KLBRENG-B), so only these exact
# suffixes are dropped. NSE's own dummy "NSETEST" tickers are dropped as well.
YAHOO_EXCLUDED_SUFFIXES = ("-SM", "-IV", "-BL", "-RR", "-E1")


def yahoo_nse_symbols(min_traded_value: float = 0.0, top: Optional[int] = None,
                      screen: Optional[Callable] = None, page: int = 250, pause: float = 0.5) -> List[str]:
    """NSE main-board stocks from Yahoo Finance's screener (the same source as every price the bot uses). No manual step and no request to NSE's sites. Compared with NSE's own list on 2026-09-26 it
    covered all 2,317 EQ-series stocks; it also includes some trade-to-trade (BE) stocks, which the scanner's own
    liquidity, volatility and trend filters then judge like any other. ETFs, SME and other segments are dropped.

    `min_traded_value` drops stocks whose Yahoo 3-month average daily value is below HALF of it (a cheap pre-cut that
    keeps the daily bar download small; the scanner applies the exact rule on the bars). A stock Yahoo gives no volume
    for is kept, never guessed away. `top` keeps the N most traded by average daily value (100 stands in for the
    Nifty 100). Not by market cap: Yahoo leaves it empty for some of the largest stocks (RELIANCE and TCS on
    2026-09-26), which sorts them near the end of the list."""
    if screen is None:
        import yfinance as yf
        from yfinance import EquityQuery

        query = EquityQuery("eq", ["exchange", "NSI"])

        def screen(offset):
            return yf.screen(query, size=page, offset=offset, sortField="intradaymarketcap", sortAsc=False)
    quotes: list = []
    while True:
        r = retry_call(lambda: screen(len(quotes)), attempts=3, delay=2.0, what="Yahoo NSE stock list")
        got = r.get("quotes", [])
        quotes += got
        if not got or len(quotes) >= r.get("total", 0):
            break
        time.sleep(pause)
    kept = []
    for q in quotes:
        sym = str(q.get("symbol", ""))
        if not sym.endswith(SUFFIX):
            continue
        sym = sym[:-len(SUFFIX)]
        if sym.endswith(YAHOO_EXCLUDED_SUFFIXES) or "NSETEST" in sym:
            continue
        # Yahoo types ETFs as equities too. Funds carry no financial reporting currency: measured 2026-09-26, missing on
        # all 309 named ETFs/BeES but on only 10 of NSE's 2,317 EQ stocks (2 of them liquid). Stocks, not funds.
        if not q.get("financialCurrency"):
            continue
        volume, price = q.get("averageDailyVolume3Month"), q.get("regularMarketPrice")
        value = volume * price if volume and price else None
        if min_traded_value and value is not None and value < min_traded_value / 2:
            continue
        kept.append((sym, value or 0.0))
    if top is not None:
        kept = sorted(kept, key=lambda k: k[1], reverse=True)[:top]
    return [sym for sym, _ in kept]


def load_nse_symbols(directory: Optional[Path] = None, min_traded_value: float = 0.0,
                     yahoo: Optional[Callable[[float], List[str]]] = None) -> List[str]:
    """The stocks the whole-market scan covers. A hand-saved EQUITY_L.csv wins when present (NSE's own list); otherwise
    Yahoo's NSE list, fetched automatically; the hand-saved Nifty 500 file is the last resort. Raises NseFileMissing (a
    RuntimeError, so the pipeline's scan-failure alert fires) when none of them is usable."""
    try:
        return _nse_symbols_from_files(directory, (EQUITY_LIST_FILE,))
    except NseFileMissing:
        pass
    try:
        symbols = (yahoo or (lambda mtv: yahoo_nse_symbols(mtv)))(min_traded_value)
        if len(symbols) > 100:
            log.info("Yahoo NSE list: %d symbols", len(symbols))
            return symbols
        log.warning("Yahoo's NSE list returned only %d symbols; ignoring it", len(symbols))
    except Exception as e:
        log.warning("Yahoo's NSE list unavailable (%s: %s)", type(e).__name__, e)
    return _nse_symbols_from_files(directory, (INDEX_LIST_FILES[500],))


def _nse_symbols_from_files(directory: Optional[Path], names) -> List[str]:
    """The first usable stock list among the hand-saved files `names`."""
    for name in names:
        try:
            symbols = _parse_symbols(read_nse_file(name, directory))
        except NseFileMissing:
            continue
        except Exception as e:  # a half-saved or wrong file: say so and try the next one
            log.warning("%s could not be read (%s: %s)", name, type(e).__name__, e)
            continue
        if len(symbols) > 100:
            log.info("%s: %d symbols", name, len(symbols))
            return symbols
        log.warning("%s lists only %d symbols; ignoring it", name, len(symbols))
    raise NseFileMissing(f"no NSE stock list: Yahoo's list was unavailable and there is no usable {' or '.join(names)} "
                         f"in {directory or nse_files_dir()} (a hand-saved copy from NSE's website works too)")


def load_index_symbols(index: int = 500, directory: Optional[Path] = None) -> List[str]:
    """Constituents of a Nifty index as NSE symbols, from its hand-downloaded list (today's membership, so historical
    tests carry survivorship bias). Raises NseFileMissing (not a raw parsing error) both when the file is absent and
    when it is present but unusable (half-saved, an HTML error page saved with a .csv extension, ...), so a caller's
    NseFileMissing fallback (e.g. run_intraday's Yahoo fallback) is never bypassed by a malformed file."""
    name = INDEX_LIST_FILES[index]
    text = read_nse_file(name, directory)  # NseFileMissing here already, if the file is absent
    try:
        return _parse_symbols(text)
    except Exception as e:  # a half-saved or wrong file: same treatment as load_nse_symbols' own file reading
        raise NseFileMissing(f"{name} could not be read ({type(e).__name__}: {e}); "
                             "download a fresh copy from NSE's website") from e


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
        frames, zero_volume, flagged = [], 0, []
        top = set(wide.columns.get_level_values(0))
        for s in symbols:
            ticker = s + SUFFIX
            if ticker not in top:
                continue
            sub = wide[ticker][["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
            if sub.empty:
                continue
            for finding in check_bars_honesty(sub):
                if finding == ZERO_VOLUME:
                    zero_volume += 1
                else:
                    flagged.append(f"{s}: {finding}")
            sub.index = pd.MultiIndex.from_product([[s], sub.index], names=["symbol", "timestamp"])
            frames.append(sub)
        # One summary per batch instead of one warning per stock: the zero-volume line alone was 83% of a 3-day log.
        if zero_volume:
            log.info("data check: %d of %d symbols have zero-volume days (normal for thinly traded stocks)",
                     zero_volume, len(symbols))
        if flagged:
            log.warning("data check: %d finding(s) that make indicators unreliable for those stocks: %s%s", len(flagged),
                        "; ".join(flagged[:8]), f"; ... and {len(flagged) - 8} more" if len(flagged) > 8 else "")
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
                 clock: Callable[[], float] = time.monotonic, retry_sleep: Callable[[float], None] = time.sleep):
        # Settling exits, valuing positions and filling an order all need the same bars within seconds of each other.
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._lock, self._ttl, self._clock = threading.Lock(), ttl_seconds, clock
        self._retry_sleep = retry_sleep
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
        df = retry_call(lambda: self._download(symbol + SUFFIX, period="5d", interval="5m"), attempts=2, delay=1.0,
                        sleep=self._retry_sleep, what=f"5-minute bars for {symbol}")
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
        index = pd.DatetimeIndex(bars.index)
        idx = index.tz_convert("UTC") if index.tz else index.tz_localize(IST).tz_convert("UTC")
        cutoff = pd.Timestamp(since)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return bars[idx > cutoff]
