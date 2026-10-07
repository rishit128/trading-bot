"""Per-stock market data and news fetching (Yahoo Finance)."""
import logging
import threading
from datetime import date, time
from typing import Callable, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from src.data.retry import retry_call
from src.data.indicators import MAX_STALE_DAYS, BarNotReadyError, Snapshot, StaleDataError, build_snapshot

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MIN_SESSION_BARS = 60  # of the 75 five-minute bars in an NSE session (09:15-15:30)
# The newest 5-minute bar must start at or after this, so the session ran to its close. Yahoo's 5-minute series ends at
# 15:10-15:15 for some large caps and 15:25 for others; its last close matched the official daily close to within 0.2%
# on every stock checked (2026-10-05/06).
LAST_BAR_START = time(15, 10)


def session_bar_from_intraday(bars: Optional[pd.DataFrame], day: date) -> Optional[pd.DataFrame]:
    """`day`'s daily bar (Open/High/Low/Close/Volume) built from its 5-minute bars, or None unless they cover the whole
    session. Yahoo posts some NSE daily bars only by ~14:20 IST the next day; the 5-minute bars are there at once."""
    if bars is None or bars.empty:
        return None
    index = pd.DatetimeIndex(bars.index)
    local = index.tz_convert(IST) if index.tz is not None else index.tz_localize(IST)
    mask = local.date == day
    session, times = bars[mask], local[mask]
    if len(session) < MIN_SESSION_BARS or times.max().time() < LAST_BAR_START:
        return None
    return pd.DataFrame({"Open": [float(session["Open"].iloc[0])], "High": [float(session["High"].max())],
                         "Low": [float(session["Low"].min())], "Close": [float(session["Close"].iloc[-1])],
                         "Volume": [float(session["Volume"].sum())]}, index=pd.DatetimeIndex([pd.Timestamp(day)]))


def fetch_snapshot(symbol: str, suffix: str = "", as_of: Optional[Callable[[], date]] = None,
                   download: Callable = yf.download, timeout: float = 30.0,
                   today: Callable[[], date] = date.today,
                   intraday: Optional[Callable[[str], Optional[pd.DataFrame]]] = None) -> Snapshot:
    """suffix is the Yahoo exchange suffix, e.g. '.NS' for NSE; the snapshot keeps the plain symbol.

    as_of returns the last COMPLETED session date; later (still-forming) bars are dropped, so the indicators - and the
    AI prompt built from them - are identical all session and match how the backtest saw the data. When that session's
    daily bar is not published yet, `intraday(symbol)` (recent 5-minute bars) can complete it; failing that, the stock
    waits (BarNotReadyError) rather than being decided on an older bar."""
    bars = download(symbol + suffix, period="2y", interval="1d", progress=False, auto_adjust=True, timeout=timeout)
    if bars.empty:
        raise ValueError(f"{symbol}: no market data returned")
    if hasattr(bars.columns, "levels"):
        bars.columns = bars.columns.get_level_values(0)
    bars = bars.dropna(subset=["Close"])
    if as_of is not None:
        idx = bars.index.tz_localize(None) if bars.index.tz is not None else bars.index
        bars = bars[idx.normalize() <= pd.Timestamp(as_of())]
    snap = build_snapshot(symbol, bars)
    expected = as_of() if as_of is not None else today()
    if snap.bar_date is not None and (expected - date.fromisoformat(snap.bar_date)).days > MAX_STALE_DAYS:
        raise StaleDataError(f"{symbol}: last bar {snap.bar_date} is more than {MAX_STALE_DAYS} days before {expected} (suspended?)")
    if snap.volume <= 0:
        raise StaleDataError(f"{symbol}: last bar {snap.bar_date} shows zero volume (no trading)")
    if as_of is not None and snap.bar_date is not None and date.fromisoformat(snap.bar_date) < expected:
        completed = None
        if intraday is not None:
            try:
                completed = session_bar_from_intraday(intraday(symbol), expected)
            except Exception as e:
                log.warning("%s: could not build the %s bar from 5-minute data: %s", symbol, expected, e)
        if completed is None:
            raise BarNotReadyError(f"{symbol}: waiting for the {expected} bar (latest published: {snap.bar_date})")
        if bars.index.tz is not None:
            completed.index = pd.DatetimeIndex(completed.index).tz_localize(bars.index.tz)
        snap = build_snapshot(symbol, pd.concat([bars, completed]))
    return snap



def cached_per_session(fetch: Callable[[str], Snapshot], session: Callable[[], date]) -> Callable[[str], Snapshot]:
    """Memoise `fetch(symbol)` for as long as the last completed session stays the same.

    The analysis only ever uses completed daily bars, so a stock's snapshot cannot change until the next session closes;
    without this every 30-minute cycle re-downloaded two years of history for each of ~20 stocks. Failures are never
    cached (the next cycle retries), and yesterday's entries are dropped when the session rolls over."""
    lock = threading.Lock()
    store: Dict[str, Tuple[date, Snapshot]] = {}

    def get(symbol: str) -> Snapshot:
        today = session()
        with lock:
            hit = store.get(symbol)
        if hit is not None and hit[0] == today:
            return hit[1]
        snapshot = fetch(symbol)
        with lock:
            for stale in [k for k, (d, _) in store.items() if d != today]:
                del store[stale]
            store[symbol] = (today, snapshot)
        return snapshot

    return get


def fetch_daily_bars(symbol: str, suffix: str = "", download: Callable = yf.download, period: str = "2y",
                     timeout: float = 30.0) -> Optional[pd.DataFrame]:
    """A stock's daily Open/High/Low/Close (adjusted, oldest first), or None when Yahoo has nothing for it.

    Used to label what happened AFTER a decision, so unlike the analysis snapshot it deliberately includes the newest
    completed bars. A transient failure is retried; a definite empty answer is not."""
    def fetch() -> pd.DataFrame:
        return download(symbol + suffix, period=period, interval="1d", progress=False, auto_adjust=True, timeout=timeout)

    bars = retry_call(fetch, attempts=2, delay=1.0, what=f"daily bars for {symbol}")
    if bars is None or bars.empty:
        return None
    if hasattr(bars.columns, "levels"):
        bars.columns = bars.columns.get_level_values(0)
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    return bars[["Open", "High", "Low", "Close"]] if len(bars) else None
