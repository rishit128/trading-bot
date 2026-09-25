"""Per-stock market data and news fetching (Yahoo Finance)."""
import logging
import threading
from datetime import date
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

from src.data.indicators import MAX_STALE_DAYS, Snapshot, StaleDataError, build_snapshot

log = logging.getLogger(__name__)


def fetch_snapshot(symbol: str, suffix: str = "", as_of: Optional[Callable[[], date]] = None,
                   download: Callable = yf.download, timeout: float = 30.0,
                   today: Callable[[], date] = date.today) -> Snapshot:
    """suffix is the Yahoo exchange suffix, e.g. '.NS' for NSE; the snapshot keeps the plain symbol.

    as_of returns the last COMPLETED session date; later (still-forming) bars are dropped, so the indicators - and the
    AI prompt built from them - are identical all session and match how the backtest saw the data."""
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
