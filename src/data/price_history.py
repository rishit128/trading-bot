"""Multi-year daily price history (adjusted, disk-cached): an index's stocks and the index itself. Shared by the live bot's
market context and by the research tools, so it lives with the data layer, not in research/."""

import logging

# The pickle files read below are the bot's own caches, written into its own cache directory; never user input.
import pickle  # nosec B403
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from src.data.india import SUFFIX, load_index_symbols, load_nse_symbols

log = logging.getLogger(__name__)


def _default_download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers, start=start, end=end, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True
    )


def load_index_close(
    ticker: str,
    years: int = 10,
    cache_dir: Path = Path("research_cache"),
    refresh: bool = False,
    download: Optional[Callable] = None,
    today: Callable[[], date] = date.today,
    max_age_hours: Optional[float] = None,
) -> pd.Series:
    """Close series for a single Yahoo ticker such as ^CRSLDX (Nifty 500 index). Cached on disk.

    `max_age_hours` makes the cache expire (a live bot must not read a frozen regime); if the re-download then fails the
    stale cache is returned with a warning. Research callers leave it None to keep their results reproducible."""
    cache = cache_dir / f"index_{ticker.strip('^')}_{years}y.pkl"
    stale = None
    if cache.exists() and not refresh:
        cached = pickle.loads(cache.read_bytes())  # nosec B301
        if max_age_hours is None or time.time() - cache.stat().st_mtime < max_age_hours * 3600:
            return cached
        stale = cached
    download = download or _default_download
    try:
        return _download_index(ticker, years, cache, cache_dir, download, today)
    except Exception as e:
        if stale is None:
            raise
        log.warning(
            "could not refresh %s (%s: %s); using the cached series ending %s",
            ticker,
            type(e).__name__,
            e,
            stale.index[-1].date(),
        )
        return stale


def _download_index(ticker, years, cache, cache_dir, download, today) -> pd.Series:
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    raw = download([ticker], start, today().isoformat())
    close = (raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw)["Close"].dropna()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(close))
    return close


def load_ohlc_universe(
    index: int = 500,
    years: int = 10,
    cache_dir: Path = Path("research_cache"),
    refresh: bool = False,
    download: Optional[Callable] = None,
    symbols: Optional[List[str]] = None,
    chunk: int = 100,
    today: Callable[[], date] = date.today,
) -> dict:
    """symbol -> daily OHLCV frame (adjusted, no NaN) for an index's stocks: the event simulator needs Open/High/Low for
    next-open fills and stop checks. Cached on disk."""
    cache = cache_dir / f"ohlc_nifty{index}_{years}y.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())  # nosec B301
    symbols = symbols if symbols is not None else load_index_symbols(index)
    return _download_ohlc(symbols, years, cache, cache_dir, download, chunk, today)


def load_ohlc_nse(
    years: int = 10,
    cache_dir: Path = Path("research_cache"),
    refresh: bool = False,
    download: Optional[Callable] = None,
    list_symbols: Optional[Callable[[], List[str]]] = None,
    chunk: int = 100,
    today: Callable[[], date] = date.today,
) -> dict:
    """symbol -> daily OHLCV for EVERY NSE main-board stock listed today -- the universe the live scanner actually scans
    (~2,300), rather than today's Nifty 500. Removes the index-membership part of the survivorship bias (small caps
    that were never in the index are now included) but not the delisting part: Yahoo has no bars for stocks that have
    since been delisted, so they are still missing. Cached on disk; the first download takes a while."""
    cache = cache_dir / f"ohlc_nse_{years}y.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())  # nosec B301
    return _download_ohlc((list_symbols or load_nse_symbols)(), years, cache, cache_dir, download, chunk, today)


def _download_ohlc(
    symbols: List[str],
    years: int,
    cache: Path,
    cache_dir: Path,
    download: Optional[Callable],
    chunk: int,
    today: Callable[[], date],
) -> dict:
    """Download daily OHLCV for `symbols` in chunks, keep the ones Yahoo has, and cache the dict at `cache`."""
    download = download or _default_download
    end = today().isoformat()
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    out = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i : i + chunk]
        wide = download([s + SUFFIX for s in batch], start, end)
        if wide is None or wide.empty:
            continue
        top = set(wide.columns.get_level_values(0))
        for s in batch:
            if s + SUFFIX in top:
                sub = pd.DataFrame(wide[s + SUFFIX][["Open", "High", "Low", "Close", "Volume"]]).dropna(
                    subset=["Open", "High", "Low", "Close"]
                )
                if len(sub):
                    out[s] = sub.fillna({"Volume": 0.0})
        log.info("downloaded %d/%d symbols", min(i + chunk, len(symbols)), len(symbols))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out
