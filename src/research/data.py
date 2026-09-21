"""Multi-year daily data for research: adjusted close and volume for an index's stocks, plus the Nifty 50 index."""
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

from src.data.india import SUFFIX, fetch_index_symbols

log = logging.getLogger(__name__)


def _default_download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker", auto_adjust=True,
                       progress=False, threads=True)


def load_index_close(ticker: str, years: int = 10, cache_dir: Path = Path("research_cache"), refresh: bool = False,
                     download: Optional[Callable] = None, today: Callable[[], date] = date.today) -> pd.Series:
    """Close series for a single Yahoo ticker such as ^CRSLDX (Nifty 500 index). Cached on disk."""
    cache = cache_dir / f"index_{ticker.strip('^')}_{years}y.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    download = download or _default_download
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    raw = download([ticker], start, today().isoformat())
    close = (raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw)["Close"].dropna()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(close))
    return close


def load_universe(index: int = 500, years: int = 10, cache_dir: Path = Path("research_cache"), refresh: bool = False,
                  download: Optional[Callable] = None, symbols: Optional[List[str]] = None, chunk: int = 100,
                  today: Callable[[], date] = date.today) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns (close, volume, nifty50_close). Cached on disk; symbols Yahoo cannot resolve are skipped."""
    cache = cache_dir / f"nifty{index}_{years}y.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    download = download or _default_download
    symbols = symbols if symbols is not None else fetch_index_symbols(index)
    end = today().isoformat()
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    closes, volumes = {}, {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        wide = download([s + SUFFIX for s in batch], start, end)
        if wide is None or wide.empty:
            continue
        top = set(wide.columns.get_level_values(0))
        for s in batch:
            t = s + SUFFIX
            if t in top:
                sub = wide[t].dropna(subset=["Close"])
                if len(sub):
                    closes[s], volumes[s] = sub["Close"], sub["Volume"]
        log.info("downloaded %d/%d symbols", min(i + chunk, len(symbols)), len(symbols))
    nifty = download(["^NSEI"], start, end)
    nifty_close = (nifty["^NSEI"] if isinstance(nifty.columns, pd.MultiIndex) else nifty)["Close"].dropna()
    close, volume = pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()
    close, volume = close.reindex(nifty_close.index), volume.reindex(nifty_close.index)  # one shared trading calendar
    result = (close, volume.fillna(0.0), nifty_close)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    return result
