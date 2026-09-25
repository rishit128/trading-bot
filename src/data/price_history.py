"""Multi-year daily price history (adjusted, disk-cached): an index's stocks and the index itself. Shared by the live bot's
market context and by the research tools, so it lives with the data layer, not in research/."""
import logging
import pickle
import time
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
                     download: Optional[Callable] = None, today: Callable[[], date] = date.today,
                     max_age_hours: Optional[float] = None) -> pd.Series:
    """Close series for a single Yahoo ticker such as ^CRSLDX (Nifty 500 index). Cached on disk.

    `max_age_hours` makes the cache expire (a live bot must not read a frozen regime); if the re-download then fails the
    stale cache is returned with a warning. Research callers leave it None to keep their results reproducible."""
    cache = cache_dir / f"index_{ticker.strip('^')}_{years}y.pkl"
    stale = None
    if cache.exists() and not refresh:
        cached = pickle.loads(cache.read_bytes())
        if max_age_hours is None or time.time() - cache.stat().st_mtime < max_age_hours * 3600:
            return cached
        stale = cached
    download = download or _default_download
    try:
        return _download_index(ticker, years, cache, cache_dir, download, today)
    except Exception as e:
        if stale is None:
            raise
        log.warning("could not refresh %s (%s: %s); using the cached series ending %s", ticker, type(e).__name__, e,
                    stale.index[-1].date())
        return stale


def _download_index(ticker, years, cache, cache_dir, download, today) -> pd.Series:
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    raw = download([ticker], start, today().isoformat())
    close = (raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw)["Close"].dropna()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(close))
    return close


def load_universe(index: int = 500, years: int = 10, cache_dir: Path = Path("research_cache"), refresh: bool = False,
                  download: Optional[Callable] = None, symbols: Optional[List[str]] = None, chunk: int = 100,
                  membership: Optional[dict] = None,
                  today: Callable[[], date] = date.today) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns (close, volume, nifty50_close). Cached on disk; symbols Yahoo cannot resolve are skipped.

    Pass `membership` as a {date: iterable-of-symbols} history to slash survivorship bias: everything each date
    does not list is masked to NaN in the returned frames, so a signal can never select it and it earns no returns.
    Without it the frames use today's members across the whole history (documented bias, measured in
    scripts/research_signals.py's survivorship check)."""
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
                sub = pd.DataFrame(wide[t]).dropna(subset=["Close"])
                if len(sub):
                    closes[s], volumes[s] = sub["Close"], sub["Volume"]
        log.info("downloaded %d/%d symbols", min(i + chunk, len(symbols)), len(symbols))
    nifty = download(["^NSEI"], start, end)
    nifty_close = (nifty["^NSEI"] if isinstance(nifty.columns, pd.MultiIndex) else nifty)["Close"].dropna()
    close, volume = pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()
    close, volume = close.reindex(nifty_close.index), volume.reindex(nifty_close.index)  # one shared trading calendar
    result = (close, volume.fillna(0.0), nifty_close)
    if membership is not None:
        # Survivorship-free variant: never cached, a fresh mask each call.
        return point_in_time(close, volume.fillna(0.0), membership) + (nifty_close,)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    return result


def point_in_time(close: pd.DataFrame, volume: pd.DataFrame,
                  membership: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mask price/volume frames to what a researcher could actually have seen.

    Data availability is already point-in-time: a symbol's rows before it listed and after it delisted are NaN in the
    raw download, so it can neither be selected nor earn returns there. The survivorship hazard is *today's members
    kept for the whole history*, which `membership` fixes: pass {date: iterable-of-symbols} and every date that does
    not list a symbol masks that symbol to NaN (it may still trade, it is just no longer in the studied universe).

    Returns masked copies of `close` and `volume` on the same index/columns. With membership=None the frames are
    returned as-is (documented bias, measured in scripts/research_signals.py)."""
    if membership is None:
        return close.copy(), volume.copy()
    hist = pd.Series({pd.Timestamp(d): frozenset(s) for d, s in sorted(membership.items())})
    hist = hist.reindex(close.index).ffill()
    active = hist.dropna()
    if active.empty:
        nothing = pd.DataFrame(False, index=close.index, columns=close.columns)  # nothing is a member, so everything is masked
        return close.where(nothing), volume.where(nothing)
    members = set().union(*active)
    allowed = {col: hist.apply(lambda s: col in s) for col in close.columns if col in members}
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    mask.update(pd.DataFrame(allowed))
    return close.where(mask), volume.where(mask)


def load_ohlc_universe(index: int = 500, years: int = 10, cache_dir: Path = Path("research_cache"),
                       refresh: bool = False, download: Optional[Callable] = None,
                       symbols: Optional[List[str]] = None, chunk: int = 100,
                       today: Callable[[], date] = date.today) -> dict:
    """symbol -> daily OHLCV frame (adjusted, no NaN) for an index's stocks; the event simulator needs Open/High/Low for
    next-open fills and stop checks, which `load_universe` (Close and Volume only) does not keep. Cached on disk."""
    cache = cache_dir / f"ohlc_nifty{index}_{years}y.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    download = download or _default_download
    symbols = symbols if symbols is not None else fetch_index_symbols(index)
    end = today().isoformat()
    start = (today() - timedelta(days=int(years * 365.25))).isoformat()
    out = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        wide = download([s + SUFFIX for s in batch], start, end)
        if wide is None or wide.empty:
            continue
        top = set(wide.columns.get_level_values(0))
        for s in batch:
            if s + SUFFIX in top:
                sub = pd.DataFrame(wide[s + SUFFIX][["Open", "High", "Low", "Close", "Volume"]]).dropna(
                    subset=["Open", "High", "Low", "Close"])
                if len(sub):
                    out[s] = sub.fillna({"Volume": 0.0})
        log.info("downloaded %d/%d symbols", min(i + chunk, len(symbols)), len(symbols))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out
