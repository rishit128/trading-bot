"""Each stock's sector (Yahoo Finance's eleven sectors: Financial Services, Technology, Healthcare, ...), kept in a cache.

The intraday accounts use it to see which sectors are rising today (src.intraday.strategy.sector_gains). Yahoo is asked one
stock at a time, so the stocks the cache does not hold yet are looked up in a background thread and the shared dict fills in as
answers arrive; until then, and for a stock Yahoo has no sector for, no sector check applies. NSE's own industry list is not
used: NSE's terms forbid automated downloads (src/data/india.py)."""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

log = logging.getLogger(__name__)

SECTORS_FILE = "sectors.json"
SECTORS_MAX_AGE = 90 * 86400  # seconds: a company's sector almost never changes
PACE = 0.5  # seconds between two Yahoo requests
SAVE_EVERY = 25


def yahoo_broad_sector(symbol: str) -> Optional[str]:
    """An NSE company's sector from Yahoo Finance, or None (never raises)."""
    try:
        import yfinance as yf

        sector = (yf.Ticker(symbol + ".NS").info or {}).get("sector")
        return str(sector) if sector else None
    except Exception as e:
        log.warning("no sector for %s: %s: %s", symbol, type(e).__name__, e)
        return None


def fill_sectors(
    sectors: Dict[str, str],
    symbols: Sequence[str],
    path: Path,
    lookup: Callable[[str], Optional[str]] = yahoo_broad_sector,
    pace: float = PACE,
) -> None:
    """Look up every stock of `symbols` that `sectors` lacks, adding each answer to it and saving as it goes."""
    missing = [s for s in symbols if s not in sectors]
    for i, symbol in enumerate(missing, 1):
        sector = lookup(symbol)
        if sector:
            sectors[symbol] = sector
        if i % SAVE_EVERY == 0 or i == len(missing):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sectors), encoding="utf-8")
        if pace:
            time.sleep(pace)
    if missing:
        log.info("sector list: %d of %d stocks now have a sector", sum(s in sectors for s in symbols), len(symbols))


def load_sectors(
    cache_dir: Path, symbols: Sequence[str], lookup: Callable[[str], Optional[str]] = yahoo_broad_sector, background: bool = True
) -> Dict[str, str]:
    """Symbol -> sector from the cache (dropped and rebuilt when older than SECTORS_MAX_AGE). The stocks it lacks are looked up
    in a background thread that fills the returned dict; `background=False` waits for them. Never raises."""
    path = cache_dir / SECTORS_FILE
    sectors: Dict[str, str] = {}
    try:
        if path.exists() and time.time() - path.stat().st_mtime < SECTORS_MAX_AGE:
            sectors = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("could not read the sector cache (%s: %s); rebuilding it", type(e).__name__, e)
    if any(s not in sectors for s in symbols):
        if background:
            threading.Thread(target=fill_sectors, args=(sectors, symbols, path, lookup), daemon=True, name="sectors").start()
        else:
            fill_sectors(sectors, symbols, path, lookup, pace=0.0)
    return sectors
