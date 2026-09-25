"""NSE daily delivery percentage (share of traded quantity that was actually taken for delivery, not squared off intraday).

Source: NSE's public full bhavcopy, one CSV per trading day. Weekends and holidays return 404 and are skipped."""
import io
import logging
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd

log = logging.getLogger(__name__)
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"


def _fetch_day(day: date, client: httpx.Client) -> Optional[str]:
    """The CSV text for one day, or None when NSE has no file (holiday/weekend)."""
    r = client.get(URL.format(d=day.strftime("%d%m%Y")))
    return r.text if r.status_code == 200 else None


def parse_bhavcopy(text: str) -> pd.Series:
    """Symbol -> delivery percent for the EQ series of one day's file."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = pd.Index([c.strip() for c in df.columns])
    df = df[df["SERIES"].str.strip() == "EQ"]
    return pd.to_numeric(df.set_index("SYMBOL")["DELIV_PER"], errors="coerce")


def load_delivery(years: float = 3, cache_dir: Path = Path("research_cache"), refresh: bool = False,
                  fetch_day: Optional[Callable[[date], Optional[str]]] = None, today: Callable[[], date] = date.today,
                  pause: float = 0.3) -> pd.DataFrame:
    """Dates x symbols delivery percent. Cached on disk; only days missing from the cache are downloaded."""
    cache = cache_dir / f"delivery_{years}y.pkl"
    have = pickle.loads(cache.read_bytes()) if cache.exists() and not refresh else pd.DataFrame()
    known = {d.date() for d in have.index} if len(have) else set()
    client = None
    if fetch_day is None:
        client = httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
        fetch_day = lambda d: _fetch_day(d, client)  # noqa: E731
    rows, day = {}, today() - timedelta(days=int(years * 365.25))
    while day < today():
        if day.weekday() < 5 and day not in known:
            try:
                text = fetch_day(day)
            except Exception as e:  # a network blip costs one day, not the whole download
                log.warning("delivery %s failed: %s", day, type(e).__name__)
                text = None
            if text:
                rows[pd.Timestamp(day)] = parse_bhavcopy(text)
            time.sleep(pause) if client else None
        day += timedelta(days=1)
    result = pd.concat([have, pd.DataFrame(rows).T]).sort_index() if rows else have
    result = result[~result.index.duplicated(keep="last")]
    if len(result):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(result))
    return result
