"""The news universe: the 200 most liquid of the intraday universe's stocks, priced Rs 100 or more, with at least Rs 50 crore
traded a day on average (the last 20 sessions, Yahoo daily bars). Refreshed once a day; the watcher and the engine share one
copy, so a filing is kept exactly when its stock can be traded."""

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence

import pandas as pd

from src.data.india import IST, SUFFIX

log = logging.getLogger(__name__)

SIZE, MIN_PRICE, MIN_TRADED = 200, 100.0, 50e7  # Rs 50 crore
SESSIONS = 20


def _yahoo_daily(tickers: List[str]) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers, period="2mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True, timeout=60
    )


def select_news_universe(base: Sequence[str], download: Callable[[List[str]], pd.DataFrame] = _yahoo_daily) -> List[str]:
    """The base stocks that pass the price and liquidity floors, most traded first, at most SIZE."""
    ranked = []
    for start in range(0, len(base), 100):
        chunk = list(base[start : start + 100])
        wide = download([s + SUFFIX for s in chunk])
        if wide is None or wide.empty:
            continue
        top = set(wide.columns.get_level_values(0))
        for s in chunk:
            if s + SUFFIX not in top:
                continue
            df = pd.DataFrame(wide[s + SUFFIX]).dropna(subset=["Close"]).tail(SESSIONS)
            if len(df) < SESSIONS // 2:
                continue
            traded = float((df["Close"] * df["Volume"]).mean())
            if float(df["Close"].iloc[-1]) >= MIN_PRICE and traded >= MIN_TRADED:
                ranked.append((traded, s))
    return [s for _, s in sorted(ranked, reverse=True)[:SIZE]]


class NewsUniverse:
    """Today's news universe, computed at most once per IST date. A failed refresh keeps the last list (or, on the first day,
    falls back to the first SIZE base stocks, which are already ordered by traded value)."""

    def __init__(
        self,
        base: Sequence[str],
        select: Callable[[Sequence[str]], List[str]] = select_news_universe,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.base, self.select, self.now_fn = list(base), select, now_fn
        self._symbols: List[str] = []
        self._day: Optional[str] = None
        self._lock = threading.Lock()

    def symbols(self) -> List[str]:
        today = self.now_fn().astimezone(IST).date().isoformat()
        with self._lock:
            if self._day != today:
                try:
                    chosen = self.select(self.base)
                    if not chosen:
                        raise ValueError("no stock passed the floors")
                    self._symbols = chosen
                    log.info("news universe: %d stocks", len(chosen))
                except Exception as e:
                    log.warning("could not refresh the news universe (%s: %s)", type(e).__name__, e)
                    self._symbols = self._symbols or self.base[:SIZE]
                self._day = today
            return list(self._symbols)
