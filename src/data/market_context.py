"""Market context: the regime the single-stock analysis sits inside, for the agent's context check.

The bot's own research (STRATEGY.md) found that on a decade of Nifty 50 data almost all of the edge lived in holding
stocks while the index was above its 200-day average. This gives the AI that same lens per decision: Nifty 50
above/below its 200-day MA, index momentum and RSI, the VIX and where today's level sits in the past year, and the
stock's return relative to the index over the last 6 months.

Honesty notes:
  * The daily index/VIX series come from Yahoo (same source as the stock bars) and are cached on disk under
    research_cache; any failure returns None so the analysis proceeds without the context step instead of erroring.
  * The provider memoises the index/VIX series in memory, so the whole stock cycle shares one download.
  * VIX percentile needs ~a year of history; below that it is reported as None (neutral), never as a made-up level."""
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.data.indicators import Snapshot, compute_rsi
from src.data.price_history import load_index_close

log = logging.getLogger(__name__)

INDEX_TICKER = "^NSEI"  # Nifty 50
VIX_TICKER = "^INDIAVIX"
CACHE_MAX_AGE_HOURS = 12.0  # the live bot re-downloads the index/VIX series at most this stale


@dataclass(frozen=True)
class MarketContext:
    """What the stock analysis happens against, as of the latest completed bars."""
    index_name: str = "NIFTY 50"
    index_price: Optional[float] = None
    index_ma200: Optional[float] = None
    index_above_ma200: Optional[bool] = None
    index_rsi: Optional[float] = None
    index_ytd: Optional[float] = None
    index_mom_6m: Optional[float] = None
    vix: Optional[float] = None
    vix_percentile: Optional[float] = None  # 0-1, where today's VIX sits in the last year's range
    vs_index_6m: Optional[float] = None  # the stock's 6-month return minus the index's (stock price data available)
    # Extras from the review: each defaults to None, meaning "no data source - treat as neutral". The provider
    # fills what it can honestly (the beta proxy from real return data); sector/earnings/correlation need feeds the
    # bot does not have yet, so they stay None and the prompt tells the model to answer neutrally.
    beta_6m: Optional[float] = None  # crude 6-month momentum ratio (stock 6m return / index 6m return); NOT a regression beta
    sector_trend: Optional[str] = None  # e.g. "+8.1% over 3 months"; None when there is no sector feed
    earnings_days_until: Optional[int] = None  # sessions until the next earnings event; None when unknown
    correlation_with_index: Optional[float] = None  # None when the return series needed is unavailable

    @property
    def regime_label(self) -> str:
        """A short stable label for the record: bull / bear / risk_off."""
        if (self.vix_percentile is not None and self.vix_percentile >= 0.8) or self.index_above_ma200 is False:
            return "risk_off"
        return "bull" if self.index_above_ma200 else "bear"

    def is_notable(self) -> bool:
        """True only when the regime is worth an extra LLM call: stress (VIX spike, index below the 200-day) or a
        euphoric melt-up. A normal regime needs no adjustment and no extra call."""
        risk_off = (self.vix_percentile is not None and self.vix_percentile >= 0.8) or self.index_above_ma200 is False
        euphoric = self.index_rsi is not None and self.index_rsi >= 75
        return risk_off or euphoric


class MarketContextProvider:
    """Builds a MarketContext per stock. Index/VIX series are fetched once (Yahoo, disk-cached) and reused.

    `download` is injectable for tests; it must match the (tickers, start, end) signature used by the data layer."""

    def __init__(self, index_ticker: str = INDEX_TICKER, vix_ticker: str = VIX_TICKER, years: int = 3,
                 cache_dir: Path = Path("research_cache"), download: Optional[Callable] = None,
                 today: Callable[[], date] = date.today):
        self._index_ticker, self._vix_ticker = index_ticker, vix_ticker
        self._years, self._cache_dir, self._today = years, cache_dir, today
        self._download = download
        self._lock = threading.Lock()
        self._memo: dict = {}  # ticker -> (monotonic load time, series or None)

    def _load(self, ticker: str) -> Optional[pd.Series]:
        return load_index_close(ticker, years=self._years, cache_dir=self._cache_dir, download=self._download,
                                today=self._today,
                                max_age_hours=CACHE_MAX_AGE_HOURS)

    def _series(self, ticker: str) -> Optional[pd.Series]:
        """The series for a ticker, memoised in memory for CACHE_MAX_AGE_HOURS so a long-running bot re-reads it daily."""
        with self._lock:
            hit = self._memo.get(ticker)
            if hit is not None and time.monotonic() - hit[0] < CACHE_MAX_AGE_HOURS * 3600:
                return hit[1]
            try:
                series = self._load(ticker)
            except Exception as e:
                if ticker == self._index_ticker:
                    raise
                log.warning("no VIX data (%s); VIX context will be neutral", type(e).__name__)
                series = None
            self._memo[ticker] = (time.monotonic(), series)
            return series

    def context(self, symbol: str, snapshot: Snapshot) -> Optional[MarketContext]:
        """The market context for one stock, or None when the index cannot be loaded (analysis continues unchanged)."""
        try:
            nifty = self._series(self._index_ticker)
        except Exception as e:
            log.warning("no index data for market context: %s: %s", type(e).__name__, e)
            return None
        if nifty is None or len(nifty) < 200:
            return None
        last, ma200 = float(nifty.iloc[-1]), float(nifty.tail(200).mean())
        years_ended = nifty[pd.DatetimeIndex(nifty.index).year == self._today().year]
        ytd = last / float(years_ended.iloc[0]) - 1 if len(years_ended) >= 2 else None
        vix = self._series(self._vix_ticker)
        vix_last = float(vix.iloc[-1]) if vix is not None and len(vix) else None
        vix_pct = None
        if vix is not None and len(vix) >= 30:
            vix_pct = float(vix.tail(252).rank(pct=True).iloc[-1])  # the last year, as the prompt says
        stock_mom = snapshot.momentum_6m
        index_mom = float(nifty.iloc[-1] / nifty.iloc[-127] - 1) if len(nifty) > 126 else None
        # Crude but real: the ratio of the two six-month returns (honest label, not a regression beta). When the
        # index is flat this proxy is meaningless, so it is None and the prompt says "no beta feed".
        beta_6m = None
        if stock_mom is not None and index_mom is not None and abs(index_mom) > 0.005:
            beta_6m = stock_mom / index_mom
        return MarketContext(
            index_price=last,
            index_ma200=ma200,
            index_above_ma200=last > ma200,
            index_rsi=compute_rsi(nifty.tail(300)),
            index_ytd=ytd,
            index_mom_6m=index_mom,
            vix=vix_last,
            vix_percentile=vix_pct,
            vs_index_6m=(stock_mom - index_mom) if stock_mom is not None and index_mom is not None else None,
            beta_6m=beta_6m,
            # sector trend, earnings proximity and correlation need feeds the bot does not have; stay None (neutral).
        )