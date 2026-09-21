"""Whole-market scanner. Stage 1 (this module) is cheap deterministic code over every listed NSE stock;
only the few survivors go on to the (rate-limited, free) LLM agents."""
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, List, Optional, Sequence

import pandas as pd

from src.data.indicators import MAX_STALE_DAYS, MIN_BARS, compute_rsi

log = logging.getLogger(__name__)

MOMENTUM_BARS = 253  # bars needed for 12-1 month momentum (252 trading days back, skipping the latest 21)


@dataclass(frozen=True)
class Candidate:
    """A stock that passed the scanner, with the numbers it was ranked on."""
    symbol: str
    price: float
    momentum_63d: float
    rsi: float
    avg_traded_value: float
    daily_volatility: float
    momentum_12_1: Optional[float] = None  # return from 12 months ago to 1 month ago (skips the latest month)

    @property
    def score(self) -> float:
        """Ranking score: 12-1 month momentum when known (the only signal that beat holding everything in the 10-year
        research, see STRATEGY.md); otherwise 63-day return per unit of 63-day volatility."""
        if self.momentum_12_1 is not None:
            return self.momentum_12_1
        return self.momentum_63d / (self.daily_volatility * math.sqrt(63)) if self.daily_volatility > 0 else 0.0


@dataclass(frozen=True)
class ScreenConfig:
    """Scanner thresholds."""
    max_candidates: int = 15
    min_price: float = 10.0
    min_traded_value: float = 20_000_000.0
    max_daily_volatility: float = 0.04


def screen_bars(bars: pd.DataFrame, cfg: ScreenConfig) -> List[Candidate]:
    """bars: MultiIndex (symbol, timestamp) daily OHLCV of COMPLETED sessions. Returns every symbol that passes."""
    found = []
    if bars.empty:
        return found
    newest = bars.index.get_level_values(1).max()
    for symbol, g in bars.groupby(level=0):
        if len(g) < max(MIN_BARS, MOMENTUM_BARS):  # a full year is needed to rank by 12-1 month momentum
            continue
        if (newest - g.index.get_level_values(1)[-1]).days > MAX_STALE_DAYS:
            continue  # data stops early: suspended or delisted, not a tradable candidate
        close = g["Close"].astype(float) if "Close" in g else g["close"].astype(float)
        volume = g["Volume"] if "Volume" in g else g["volume"]
        price = float(close.iloc[-1])
        traded_value = float((close * volume).tail(20).mean())
        if price < cfg.min_price or traded_value < cfg.min_traded_value:
            continue
        volatility = float(close.pct_change().tail(63).std())
        if not volatility <= cfg.max_daily_volatility:  # also rejects NaN
            continue
        ma50, ma200 = float(close.tail(50).mean()), float(close.tail(200).mean())
        if not price > ma50 > ma200:
            continue
        rsi = compute_rsi(close)
        if rsi >= 70:
            continue
        mom_12_1 = float(close.iloc[-22] / close.iloc[-253] - 1)
        found.append(Candidate(symbol, price, price / float(close.iloc[-64]) - 1, rsi, traded_value, volatility, mom_12_1))
    return found


def rank(candidates: Sequence[Candidate], n: int) -> List[Candidate]:
    """The top n candidates by score (12-1 month momentum when available)."""
    return sorted(candidates, key=lambda c: c.score, reverse=True)[:n]


class UniverseScreener:
    """Scans the full market once per day (and caches it); symbols_for() adds anything currently held."""

    def __init__(self, list_symbols: Callable[[], List[str]], fetch_bars: Callable[[List[str]], pd.DataFrame],
                 cfg: ScreenConfig, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
                 chunk: int = 500, sleep: Callable[[float], None] = time.sleep):
        self.list_symbols, self.fetch_bars, self.cfg, self.today, self.chunk = list_symbols, fetch_bars, cfg, today, chunk
        self.sleep = sleep
        self._day: Optional[date] = None
        self._cached: List[Candidate] = []

    def candidates(self) -> List[Candidate]:
        """Today's top candidates; the full scan runs once per day and is cached."""
        if self._day == self.today():
            return self._cached
        symbols = self.list_symbols()
        log.info("screening %d symbols", len(symbols))
        passed: List[Candidate] = []
        for i in range(0, len(symbols), self.chunk):
            passed.extend(screen_bars(self._fetch_with_retry(symbols[i:i + self.chunk]), self.cfg))
        self._cached = rank(passed, self.cfg.max_candidates)
        self._day = self.today()
        log.info("screen: %d passed filters, keeping top %d: %s", len(passed), len(self._cached),
                 ", ".join(c.symbol for c in self._cached))
        return self._cached

    def _fetch_with_retry(self, symbols: List[str], attempts: int = 3) -> pd.DataFrame:
        for attempt in range(1, attempts + 1):
            try:
                return self.fetch_bars(symbols)
            except Exception as e:
                if attempt == attempts:
                    raise
                log.warning("bar fetch failed (attempt %d/%d): %s: %s", attempt, attempts, type(e).__name__, e)
                self.sleep(3 * attempt)

    def symbols_for(self, portfolio) -> List[str]:
        """Current holdings first, then today's candidates: everything to analyse this cycle."""
        held = sorted(portfolio.positions)
        return held + [c.symbol for c in self.candidates() if c.symbol not in portfolio.positions]
