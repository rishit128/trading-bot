"""Whole-market scanner. Stage 1 (this module) is cheap deterministic code over every listed NSE stock;
only the few survivors go on to the (rate-limited, free) LLM agents."""
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.indicators import MAX_STALE_DAYS, MIN_BARS, rsi_from_array

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
    # One share must fit in this fraction of equity (the smallest position the risk engine will size). On a small
    # account a Rs 9,000 stock can never be bought, yet it would still be ranked, analysed by the AI and reported;
    # None disables the rule (large accounts, the plain scanner report).
    affordable_pct: Optional[float] = None


def screen_arrays(symbol: str, close: np.ndarray, volume: np.ndarray, last_bar: pd.Timestamp, newest: pd.Timestamp,
                  cfg: ScreenConfig) -> Optional[Candidate]:
    """The scanner's verdict on ONE stock from plain arrays (completed daily closes and volumes, oldest first, no NaN
    closes): a Candidate, or None. `last_bar` is the date of the newest bar; `newest` the freshest bar in the whole scan,
    to spot stocks whose data stopped.

    This is the single implementation of the entry filter: the live scanner (`screen_bars`) and the live-configuration
    backtest both end up here, so what is tested is what runs. Arrays, not Series: the backtest calls it millions of
    times and pandas overhead dominated the cost."""
    if len(close) < max(MIN_BARS, MOMENTUM_BARS):  # a full year is needed to rank by 12-1 month momentum
        return None
    if (newest - last_bar).days > MAX_STALE_DAYS:
        return None  # data stops early: suspended or delisted, not a tradable candidate
    price = float(close[-1])
    traded_value = float(np.nanmean(close[-20:] * volume[-20:]))
    if price < cfg.min_price or traded_value < cfg.min_traded_value:
        return None
    recent = close[-64:]
    volatility = float(np.std(recent[1:] / recent[:-1] - 1.0, ddof=1))
    if not volatility <= cfg.max_daily_volatility:  # also rejects NaN
        return None
    ma50, ma200 = float(close[-50:].mean()), float(close[-200:].mean())
    if not price > ma50 > ma200:
        return None
    rsi = rsi_from_array(close)
    if rsi >= 70:
        return None
    mom_12_1 = float(close[-22] / close[-253] - 1)
    return Candidate(symbol, price, price / float(close[-64]) - 1, rsi, traded_value, volatility, mom_12_1)


def screen_symbol(symbol: str, close: pd.Series, volume: pd.Series, newest: pd.Timestamp,
                  cfg: ScreenConfig) -> Optional[Candidate]:
    """`screen_arrays` for one stock's Series (completed daily bars, oldest first, no NaN closes)."""
    if len(close) == 0:
        return None
    return screen_arrays(symbol, close.to_numpy(dtype=float), volume.to_numpy(dtype=float), close.index[-1], newest, cfg)


def screen_bars(bars: pd.DataFrame, cfg: ScreenConfig) -> List[Candidate]:
    """bars: MultiIndex (symbol, timestamp) daily OHLCV of COMPLETED sessions. Returns every symbol that passes."""
    found = []
    if bars.empty:
        return found
    newest = bars.index.get_level_values(1).max()
    for symbol, g in bars.groupby(level=0):
        close = (g["Close"] if "Close" in g else g["close"]).astype(float)
        close.index = g.index.get_level_values(1)
        volume = (g["Volume"] if "Volume" in g else g["volume"]).copy()
        volume.index = close.index
        candidate = screen_symbol(symbol, close, volume, newest, cfg)
        if candidate is not None:
            found.append(candidate)
    return found


def rank(candidates: Sequence[Candidate], n: int) -> List[Candidate]:
    """The top n candidates by score (12-1 month momentum when available)."""
    return sorted(candidates, key=lambda c: c.score, reverse=True)[:n]


def select_candidates(ranked: Sequence[Candidate], n: int, equity: Optional[float] = None,
                      affordable_pct: Optional[float] = None) -> List[Candidate]:
    """The top n of an already-ranked list that the account can actually buy at least one share of.

    Shared by the live screener and the live-configuration backtest. With no equity or rule every candidate qualifies."""
    if equity is not None and affordable_pct is not None:
        ranked = [c for c in ranked if c.price <= equity * affordable_pct]
    return list(ranked[:n])


def delivery_average(deliv: pd.DataFrame, window: int = 20, min_days: int = 10) -> pd.DataFrame:
    """Rolling average delivery % per symbol (dates x symbols): the series the delivery filter ranks on."""
    return deliv.rolling(window, min_periods=min_days).mean()


def above_median_delivery(avg: pd.DataFrame, as_of: date) -> Optional[set]:
    """Symbols above that day's cross-sectional median of `avg` (latest row on or before as_of), or None when no row exists."""
    usable = avg.index[avg.index <= pd.Timestamp(as_of)]
    if len(usable) == 0:
        return None
    row = avg.loc[usable[-1]]
    return set(row[row > row.median()].dropna().index)


def delivery_filter(as_of: date, load_delivery: Optional[Callable] = None, window: int = 20,
                    min_days: int = 10) -> Optional[set]:
    """Symbols whose 20-day average NSE delivery % is above that day's cross-sectional median, or None when delivery
    data is unavailable (any failure disables the filter for the day rather than blocking the scan).

    Research finding (STRATEGY.md, 2026-09-22): combined with 12-1 month momentum ranking (the live default), this
    added ~14%/yr over the same-window equal-weight basket in a single 3-year test. Promising, not proven."""
    if load_delivery is None:
        from src.research.delivery import load_delivery as _load

        load_delivery = _load
    try:
        deliv = load_delivery(years=3)
    except Exception as e:
        log.warning("delivery filter unavailable, screening without it today: %s", type(e).__name__)
        return None
    if deliv.empty:
        return None
    return above_median_delivery(delivery_average(deliv, window, min_days), as_of)


class UniverseScreener:
    """Scans the full market once per day (and caches it); symbols_for() adds anything currently held."""

    def __init__(self, list_symbols: Callable[[], List[str]], fetch_bars: Callable[[List[str]], pd.DataFrame],
                 cfg: ScreenConfig, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
                 chunk: int = 500, sleep: Callable[[float], None] = time.sleep, use_delivery_filter: bool = False,
                 delivery_loader: Optional[Callable] = None):
        self.list_symbols, self.fetch_bars, self.cfg, self.today, self.chunk = list_symbols, fetch_bars, cfg, today, chunk
        self.sleep, self.use_delivery_filter, self.delivery_loader = sleep, use_delivery_filter, delivery_loader
        self._day: Optional[date] = None
        self._ranked: List[Candidate] = []

    def ranked(self) -> List[Candidate]:
        """Every stock that passed today's filters, best first; the full scan runs once per day and is cached."""
        if self._day == self.today():
            return self._ranked
        symbols = self.list_symbols()
        log.info("screening %d symbols", len(symbols))
        passed: List[Candidate] = []
        for i in range(0, len(symbols), self.chunk):
            passed.extend(screen_bars(self._fetch_with_retry(symbols[i:i + self.chunk]), self.cfg))
        if self.use_delivery_filter:
            ok = delivery_filter(self.today(), self.delivery_loader)
            if ok is not None:
                before = len(passed)
                passed = [c for c in passed if c.symbol in ok]
                log.info("delivery filter: kept %d/%d (above the day's median 20-day delivery %%)", len(passed), before)
        self._ranked = rank(passed, len(passed))
        self._day = self.today()
        log.info("screen: %d passed filters, best by momentum (before the affordability rule): %s", len(passed),
                 ", ".join(c.symbol for c in self._ranked[:self.cfg.max_candidates]))
        return self._ranked

    def candidates(self, equity: Optional[float] = None) -> List[Candidate]:
        """Today's top candidates; with `equity`, only stocks the account can afford one share of (cfg.affordable_pct)."""
        return select_candidates(self.ranked(), self.cfg.max_candidates, equity, self.cfg.affordable_pct)

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
        picks = self.candidates(portfolio.equity)
        if self.cfg.affordable_pct is not None:
            budget = portfolio.equity * self.cfg.affordable_pct
            skipped = [c.symbol for c in self.ranked()[:self.cfg.max_candidates] if c.price > budget]
            if skipped:
                log.info("affordability: one share must fit in %.0f (%.0f%% of equity); skipped %s",
                         budget, self.cfg.affordable_pct * 100, ", ".join(skipped))
        return held + [c.symbol for c in picks if c.symbol not in portfolio.positions]
