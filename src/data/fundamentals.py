"""Company fundamentals for the AI's company check: recent quarters' sales and profit, and a few health ratios.

This is the information the price-based rule never sees, which is what gives the AI a real job. Source: Yahoo Finance
(same as prices; checked 2026-10-09 to cover every stock the bot held or ranked). Live only: Yahoo serves today's figures,
not what was known on a past date, so it cannot be backtested honestly and research never uses it. Any failure means
"no company data": the prompt then says so and the AI judges without it."""

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

SUFFIX = ".NS"
MAX_QUARTERS = 6
CACHE_SECONDS = 6 * 3600  # results change a few times a year; a re-read every 6 hours is plenty
FAILURE_CACHE_SECONDS = 1800  # a failed lookup is retried after half an hour, not on every cycle


@dataclass(frozen=True)
class Quarter:
    period_end: str  # YYYY-MM-DD
    revenue: Optional[float]  # rupees
    net_income: Optional[float]  # rupees


@dataclass(frozen=True)
class Fundamentals:
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None  # rupees
    pe: Optional[float] = None  # trailing price / earnings
    debt_to_equity: Optional[float] = None  # a ratio (Yahoo reports it in percent; converted)
    profit_margin: Optional[float] = None  # fraction of sales
    revenue_growth: Optional[float] = None  # latest quarter vs the same quarter a year earlier, fraction
    earnings_growth: Optional[float] = None  # the same for profit
    promoter_holding: Optional[float] = None  # insiders/promoters' share of the company, fraction
    quarters: Tuple[Quarter, ...] = ()  # newest first


def _number(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _row(statement, *labels):
    for label in labels:
        if label in statement.index:
            return statement.loc[label]
    return None


def yahoo_fundamentals(symbol: str, ticker: Optional[Callable] = None) -> Optional[Fundamentals]:
    """An NSE company's fundamentals from Yahoo, or None when Yahoo has nothing useful for it."""
    if ticker is None:
        import yfinance as yf

        ticker = yf.Ticker
    t = ticker(symbol + SUFFIX)
    info = t.info or {}
    statement = t.quarterly_income_stmt
    quarters: Tuple[Quarter, ...] = ()
    if statement is not None and not statement.empty:
        revenue = _row(statement, "Total Revenue", "Operating Revenue")
        profit = _row(statement, "Net Income", "Net Income Common Stockholders")
        columns = sorted(statement.columns, reverse=True)[:MAX_QUARTERS]
        quarters = tuple(
            Quarter(
                str(c.date()),
                _number(revenue[c]) if revenue is not None else None,
                _number(profit[c]) if profit is not None else None,
            )
            for c in columns
        )
    debt = _number(info.get("debtToEquity"))
    found = Fundamentals(
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=_number(info.get("marketCap")),
        pe=_number(info.get("trailingPE")),
        debt_to_equity=debt / 100 if debt is not None else None,
        profit_margin=_number(info.get("profitMargins")),
        revenue_growth=_number(info.get("revenueGrowth")),
        earnings_growth=_number(info.get("earningsGrowth")),
        promoter_holding=_number(info.get("heldPercentInsiders")),
        quarters=quarters,
    )
    return found if found != Fundamentals() else None


class FundamentalsProvider:
    """`provider(symbol)` -> Fundamentals or None, cached so one cycle (and the next several) read Yahoo once per stock."""

    def __init__(
        self, fetch: Callable[[str], Optional[Fundamentals]] = yahoo_fundamentals, clock: Callable[[], float] = time.monotonic
    ):
        self.fetch, self.clock = fetch, clock
        self._cache: Dict[str, Tuple[float, Optional[Fundamentals]]] = {}
        self._lock = threading.Lock()

    def __call__(self, symbol: str) -> Optional[Fundamentals]:
        with self._lock:
            hit = self._cache.get(symbol)
        if hit is not None and self.clock() < hit[0]:
            return hit[1]
        try:
            found, keep = self.fetch(symbol), CACHE_SECONDS
        except Exception as e:
            log.warning("no company data for %s: %s: %s", symbol, type(e).__name__, str(e)[:100])
            found, keep = None, FAILURE_CACHE_SECONDS
        with self._lock:
            self._cache[symbol] = (self.clock() + keep, found)
        return found
