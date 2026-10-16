"""Today's market pulse: which stocks, and which sectors, are surging right now.

The swing scanner buys long-established uptrends (price above a rising 50- and 200-day average, ranked by 12-month
momentum). A sector that turns up after a long slump (IT in October 2026: up 8-15% in a month, still under its 200-day
average) never passes that filter, however strong the move. This module looks at the same daily bars the scanner has
already downloaded for the whole market and finds what is surging today, without any new data source:

  * a HOT STOCK is liquid, above its 50-day average, up at least `HOT_RET_21D` over 21 sessions and not yet extreme
    (RSI(14) below `HOT_MAX_RSI`, a looser line than the entry filter's 70) and not past a +35% month;
  * a HOT SECTOR is an industry group with at least `MIN_SECTOR_STOCKS` hot stocks among the strongest `LOOKUPS` movers
    (Yahoo Finance's industry label, falling back to its broader sector: the source the bot already uses for company
    data; a stock with neither is ignored). Broad sectors would all look hot in any wide rally, so the finer label is used.

The pulse only names candidates and gives the AI the context. The AI, the risk engine, the position cap and the stop
decide as for any other stock, and a position bought this way is sold on a close below its 50-day average (not the
200-day one, which it has not yet reclaimed). Nothing here trades."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable, Collection, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.indicators import MAX_STALE_DAYS, rsi_from_array

log = logging.getLogger(__name__)

HOT_RET_21D = 0.08  # up at least 8% over 21 sessions...
HOT_MAX_RET_21D = 0.35  # ...but a month already past +35% is a blow-off, not a trend to join
HOT_MAX_RSI = 80.0  # above this a surge is too stretched to start a position
MIN_SECTOR_STOCKS = 3  # hot stocks needed for a sector to count as hot
LOOKUPS = 150  # the strongest movers whose group is looked up (one Yahoo call each, cached)
MAX_PICKS = 5  # hot-sector stocks added to one cycle's candidates
MAX_OPEN = 3  # positions open at once that were bought this way
BARS_NEEDED = 64  # 63-session return and a 50-day average


@dataclass(frozen=True)
class Mover:
    """A stock surging by the pulse's own test, with the numbers it was judged on."""

    symbol: str
    price: float
    ret21: float
    ret63: float
    avg_traded_value: float


@dataclass(frozen=True)
class HotStock:
    """A surging stock in a hot sector: what the AI is shown and what a replay needs. The sector figures are computed
    once per day, so the JSON is identical across the day's cycles (a remembered answer is keyed on it)."""

    symbol: str
    sector: str
    sector_stocks: int  # hot stocks found in the sector
    sector_avg_ret21: float  # their average 21-session return
    ret21: float
    ret63: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "HotStock":
        return cls(
            d["symbol"], d["sector"], int(d["sector_stocks"]), float(d["sector_avg_ret21"]), float(d["ret21"]), float(d["ret63"])
        )


def pulse_line(h: HotStock) -> str:
    """The pulse as the AI is shown it: the figures only, no advice."""
    return (
        f"the {h.sector} sector is surging: {h.sector_stocks} liquid stocks in it are up {HOT_RET_21D:.0%} or more over "
        f"the past month (average {h.sector_avg_ret21:+.0%}); {h.symbol} is {h.ret21:+.0%} over 21 sessions and "
        f"{h.ret63:+.0%} over 63"
    )


def mover(
    symbol: str, close: np.ndarray, volume: np.ndarray, last_bar: pd.Timestamp, newest: pd.Timestamp, min_price: float,
    min_traded_value: float, max_daily_volatility: float,
) -> Optional[Mover]:  # fmt: skip
    """The pulse's verdict on one stock from plain arrays (completed daily closes and volumes, oldest first)."""
    if len(close) < BARS_NEEDED or (newest - last_bar).days > MAX_STALE_DAYS:
        return None
    price = float(close[-1])
    traded_value = float(np.nanmean(close[-20:] * volume[-20:]))
    if not (price >= min_price and traded_value >= min_traded_value):
        return None
    recent = close[-BARS_NEEDED:]
    volatility = float(np.std(recent[1:] / recent[:-1] - 1.0, ddof=1))
    if not volatility <= max_daily_volatility:  # also rejects NaN
        return None
    ret21 = price / float(close[-22]) - 1
    if price <= float(close[-50:].mean()) or not HOT_RET_21D <= ret21 <= HOT_MAX_RET_21D:
        return None
    if not rsi_from_array(close) < HOT_MAX_RSI:
        return None
    return Mover(symbol, price, ret21, price / float(close[-BARS_NEEDED]) - 1, traded_value)


def movers_from_bars(bars: pd.DataFrame, cfg) -> List[Mover]:
    """Every surging stock in a screener-format frame (MultiIndex symbol, timestamp; completed daily bars)."""
    found: List[Mover] = []
    if bars.empty:
        return found
    newest = bars.index.get_level_values(1).max()
    for symbol, g in bars.groupby(level=0):
        m = mover(
            str(symbol),
            (g["Close"] if "Close" in g else g["close"]).to_numpy(dtype=float),  # the spellings screen_bars accepts
            (g["Volume"] if "Volume" in g else g["volume"]).to_numpy(dtype=float),
            g.index.get_level_values(1)[-1],
            newest,
            cfg.min_price,
            cfg.min_traded_value,
            cfg.max_daily_volatility,
        )
        if m is not None:
            found.append(m)
    return found


def hot_stocks(movers: Sequence[Mover], sector_of: Callable[[str], Optional[str]]) -> List[HotStock]:
    """The surging stocks that sit in a hot sector, strongest first. Looks up the sector of the strongest `LOOKUPS`
    movers only."""
    top = sorted(movers, key=lambda m: m.ret21, reverse=True)[:LOOKUPS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        sectors = list(pool.map(sector_of, [m.symbol for m in top]))
    groups: Dict[str, List[Mover]] = {}
    for m, sector in zip(top, sectors):
        if sector:
            groups.setdefault(sector, []).append(m)
    out: List[HotStock] = []
    for sector, members in groups.items():
        if len(members) < MIN_SECTOR_STOCKS:
            continue
        avg = float(np.mean([m.ret21 for m in members]))
        out.extend(HotStock(m.symbol, sector, len(members), avg, m.ret21, m.ret63) for m in members)
    return sorted(out, key=lambda h: h.ret21, reverse=True)


def yahoo_sector(symbol: str, ticker: Optional[Callable] = None) -> Optional[str]:
    """An NSE company's industry group from Yahoo Finance (its sector when it has no industry), or None (never raises)."""
    try:
        if ticker is None:
            import yfinance as yf

            ticker = yf.Ticker
        info = ticker(symbol + ".NS").info or {}
        group = info.get("industry") or info.get("sector")
        return str(group) if group else None
    except Exception as e:
        log.warning("no sector for %s: %s: %s", symbol, type(e).__name__, e)
        return None


class MarketPulse:
    """Today's hot stocks, computed once per day from the screener's movers and cached (sectors rarely change, so they
    are remembered for the life of the process)."""

    def __init__(
        self,
        movers_fn: Callable[[], List[Mover]],
        sector_fn: Callable[[str], Optional[str]] = yahoo_sector,
        today: Callable[[], date] = date.today,
    ):
        self.movers_fn, self.today = movers_fn, today
        self._sectors: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()
        self._sector_fn = sector_fn
        self._day: Optional[date] = None
        self._hot: List[HotStock] = []
        self._movers: Dict[str, Mover] = {}

    def _sector(self, symbol: str) -> Optional[str]:
        if symbol not in self._sectors:
            self._sectors[symbol] = self._sector_fn(symbol)
        return self._sectors[symbol]

    def hot(self) -> List[HotStock]:
        """The day's hot stocks (empty when nothing is surging or the scan is unavailable: the bot then runs as before)."""
        with self._lock:
            if self._day == self.today():
                return self._hot
            movers = self.movers_fn()
            self._movers = {m.symbol: m for m in movers}
            self._hot = hot_stocks(movers, self._sector)
            self._day = self.today()
            return self._hot

    def picks(self, budget: float, held: Collection[str], slots: int) -> List[HotStock]:
        """Up to `slots` (at most MAX_PICKS) hot stocks not already held, one share of which costs no more than `budget`; the
        most liquid first (a leader of a surging group, not its biggest one-month spike)."""
        out: List[HotStock] = []
        if slots <= 0:
            return out
        for h in sorted(self.hot(), key=lambda h: self._movers[h.symbol].avg_traded_value, reverse=True):
            if h.symbol in held or self._movers[h.symbol].price > budget:
                continue
            out.append(h)
            if len(out) >= min(slots, MAX_PICKS):
                break
        return out


def summary(hot: Sequence[HotStock]) -> str:
    """One line per hot sector for the user (Telegram), or an empty string when nothing is surging."""
    sectors: Dict[str, List[HotStock]] = {}
    for h in hot:
        sectors.setdefault(h.sector, []).append(h)
    lines = [
        f"{name}: {len(members)} stocks, avg {members[0].sector_avg_ret21:+.0%} in a month ("
        + ", ".join(m.symbol for m in members[:6])
        + ")"
        for name, members in sorted(sectors.items(), key=lambda kv: -kv[1][0].sector_avg_ret21)
    ]
    return "\n".join(lines)
