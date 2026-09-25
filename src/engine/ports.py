"""The interfaces the trading engine depends on, written down instead of assumed.

A broker, a price feed and a market clock used to be typed `object` and probed with `getattr`/`hasattr`, so a missing or
misspelled method surfaced at run time, in the middle of a cycle. Now the required surface is a Protocol (checked by mypy)
and each OPTIONAL broker capability is its own Protocol the pipeline tests for explicitly with `isinstance`."""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

from src.engine.risk_engine import Portfolio


@dataclass(frozen=True)
class Fill:
    """What the broker reported for an order: id, status, and the quantity and price actually filled when known."""
    broker_order_id: str
    status: str
    filled_qty: Optional[int] = None
    avg_price: Optional[float] = None


class PriceFeed(Protocol):
    """Prices for a symbol: the latest, and the bars since a moment (to replay stops and targets)."""
    def last_price(self, symbol: str) -> float: ...
    def bars_since(self, symbol: str, since: datetime) -> pd.DataFrame: ...


class MarketClock(Protocol):
    """When the market is open, and today's date there."""
    def is_open(self, now: Optional[datetime] = None) -> bool: ...
    def today_ist(self, now: Optional[datetime] = None) -> str: ...


class Broker(Protocol):
    """What every broker must do. The paper broker is the implementation today; a real one would provide the same."""
    def portfolio(self) -> Portfolio: ...
    def is_market_open(self) -> bool: ...
    def has_open_order(self, symbol: str) -> bool: ...
    def buy_with_bracket(self, symbol: str, qty: int, price: float, stop_pct: float, take_pct: float) -> Fill: ...
    def sell(self, symbol: str, qty: int) -> Fill: ...


@runtime_checkable
class ProtectsPositions(Protocol):
    """A broker that can tell which positions have no working stop, and place one."""
    def unprotected_positions(self) -> List[Tuple[str, int, float]]: ...
    def protect(self, symbol: str, qty: int, stop_price: float) -> Fill: ...


@runtime_checkable
class ConfirmsFills(Protocol):
    """A broker that can report what an order really did (instead of trusting "accepted")."""
    def confirm(self, fill: Fill, qty: int) -> Fill: ...


@runtime_checkable
class ChargesFees(Protocol):
    """A broker with a known fee schedule, so the risk engine can refuse fee-dominated positions."""
    fees: Callable[[str, float], float]


@runtime_checkable
class ListsHoldings(Protocol):
    """A broker that can list its open positions in detail (price paid, live price, profit, stop)."""
    def holdings(self) -> List[dict]: ...


@runtime_checkable
class ListsTrades(Protocol):
    """A broker that can list its closed trades."""
    def trade_history(self, limit: Optional[int] = None) -> List[dict]: ...
