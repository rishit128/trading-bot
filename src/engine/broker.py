"""Alpaca paper-trading broker (US market)."""
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, MarketOrderRequest, StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

from src.engine.risk_engine import Portfolio
from src.net import apply_timeout

STOP_TYPES = ("stop", "stop_limit", "trailing_stop")
TERMINAL = ("filled", "canceled", "cancelled", "expired", "rejected", "done_for_day")


@dataclass(frozen=True)
class Fill:
    """What the broker reported for an order: id, status, and the quantity and price actually filled when known."""
    broker_order_id: str
    status: str
    filled_qty: Optional[int] = None
    avg_price: Optional[float] = None


def _value(enum_or_str) -> str:
    return str(getattr(enum_or_str, "value", enum_or_str)).lower()


class AlpacaBroker:
    """Alpaca paper trading. Entries are market orders with a broker-side bracket (protective stop + target)."""

    def __init__(self, client: Optional[TradingClient] = None):
        self.client = client or apply_timeout(
            TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
        )

    def portfolio(self) -> Portfolio:
        """Current cash, equity and positions from Alpaca."""
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        return Portfolio(
            cash=float(account.cash),
            equity=float(account.equity),
            positions={p.symbol: abs(float(p.market_value)) for p in positions},
            position_qty={p.symbol: int(float(p.qty)) for p in positions},
            start_of_day_equity=float(account.last_equity),
        )

    def is_market_open(self) -> bool:
        """True when the US market is open."""
        return bool(self.client.get_clock().is_open)

    def has_open_order(self, symbol: str) -> bool:
        """True when the symbol has any working order."""
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        return len(orders) > 0

    def buy_with_bracket(self, symbol: str, qty: int, price: float, stop_pct: float, take_pct: float) -> Fill:
        """Market buy with a broker-side protective stop and target."""
        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                # Bracket legs inherit the parent's time-in-force; DAY legs could expire and leave the position unprotected.
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(price * (1 + take_pct), 2)),
                stop_loss=StopLossRequest(stop_price=round(price * (1 - stop_pct), 2)),
            )
        )
        return Fill(str(order.id), _value(order.status))

    def sell(self, symbol: str, qty: int) -> Fill:
        # Bracket legs reserve the shares; Alpaca rejects a sell until they are cancelled.
        """Cancel the symbol's stop/target legs, then sell at market (one retry; a loud error if the position may be left unprotected)."""
        for open_order in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])):
            self.client.cancel_order_by_id(open_order.id)
        request = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            order = self.client.submit_order(request)
        except Exception:
            try:
                order = self.client.submit_order(request)
            except Exception as e:
                raise RuntimeError(
                    f"sell of {qty} {symbol} failed after cancelling its stop/target orders; "
                    f"the position may be UNPROTECTED: {e}"
                ) from e
        return Fill(str(order.id), _value(order.status))

    def confirm(self, fill: Fill, qty: int, wait: float = 10.0, poll: float = 1.0,
                sleep: Callable[[float], None] = time.sleep) -> Fill:
        """Poll the order until it is final (or `wait` seconds pass) and report what was actually filled."""
        waited, order = 0.0, None
        while True:
            order = self.client.get_order_by_id(fill.broker_order_id)
            if _value(order.status) in TERMINAL or waited >= wait:
                break
            sleep(poll)
            waited += poll
        filled = int(float(order.filled_qty or 0))
        price = float(order.filled_avg_price) if order.filled_avg_price else None
        return Fill(fill.broker_order_id, _value(order.status), filled, price)

    def unprotected_positions(self) -> List[Tuple[str, int, float]]:
        """Long positions with no working stop order. Bracket legs only appear with nested=True, and a take-profit
        (limit) leg alone is not protection, so only stop-type SELL orders count."""
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500))
        flat = []
        for o in orders:
            flat.append(o)
            flat.extend(o.legs or [])
        protected = {o.symbol for o in flat if _value(o.side) == "sell" and _value(o.order_type) in STOP_TYPES}
        return [(p.symbol, int(float(p.qty)), float(p.avg_entry_price))
                for p in self.client.get_all_positions()
                if float(p.qty) > 0 and p.symbol not in protected]

    def protect(self, symbol: str, qty: int, stop_price: float) -> Fill:
        """Place a standalone good-til-cancelled protective stop for an existing long position."""
        order = self.client.submit_order(
            StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                             stop_price=round(stop_price, 2))
        )
        return Fill(str(order.id), _value(order.status))
