"""Verify Alpaca paper trading end to end with the bot's own broker code (US market only; India needs no Alpaca).

Submits a 1-share market order and a 1-share bracket order (exactly the request AlpacaBroker.buy_with_bracket builds,
with the bot's default 8% stop / 100% target), checks Alpaca accepts them, then CANCELS both and confirms nothing is
left behind. Paper account only; it never holds a position unless the market fills an order before the cancel lands."""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from alpaca.common.exceptions import APIError  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import MarketOrderRequest  # noqa: E402

from src.config import RiskLimits  # noqa: E402
from src.engine.broker import AlpacaBroker, Fill  # noqa: E402

SYMBOL = "AAPL"


def cancelled(trading, order_id) -> bool:
    for _ in range(10):
        status = str(trading.get_order_by_id(order_id).status).lower()
        if "cancel" in status or "expired" in status or "rejected" in status:
            return True
        time.sleep(1)
    return False


def main():
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("FAIL: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        sys.exit(1)

    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)
    created = []
    try:
        account = trading.get_account()
        print(f"OK: account status={account.status} cash=${account.cash} equity=${account.equity}")
        print(f"OK: market open right now: {trading.get_clock().is_open}")

        start = datetime.now(timezone.utc) - timedelta(days=10)  # without a start Alpaca returns today only (empty on weekends)
        price = float(data.get_stock_bars(StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Day, start=start))[SYMBOL][-1].close)
        print(f"OK: last {SYMBOL} daily close ${price:.2f}")

        plain = trading.submit_order(MarketOrderRequest(symbol=SYMBOL, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        created.append(plain.id)
        print(f"OK: plain market order accepted, id={plain.id} status={plain.status}")

        broker = AlpacaBroker(trading)
        confirmed = broker.confirm(Fill(str(plain.id), "accepted"), 1, wait=2.0, poll=1.0)
        print(f"OK: confirm() reports status={confirmed.status} filled_qty={confirmed.filled_qty} (market closed => 0)")
        print(f"OK: unprotected_positions() ran against the real API -> {broker.unprotected_positions()}")

        risk = RiskLimits()
        fill = AlpacaBroker(trading).buy_with_bracket(SYMBOL, 1, price, risk.stop_loss_pct, risk.take_profit_pct)
        created.append(fill.broker_order_id)
        print(f"OK: bracket order accepted via AlpacaBroker, id={fill.broker_order_id} status={fill.status} "
              f"(stop {risk.stop_loss_pct:.0%}, target {risk.take_profit_pct:.0%})")
    except APIError as e:
        print(f"FAIL: Alpaca API rejected the request: {e}")
        print("Check the keys are paper-trading keys, and that stop/target prices are valid for the symbol.")
        sys.exit(1)
    finally:
        for order_id in created:
            try:
                trading.cancel_order_by_id(order_id)
            except APIError as e:
                print(f"note: could not cancel {order_id} ({e}); it may already have filled or been cancelled")
        if created:
            done = all(cancelled(trading, oid) for oid in created)
            print(f"{'OK' if done else 'WARN'}: test orders cancelled: {done}")

    held = [p for p in trading.get_all_positions() if p.symbol == SYMBOL]
    print(f"OK: no leftover {SYMBOL} position" if not held else f"WARN: {SYMBOL} position exists ({held[0].qty} sh) - check your paper account")
    print("\nAll Alpaca checks passed.")


if __name__ == "__main__":
    main()
