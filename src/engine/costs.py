"""What a trade costs, and the one execution convention every evaluation path must agree on.

A signal decided from data up to the close of day T executes at the NEXT session's open, T+1:
    * backtest `simulate` fills BUY and SIGNAL-SELL at Open[T+1];
    * the paper broker fills at the current 5-minute close (decisions are made after the close, so this is the
      same session) PLUS the documented slippage;
adversarial slippage (buy higher, sell lower) is charged on market orders only, and the full delivery fee
schedule applies on every fill.

Stop/target levels are set off the signal close. In the paper broker a stop is a stop-market order: it triggers at
its level (or the bar open when a bar gaps through it) and fills with the same slippage as any market sell; a target
is a limit order and fills at its level with none. The daily backtest `simulate` still fills stops at the exact
level (a known, small optimism on daily bars).

Admitted, quantified divergence: the paper broker executes at the *current* 5-minute close, not tomorrow's open.
When a decision is made after the close, those are the same session; the residual gap between that close and the
next open is the paper-vs-backtest difference and is intentionally left visible (see the parity test rather than
being smoothed away). Other simplifications to remember: no circuit-limit modelling and no partial fills."""

from typing import Callable

from src.engine.enums import Action

SLIPPAGE = 0.0005  # 5 bps per side: the paper broker's default and every backtest path's (buy fills higher, sell lower)
# This account's Angel One brokerage plan (the owner confirmed it from the app on 2026-11-11, and Angel One's own calculator,
# SmartAPI estimateCharges, returned exactly these amounts the same day): 0.03% of the order value intraday and 0.3% on delivery,
# NSE and BSE, with no minimum and no cap. It is NOT the public "lower of Rs 20 or 0.1%, minimum Rs 5" plan of the pricing page,
# which the model wrongly used from 2026-11-06 to 2026-11-11. Statutory charges as the calculator returned them.
INTRADAY_BROKERAGE = 0.0003
DELIVERY_BROKERAGE = 0.003
INTRADAY_STT = 0.00025  # on the sell
DELIVERY_STT = 0.001  # both sides
INTRADAY_STAMP = 0.00003  # on the buy
DELIVERY_STAMP = 0.00015  # on the buy
NSE_TRANSACTION = 0.000030699  # both sides
SEBI_FEE = 0.000001  # Rs 10 a crore, both sides
NSE_IPFT = 0.000000001  # investor protection fund, both sides
GST = 0.18  # on brokerage, transaction, SEBI and IPFT charges
DP_CHARGE = 23.60  # per stock per day sold from the demat account: Rs 20 + GST (Angel One support page; not in the calculator)


def _charges(value: float, brokerage: float, stt: float, stamp: float) -> float:
    taxed = (brokerage + NSE_TRANSACTION + SEBI_FEE + NSE_IPFT) * value
    return taxed * (1 + GST) + (stt + stamp) * value


def india_delivery_fees(side: str, value: float) -> float:
    """This account's NSE delivery (CNC) charges on one executed order of `value` rupees: 0.3% brokerage, STT 0.1% both sides, stamp
    duty 0.015% on buys, exchange and SEBI fees, 18% GST, and the DP charge on each sale. About 0.93% of the value a round trip plus
    Rs 23.60: 2.0% on Rs 2,200, 1.2% on Rs 10,000. Rates change."""
    buy = side == Action.BUY
    return _charges(value, DELIVERY_BROKERAGE, DELIVERY_STT, DELIVERY_STAMP if buy else 0.0) + (0.0 if buy else DP_CHARGE)


def intraday_fees(side: str, value: float) -> float:
    """This account's NSE equity intraday (MIS) charges on one executed order of `value` rupees: 0.03% brokerage, STT 0.025% on the
    sell, stamp duty 0.003% on the buy, exchange and SEBI fees, 18% GST. About 0.106% a round trip at any size. Rates change."""
    buy = side == Action.BUY
    return _charges(value, INTRADAY_BROKERAGE, 0.0 if buy else INTRADAY_STT, INTRADAY_STAMP if buy else 0.0)


# Fees and slippage of a delivery round trip on a typical Rs 2,200 swing position, used to judge outcomes net of costs.
ROUND_TRIP_COST = (india_delivery_fees(Action.BUY, 2200.0) + india_delivery_fees(Action.SELL, 2200.0)) / 2200.0 + 2 * SLIPPAGE


def net_return(entry: float, exit_price: float, qty: float, fees: Callable[[str, float], float]) -> float:
    """A closed long trade's return after `fees` on both fills, as a fraction of the money put in. The fill prices already carry
    the slippage. Reports use it to judge every recorded trade at today's fee schedule, whatever was booked at the time."""
    bought, sold = entry * qty, exit_price * qty
    return (sold - bought - fees(Action.BUY, bought) - fees(Action.SELL, sold)) / bought
