"""What a trade costs, and the one execution convention every evaluation path must agree on.

    A signal decided from data up to the close of day T executes at the NEXT session's open, T+1:
        * backtest `simulate` fills BUY and SIGNAL-SELL at Open[T+1];
        * the paper broker fills at the current 5-minute close (decisions are made after the close, so this is the
          same session) PLUS the documented slippage;
    adversarial slippage (buy higher, sell lower) is charged on market orders only, and the full delivery fee
    schedule applies on every fill.

    Stop/target exits are NOT market orders: they are limit-like levels set off the signal close and are hit at the
    exact level (or at the bar open when a bar gaps through them), with no extra slippage.

    Admitted, quantified divergence: the paper broker executes at the *current* 5-minute close, not tomorrow's open.
    When a decision is made after the close, those are the same session; the residual gap between that close and the
    next open is the paper-vs-backtest difference and is intentionally left visible (see the parity test rather than
    being smoothed away). Other simplifications to remember: no circuit-limit modelling and no partial fills."""
from src.engine.enums import Action

SLIPPAGE = 0.0005  # 5 bps per side: the paper broker's default and every backtest path's (buy fills higher, sell lower)


def india_delivery_fees(side: str, value: float) -> float:
    """Approximate NSE delivery (CNC) charges with a zero-brokerage discount broker: STT 0.1% both sides, stamp duty
    0.015% on buys, exchange 0.00297%, SEBI 0.0001%, 18% GST on those two, DP charge Rs 15.93 per sell. Rates change."""
    stt = 0.001 * value
    stamp = 0.00015 * value if side == Action.BUY else 0.0
    exchange, sebi = 0.0000297 * value, 0.000001 * value
    gst = 0.18 * (exchange + sebi)
    dp = 15.93 if side == Action.SELL else 0.0
    return stt + stamp + exchange + sebi + gst + dp
