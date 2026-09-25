"""The one execution convention every evaluation path must agree on (P0, D1/D2).

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
from typing import Callable

from src.engine.paper_broker import india_delivery_fees

SLIPPAGE = 0.0005  # 5 bps per side, the paper broker's default and the backtest's default through RunBacktestCosts
EXEC_LAG_DAYS = 1  # a decision at the close of T is quantified at the open of T + 1
BACKTEST_FILL = "next_open"  # documented, constant; see simulate() in src/backtest.py
COSTS: Callable[[str, float], float] = india_delivery_fees  # the shared delivery (CNC) fee schedule


def apply_slippage(price: float, side: str, slippage: float = SLIPPAGE) -> float:
    """Adversarial slippage: a buy fills higher, a sell fills lower (stop/target exits must NOT use this)."""
    return price * (1 + slippage) if side == "BUY" else price * (1 - slippage)