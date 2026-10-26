"""Portfolio simulator that mirrors live behaviour: signals at a close execute at the next open, sizing comes from
RiskEngine, bracket stop/target levels are set off the signal-day close, and a symbol with an open position is not added to."""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import RiskLimits
from src.engine.enums import Action
from src.engine.risk_engine import Portfolio, RiskEngine, drawdown_pause
from src.engine.rules import trend_broken

START_EQUITY = 100_000.0

SignalTable = Dict[str, Dict[pd.Timestamp, Tuple[str, float]]]  # symbol -> date -> (action, confidence)
# (signal day, account state at the fill day's open) -> ordered (symbol, confidence) BUY picks, best first
BuySource = Callable[[pd.Timestamp, Portfolio], Sequence[Tuple[str, float]]]


@dataclass(frozen=True)
class Trade:
    """A closed simulated trade; return_pct is net of fees when a fee model was used. `stop`/`target` are the levels
    the bracket set off the signal-day close, so a report can verify no exit ever slipped past its protective level."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    qty: int
    reason: str
    fees: float = 0.0
    stop: Optional[float] = None
    target: Optional[float] = None

    @property
    def return_pct(self) -> float:
        """Net of fees when a fee model was used."""
        return self.exit_price / self.entry_price - 1 - self.fees / (self.entry_price * self.qty)


@dataclass(frozen=True)
class Result:
    """Simulation output: the equity curve, closed trades, and how many positions were still open."""

    equity: pd.Series
    trades: List[Trade]
    open_at_end: int


def simulate(
    bars: Dict[str, pd.DataFrame],
    signals: SignalTable,
    limits: RiskLimits,
    start_equity: float = START_EQUITY,
    trend_exit: bool = True,
    fees: Optional[Callable[[str, float], float]] = None,
    slippage: float = 0.0,
    buy_source: Optional[BuySource] = None,
    cash_yield: float = 0.0,
    cash_yield_from: Optional[pd.Timestamp] = None,
    trail_pct: Optional[float] = None,
    lock_at: Optional[float] = None,
) -> Result:
    """Replay signals through the real risk engine: next-open fills, stop/target levels off the signal-day close.

    `cash_yield` is the annual rate idle cash earns (a liquid-fund sweep; 0 = cash earns nothing, the old behaviour). It
    accrues per calendar day between bars, before that day's trading, the same way the paper broker accrues it.
    `cash_yield_from` starts the accrual at that date, so a warm-up stretch (history only, no trading) earns nothing
    and cannot show up as a gain in a curve reported from the real start.

    `slippage` is charged adversarially (buy higher, sell lower) on market fills only; stop/target exits hit at their
    exact levels. `fees` mirrors the paper broker's schedule (india_delivery_fees) for net-of-cost parity (D1/D2).

    `trail_pct` raises each position's stop to that fraction below its highest high since entry (never lowers it);
    `lock_at` (a gain such as 0.20) first moves the stop to the entry price once the high has gained that much.
    Both apply from the next day's open, like the live broker-side stop moves. Research only: the live bot does neither.

    `trend_exit` (the live exit, on by default) sells at the next open after a close below the 200-day average, so a
    simulation tests the strategy that is actually running.

    Entries come from `signals` (BUYs processed in symbol order) or, when given, from `buy_source(signal_day, portfolio)`:
    an ORDERED list of (symbol, confidence) picks, best first, decided from the account state at the fill day's open.
    The order matters whenever slots are scarce: live fills the best-ranked candidate first, not the alphabetical one.

    Symbols may have different histories (later listings, halts): the calendar is the union of all dates, a symbol
    without a bar on a day cannot be traded or stopped that day, and its position is valued at its last known close."""
    engine = RiskEngine(limits, fees=fees)
    ma200 = {sym: df["Close"].astype(float).rolling(200).mean() for sym, df in bars.items()} if trend_exit else {}
    calendar = sorted(set().union(*[set(df.index) for df in bars.values()]))
    cash, peak, prev_equity = start_equity, start_equity, start_equity
    halted_since = None  # start of the current continuous drawdown halt (see drawdown_pause)
    positions: dict = {}
    trades: List[Trade] = []
    curve = {}

    def has_bar(sym, day):
        """Whether the symbol traded on this day."""
        return day in bars[sym].index

    def price_at(sym, day, column):
        """The day's value, or the last known close when the symbol has no bar that day."""
        df = bars[sym]
        if day in df.index:
            return float(df.at[day, column])
        i = df.index.searchsorted(day) - 1
        return float(df["Close"].iloc[i]) if i >= 0 else float("nan")

    def close_position(sym, day, price, reason):
        """Close a simulated position at a price and record the trade."""
        nonlocal cash
        pos = positions.pop(sym)
        exit_fee = fees(Action.SELL, pos["qty"] * price) if fees else 0.0
        cash += pos["qty"] * price - exit_fee
        trades.append(
            Trade(
                sym,
                pos["entry_date"],
                pos["entry_price"],
                day,
                price,
                pos["qty"],
                reason,
                fees=exit_fee + pos["entry_fee"],
                stop=pos["stop"],
                target=pos["target"],
            )
        )

    def portfolio_at(day, column):
        """The portfolio as the risk engine would see it at the day's open or close prices."""
        values = {s: p["qty"] * price_at(s, day, column) for s, p in positions.items()}
        return Portfolio(
            cash, cash + sum(values.values()), values, {s: p["qty"] for s, p in positions.items()}, prev_equity, peak
        )

    def try_buy(sym, confidence, prev, day):
        """Size a BUY through the risk engine and fill it at the day's open."""
        nonlocal cash
        if sym in positions or not (has_bar(sym, prev) and has_bar(sym, day)):
            return
        ref = float(bars[sym].loc[prev, "Close"])
        decision = engine.evaluate(Action.BUY, confidence, sym, ref, portfolio_at(day, "Open"))
        if not decision.approved:
            return
        open_price = float(bars[sym].loc[day, "Open"]) * (1 + slippage)
        qty = min(decision.quantity, int(cash // open_price))
        while qty > 0 and fees and qty * open_price + fees(Action.BUY, qty * open_price) > cash:
            qty -= 1
        if qty <= 0:
            return
        entry_fee = fees(Action.BUY, qty * open_price) if fees else 0.0
        cash -= qty * open_price + entry_fee
        positions[sym] = dict(
            qty=qty,
            entry_price=open_price,
            entry_date=day,
            entry_fee=entry_fee,
            stop=ref * (1 - limits.stop_loss_pct),
            target=ref * (1 + limits.take_profit_pct),
        )

    for i, day in enumerate(calendar):
        prev = calendar[i - 1] if i else None
        if cash_yield and prev is not None and cash > 0 and (cash_yield_from is None or prev >= cash_yield_from):
            cash *= (1 + cash_yield) ** ((day - prev).days / 365)
        peak, halted_since = drawdown_pause(
            peak, prev_equity, halted_since, day, limits.max_drawdown_pct, limits.drawdown_pause_days
        )

        if prev is not None:
            for sym in list(positions):
                if not has_bar(sym, day):
                    continue
                sig = signals.get(sym, {}).get(prev)
                if (
                    trend_exit
                    and prev in ma200[sym].index
                    and trend_broken(float(bars[sym].loc[prev, "Close"]), ma200[sym].loc[prev])
                ):
                    sig = (Action.SELL, 1.0)
                if (
                    sig
                    and sig[0] == Action.SELL
                    and engine.evaluate(Action.SELL, sig[1], sym, 1.0, portfolio_at(day, "Open")).approved
                ):
                    fill = float(bars[sym].loc[day, "Open"]) * (1 - slippage)
                    close_position(sym, day, fill, "SIGNAL")

            if buy_source is not None:
                for sym, confidence in buy_source(prev, portfolio_at(day, "Open")):
                    try_buy(sym, confidence, prev, day)
            else:
                for sym in sorted(bars):
                    sig = signals.get(sym, {}).get(prev)
                    if sig and sig[0] == Action.BUY:
                        try_buy(sym, sig[1], prev, day)

        for sym in list(positions):
            if not has_bar(sym, day):
                continue
            pos, row = positions[sym], bars[sym].loc[day]
            if row.Open <= pos["stop"]:
                close_position(sym, day, row.Open, "STOP")
            elif row.Open >= pos["target"]:
                close_position(sym, day, row.Open, "TARGET")
            elif row.Low <= pos["stop"]:
                close_position(sym, day, pos["stop"], "STOP")
            elif row.High >= pos["target"]:
                close_position(sym, day, pos["target"], "TARGET")

        if trail_pct is not None or lock_at is not None:
            for sym, pos in positions.items():
                if not has_bar(sym, day):
                    continue
                high = float(bars[sym].at[day, "High"])
                pos["peak"] = max(pos.get("peak", pos["entry_price"]), high)
                if trail_pct is not None:
                    pos["stop"] = max(pos["stop"], pos["peak"] * (1 - trail_pct))
                if lock_at is not None and pos["peak"] >= pos["entry_price"] * (1 + lock_at):
                    pos["stop"] = max(pos["stop"], pos["entry_price"])

        equity = cash + sum(p["qty"] * price_at(s, day, "Close") for s, p in positions.items())
        curve[day] = equity
        peak, prev_equity = max(peak, equity), equity

    return Result(pd.Series(curve), trades, len(positions))


def curve_metrics(curve: pd.Series, start_equity: Optional[float] = None) -> dict:
    """total_return is measured from start_equity (default: first point), so day-one moves are not dropped."""
    start = start_equity if start_equity is not None else curve.iloc[0]
    daily = curve.pct_change().dropna()
    drawdown = (curve / curve.cummax() - 1).min()
    sharpe = float(daily.mean() / daily.std() * math.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0
    return {"total_return": float(curve.iloc[-1] / start - 1), "max_drawdown": float(drawdown), "sharpe": sharpe}
