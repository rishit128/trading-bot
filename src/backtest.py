"""Portfolio simulator that mirrors live behaviour: signals at a close execute at the next open, sizing comes from
RiskEngine, bracket stop/target levels are set off the signal-day close, and a symbol with an open position is not added to."""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import RiskLimits
from src.data.indicators import build_snapshot
from src.engine.risk_engine import Portfolio, RiskEngine, drawdown_pause

START_EQUITY = 100_000.0

Signals = Dict[str, Dict[pd.Timestamp, Tuple[str, float]]]  # symbol -> date -> (action, confidence)
# (signal day, account state at the fill day's open) -> ordered (symbol, confidence) BUY picks, best first
BuySource = Callable[[pd.Timestamp, Portfolio], Sequence[Tuple[str, float]]]

# Keyword arguments that make `simulate` follow the live exits (8% stop from the limits + MA200 trend exit, no time exit).
LIVE_PARITY = dict(max_hold_days=10 ** 9, trend_exit=True)


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


def simulate(bars: Dict[str, pd.DataFrame], signals: Signals, limits: RiskLimits,
             start_equity: float = START_EQUITY, max_hold_days: int = 10,
             fees: Optional[Callable[[str, float], float]] = None, slippage: float = 0.0,
             trend_exit: bool = False, buy_source: Optional[BuySource] = None) -> Result:
    """Replay signals through the real risk engine: next-open fills, stop/target levels off the signal-day close.

    `slippage` is charged adversarially (buy higher, sell lower) on market fills only; stop/target exits hit at their
    exact levels. `fees` mirrors the paper broker's schedule (india_delivery_fees) for net-of-cost parity (D1/D2).

    `trend_exit=True` mirrors the live deterministic exit: a position whose previous close is below its 200-day average
    is sold at the next open, whatever the signals say. The live bot has no time exit, so live-parity callers also pass
    a huge `max_hold_days` (see LIVE_PARITY).

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
        exit_fee = fees("SELL", pos["qty"] * price) if fees else 0.0
        cash += pos["qty"] * price - exit_fee
        trades.append(Trade(sym, pos["entry_date"], pos["entry_price"], day, price, pos["qty"], reason,
                            fees=exit_fee + pos["entry_fee"], stop=pos["stop"], target=pos["target"]))

    def portfolio_at(day, column):
        """The portfolio as the risk engine would see it at the day's open or close prices."""
        values = {s: p["qty"] * price_at(s, day, column) for s, p in positions.items()}
        return Portfolio(cash, cash + sum(values.values()), values, {s: p["qty"] for s, p in positions.items()},
                         prev_equity, peak)

    def try_buy(sym, confidence, prev, day):
        """Size a BUY through the risk engine and fill it at the day's open."""
        nonlocal cash
        if sym in positions or not (has_bar(sym, prev) and has_bar(sym, day)):
            return
        ref = float(bars[sym].loc[prev, "Close"])
        decision = engine.evaluate("BUY", confidence, sym, ref, portfolio_at(day, "Open"))
        if not decision.approved:
            return
        open_price = float(bars[sym].loc[day, "Open"]) * (1 + slippage)
        qty = min(decision.quantity, int(cash // open_price))
        while qty > 0 and fees and qty * open_price + fees("BUY", qty * open_price) > cash:
            qty -= 1
        if qty <= 0:
            return
        entry_fee = fees("BUY", qty * open_price) if fees else 0.0
        cash -= qty * open_price + entry_fee
        positions[sym] = dict(qty=qty, entry_price=open_price, entry_date=day, days=0, entry_fee=entry_fee,
                              stop=ref * (1 - limits.stop_loss_pct), target=ref * (1 + limits.take_profit_pct))

    for i, day in enumerate(calendar):
        prev = calendar[i - 1] if i else None
        peak, halted_since = drawdown_pause(peak, prev_equity, halted_since, day, limits.max_drawdown_pct,
                                            limits.drawdown_pause_days)

        if prev is not None:
            for sym in list(positions):
                if not has_bar(sym, day):
                    continue
                sig = signals.get(sym, {}).get(prev)
                if trend_exit and prev in ma200[sym].index and float(bars[sym].loc[prev, "Close"]) < ma200[sym].loc[prev]:
                    sig = ("SELL", 1.0)
                if sig and sig[0] == "SELL" and engine.evaluate("SELL", sig[1], sym, 1.0, portfolio_at(day, "Open")).approved:
                    fill = float(bars[sym].loc[day, "Open"]) * (1 - slippage)
                    close_position(sym, day, fill, "SIGNAL")

            if buy_source is not None:
                for sym, confidence in buy_source(prev, portfolio_at(day, "Open")):
                    try_buy(sym, confidence, prev, day)
            else:
                for sym in sorted(bars):
                    sig = signals.get(sym, {}).get(prev)
                    if sig and sig[0] == "BUY":
                        try_buy(sym, sig[1], prev, day)

        for sym in list(positions):
            if not has_bar(sym, day):
                continue
            pos, row = positions[sym], bars[sym].loc[day]
            pos["days"] += 1
            if row.Open <= pos["stop"]:
                close_position(sym, day, row.Open, "STOP")
            elif row.Open >= pos["target"]:
                close_position(sym, day, row.Open, "TARGET")
            elif row.Low <= pos["stop"]:
                close_position(sym, day, pos["stop"], "STOP")
            elif row.High >= pos["target"]:
                close_position(sym, day, pos["target"], "TARGET")
            elif pos["days"] >= max_hold_days:
                close_position(sym, day, float(row.Close) * (1 - slippage), "TIME")

        equity = cash + sum(p["qty"] * price_at(s, day, "Close") for s, p in positions.items())
        curve[day] = equity
        peak, prev_equity = max(peak, equity), equity

    return Result(pd.Series(curve), trades, len(positions))


def buy_and_hold_curve(bars: Dict[str, pd.DataFrame], start_equity: float = START_EQUITY) -> pd.Series:
    """Equity of holding every symbol equally from the first day."""
    calendar = sorted(set.intersection(*[set(df.index) for df in bars.values()]))
    alloc = start_equity / len(bars)
    total = pd.Series(0.0, index=calendar)
    for df in bars.values():
        shares = alloc / df.loc[calendar[0], "Open"]
        total += shares * df.loc[calendar, "Close"]
    return total


def rule_signals(bars: Dict[str, pd.DataFrame], dates) -> Signals:
    """Trivial baseline the LLM has to beat: trend-following on the same indicators, no model involved."""
    out: Signals = {}
    for sym, df in bars.items():
        out[sym] = {}
        for d in dates:
            snap = build_snapshot(sym, df.loc[:d])
            if snap.price > snap.ma50 > snap.ma200 and snap.rsi < 70:
                out[sym][d] = ("BUY", 0.9)
            elif snap.price < snap.ma200:
                out[sym][d] = ("SELL", 0.9)
    return out


def curve_metrics(curve: pd.Series, start_equity: Optional[float] = None) -> dict:
    """total_return is measured from start_equity (default: first point), so day-one moves are not dropped."""
    start = start_equity if start_equity is not None else curve.iloc[0]
    daily = curve.pct_change().dropna()
    drawdown = (curve / curve.cummax() - 1).min()
    sharpe = float(daily.mean() / daily.std() * math.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0
    return {"total_return": float(curve.iloc[-1] / start - 1), "max_drawdown": float(drawdown), "sharpe": sharpe}


def trade_metrics(trades: List[Trade]) -> dict:
    """Trade count, win rate, average return and exits by reason."""
    if not trades:
        return {"trades": 0}
    rets = [t.return_pct for t in trades]
    wins = [r for r in rets if r > 0]
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(rets),
        "avg_trade_return": sum(rets) / len(rets),
        "exits": {r: sum(1 for t in trades if t.reason == r) for r in ("STOP", "TARGET", "TIME", "SIGNAL")},
    }


def analyze_trades(trades: List[Trade]) -> dict:
    """Where the money was actually made or lost: by exit reason, holding period, year and symbol (P&L is net of fees)."""
    if not trades:
        return {}
    df = pd.DataFrame([{
        "symbol": t.symbol, "reason": t.reason, "ret": t.return_pct, "year": t.exit_date.year,
        "days": (t.exit_date - t.entry_date).days,
        "pnl": (t.exit_price - t.entry_price) * t.qty - t.fees,
    } for t in trades])
    df["holding"] = pd.cut(df["days"], [-1, 2, 7, 14, 30, 10 ** 6], labels=["<=2d", "3-7d", "8-14d", "15-30d", ">30d"])
    agg = dict(n=("ret", "size"), win_rate=("ret", lambda s: float((s > 0).mean())), avg_return=("ret", "mean"), pnl=("pnl", "sum"))
    return {
        "by_reason": df.groupby("reason").agg(**agg),
        "by_holding": df.groupby("holding", observed=True).agg(**agg),
        "by_year": df.groupby("year").agg(**agg),
        "worst_symbols": df.groupby("symbol")["pnl"].sum().nsmallest(5),
        "median_days_stopped_out": float(df.loc[df["reason"] == "STOP", "days"].median()) if (df["reason"] == "STOP").any() else None,
    }
