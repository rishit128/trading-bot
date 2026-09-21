"""Portfolio simulator that mirrors live behaviour: signals at a close execute at the next open, sizing comes from
RiskEngine, bracket stop/target levels are set off the signal-day close, and a symbol with an open position is not added to."""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.config import RiskLimits
from src.data.indicators import build_snapshot
from src.engine.risk_engine import Portfolio, RiskEngine

START_EQUITY = 100_000.0

Signals = Dict[str, Dict[pd.Timestamp, Tuple[str, float]]]  # symbol -> date -> (action, confidence)


@dataclass(frozen=True)
class Trade:
    """A closed simulated trade; return_pct is net of fees when a fee model was used."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    qty: int
    reason: str
    fees: float = 0.0

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
             fees: Optional[Callable[[str, float], float]] = None) -> Result:
    """Replay signals through the real risk engine: next-open fills, stop/target levels off the signal-day close, optional fees."""
    engine = RiskEngine(limits)
    calendar = sorted(set.intersection(*[set(df.index) for df in bars.values()]))
    cash, peak, prev_equity = start_equity, start_equity, start_equity
    positions: dict = {}
    trades: List[Trade] = []
    curve = {}

    def close_position(sym, day, price, reason):
        """Close a simulated position at a price and record the trade."""
        nonlocal cash
        pos = positions.pop(sym)
        exit_fee = fees("SELL", pos["qty"] * price) if fees else 0.0
        cash += pos["qty"] * price - exit_fee
        trades.append(Trade(sym, pos["entry_date"], pos["entry_price"], day, price, pos["qty"], reason,
                            fees=exit_fee + pos["entry_fee"]))

    def portfolio_at(day, column):
        """The portfolio as the risk engine would see it at the day's open or close prices."""
        values = {s: p["qty"] * bars[s].loc[day, column] for s, p in positions.items()}
        return Portfolio(cash, cash + sum(values.values()), values, {s: p["qty"] for s, p in positions.items()},
                         prev_equity, peak)

    for i, day in enumerate(calendar):
        prev = calendar[i - 1] if i else None

        if prev is not None:
            for sym in list(positions):
                sig = signals.get(sym, {}).get(prev)
                if sig and sig[0] == "SELL" and engine.evaluate("SELL", sig[1], sym, 1.0, portfolio_at(day, "Open")).approved:
                    close_position(sym, day, bars[sym].loc[day, "Open"], "SIGNAL")

            for sym in sorted(bars):
                sig = signals.get(sym, {}).get(prev)
                if not sig or sig[0] != "BUY" or sym in positions:
                    continue
                ref = float(bars[sym].loc[prev, "Close"])
                decision = engine.evaluate("BUY", sig[1], sym, ref, portfolio_at(day, "Open"))
                if not decision.approved:
                    continue
                open_price = float(bars[sym].loc[day, "Open"])
                qty = min(decision.quantity, int(cash // open_price))
                while qty > 0 and fees and qty * open_price + fees("BUY", qty * open_price) > cash:
                    qty -= 1
                if qty <= 0:
                    continue
                entry_fee = fees("BUY", qty * open_price) if fees else 0.0
                cash -= qty * open_price + entry_fee
                positions[sym] = dict(qty=qty, entry_price=open_price, entry_date=day, days=0, entry_fee=entry_fee,
                                      stop=ref * (1 - limits.stop_loss_pct), target=ref * (1 + limits.take_profit_pct))

        for sym in list(positions):
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
                close_position(sym, day, row.Close, "TIME")

        equity = cash + sum(p["qty"] * bars[s].loc[day, "Close"] for s, p in positions.items())
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
