"""Simulated broker for markets with no broker sandbox (India).

Simplifications, all optimistic and worth remembering: fills happen at the latest 5-minute close plus a fixed slippage;
stop/target exits fill at the stop/target level (or the bar open on a gap) with no slippage; no circuit-limit modelling
(a real stock locked at its lower circuit may not let you sell at your stop)."""
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from sqlalchemy import func, select

from src.database import PaperAccountRecord, PaperPositionRecord, PaperTradeRecord
from src.engine.risk_engine import Portfolio

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    """What the broker reported for an order: id, status, and the quantity and price actually filled when known."""
    broker_order_id: str
    status: str
    filled_qty: Optional[int] = None
    avg_price: Optional[float] = None


def _utc(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; everything we store is UTC, so make them aware before comparing to market data."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def india_delivery_fees(side: str, value: float) -> float:
    """Approximate NSE delivery (CNC) charges with a zero-brokerage discount broker: STT 0.1% both sides, stamp duty
    0.015% on buys, exchange 0.00297%, SEBI 0.0001%, 18% GST on those two, DP charge Rs 15.93 per sell. Rates change."""
    stt = 0.001 * value
    stamp = 0.00015 * value if side == "BUY" else 0.0
    exchange, sebi = 0.0000297 * value, 0.000001 * value
    gst = 0.18 * (exchange + sebi)
    dp = 15.93 if side == "SELL" else 0.0
    return stt + stamp + exchange + sebi + gst + dp


@dataclass
class PaperBroker:
    """Simulated broker for markets with no sandbox: state kept in the database."""
    sessions: Callable
    feed: object  # needs last_price(symbol) and bars_since(symbol, since)
    clock: object  # needs is_open() and today_ist()
    initial_cash: float = 1_000_000.0
    fees: Callable[[str, float], float] = india_delivery_fees
    slippage: float = 0.0005
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def __post_init__(self):
        self._lock = threading.RLock()
        with self.sessions() as s:
            if s.get(PaperAccountRecord, 1) is None:
                s.add(PaperAccountRecord(id=1, cash=self.initial_cash, initial_cash=self.initial_cash))
                s.commit()

    # ---- interface used by the pipeline -------------------------------------------------------------
    def is_market_open(self) -> bool:
        """True when the market is open (via the market clock)."""
        return self.clock.is_open(self.now_fn())

    def has_open_order(self, symbol: str) -> bool:
        """A held position has its stop/target 'legs' working, like a broker-side bracket."""
        with self.sessions() as s:
            return s.get(PaperPositionRecord, symbol) is not None

    def portfolio(self) -> Portfolio:
        """Settle any hit stops/targets, then value the account at live prices."""
        with self._lock:
            self._settle_exits()
            with self.sessions() as s:
                acct = s.get(PaperAccountRecord, 1)
                positions = s.scalars(select(PaperPositionRecord)).all()
                values, qtys = {}, {}
                for p in positions:
                    try:
                        price = self.feed.last_price(p.symbol)
                    except Exception as e:
                        log.warning("no price for %s, valuing at cost: %s", p.symbol, e)
                        price = p.avg_price
                    values[p.symbol], qtys[p.symbol] = p.qty * price, p.qty
                equity = acct.cash + sum(values.values())
                today = self.clock.today_ist(self.now_fn())
                if acct.day != today:
                    acct.day_start_equity = acct.last_mark if acct.last_mark is not None else equity
                    acct.day = today
                acct.last_mark = equity
                s.commit()
                return Portfolio(cash=acct.cash, equity=equity, positions=values, position_qty=qtys,
                                 start_of_day_equity=acct.day_start_equity)

    def buy_with_bracket(self, symbol: str, qty: int, price: float, stop_pct: float, take_pct: float) -> Fill:
        """Simulated market buy with slippage and Indian costs; records the stop and target levels."""
        with self._lock:
            fill_price = self.feed.last_price(symbol) * (1 + self.slippage)
            value = qty * fill_price
            cost = value + self.fees("BUY", value)
            now = self.now_fn()
            with self.sessions() as s:
                acct = s.get(PaperAccountRecord, 1)
                if s.get(PaperPositionRecord, symbol) is not None:
                    raise ValueError(f"{symbol}: position already open")
                if cost > acct.cash:
                    raise ValueError(f"insufficient cash: need {cost:.2f}, have {acct.cash:.2f}")
                acct.cash -= cost
                s.add(PaperPositionRecord(symbol=symbol, qty=qty, avg_price=fill_price,
                                          stop=round(price * (1 - stop_pct), 2), target=round(price * (1 + take_pct), 2),
                                          opened_at=now, last_checked=now))
                s.commit()
            return Fill(f"paper-{uuid.uuid4().hex[:12]}", "filled", qty, fill_price)

    def sell(self, symbol: str, qty: int) -> Fill:
        """Simulated market sell of some or all of a position."""
        with self._lock:
            with self.sessions() as s:
                pos = s.get(PaperPositionRecord, symbol)
                if pos is None or qty > pos.qty:
                    raise ValueError(f"{symbol}: cannot sell {qty}, holding {pos.qty if pos else 0}")
            price = self.feed.last_price(symbol) * (1 - self.slippage)
            self._close(symbol, qty, price, "SIGNAL")
            return Fill(f"paper-{uuid.uuid4().hex[:12]}", "filled", qty, price)

    def unprotected_positions(self) -> List[Tuple[str, int, float]]:
        """Always empty: every simulated position is opened with its stop level and checked against 5-minute bars."""
        return []

    def protect(self, symbol: str, qty: int, stop_price: float) -> Fill:
        """Not applicable: simulated positions always carry their stop."""
        raise NotImplementedError("the paper broker's positions are always protected")

    # ---- internals ----------------------------------------------------------------------------------
    def _close(self, symbol: str, qty: int, price: float, reason: str) -> None:
        now = self.now_fn()
        with self.sessions() as s:
            pos = s.get(PaperPositionRecord, symbol)
            acct = s.get(PaperAccountRecord, 1)
            proceeds = qty * price
            fees = self.fees("SELL", proceeds)
            entry_fees = self.fees("BUY", qty * pos.avg_price)
            acct.cash += proceeds - fees
            s.add(PaperTradeRecord(symbol=symbol, qty=qty, entry_price=pos.avg_price, exit_price=price,
                                   opened_at=pos.opened_at, closed_at=now, reason=reason, fees=fees + entry_fees,
                                   net_pnl=(price - pos.avg_price) * qty - fees - entry_fees))
            if qty >= pos.qty:
                s.delete(pos)
            else:
                pos.qty -= qty
            s.commit()

    def _settle_exits(self) -> None:
        """Replay 5-minute bars since each position was last checked to see whether its stop or target was hit."""
        with self.sessions() as s:
            positions = [(p.symbol, p.qty, p.stop, p.target, _utc(p.last_checked))
                         for p in s.scalars(select(PaperPositionRecord))]
        for symbol, qty, stop, target, last_checked in positions:
            try:
                bars = self.feed.bars_since(symbol, last_checked)
            except Exception as e:
                log.warning("could not check exits for %s: %s", symbol, e)
                continue
            if bars is None or len(bars) == 0:
                continue
            exit_price, reason = None, None
            for _, bar in bars.iterrows():
                if bar["Open"] <= stop:
                    exit_price, reason = float(bar["Open"]), "STOP"
                elif bar["Open"] >= target:
                    exit_price, reason = float(bar["Open"]), "TARGET"
                elif bar["Low"] <= stop:
                    exit_price, reason = stop, "STOP"
                elif bar["High"] >= target:
                    exit_price, reason = target, "TARGET"
                if reason:
                    break
            if reason:
                log.info("paper %s hit on %s at %.2f", reason, symbol, exit_price)
                self._close(symbol, qty, exit_price, reason)
            else:
                with self.sessions() as s:
                    pos = s.get(PaperPositionRecord, symbol)
                    pos.last_checked = self.now_fn()
                    s.commit()

    def holdings(self) -> List[dict]:
        """Open positions in detail: buy date and price, live price, profit or loss (amount and %), stop level."""
        with self._lock:
            self._settle_exits()
            with self.sessions() as s:
                rows = s.scalars(select(PaperPositionRecord)).all()
                out = []
                for p in rows:
                    try:
                        price = self.feed.last_price(p.symbol)
                    except Exception as e:
                        log.warning("no price for %s, showing cost: %s", p.symbol, e)
                        price = p.avg_price
                    out.append({"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price, "price": price,
                                "value": p.qty * price, "pnl": p.qty * (price - p.avg_price),
                                "pnl_pct": price / p.avg_price - 1, "stop": p.stop, "opened_at": _utc(p.opened_at)})
                return sorted(out, key=lambda h: h["pnl"], reverse=True)

    def trade_history(self, limit: Optional[int] = None) -> List[dict]:
        """Closed trades, newest first (optionally only the latest `limit`): dates, prices, why, fees and net P&L."""
        with self.sessions() as s:
            query = select(PaperTradeRecord).order_by(PaperTradeRecord.closed_at.desc(), PaperTradeRecord.id.desc())
            trades = s.scalars(query.limit(limit) if limit else query).all()
            return [{"symbol": t.symbol, "qty": t.qty, "entry_price": t.entry_price, "exit_price": t.exit_price,
                     "opened_at": _utc(t.opened_at), "closed_at": _utc(t.closed_at), "reason": t.reason,
                     "fees": t.fees, "net_pnl": t.net_pnl} for t in trades]

    def summary(self) -> dict:
        """Account summary for --report: equity, return, trades, win rate, fees, exits."""
        with self._lock:
            pf = self.portfolio()
            with self.sessions() as s:
                acct = s.get(PaperAccountRecord, 1)
                trades = s.scalars(select(PaperTradeRecord)).all()
                fees = s.scalar(select(func.coalesce(func.sum(PaperTradeRecord.fees), 0.0)))
            wins = [t for t in trades if t.net_pnl > 0]
            return {
                "initial_cash": acct.initial_cash, "equity": pf.equity, "cash": pf.cash,
                "return_pct": pf.equity / acct.initial_cash - 1, "open_positions": len(pf.positions),
                "closed_trades": len(trades), "win_rate": (len(wins) / len(trades)) if trades else None,
                "realized_net_pnl": sum(t.net_pnl for t in trades), "fees_paid": fees,
                "exits": {r: sum(1 for t in trades if t.reason == r) for r in ("STOP", "TARGET", "SIGNAL")},
            }
