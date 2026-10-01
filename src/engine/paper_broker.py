"""Simulated broker for markets with no broker sandbox (India).

Simplifications, all optimistic and worth remembering: fills happen at the latest 5-minute close plus a fixed slippage;
stop/target exits fill at the stop/target level (or the bar open on a gap) with no slippage; no circuit-limit modelling
(a real stock locked at its lower circuit may not let you sell at your stop). Splits, bonuses and dividends are applied
on their ex-dates (`corporate_actions`), without which a 1:1 bonus halves the price and fires the stop as a fake loss;
fractional entitlements are not paid out (in reality they are settled in cash).

Stop/target replay: every 5-minute bar after the buy has its full high/low examined. The bar still forming is examined
too (its highs and lows so far are real prints, so a breach is not delayed) but is only marked as checked once it has
finished, so the rest of its range is seen next time. The bar the buy happened in is never examined: it also holds
prints from before the fill."""
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func, select

from src.engine.costs import SLIPPAGE, india_delivery_fees
from src.engine.enums import Action, OrderStatus
from src.engine.ports import BAR, Fill, MarketClock, PriceFeed
from src.database import PaperAccountRecord, PaperPositionRecord, PaperTradeRecord
from src.engine.risk_engine import Portfolio

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
# (splits, dividends) for one stock: ex-date -> split ratio (2.0 for a 2:1 split or a 1:1 bonus), ex-date -> rupees a share
CorporateActions = Tuple[Dict[date, float], Dict[date, float]]


def _utc(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; everything we store is UTC, so make them aware before comparing to market data."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _newest_finished(bars: pd.DataFrame, now: datetime) -> Optional[datetime]:
    """Start time (UTC) of the newest bar that had finished by `now`, or None when every bar is still in progress."""
    newest: Optional[datetime] = None
    for ts in bars.index:
        start = _utc(ts.to_pydatetime()).astimezone(timezone.utc)
        if start + BAR <= now and (newest is None or start > newest):
            newest = start
    return newest


@dataclass
class PaperBroker:
    """Simulated broker for markets with no sandbox: state kept in the database."""
    sessions: Callable
    feed: PriceFeed
    clock: MarketClock
    initial_cash: float = 1_000_000.0
    fees: Callable[[str, float], float] = india_delivery_fees
    slippage: float = SLIPPAGE
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    # Annual rate idle cash earns (a liquid mutual fund sweep: direct plan held outside demat, so no DP charge or STT
    # when redeemed). 0 = cash earns nothing, the original behaviour. Credited per elapsed calendar day, like the
    # backtest (src.research.backtest.simulate's cash_yield).
    cash_yield: float = 0.0
    corporate_actions: Optional[Callable[[str], CorporateActions]] = None  # None: not modelled (tests, intraday)
    actions_ttl_seconds: float = 3600.0  # how long one stock's corporate actions are reused before asking again
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self):
        self._lock = threading.RLock()
        self._actions_cache: Dict[str, Tuple[float, CorporateActions]] = {}
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

    def _accrue_interest(self, acct: PaperAccountRecord) -> None:
        """Credit idle cash with the sweep rate for the time since the last credit (no-op when cash_yield is 0).

        The first call only starts the clock: interest is never back-dated to before the sweep was switched on."""
        if self.cash_yield <= 0:
            return
        now = self.now_fn()
        if acct.interest_accrued_at is not None and acct.cash > 0:
            days = (now - _utc(acct.interest_accrued_at)).total_seconds() / 86400
            if days > 0:
                interest = acct.cash * ((1 + self.cash_yield) ** (days / 365) - 1)
                acct.cash += interest
                acct.interest_earned = (acct.interest_earned or 0.0) + interest
        acct.interest_accrued_at = now

    def portfolio(self) -> Portfolio:
        """Apply corporate actions, settle any hit stops/targets, credit idle-cash interest, then value the account."""
        with self._lock:
            self._apply_corporate_actions()
            self._settle_exits()
            with self.sessions() as s:
                acct = s.get(PaperAccountRecord, 1)
                self._accrue_interest(acct)
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
            cost = value + self.fees(Action.BUY, value)
            now = self.now_fn()
            with self.sessions() as s:
                acct = s.get(PaperAccountRecord, 1)
                if s.get(PaperPositionRecord, symbol) is not None:
                    raise ValueError(f"{symbol}: position already open")
                self._accrue_interest(acct)  # interest on the balance as it stood until now, before it changes
                if cost > acct.cash:
                    raise ValueError(f"insufficient cash: need {cost:.2f}, have {acct.cash:.2f}")
                acct.cash -= cost
                s.add(PaperPositionRecord(symbol=symbol, qty=qty, avg_price=fill_price,
                                          stop=round(price * (1 - stop_pct), 2), target=round(price * (1 + take_pct), 2),
                                          opened_at=now, last_checked=now))
                s.commit()
            return Fill(f"paper-{uuid.uuid4().hex[:12]}", OrderStatus.FILLED, qty, fill_price)

    def sell(self, symbol: str, qty: int) -> Fill:
        """Simulated market sell of some or all of a position."""
        with self._lock:
            with self.sessions() as s:
                pos = s.get(PaperPositionRecord, symbol)
                if pos is None or qty > pos.qty:
                    raise ValueError(f"{symbol}: cannot sell {qty}, holding {pos.qty if pos else 0}")
            price = self.feed.last_price(symbol) * (1 - self.slippage)
            self._close(symbol, qty, price, "SIGNAL")
            return Fill(f"paper-{uuid.uuid4().hex[:12]}", OrderStatus.FILLED, qty, price)

    def unprotected_positions(self) -> List[Tuple[str, int, float]]:
        """Always empty: every simulated position is opened with its stop level and checked against 5-minute bars."""
        return []

    def protect(self, symbol: str, qty: int, stop_price: float) -> Fill:
        """Not applicable: simulated positions always carry their stop."""
        raise NotImplementedError("the paper broker's positions are always protected")

    # ---- internals ----------------------------------------------------------------------------------
    def _actions_for(self, symbol: str) -> Optional[CorporateActions]:
        assert self.corporate_actions is not None
        hit = self._actions_cache.get(symbol)
        if hit is not None and self.monotonic() - hit[0] < self.actions_ttl_seconds:
            return hit[1]
        try:
            events = self.corporate_actions(symbol)
        except Exception as e:
            log.warning("could not read corporate actions for %s: %s", symbol, e)
            return None
        self._actions_cache[symbol] = (self.monotonic(), events)
        return events

    def _apply_corporate_actions(self) -> None:
        """Bring every open position through the splits, bonuses and dividends whose ex-date has passed since it was
        opened (or since the last one applied). A split scales the quantity and divides the stop and target by its
        ratio, keeping the cost basis; a dividend is credited as cash. Must run before stops are replayed: the price
        feed is already split-adjusted."""
        if self.corporate_actions is None:
            return
        today = date.fromisoformat(self.clock.today_ist(self.now_fn()))
        with self.sessions() as s:
            held = [(p.symbol, date.fromisoformat(p.actions_through) if p.actions_through
                     else _utc(p.opened_at).astimezone(IST).date()) for p in s.scalars(select(PaperPositionRecord))]
        for symbol, through in held:
            events = self._actions_for(symbol)
            if events is None:
                continue
            splits, dividends = ({d: v for d, v in e.items() if through < d <= today} for e in events)
            if not splits and not dividends:
                continue
            ratio = math.prod(r for r in splits.values() if r > 0)
            with self.sessions() as s:
                pos, acct = s.get(PaperPositionRecord, symbol), s.get(PaperAccountRecord, 1)
                if pos is None:
                    continue
                if ratio != 1:
                    new_qty = int(pos.qty * ratio)
                    if new_qty < 1:
                        log.warning("%s: a %.4g-for-1 consolidation leaves less than one share of %d; not modelled",
                                    symbol, ratio, pos.qty)
                    else:
                        log.info("%s: split/bonus %.4g-for-1: %d -> %d shares", symbol, ratio, pos.qty, new_qty)
                        pos.avg_price = pos.qty * pos.avg_price / new_qty
                        pos.qty = new_qty
                        pos.stop, pos.target = round(pos.stop / ratio, 2), round(pos.target / ratio, 2)
                amount = sum(dividends.values()) * pos.qty  # Yahoo's dividends are per share after any split
                if amount > 0:
                    self._accrue_interest(acct)
                    acct.cash += amount
                    acct.dividends_received = (acct.dividends_received or 0.0) + amount
                    log.info("%s: dividend %.2f credited on %d shares", symbol, amount, pos.qty)
                pos.actions_through = max([*splits, *dividends]).isoformat()
                s.commit()

    def _close(self, symbol: str, qty: int, price: float, reason: str) -> None:
        now = self.now_fn()
        with self.sessions() as s:
            pos = s.get(PaperPositionRecord, symbol)
            acct = s.get(PaperAccountRecord, 1)
            proceeds = qty * price
            fees = self.fees(Action.SELL, proceeds)
            entry_fees = self.fees(Action.BUY, qty * pos.avg_price)
            self._accrue_interest(acct)  # before the proceeds land, so they earn nothing for time they were not held
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
        """Replay the 5-minute bars since each position was last checked to see whether its stop or target was hit."""
        now = _utc(self.now_fn())
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
            exit_price: Optional[float] = None
            reason: Optional[str] = None
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
            if reason is not None and exit_price is not None:
                log.info("paper %s hit on %s at %.2f", reason, symbol, exit_price)
                self._close(symbol, qty, exit_price, reason)
            else:
                finished = _newest_finished(bars, now)  # the bar still forming is looked at again next time
                if finished is not None:
                    with self.sessions() as s:
                        pos = s.get(PaperPositionRecord, symbol)
                        pos.last_checked = finished
                        s.commit()

    def holdings(self) -> List[dict]:
        """Open positions in detail: buy date and price, live price, profit or loss (amount and %), stop level."""
        with self._lock:
            self._apply_corporate_actions()
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
                "interest_earned": acct.interest_earned or 0.0,
                "dividends_received": acct.dividends_received or 0.0,
            }
