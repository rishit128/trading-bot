"""The deterministic risk gate. It is the only place order quantity is decided."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.config import RiskLimits
from src.engine.enums import Action


@dataclass(frozen=True)
class Portfolio:
    """Account state the risk engine judges a trade against."""
    cash: float
    equity: float
    positions: dict = field(default_factory=dict)  # symbol -> market value in account currency
    position_qty: dict = field(default_factory=dict)  # symbol -> shares held
    start_of_day_equity: Optional[float] = None
    peak_equity: Optional[float] = None


@dataclass(frozen=True)
class RiskDecision:
    """Approved or rejected, with the share quantity and the reason."""
    approved: bool
    quantity: int
    reason: str


def drawdown_pause(peak: float, equity: float, halted_since, today, limit: float, pause_days: float):
    """The drawdown circuit breaker's cool-off, shared by the live pipeline and the simulator.

    New buys stop while equity is `limit` below its peak. With no rule to end that, the halt was permanent: an idle
    account earns nothing, so it never recovers (in a 9-year replay the bot stopped buying in March 2018 and never
    traded again). After `pause_days` calendar days of continuous halt the peak is rebased to today's equity and buying
    resumes; the limit then applies to losses from that new base. `pause_days <= 0` keeps the permanent halt.

    `today` and `halted_since` are dates (or timestamps). Returns (peak, halted_since)."""
    if pause_days <= 0 or peak <= 0:
        return peak, halted_since
    if (peak - equity) / peak < limit:
        return peak, None  # not in drawdown (or recovered): nothing pending
    if halted_since is None:
        return peak, today
    if (today - halted_since).days >= pause_days:
        return equity, None
    return peak, halted_since


def _reject(reason: str) -> RiskDecision:
    return RiskDecision(False, 0, reason)


class RiskEngine:
    """Deterministic gate. The only source of order quantity: callers must not size trades themselves."""

    def __init__(self, limits: RiskLimits, fees=None):
        """`fees(side, value)` is the broker's fee schedule; with it, buys whose round-trip fees are too large a share
        of the position are refused (limits.max_fee_drag_pct). Without it that rule is simply not applied."""
        self.limits = limits
        self.fees = fees

    def evaluate(self, action: str, confidence: float, symbol: str, price: float, portfolio: Portfolio) -> RiskDecision:
        """Decide whether, and how much, to trade for a BUY or SELL signal."""
        if action == Action.HOLD:
            return _reject("HOLD: nothing to do")
        if action not in (Action.BUY, Action.SELL):
            return _reject(f"unknown action {action!r}")
        if price <= 0 or portfolio.equity <= 0:
            return _reject("invalid price or equity")
        if confidence < self.limits.min_confidence:
            return _reject(f"confidence {confidence:.2f} below minimum {self.limits.min_confidence:.2f}")

        if action == Action.SELL:
            held = portfolio.position_qty.get(symbol, 0)
            if held <= 0:
                return _reject(f"no {symbol} position to sell (shorting disabled)")
            return RiskDecision(True, int(held), f"closing full {symbol} position")

        return self._evaluate_buy(symbol, confidence, price, portfolio)

    def halt_reason(self, p: Portfolio) -> Optional[Tuple[str, str]]:
        """(kind, message) when new buys are blocked at portfolio level; kind is stable across cycles."""
        lim = self.limits
        if p.start_of_day_equity and p.start_of_day_equity > 0:
            day_loss = (p.start_of_day_equity - p.equity) / p.start_of_day_equity
            if day_loss >= lim.max_daily_loss_pct:
                return "daily_loss", f"daily loss {day_loss:.2%} reached limit {lim.max_daily_loss_pct:.2%}"
        if p.peak_equity and p.peak_equity > 0:
            drawdown = (p.peak_equity - p.equity) / p.peak_equity
            if drawdown >= lim.max_drawdown_pct:
                return "drawdown", f"drawdown {drawdown:.2%} reached limit {lim.max_drawdown_pct:.2%}"
        return None

    def position_pct(self, confidence: float) -> float:
        """Position size as a fraction of equity, scaled linearly with confidence: min_position_pct at
        min_confidence, max_position_pct at confidence 1.0 (and beyond min_confidence..1.0, clamped)."""
        lim = self.limits
        span = 1.0 - lim.min_confidence
        frac = (confidence - lim.min_confidence) / span if span > 0 else 1.0
        frac = min(1.0, max(0.0, frac))
        return lim.min_position_pct + frac * (lim.max_position_pct - lim.min_position_pct)

    def _evaluate_buy(self, symbol: str, confidence: float, price: float, p: Portfolio) -> RiskDecision:
        lim = self.limits

        halt = self.halt_reason(p)
        if halt:
            return _reject(halt[1])

        if symbol not in p.positions and len(p.positions) >= lim.max_open_positions:
            return _reject(f"max open positions reached ({lim.max_open_positions})")

        pct = self.position_pct(confidence)
        existing_value = p.positions.get(symbol, 0.0)
        position_room = p.equity * pct - existing_value
        total_exposure = sum(p.positions.values())
        exposure_room = p.equity * lim.max_portfolio_exposure_pct - total_exposure
        budget = min(position_room, exposure_room, p.cash)

        if budget < price:
            return _reject(
                f"no room: position_room={position_room:,.0f} exposure_room={exposure_room:,.0f} cash={p.cash:,.0f}"
            )
        qty = int(budget // price)
        if self.fees is not None and lim.max_fee_drag_pct > 0:
            value = qty * price
            drag = (self.fees(Action.BUY, value) + self.fees(Action.SELL, value)) / value
            if drag > lim.max_fee_drag_pct:
                return _reject(f"fees would cost {drag:.1%} of a {value:,.0f} position (limit {lim.max_fee_drag_pct:.1%}): too small to be worth trading")
        return RiskDecision(True, qty, f"approved {qty} shares (value {qty * price:,.0f}, sized at {pct:.1%} for confidence {confidence:.2f})")
