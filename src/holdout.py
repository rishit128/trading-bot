"""P0: the holdout reserve account - one independent ruler that live tuning can never contaminate.

The paper account measures the bot, but every learning/tuning decision in this repo is made *against*
that account's outcomes (learning phase, calibration, reconciliation). If those decisions are then
graded on the same data, the validation is self-referential. The reserve is the escape: it holds no
orders and is never tuned; it is a *re-run* of the stored approved decisions through the deterministic
simulator, so whatever the model or the process changes next, the reserve only ever answers "did the
decisions themselves make money?" - in the exact D1/D2 conventions used by every backtest, sealed as of
each mark.

    HoldoutReserve(sessions, fetch_bars).mark()      # replay decisions -> append a HoldoutMark
    HoldoutReserve(sessions, fetch_bars).report()    # the reserve's history + drift vs the paper account

`fetch_bars(symbol)` returns a daily OHLCV frame (the same shape yfinance/backtests use) or None when
the symbol cannot be priced; a symbol with no bars is skipped, never guessed. The reserve seeds from the
paper account's own starting cash so returns are directly comparable."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pandas as pd
from sqlalchemy import select

from src.backtest import LIVE_PARITY, RiskLimits, Signals, simulate
from src.database import DecisionRecord, HoldoutMark, PaperAccountRecord
from src.engine.paper_broker import india_delivery_fees
from datetime import timezone

log = logging.getLogger(__name__)

DEFAULT_MIN_CONFIDENCE = 0.60


def _bar_date(created_at) -> Optional[pd.Timestamp]:
    """A decision expects a daily bar on its UTC calendar day; return that naive timestamp (or None)."""
    if created_at is None:
        return None
    dt = created_at if created_at.tzinfo is None else created_at.astimezone(timezone.utc)
    return pd.Timestamp(dt.year, dt.month, dt.day)


@dataclass
class HoldoutReserve:
    """Decision-only re-simulator producing point-in-time equity marks that live tuning never touches."""
    sessions: Callable
    fetch_bars: Callable[[str], Optional[pd.DataFrame]]  # symbol -> daily OHLCV or None
    initial_cash: Optional[float] = None  # None -> seed from the paper account's own starting cash
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    slippage: float = 0.0005  # D1 convention: only market fills pay it (buy high, sell low)
    fees: Callable[[str, float], float] = india_delivery_fees

    def __post_init__(self):
        if self.initial_cash is None:
            with self.sessions() as s:
                paper = s.get(PaperAccountRecord, 1)
            self.initial_cash = float(paper.initial_cash) if paper else 100_000.0

    def _approved_buys(self) -> list:
        with self.sessions() as s:
            decisions = s.scalars(select(DecisionRecord).order_by(DecisionRecord.id)).all()
        return [d for d in decisions
                if d.final_action == "BUY" and d.risk_approved and d.final_confidence >= self.min_confidence]

    def mark(self) -> dict:
        """Replay every stored approved BUY through the simulator and append one HoldoutMark.

        Deterministic: given the same decisions and bars, repeated calls yield the same values (only the
        timestamped ledger rows accumulate). Never reads or writes the paper account's orders/positions."""
        buys = self._approved_buys()
        bars: Dict[str, pd.DataFrame] = {}
        signals: Signals = {}
        for sym in sorted({d.symbol for d in buys}):
            try:
                df = self.fetch_bars(sym)
            except Exception as e:
                log.warning("holdout: no bars for %s (%s); excluded", sym, e)
                df = None
            if df is not None and not df.empty:
                bars[sym] = df
        for d in buys:
            day = _bar_date(d.created_at)
            if day is None or d.symbol not in bars or day not in bars[d.symbol].index:
                continue
            signals.setdefault(d.symbol, {})[day] = ("BUY", d.final_confidence)

        equity = float(self.initial_cash)
        trades, open_positions = 0, 0
        if bars:
            result = simulate(bars, signals, RiskLimits(min_confidence=self.min_confidence),
                              start_equity=self.initial_cash, fees=self.fees, slippage=self.slippage, **LIVE_PARITY)
            if len(result.equity):
                equity = float(result.equity.iloc[-1])
            trades, open_positions = len(result.trades), result.open_at_end

        return_pct = equity / self.initial_cash - 1
        with self.sessions() as s:
            s.add(HoldoutMark(equity=equity, return_pct=return_pct, decisions=len(signals),
                              closed_trades=trades, open_positions=open_positions))
            s.commit()
        return {"equity": equity, "return_pct": return_pct, "decisions": len(signals),
                "closed_trades": trades, "open_positions": open_positions}

    def report(self, limit: int = 10) -> dict:
        """The reserve's own history (newest first) plus its latest drift vs the paper account's last mark."""
        with self.sessions() as s:
            marks = s.scalars(select(HoldoutMark).order_by(HoldoutMark.recorded_at.desc(),
                                                          HoldoutMark.id.desc()).limit(limit)).all()
            paper = s.get(PaperAccountRecord, 1)
        supply = {"initial_cash": self.initial_cash,
                  "marks": [{"recorded_at": m.recorded_at, "equity": m.equity, "return_pct": m.return_pct,
                             "decisions": m.decisions, "closed_trades": m.closed_trades,
                             "open_positions": m.open_positions} for m in marks]}
        if not marks:
            supply["drift_vs_paper"] = None
            return supply
        paper_mark = float(paper.last_mark) if paper and paper.last_mark is not None else None
        supply["drift_vs_paper"] = (marks[0].equity - paper_mark) if paper_mark is not None else None
        return supply