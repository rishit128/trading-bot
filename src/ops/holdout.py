"""The holdout reserve account - one independent ruler that live tuning can never contaminate.

The paper account measures the bot, but every learning/tuning decision in this repo is made *against*
that account's outcomes (learning phase, calibration, reconciliation). If those decisions are then
graded on the same data, the validation is self-referential. The reserve is the escape: it holds no
orders and is never tuned; it is a *re-run* of the stored approved decisions through the deterministic
simulator, so whatever the model or the process changes next, the reserve only ever answers "did the
decisions themselves make money?" - in the execution convention every backtest uses (src/engine/costs.py), sealed as of
each mark.

    HoldoutReserve(sessions, fetch_bars).mark()      # replay decisions -> append a HoldoutMark
    HoldoutReserve(sessions, fetch_bars).report()    # the reserve's history + drift vs the paper account

`fetch_bars(symbol)` returns a daily OHLCV frame (the same shape yfinance/backtests use) or None when
the symbol cannot be priced; a symbol with no bars is skipped, never guessed. The reserve seeds from the
paper account's own starting cash so returns are directly comparable."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.database import DecisionRecord, HoldoutMark, PaperAccountRecord
from src.engine.costs import SLIPPAGE, india_delivery_fees
from src.engine.enums import Action
from src.learning.mistakes import rule_action_from_snapshot
from src.research.backtest import BuySource, RiskLimits, SignalTable, simulate

log = logging.getLogger(__name__)

DEFAULT_MIN_CONFIDENCE = 0.60


def _bar_date(created_at) -> Optional[pd.Timestamp]:
    """A decision expects a daily bar on its UTC calendar day; return that naive timestamp (or None)."""
    if created_at is None:
        return None
    dt = created_at if created_at.tzinfo is None else created_at.astimezone(timezone.utc)
    return pd.Timestamp(dt.year, dt.month, dt.day)


def _picks_by_day(
    picks: List[Tuple[DecisionRecord, float]], bars: Dict[str, pd.DataFrame]
) -> Dict[pd.Timestamp, List[Tuple[str, float]]]:
    """Per decision day, the (symbol, confidence) picks in the order recorded, one per stock per day, for stocks that have a
    bar on that day."""
    out: Dict[pd.Timestamp, List[Tuple[str, float]]] = {}
    seen = set()
    for d, confidence in picks:
        day = _bar_date(d.created_at)
        key = (d.symbol, day)
        if day is None or key in seen or d.symbol not in bars or day not in bars[d.symbol].index:
            continue
        seen.add(key)
        out.setdefault(day, []).append((d.symbol, confidence))
    return out


def _buy_source(queue: Dict[pd.Timestamp, List[Tuple[str, float]]]) -> BuySource:
    def source(day: pd.Timestamp, _portfolio) -> List[Tuple[str, float]]:
        return queue.get(day, [])

    return source


@dataclass
class HoldoutReserve:
    """Decision-only re-simulator producing point-in-time equity marks that live tuning never touches."""

    sessions: Callable
    fetch_bars: Callable[[str], Optional[pd.DataFrame]]  # symbol -> daily OHLCV or None
    initial_cash: Optional[float] = None  # None -> seed from the paper account's own starting cash
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    slippage: float = SLIPPAGE  # only market fills pay it (buy high, sell low); see engine/costs.py
    fees: Callable[[str, float], float] = india_delivery_fees
    cash_yield: float = 0.0  # match the paper account's CASH_YIELD_PCT, or the drift between the two is not like for like
    starting_cash: float = field(init=False, default=0.0)  # the resolved starting balance (initial_cash, or the paper account's)

    def __post_init__(self):
        if self.initial_cash is None:
            with self.sessions() as s:
                paper = s.get(PaperAccountRecord, 1)
            self.initial_cash = float(paper.initial_cash) if paper else 100_000.0
        self.starting_cash = float(self.initial_cash)

    def _decisions(self) -> list:
        with self.sessions() as s:
            return list(s.scalars(select(DecisionRecord).order_by(DecisionRecord.id)))

    def _approved_buys(self, decisions: Optional[list] = None) -> list:
        return [
            d
            for d in (self._decisions() if decisions is None else decisions)
            if d.final_action == Action.BUY and d.risk_approved and d.final_confidence >= self.min_confidence
        ]

    def _load_bars(self, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
        """Daily bars for each symbol that can be priced; one that cannot is left out, never guessed."""
        bars: Dict[str, pd.DataFrame] = {}
        for sym in sorted(set(symbols)):
            try:
                df = self.fetch_bars(sym)
            except Exception as e:
                log.warning("holdout: no bars for %s (%s); excluded", sym, e)
                df = None
            if df is not None and not df.empty:
                bars[sym] = df
        return bars

    def mark(self) -> dict:
        """Replay every stored approved BUY through the simulator and append one HoldoutMark.

        Deterministic: given the same decisions and bars, repeated calls yield the same values (only the
        timestamped ledger rows accumulate). Never reads or writes the paper account's orders/positions."""
        buys = self._approved_buys()
        bars = self._load_bars(d.symbol for d in buys)
        signals: SignalTable = {}
        for d in buys:
            day = _bar_date(d.created_at)
            if day is None or d.symbol not in bars or day not in bars[d.symbol].index:
                continue
            signals.setdefault(d.symbol, {})[day] = (Action.BUY, d.final_confidence)

        equity = self.starting_cash
        trades, open_positions = 0, 0
        if signals:  # not `bars`: bars can exist for a symbol whose decision day never matched an index date (e.g. a
            # fresh BUY made before that day's bar was posted) - simulating then would let cash_yield accrue
            # from day one of the whole fetched history (cash_yield_from=None), a fictitious return for a
            # mark that in truth replayed nothing.
            first_decision = min((day for days in signals.values() for day in days), default=None)
            result = simulate(
                bars,
                signals,
                RiskLimits(min_confidence=self.min_confidence),
                start_equity=self.starting_cash,
                fees=self.fees,
                slippage=self.slippage,
                cash_yield=self.cash_yield,
                cash_yield_from=first_decision,
            )
            if len(result.equity):
                equity = float(result.equity.iloc[-1])
            trades, open_positions = len(result.trades), result.open_at_end

        return_pct = equity / self.starting_cash - 1
        with self.sessions() as s:
            s.add(
                HoldoutMark(
                    equity=equity,
                    return_pct=return_pct,
                    decisions=len(signals),
                    closed_trades=trades,
                    open_positions=open_positions,
                )
            )
            s.commit()
        return {
            "equity": equity,
            "return_pct": return_pct,
            "decisions": len(signals),
            "closed_trades": trades,
            "open_positions": open_positions,
        }

    def compare(self) -> Optional[dict]:
        """The AI against the plain entry rule, on the same stocks and days: the recorded AI-approved BUYs replayed beside
        every BUY the rule alone would have made (the first per stock per day, at full confidence), through the same
        simulator, risk limits, costs and best-first order (the order the bot analysed them in). Nothing is written. None
        when there is nothing to replay; else each arm's final equity and trades, and the paired difference of their daily
        returns (AI minus rule): its mean, t statistic and number of days."""
        decisions = self._decisions()
        arms = {
            "ai": [(d, d.final_confidence) for d in self._approved_buys(decisions)],
            "rule": [(d, 1.0) for d in decisions if (d.rule_action or rule_action_from_snapshot(d.snapshot_json)) == Action.BUY],
        }
        bars = self._load_bars(d.symbol for picks in arms.values() for d, _ in picks)
        queues = {name: _picks_by_day(picks, bars) for name, picks in arms.items()}
        start = min((day for queue in queues.values() for day in queue), default=None)
        if start is None:
            return None
        limits = RiskLimits(min_confidence=self.min_confidence)
        results = {
            name: simulate(
                bars,
                {},
                limits,
                start_equity=self.starting_cash,
                fees=self.fees,
                slippage=self.slippage,
                cash_yield=self.cash_yield,
                cash_yield_from=start,
                buy_source=_buy_source(queue),
            )
            for name, queue in queues.items()
        }
        daily = {name: r.equity.pct_change().dropna().loc[start:] for name, r in results.items()}  # from the first decision on
        diff = (daily["ai"] - daily["rule"]).dropna()
        spread = float(diff.std()) if len(diff) > 1 else 0.0
        return {
            **{
                f"{name}_{key}": value
                for name, r in results.items()
                for key, value in (("equity", float(r.equity.iloc[-1])), ("trades", len(r.trades)))
            },
            "days": len(diff),
            "mean_diff": float(diff.mean()) if len(diff) else 0.0,
            "t": float(diff.mean() / (spread / np.sqrt(len(diff)))) if spread > 1e-12 else 0.0,
        }

    def report(self, limit: int = 10) -> dict:
        """The reserve's own history (newest first) plus its latest drift vs the paper account's last mark."""
        with self.sessions() as s:
            marks = s.scalars(
                select(HoldoutMark).order_by(HoldoutMark.recorded_at.desc(), HoldoutMark.id.desc()).limit(limit)
            ).all()
            paper = s.get(PaperAccountRecord, 1)
        supply = {
            "initial_cash": self.initial_cash,
            "marks": [
                {
                    "recorded_at": m.recorded_at,
                    "equity": m.equity,
                    "return_pct": m.return_pct,
                    "decisions": m.decisions,
                    "closed_trades": m.closed_trades,
                    "open_positions": m.open_positions,
                }
                for m in marks
            ],
        }
        if not marks:
            supply["drift_vs_paper"] = None
            return supply
        paper_mark = float(paper.last_mark) if paper and paper.last_mark is not None else None
        supply["drift_vs_paper"] = (marks[0].equity - paper_mark) if paper_mark is not None else None
        return supply
