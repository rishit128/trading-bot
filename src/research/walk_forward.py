"""Walk-forward (rolling block) out-of-sample validation harness.

The strategy has no tunable hyperparameters - every indicator keeps its literature default and the
review forbids fitting on data - so training cannot sneak in through parameter selection. The
discipline this module enforces is therefore structural:

  * the decision timeline is carved into non-overlapping blocks;
  * in each fold the hold-out block is evaluated with NO access to future data: learning/history is
    point-in-time (pinned by A1c/B1), and folds only ever see dates up to the block's start;
  * the walk-forward OOS number is ONE continuous equity curve built from every hold-out block
    stitched together - never a cherry-picked window;
  * per-fold metrics are reported so instability between windows is visible instead of averaged away.

Mechanical DET (rule_signals) runs fully offline; pass an `llm` to walk the full AI decision stack."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.research.backtest import RiskLimits, START_EQUITY, SignalTable, simulate, curve_metrics, trade_metrics
from src.engine.costs import SLIPPAGE
from src.engine.costs import india_delivery_fees
from src.research.ablation import DET, PHASES, signals_for_phase

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward hold-out block plus the expanding train segment that preceded it."""
    index: int
    train_dates: Tuple
    val_dates: Tuple
    train: dict = field(default_factory=dict)
    val: dict = field(default_factory=dict)
    val_degraded_share: float = 0.0


@dataclass(frozen=True)
class WalkForwardResult:
    symbol: str
    phase: str
    folds: List[Fold]
    oos: dict
    oos_degraded_share: float = 0.0

    def summary(self) -> str:
        lines = [f"walk-forward {self.phase} on {self.symbol}: {len(self.folds)} non-overlapping hold-out blocks"]
        header = f"{'fold':<5}{'train':<11}{'val':<11}{'val ret':>9}{'val maxDD':>10}{'val trades':>11}{'trades/train':>12}"
        lines.append(header)
        for f in self.folds:
            t, v = f.train, f.val
            lines.append(f"{f.index:<5}{len(f.train_dates):<11}{len(f.val_dates):<11}"
                         f"{v.get('total_return', 0):>9.2%}{v.get('max_drawdown', 0):>10.2%}"
                         f"{v.get('trades', 0):>11}{t.get('trades', 0):>12}")
        lines.append(f"\nOOS (continuous: all hold-out blocks stitched together): "
                     f"return {self.oos.get('total_return', 0):.2%}, max drawdown {self.oos.get('max_drawdown', 0):.2%}, "
                     f"Sharpe {self.oos.get('sharpe', 0):.2f}, trades {self.oos.get('trades', 0)}")
        return "\n".join(lines)


def walk_forward_folds(dates: Sequence, n_windows: int, min_train: int = 0) -> List[Tuple[tuple, tuple]]:
    """Split `dates` into n non-overlapping chronological blocks; fold i trains on everything before block i.

    The first fold's train segment is empty unless `min_train` guarantees earlier dates exist; folds with fewer
    than `min_train` training dates are dropped so no fold reports a meaningless in-sample number."""
    if n_windows < 2 or len(dates) < n_windows:
        raise ValueError(f"need at least {n_windows} decision dates for {n_windows} windows, got {len(dates)}")
    blocks = [list(b) for b in np.array_split(list(dates), n_windows)]
    folds: List[Tuple[tuple, tuple]] = []
    seen_so_far: List = []
    for block in blocks:
        train = tuple(seen_so_far)
        seen_so_far.extend(block)
        if len(train) >= min_train:
            folds.append((train, tuple(block)))
    return folds


def _measure(symbol: str, bars: Dict[str, pd.DataFrame], dates, signals: SignalTable, limits,
             start_equity: float, slippage: float, fees) -> dict:
    result = simulate(bars, signals, limits, start_equity=start_equity, fees=fees, slippage=slippage)
    metrics = dict(curve_metrics(result.equity, start_equity))
    metrics.update(trade_metrics(result.trades))
    metrics["open_positions"] = result.open_at_end  # live exits have no time limit, so entries may still be open
    metrics["decisions"] = len(dates)
    return metrics


def run_walk_forward(
    symbol: str,
    bars: Dict[str, pd.DataFrame],
    dates,
    phase: str = DET,
    n_windows: int = 4,
    min_train: int = 0,
    limits: Optional[RiskLimits] = None,
    llm: object = None,
    sessions=None,
    history_fn_factory: Optional[Callable[[pd.Timestamp], Callable]] = None,
    market_fn: Optional[Callable[[pd.Timestamp], Optional[object]]] = None,
    start_equity: float = START_EQUITY,
    slippage: float = SLIPPAGE,
    fees: Optional[Callable[[str, float], float]] = india_delivery_fees,
) -> WalkForwardResult:
    """Walk one phase through `dates` in non-overlapping hold-out blocks and stitch the OOS curve.

    Every hold-out block calls `signals_for_phase` with only its own dates, so no decision in a block
    can see data from a later block (the simulator is fed only the block's signals)."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {PHASES}")
    if phase != DET and llm is None:
        raise ValueError(f"phase {phase} needs a model; pass llm= (or use {DET})")
    norms = limits if limits is not None else RiskLimits()
    folds = walk_forward_folds(dates, n_windows, min_train)

    built: List[Fold] = []
    oos_signals: SignalTable = {symbol: {}}
    oos_decisions = oos_degraded = 0
    phase_llm = None if phase == DET else llm
    for i, (train_dates, val_dates) in enumerate(folds):
        train_metrics: dict = {}
        if train_dates:
            sig_train, _, _ = signals_for_phase(symbol, bars, train_dates, phase, phase_llm, sessions, history_fn_factory,
                                                market_fn)
            train_metrics = _measure(symbol, bars, train_dates, sig_train, norms, start_equity, slippage, fees)
        sig_val, dec_val, deg_val = signals_for_phase(
            symbol, bars, val_dates, phase, phase_llm, sessions, history_fn_factory, market_fn)
        val_metrics = _measure(symbol, bars, val_dates, sig_val, norms, start_equity, slippage, fees)
        oos_signals[symbol].update(dict(sig_val[symbol]))
        oos_decisions += dec_val
        oos_degraded += deg_val

        # in-sample is reported for contrast only; the headline remains the hold-out blocks
        built.append(Fold(i, tuple(train_dates), tuple(val_dates), train=train_metrics, val=val_metrics,
                          val_degraded_share=deg_val / dec_val if dec_val else 0.0))

    oos = _measure(symbol, bars, list(oos_signals[symbol]), oos_signals, norms, start_equity, slippage, fees)
    oos["decisions"] = oos_decisions
    return WalkForwardResult(symbol, phase, built, oos,
                             oos_degraded_share=oos_degraded / oos_decisions if oos_decisions else 0.0)