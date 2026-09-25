"""Phase ablation over one history window (B1).

The review asks for one historical window in which each incremental capability of the decision
stack is measured alone and then combined, all under identical execution rules:

  DET        mechanical filter only, no AI call
  +LLM       the raw AI call on top of the filter
  +LEARNING  ... plus the closed-trade-history adjustment (decision memory)
  +CONTEXT   ... plus the market-regime adjustment (market context)
  +REFLECTION... plus the six-step self-critique (reflection)
  FULL       everything enabled

Every phase produces signals that feed the same backtest simulator, so phases differ only in
the signals, never in fills (next-open, one-day lag), costs (India delivery fees) or slippage.
LLM-dependent phases degrade to a flagged HOLD whenever the model cannot be reached; the report
exposes that share (`degraded_share`) instead of hiding it.

Historical learning lookups are point-in-time: the history hook only ever sees trades that
closed before the decision date, so decision memory never attributes future results to a past setup."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd

from src.research.backtest import RiskLimits, START_EQUITY, SignalTable, rule_signals, simulate, curve_metrics, trade_metrics
from src.agents.technical import TechnicalAgent
from src.agents.base import AgentContext
from src.agents.history import history_stats
from src.data.indicators import build_snapshot
from src.engine.costs import SLIPPAGE
from src.engine.costs import india_delivery_fees

log = logging.getLogger(__name__)

DET = "DET"
PHASES = [DET, "+LLM", "+LEARNING", "+CONTEXT", "+REFLECTION", "FULL"]

# Each row enables exactly one capability on top of the base AI call (the mechanical filter is
# always present as the prompt's baseline); FULL turns everything on. The other phases are set
# explicitly FALSE - TechnicalAgent defaults them all to True, so omitting one would silently run
# reflection (or learning/context) in a phase that claims to isolate just one capability.
_PHASE_FLAGS: Dict[str, Optional[Dict[str, bool]]] = {
    DET: None,
    "+LLM": {"use_learning": False, "use_context": False, "use_reflect": False},
    "+LEARNING": {"use_learning": True, "use_context": False, "use_reflect": False},
    "+CONTEXT": {"use_learning": False, "use_context": True, "use_reflect": False},
    "+REFLECTION": {"use_learning": False, "use_context": False, "use_reflect": True},
    "FULL": {"use_learning": True, "use_context": True, "use_reflect": True},
}


@dataclass(frozen=True)
class PhaseReport:
    """One phase's full result: equity curve summary, trade summary and how the phase actually ran."""
    phase: str
    skipped: bool = False
    curve_metrics: dict = field(default_factory=dict)
    trade_metrics: dict = field(default_factory=dict)
    decisions: int = 0
    degraded_share: float = 0.0
    open_at_end: int = 0
    action_counts: dict = field(default_factory=dict)
    mean_confidence: float = 0.0


@dataclass(frozen=True)
class AblationResult:
    symbol: str
    reports: Dict[str, PhaseReport]

    def table(self) -> str:
        """Compact comparison table: one row per phase with the headline performance + degradation."""
        header = (f"{'phase':<11}{'return':>10}{'maxDD':>9}{'trades':>8}{'win%':>7}{'avg%':>7}"
                  f"{'degr':>6}{'skip':>6}")
        rows = [header]
        for phase in PHASES:
            r = self.reports[phase]
            if r.skipped:
                rows.append(f"{phase:<11}{'skipped (no model configured)':<38}{'':>6}{'yes':>6}")
                continue
            c, t = r.curve_metrics, r.trade_metrics
            rows.append(f"{phase:<11}{c.get('total_return', 0):>10.2%}{c.get('max_drawdown', 0):>9.2%}"
                        f"{t.get('trades', 0):>8}{t.get('win_rate', 0):>7.0%}{t.get('avg_trade_return', 0):>7.2%}"
                        f"{r.degraded_share:>6.0%}{'':>6}")
        return "\n".join(rows)


def _eod_utc(d) -> "pd.Timestamp":
    """End of the bar's UTC day: all trades closed at or before this are visible to a decision on `d`."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").ceil("D")


def _action_summary(signals: SignalTable, symbol: str) -> dict:
    """Signals fingerprint: what the phase actually decided, useful when two phases size identically."""
    rows = signals.get(symbol, {})
    counts: Dict[str, int] = {}
    confidences, total = 0.0, 0
    for action, confidence in rows.values():
        counts[action] = counts.get(action, 0) + 1
        confidences += confidence
        total += 1
    return {"action_counts": counts, "mean_confidence": confidences / total if total else 0.0}


def _agent_signals(symbol: str, bars: Dict[str, pd.DataFrame], dates, flags: Dict[str, bool],
                   llm, sessions, history_fn_factory, market_fn) -> Tuple[SignalTable, int, int]:
    """Run the TechnicalAgent with `flags` over every decision date. Returns signals plus a degraded count."""
    out: Dict[str, Dict] = {symbol: {}}
    decisions, degraded = 0, 0
    df = bars[symbol]
    for d in dates:
        snap = build_snapshot(symbol, df.loc[:d])
        if snap is None:
            continue
        decisions += 1
        now = _eod_utc(d)
        history_fn = None
        if flags.get("use_learning"):
            if history_fn_factory is not None:
                history_fn = history_fn_factory(now)
            elif sessions is not None:
                def history(sn, _now=now):
                    try:
                        return history_stats(sessions, sn, now=_now)
                    except Exception as e:
                        log.warning("history lookup failed for %s: %s", symbol, e)
                        return None

                history_fn = history
        def market(_now=now):
            return market_fn(_now) if market_fn is not None else None

        ctx = AgentContext(symbol, snap, market=market)
        signal = TechnicalAgent(llm, use_cot=True, history_fn=history_fn, **flags).analyze(ctx)
        out[symbol][d] = (signal.action, signal.confidence)
        degraded += int(bool(signal.degraded))
    return out, decisions, degraded


def signals_for_phase(symbol: str, bars: Dict[str, pd.DataFrame], dates, phase: str, llm,
                      sessions=None, history_fn_factory=None, market_fn=None):
    """A grid of dates -> (action, confidence) for one ablation phase, plus how it actually ran.

    Returns `(signals, decisions, degraded)` where `signals` is `{symbol: {date: (action, conf)}}`.
    DET is the mechanical filter (no model call); every other phase runs `TechnicalAgent` with just
    that capability enabled. Learning lookups are point-in-time: they only see trades closed before
    each decision date (history_stats gets `now=` set to that date, and trades closed later are
    filtered out in the lookup itself)."""
    if phase == DET:
        return rule_signals(bars, dates), len(dates), 0
    flags = _PHASE_FLAGS[phase]
    assert flags is not None  # only DET has no flags, and it returned above
    return _agent_signals(symbol, bars, dates, flags, llm, sessions, history_fn_factory, market_fn)


def run_ablation(
    symbol: str,
    bars: Dict[str, pd.DataFrame],
    decision_dates,
    limits: Optional[RiskLimits] = None,
    llm: Union[None, object, Callable[[str], object]] = None,
    sessions=None,
    history_fn_factory: Optional[Callable[[pd.Timestamp], Callable]] = None,
    market_fn: Optional[Callable[[pd.Timestamp], Optional[object]]] = None,
    start_equity: float = START_EQUITY,
    slippage: float = SLIPPAGE,
    fees: Optional[Callable[[str, float], float]] = india_delivery_fees,
) -> AblationResult:
    """Run every phase over one window with an identical simulator and return the comparison.

    `llm` is either a ready `LLMClient` (one instance reused by all phases), a callable mapping a
    phase name to a fresh client (used by deterministic tests), or None for a fully offline
    mechanical-only run (all AI phases are skipped and the report says so)."""
    norms = limits if limits is not None else RiskLimits()
    reports: Dict[str, PhaseReport] = {}
    for phase in PHASES:
        if llm is None and phase != DET:
            reports[phase] = PhaseReport(phase=phase, skipped=True)
            continue
        phase_llm = None if phase == DET else (llm(phase) if callable(llm) else llm)
        signals, decisions, degraded = signals_for_phase(
            symbol, bars, decision_dates, phase, phase_llm, sessions, history_fn_factory, market_fn)
        result = simulate(bars, signals, norms, start_equity=start_equity, fees=fees, slippage=slippage)
        summary = _action_summary(signals, symbol)
        reports[phase] = PhaseReport(
            phase=phase,
            curve_metrics=curve_metrics(result.equity, start_equity),
            trade_metrics=trade_metrics(result.trades),
            decisions=decisions,
            degraded_share=degraded / decisions if decisions else 0.0,
            open_at_end=result.open_at_end,
            action_counts=summary["action_counts"],
            mean_confidence=summary["mean_confidence"],
        )
    return AblationResult(symbol, reports)