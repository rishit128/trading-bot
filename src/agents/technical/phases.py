"""The pieces of the technical agent's refinement phases that need no model: how a confidence change is capped, how each
phase's answer becomes audit details, how the critic's conviction chain is enforced, and how the audit record is stamped.
Pure functions over signals, so each rule is testable on its own and the agent class only orchestrates."""
import dataclasses
from typing import List, Tuple

from src.agents.base import AgentContext
from src.agents.history import HUMILITY, PatternStats, pattern_id
from src.agents.technical.schemas import ContextSignal, LearningSignal, ReflectionChain
from src.engine.agent_signal import (ContextDetails, ConvictionStage, LearningDetails, ReflectionDetails, AgentSignal, SignalDetails)

MAX_PHASE_ADJUSTMENT = 0.15  # the largest confidence swing one refinement phase may make


def adjustment_reason(details: SignalDetails, current: AgentSignal) -> str:
    """Why the refinement phases are credited with the final call, in the words of the phase that last changed it."""
    if details.learning and details.learning.reason:
        return details.learning.reason
    if details.context and details.context.reasoning:
        return details.context.reasoning
    reflection = details.reflection
    if reflection is None:
        return "a refinement phase adjusted the call"
    if reflection.upheld is True:
        return "reflection upheld the call unchanged"
    if reflection.upheld is False:
        return f"reflection vetoed the call: {reflection.biggest_risk}"
    return f"self-critique cut confidence to {current.confidence:.2f}"  # a record from before the veto-only contract


def tag(sig: AgentSignal, key: str, value) -> AgentSignal:
    """The same signal with one more audit flag in its details, so a skipped review is visible in the record."""
    if sig.details is None:
        return sig
    return AgentSignal(action=sig.action, confidence=sig.confidence, reasoning=sig.reasoning, degraded=sig.degraded,
                  details=sig.details.model_copy(update={key: value}))


def stamp_base(sig: AgentSignal) -> AgentSignal:
    """Record the untouched base confidence before any later phase moves it. Signals without details (simple
    mode) are left alone so a no-details signal stays a no-details signal."""
    if sig.details is None:
        return sig
    details = sig.details if sig.details.base_confidence is not None else sig.details.model_copy(
        update={"base_confidence": sig.confidence})
    return AgentSignal(action=sig.action, confidence=sig.confidence, reasoning=sig.reasoning,
                  degraded=sig.degraded, details=details)


def clamp_adjustment(base_conf: float, new_conf: float, cap: float) -> float:
    """A confidence change inside [-cap, +cap] (also within [0, 1]); the cap is the confidence cap per phase."""
    new_conf = min(1.0, max(0.0, new_conf))
    lower, upper = max(base_conf - cap, 0.0), min(base_conf + cap, 1.0)
    return round(max(lower, min(new_conf, upper)), 4)


def learning_details(out: LearningSignal, stats: PatternStats, applied: float) -> LearningDetails:
    return LearningDetails(
        adjusted_action=out.action, adjusted_confidence=applied, pattern_reliability=out.pattern_reliability,
        sample_size=stats.sample_size, win_rate=stats.win_rate, avg_win_pct=stats.avg_win_pct,
        avg_loss_pct=stats.avg_loss_pct, profit_factor=stats.profit_factor, best_holding_days=stats.best_holding_days,
        worst_holding_days=stats.worst_holding_days, confidence_in_pattern=stats.confidence_in_pattern,
        reason=out.reason_for_adjustment, what_could_break=out.what_could_break)


def context_details(out: ContextSignal, market, applied: float) -> ContextDetails:
    return ContextDetails(
        regime=market.regime_label, macro_support=out.macro_support, sector_support=out.sector_support,
        earnings_risk=out.earnings_risk, diversification_score=out.diversification_score,
        index_above_ma200=market.index_above_ma200, index_rsi=market.index_rsi, vix=market.vix,
        vix_percentile=market.vix_percentile, vs_index_6m=market.vs_index_6m, reasoning=out.context_reasoning,
        risks=out.key_context_risks)


def reflection_details(base: AgentSignal, out: ReflectionChain, steps: List[Tuple[str, float, str]],
                        final_confidence: float, critic_confidence: float, upheld: bool) -> ReflectionDetails:
    return ReflectionDetails(
        biggest_risk=out.biggest_risk, what_proves_us_wrong=out.what_proves_us_wrong, bias_check=out.bias_check,
        conviction_adjustments=[ConvictionStage(stage=name, conviction=round(conviction, 4)) for name, conviction, _ in steps],
        final_action=out.action, final_confidence=final_confidence, base_confidence_entering=round(base.confidence, 4),
        humility_reduction=HUMILITY, critic_confidence=critic_confidence, upheld=upheld)


def refinement_summary(ctx: AgentContext, current: AgentSignal, state: dict, market) -> str:
    lines = [f"Base reasoning: {current.reasoning}"]
    stats = state.get("stats")
    if stats is not None:
        pf = f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "n/a"
        lines.append("Historical record on similar setups: "
                     f"n={stats.sample_size}, win rate {stats.win_rate:.0%}, "
                     f"avg win {stats.avg_win_pct:+.1f}%, avg loss {stats.avg_loss_pct:+.1f}%, "
                     f"profit factor {pf}, holding {stats.worst_holding_days}-{stats.best_holding_days} days.")
    if market is not None:
        vix = f"{market.vix:.0f}" if market.vix is not None else "n/a"
        vs = ("above" if market.index_above_ma200 else "below") if market.index_above_ma200 is not None else "? vs"
        lines.append(f"Market regime: {market.regime_label}, index {vs} the 200-day MA, VIX {vix}.")
    learning = current.details.learning if current.details else None
    if learning is not None:
        lines.append(f"Pattern overlap: {learning.sample_size} similar closed trades, "
                     f"win rate {learning.win_rate:.0%}.")
    return "\n".join(lines)


def stamp_refinements(ctx: AgentContext, current: AgentSignal, market) -> None:
    """Top-level record of what any refinement did, so the decision row exposes it to SQL queries and replay."""
    details = current.details
    if details is None:
        return
    details.pattern_id = pattern_id(ctx.snapshot)
    if market is not None:
        details.market_context_json = dataclasses.asdict(market)
    if details.base_confidence is not None and (details.learning or details.context or details.reflection):
        details.adjusted_signal_confidence = current.confidence
        details.adjustment_reason = adjustment_reason(details, current)


def apply_reflection(base: AgentSignal, out: ReflectionChain) -> Tuple[List[Tuple[str, float, str]], float]:
    """The review-doc rules, enforced client-side: step 1 is the base conviction; every later stage may only keep
    or reduce it (never raise it), and a fixed humility discount lands at the very end."""
    prev = min(1.0, max(0.0, base.confidence))
    steps = [("step1_technical", prev, "the technical base from the annotated analysis")]
    for name in ("step2_reflection", "step3_fundamental", "step4_macro", "step5_integration", "step6_risk"):
        stage = getattr(out, name)
        conviction = min(prev, min(1.0, max(0.0, stage.conviction)))
        steps.append((name, conviction, stage.reason))
        prev = conviction
    final_confidence = round(max(0.0, prev - HUMILITY), 4)
    return steps, final_confidence
