"""The parts of the agent's review step that need no model: how the critic's conviction chain is enforced, how its answer
becomes audit details, and how the audit record is stamped. Pure functions over signals, so each rule is testable on its
own and the agent class only orchestrates."""
import dataclasses
import json
from typing import List, Optional, Tuple

from src.agents.technical.prompts import company_text, market_text
from src.agents.technical.schemas import ReflectionChain
from src.engine.agent_signal import AgentSignal, ConvictionStage, ReflectionDetails, SignalDetails

HUMILITY = 0.10  # the critic's fixed humility discount (review doc 4.2); audit only, it never gates a trade


def adjustment_reason(details: SignalDetails, current: AgentSignal) -> str:
    """Why the review is credited with the final call."""
    reflection = details.reflection
    if reflection is None:
        return "no review ran"
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


def stamp_base(sig: AgentSignal, market=None, company=None) -> AgentSignal:
    """Record the untouched base confidence and the market/company data the model was shown (so the prompt can be
    rebuilt on replay). Signals without details (the compact tier) are left alone."""
    if sig.details is None:
        return sig
    def plain(data):  # JSON-shaped (tuples become lists) so the stored record equals what was used
        return json.loads(json.dumps(dataclasses.asdict(data))) if data is not None else None

    update: dict = {"market_context_json": plain(market), "company_json": plain(company)}
    if sig.details.base_confidence is None:
        update["base_confidence"] = sig.confidence
    return AgentSignal(action=sig.action, confidence=sig.confidence, reasoning=sig.reasoning,
                       degraded=sig.degraded, details=sig.details.model_copy(update=update))


def reflection_details(base: AgentSignal, out: ReflectionChain, steps: List[Tuple[str, float, str]],
                       final_confidence: float, critic_confidence: float, upheld: bool) -> ReflectionDetails:
    return ReflectionDetails(
        biggest_risk=out.biggest_risk, what_proves_us_wrong=out.what_proves_us_wrong, bias_check=out.bias_check,
        conviction_adjustments=[ConvictionStage(stage=name, conviction=round(conviction, 4)) for name, conviction, _ in steps],
        final_action=out.action, final_confidence=final_confidence, base_confidence_entering=round(base.confidence, 4),
        humility_reduction=HUMILITY, critic_confidence=critic_confidence, upheld=upheld)


def review_summary(current: AgentSignal, market=None, company=None) -> str:
    """What the critic is shown: the base reasoning, and the same company and market data the analyst had."""
    return "\n".join([f"Base reasoning: {current.reasoning}", f"Company: {company_text(company)}",
                      f"Market: {market_text(market)}"])


def stamp_review(current: AgentSignal) -> None:
    """Top-level record of what the review did, so the decision row exposes it to SQL queries and replay."""
    details: Optional[SignalDetails] = current.details
    if details is not None and details.base_confidence is not None and details.reflection is not None:
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
