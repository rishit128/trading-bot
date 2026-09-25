"""The lead agent: asks the AI for BUY/SELL/HOLD on one stock and, optionally, refines the answer.

Orchestration only. The prompt text lives in `prompts`, the response models and JSON schemas in `schemas`, and the rules
for capping, auditing and vetoing in `phases`; this class decides which calls to make, in what order, and what to do when
one fails. The phases, each recorded inside `AgentSignal.details` so a decision stays audit-able and replay-able:
  * Chain-of-thought: reason in five fixed steps (trend, overbought, volume, confluence, risks) plus a rule
    check against the mechanical baseline, with a compact one-sentence prompt as the fallback when that fails.
  * Decision memory: given the closed paper-trade record for similar setups (src.agents.history), adjust the
    call - a pattern that has lost money is a reason to stand down, not to double down.
  * Market context: given the index regime / VIX (src.data.market_context), adjust once more; only a notable
    regime (risk-off or euphoric) triggers the extra LLM call.
  * Reflection: a six-stage self-critique of the assembled call. It is a veto only: if the critic changes the
    action the call is dropped; if it upholds the action the confidence is left alone.

Each later phase is optional (`use_learning`, `use_context`, `use_reflect`). A phase that fails falls back to the previous
result - the base signal is never replaced by a fail-safe - and the analysis degrades to a HOLD only if the very first
call fails. The confidence change a single phase may make is capped at `max_adjustment` so no single call moves it far."""
import logging
from typing import List, Optional, Tuple

from src.agents.base import LEAD, AgentContext, fail_safe_hold
from src.agents.history import PatternStats, pattern_id
import src.agents.technical.phases as phases
import src.agents.technical.prompts as prompts
from src.agents.technical.prompts import mechanical_action
from src.agents.technical.schemas import (CONTEXT_SCHEMA, COT_SCHEMA, LEARNING_SCHEMA, REFLECT_SCHEMA, AdvancedSignal,
                                          ContextSignal, LearningSignal, ReflectionChain)
from src.engine.enums import Action
from src.engine.agent_signal import AgentSignal, SignalDetails
from src.llm import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)


class TechnicalAgent:
    """Lead agent: asks the AI for BUY/SELL/HOLD from price, moving averages, RSI and volume.

    With chain-of-thought enabled it works through five steps first and returns the reasoning chain, so the decision
    can be audited. Optionally it then refines the call from the trade history (decision memory), the market regime (market context)
    and a self-critique (reflection); each refinement is recorded in `AgentSignal.details`."""

    name = "technical"
    role = LEAD

    # The exact prompt text for each call. Kept reachable from the agent because a stored decision rebuilds the prompt it
    # was shown (replay) and tests fingerprint it; the wording itself lives in `prompts`.
    build_prompt = staticmethod(prompts.cot_prompt)
    build_simple_prompt = staticmethod(prompts.simple_prompt)
    build_learning_prompt = staticmethod(prompts.learning_prompt)
    build_context_prompt = staticmethod(prompts.context_prompt)
    build_reflection_prompt = staticmethod(prompts.reflection_prompt)

    def __init__(self, llm: LLMClient, use_cot: bool = True, use_learning: bool = True, use_context: bool = True,
                 use_reflect: bool = True, history_fn=None, min_pattern_sample: int = 3,
                 max_adjustment: float = phases.MAX_PHASE_ADJUSTMENT):
        self.llm = llm
        self.use_cot = use_cot
        self.use_learning = use_learning
        self.use_context = use_context
        self.use_reflect = use_reflect
        self.history_fn = history_fn  # (snapshot) -> Optional[PatternStats]; None disables decision memory
        self.min_pattern_sample = min_pattern_sample  # fewer closed trades than this are not treated as evidence
        self.max_adjustment = min(1.0, max(0.0, max_adjustment))  # confidence cap per refinement phase

    def _base(self, ctx: AgentContext) -> AgentSignal:
        """The first call, with a fallback ladder for unreliable free models: the full chain-of-thought answer; if every
        model fails at that, the compact one-sentence prompt (a smaller schema a weak model can usually still satisfy,
        recorded as tier "compact"); only when that fails too does the caller stand down with a HOLD."""
        if not self.use_cot:
            return self.llm.signal(prompts.simple_prompt(ctx.snapshot))
        try:
            advanced = self.llm.structured_call(prompts.cot_prompt(ctx.snapshot), AdvancedSignal, COT_SCHEMA)
        except LLMUnavailable as e:
            log.warning("chain-of-thought failed for %s (%s); trying the compact prompt", ctx.symbol, str(e)[:200])
            compact = self.llm.signal(prompts.simple_prompt(ctx.snapshot))
            details = SignalDetails(tier="compact", cot_failure=str(e)[:300], raw_model=compact.raw_model or None)
            return AgentSignal(action=compact.action, confidence=compact.confidence, reasoning=compact.reasoning,
                          details=details)
        # Whether the call departs from the mechanical filter is a fact we can compute; a model's own claim about it is
        # unreliable (the live logs showed it labelling a HOLD "deviate" while quoting the filter's BUY), so overwrite it.
        advanced.rule_alignment = "agree" if advanced.action == mechanical_action(ctx.snapshot) else "deviate"
        return advanced.to_signal()

    def _market(self, ctx: AgentContext):
        """Fetch the market context lazily; any failure means no context adjustment."""
        try:
            return ctx.market()
        except Exception as e:
            log.warning("market context failed for %s: %s", ctx.symbol, e)
            return None

    def _refine(self, ctx: AgentContext, current: AgentSignal, phase: str, prompt: str,
                model, schema, extra, marker, cap: Optional[float] = None) -> AgentSignal:
        """Run one refinement phase (learning/context): ask the model, record its answer, and rewrap the reasoning.

        The applied confidence is clamped into [current - cap, current + cap] (review doc 2.4) so one LLM call cannot
        swing the call radically. Any failure keeps the current signal untouched - the base call must never be replaced
        by a fail-safe."""
        try:
            out = self.llm.structured_call(prompt, model, schema)
        except LLMUnavailable:
            log.warning("phase %s failed for %s, keeping current call", phase, ctx.symbol)
            return phases.tag(current, f"{phase}_skipped", True)
        applied = phases.clamp_adjustment(current.confidence, out.confidence, cap if cap is not None else self.max_adjustment)
        details = (current.details or SignalDetails()).model_copy(update={phase: extra(out, applied)})
        return AgentSignal(action=out.action, confidence=applied,
                      reasoning=f"{current.reasoning} [{marker(out)}]",
                      degraded=current.degraded, details=details)

    def _learn(self, ctx, base: AgentSignal, stats: PatternStats) -> AgentSignal:
        return self._refine(ctx, base, "learning",
                            prompts.learning_prompt(base, ctx.symbol, stats, pattern_id(ctx.snapshot)),
                            LearningSignal, LEARNING_SCHEMA,
                            lambda out, applied: phases.learning_details(out, stats, applied),
                            lambda out: f"pattern history: {out.reason_for_adjustment}")

    def _context(self, ctx, base: AgentSignal, market) -> AgentSignal:
        return self._refine(ctx, base, "context", prompts.context_prompt(base, ctx.symbol, market),
                            ContextSignal, CONTEXT_SCHEMA,
                            lambda out, applied: phases.context_details(out, market, applied),
                            lambda out: f"market: {out.context_reasoning}")

    def _reflect(self, ctx, base: AgentSignal, summary: str) -> AgentSignal:
        try:
            out = self.llm.structured_call(prompts.reflection_prompt(base, ctx.symbol, summary),
                                           ReflectionChain, REFLECT_SCHEMA)
        except LLMUnavailable:
            log.warning("phase reflection failed for %s, keeping current call", ctx.symbol)
            return phases.tag(base, "reflection_skipped", True)
        steps, critic_confidence = phases.apply_reflection(base, out)
        # Veto only: the critic may turn the call into a HOLD/SELL, but it does not shave the confidence of a call it
        # upholds. Its monotone stages plus the humility discount always land below min_confidence (0 of 182 live
        # decisions reached 0.60), which silently blocked every entry; the discounted number stays in the audit trail.
        upheld = out.action == base.action
        final_confidence = round(base.confidence, 4) if upheld else critic_confidence
        section = phases.reflection_details(base, out, steps, final_confidence, critic_confidence, upheld)
        details = (base.details or SignalDetails()).model_copy(update={"reflection": section})
        verdict = "upheld" if upheld else f"vetoed to {out.action}"
        return AgentSignal(action=out.action, confidence=final_confidence,
                      reasoning=f"{base.reasoning} [reflection {verdict}: {steps[-1][2]} (critic conviction {critic_confidence:.2f})]",
                      degraded=base.degraded, details=details)

    def analyze(self, ctx: AgentContext) -> AgentSignal:
        """Return the AI's signal, or a fail-safe HOLD flagged as degraded if every model fails on the FIRST call."""
        try:
            base = self._base(ctx)
        except LLMUnavailable as e:
            return fail_safe_hold(f"LLM unavailable: {e}")
        current = phases.stamp_base(base)
        if current.degraded or current.action == Action.HOLD:
            return current  # nothing to refine from history/market; the fail-safe must never pretend otherwise
        state: dict = {}
        market = None
        if self.use_learning and self.history_fn is not None:
            try:
                stats = self.history_fn(ctx.snapshot)
            except Exception as e:
                log.warning("history lookup failed for %s: %s", ctx.symbol, e)
                stats = None
            if stats is not None and stats.sample_size >= self.min_pattern_sample:
                current = self._learn(ctx, current, stats)
                state["stats"] = stats
        if self.use_context:
            market = self._market(ctx)
            if market is not None and market.is_notable() and current.action != Action.HOLD:
                current = self._context(ctx, current, market)
                state["market"] = market
        if self.use_reflect and not current.degraded and current.action != Action.HOLD:
            current = self._reflect(ctx, current, phases.refinement_summary(ctx, current, state, market))
        phases.stamp_refinements(ctx, current, market)
        return current

    # Thin wrappers kept for the tests and tools that drive these two rules directly.
    _clamp_adjustment = staticmethod(phases.clamp_adjustment)

    def _apply_reflection(self, base: AgentSignal, out: ReflectionChain) -> Tuple[List[Tuple[str, float, str]], float]:
        """The critic's conviction chain with the monotone rule and the humility discount enforced (see phases)."""
        return phases.apply_reflection(base, out)
