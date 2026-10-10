"""The lead agent: asks the AI for BUY/SELL/HOLD on one stock, then optionally has a critic review the call.

Orchestration only. The prompt text lives in `prompts`, the response models and JSON schemas in `schemas`, and the
model-free review rules in `phases`. What the model is shown:
  * the chart (price, moving averages, RSI, volume and the other indicators) and the mechanical filter's verdict;
  * why the stock was picked: its 12-month momentum, the scanner's ranking signal;
  * what the chart cannot show: the company's recent quarters and ratios, and the market situation.
It follows the mechanical filter unless the company data gives a concrete reason against it. A model asked to second-guess
the chart alone rejected two thirds of the filter's picks with no measurable difference between the ones it took and the
ones it passed, and pausing buys in a weak market halved the 9-year return; so the market is shown for risk and
confidence, not as a veto (STRATEGY.md, 2026-10-09).

Reflection (`use_reflect`) is a veto-only critic of a BUY/SELL: if it changes the action the call is dropped; if it
upholds it the confidence is left alone. The analysis degrades to a HOLD only if the first call fails on every model."""
import logging

from src.agents.base import LEAD, AgentContext, fail_safe_hold
import src.agents.technical.phases as phases
import src.agents.technical.prompts as prompts
from src.agents.technical.prompts import mechanical_action
from src.agents.technical.schemas import COT_SCHEMA, REFLECT_SCHEMA, AdvancedSignal, ReflectionChain
from src.engine.enums import Action
from src.engine.agent_signal import AgentSignal, SignalDetails
from src.llm import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)


class TechnicalAgent:
    """Lead agent: asks the AI for BUY/SELL/HOLD from the chart, the company and the market."""

    name = "technical"
    role = LEAD

    def __init__(self, llm: LLMClient, use_cot: bool = True, use_reflect: bool = True):
        self.llm = llm
        self.use_cot = use_cot
        self.use_reflect = use_reflect

    @staticmethod
    def _lazy(fetch, what: str, symbol: str):
        """Market or company data for the prompt; a failure means the model is told there is none, never an error."""
        try:
            return fetch()
        except Exception as e:
            log.warning("%s data failed for %s: %s", what, symbol, e)
            return None

    def _base(self, ctx: AgentContext, market, company) -> AgentSignal:
        """The first call, with a fallback ladder for unreliable free models: the full analysis; if every model fails at
        that, the compact one-sentence prompt (recorded as tier "compact"); only when that fails too does the caller
        stand down with a HOLD."""
        if not self.use_cot:
            return self.llm.signal(prompts.simple_prompt(ctx.snapshot))
        try:
            advanced = self.llm.structured_call(prompts.cot_prompt(ctx.snapshot, market, company), AdvancedSignal,
                                                COT_SCHEMA)
        except LLMUnavailable as e:
            log.warning("full analysis failed for %s (%s); trying the compact prompt", ctx.symbol, str(e)[:200])
            compact = self.llm.signal(prompts.simple_prompt(ctx.snapshot))
            details = SignalDetails(tier="compact", cot_failure=str(e)[:300], raw_model=compact.raw_model or None)
            return AgentSignal(action=compact.action, confidence=compact.confidence, reasoning=compact.reasoning,
                               details=details)
        # Whether the call departs from the mechanical filter is a fact we can compute; a model's own claim about it is
        # unreliable (the live logs showed it labelling a HOLD "deviate" while quoting the filter's BUY), so overwrite it.
        advanced.rule_alignment = "agree" if advanced.action == mechanical_action(ctx.snapshot) else "deviate"
        return advanced.to_signal()

    def _reflect(self, ctx: AgentContext, base: AgentSignal, market, company) -> AgentSignal:
        try:
            out = self.llm.structured_call(
                prompts.reflection_prompt(base, ctx.symbol, phases.review_summary(base, market, company)),
                ReflectionChain, REFLECT_SCHEMA)
        except LLMUnavailable:
            log.warning("reflection failed for %s, keeping current call", ctx.symbol)
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
                           reasoning=f"{base.reasoning} [reflection {verdict}: {steps[-1][2]} (critic conviction "
                                     f"{critic_confidence:.2f})]",
                           degraded=base.degraded, details=details)

    def analyze(self, ctx: AgentContext) -> AgentSignal:
        """Return the AI's signal, or a fail-safe HOLD flagged as degraded if every model fails on the FIRST call."""
        market = self._lazy(ctx.market, "market", ctx.symbol) if self.use_cot else None
        company = self._lazy(ctx.company, "company", ctx.symbol) if self.use_cot else None
        try:
            current = phases.stamp_base(self._base(ctx, market, company), market, company)
        except LLMUnavailable as e:
            return fail_safe_hold(f"LLM unavailable: {e}")
        if self.use_reflect and not current.degraded and current.action != Action.HOLD:
            current = self._reflect(ctx, current, market, company)
            phases.stamp_review(current)
        return current
