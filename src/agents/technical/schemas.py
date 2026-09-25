"""What the technical agent asks the model to return: pydantic models (validated on the way in) and the JSON schemas sent
with each request (so the model is constrained on the way out). One pair per call: the full chain of thought, the
decision-memory adjustment, the market-context adjustment and the six-stage critique."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.engine.enums import ACTION_VALUES, Action
from src.engine.agent_signal import ReasoningChain, AgentSignal, SignalDetails

PLACEHOLDERS = {"n/a", "na", "none", "null", "unknown", "tbd", "todo", "string", "...", "-", "--"}


class AdvancedSignal(BaseModel):
    """The chain-of-thought output: a decision plus its step-by-step reasoning, risks and edge rating."""
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    edge_confidence: float = Field(ge=0.0, le=1.0)
    step1_trend: str
    step2_overbought: str
    step3_volume: str
    step4_confluence: int = Field(ge=1, le=10)
    step5_risks: List[str]
    final_reasoning: str
    raw_model: Optional[str] = None  # stamped client-side by the LLM client, never read from the model's JSON
    rule_alignment: Optional[Literal["agree", "deviate"]] = None  # does the call depart from the mechanical filter?
    falsification: Optional[str] = None  # the single most concrete thing that would prove this call wrong

    @field_validator("step1_trend", "step2_overbought", "step3_volume", "final_reasoning")
    @classmethod
    def _substantive(cls, value: str) -> str:
        """Schema-valid but empty answers ("", "n/a") are what a starved free model produces; reject them so the client
        retries (quoting the problem) instead of trading on a blank rationale."""
        value = value.strip()
        if len(value) < 4 or value.lower().strip(".") in PLACEHOLDERS:
            raise ValueError("a reasoning step must say something, not be blank or a placeholder like 'n/a'")
        return value

    @field_validator("step5_risks")
    @classmethod
    def _real_risks(cls, value: List[str]) -> List[str]:
        risks = [r.strip() for r in value if r and r.strip()]
        if not risks:
            raise ValueError("step5_risks must name at least one concrete risk")
        return risks

    @field_validator("falsification")
    @classmethod
    def _blank_falsification_is_absent(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() or None if value is not None else None

    def to_signal(self) -> AgentSignal:
        """Map to the pipeline's AgentSignal, stashing the reasoning chain so it is logged, persisted and replay-able."""
        details = SignalDetails(
            tier="cot", edge_confidence=self.edge_confidence, confluence_score=self.step4_confluence, risks=self.step5_risks,
            reasoning_chain=ReasoningChain(trend=self.step1_trend, overbought=self.step2_overbought, volume=self.step3_volume),
            raw_model=self.raw_model, rule_alignment=self.rule_alignment, falsification=self.falsification)
        return AgentSignal(action=self.action, confidence=self.confidence, reasoning=self.final_reasoning, details=details)


COT_SCHEMA = {
    "name": "trading_signal_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTION_VALUES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "edge_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "step1_trend": {"type": "string"},
            "step2_overbought": {"type": "string"},
            "step3_volume": {"type": "string"},
            "step4_confluence": {"type": "integer", "minimum": 1, "maximum": 10},
            "step5_risks": {"type": "array", "items": {"type": "string"}},
            "final_reasoning": {"type": "string"},
            "rule_alignment": {"type": "string", "enum": ["agree", "deviate"]},
            "falsification": {"type": "string"},
        },
        "required": ["action", "confidence", "edge_confidence", "step1_trend", "step2_overbought", "step3_volume",
                     "step4_confluence", "step5_risks", "final_reasoning", "rule_alignment", "falsification"],
        "additionalProperties": False,
    },
}


class LearningSignal(BaseModel):
    """Decision-memory output: the call restated after checking the historical record of similar setups."""
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    pattern_reliability: Literal["yes", "maybe", "no"]  # review doc 2.2: does the record support the pattern?
    reason_for_adjustment: str
    what_could_break: str


LEARNING_SCHEMA = {
    "name": "learning_adjustment_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTION_VALUES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "pattern_reliability": {"type": "string", "enum": ["yes", "maybe", "no"]},
            "reason_for_adjustment": {"type": "string"},
            "what_could_break": {"type": "string"},
        },
        "required": ["action", "confidence", "pattern_reliability", "reason_for_adjustment", "what_could_break"],
        "additionalProperties": False,
    },
}


class ContextSignal(BaseModel):
    """Market-context output: the call restated after checking the market and sector regime.

    `sector_support`, `earnings_risk` and `diversification_score` are what the context prompt asks for; when the
    underlying data is unavailable the prompt says so and the model is expected to answer neutrally."""
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    macro_support: Literal["yes", "neutral", "no"]
    sector_support: Literal["yes", "neutral", "no"]
    earnings_risk: bool
    diversification_score: int = Field(ge=1, le=10)
    context_reasoning: str
    key_context_risks: List[str]


CONTEXT_SCHEMA = {
    "name": "market_context_adjustment_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTION_VALUES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "macro_support": {"type": "string", "enum": ["yes", "neutral", "no"]},
            "sector_support": {"type": "string", "enum": ["yes", "neutral", "no"]},
            "earnings_risk": {"type": "boolean"},
            "diversification_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "context_reasoning": {"type": "string"},
            "key_context_risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "confidence", "macro_support", "sector_support", "earnings_risk",
                     "diversification_score", "context_reasoning", "key_context_risks"],
        "additionalProperties": False,
    },
}


class ReflectionStage(BaseModel):
    """One stage of the reflection self-critique: a conviction level 0-1 plus a one-sentence reason."""
    conviction: float = Field(ge=0.0, le=1.0)
    reason: str


class ReflectionChain(BaseModel):
    """Reflection output: the critic's case at each stage. Steps 2-6 come from the model; step 1 (the technical base) is
    seeded client-side from the base call. The client then enforces that conviction only ever stays put or falls, and
    applies the fixed humility discount - the model's `confidence` is its own proposal, not the last word.

    step1 seed = the call entering this phase. step2 reflection critiques the technical case; step3 fundamental
    (honest: the bot has no fundamentals feed, so if no data is visible the critic keeps conviction unchanged and says
    so); step4 macro critiques the regime; step5 integration weighs it together; step6 risk restates the call with the
    single biggest risk."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)  # the model's proposed final confidence (still subject to the rules)
    step2_reflection: ReflectionStage
    step3_fundamental: ReflectionStage
    step4_macro: ReflectionStage
    step5_integration: ReflectionStage
    step6_risk: ReflectionStage
    biggest_risk: str
    what_proves_us_wrong: str
    bias_check: List[str]


def _stage_schema():
    return {
        "type": "object",
        "properties": {
            "conviction": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["conviction", "reason"],
        "additionalProperties": False,
    }


REFLECT_SCHEMA = {
    "name": "reflection_check_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTION_VALUES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "step2_reflection": _stage_schema(),
            "step3_fundamental": _stage_schema(),
            "step4_macro": _stage_schema(),
            "step5_integration": _stage_schema(),
            "step6_risk": _stage_schema(),
            "biggest_risk": {"type": "string"},
            "what_proves_us_wrong": {"type": "string"},
            "bias_check": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "confidence", "step2_reflection", "step3_fundamental", "step4_macro",
                     "step5_integration", "step6_risk", "biggest_risk", "what_proves_us_wrong", "bias_check"],
        "additionalProperties": False,
    },
}
