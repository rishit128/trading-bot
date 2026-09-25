"""An agent's opinion and the typed audit trail that travels with it.

`AgentSignal.details` used to be a free-form dict with ~30 string keys ("learning", "reflection", "tier", ...) written by the
agent and read back, key by key, by the pipeline. A typo failed silently and adding a phase meant editing three files.
It is now `SignalDetails`: every key is a typed field, absent sections are None, and the JSON written to the audit table
has the same shape as before (see `AgentSignal.to_json_dict`), so stored rows and the analysis scripts are unaffected."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.engine.enums import Action


class ReasoningChain(BaseModel):
    """The three one-sentence steps of the chain-of-thought answer."""
    trend: str
    overbought: str
    volume: str


class LearningDetails(BaseModel):
    """What the decision-memory phase found in the closed-trade record of similar setups, and what it did about it."""
    adjusted_action: Action
    adjusted_confidence: float
    pattern_reliability: str
    sample_size: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: Optional[float] = None
    best_holding_days: int
    worst_holding_days: int
    confidence_in_pattern: float
    reason: str
    what_could_break: str


class ContextDetails(BaseModel):
    """What the market-context phase saw (index regime, VIX, relative strength) and concluded."""
    regime: str
    macro_support: str
    sector_support: str
    earnings_risk: bool
    diversification_score: int
    index_above_ma200: Optional[bool] = None
    index_rsi: Optional[float] = None
    vix: Optional[float] = None
    vix_percentile: Optional[float] = None
    vs_index_6m: Optional[float] = None
    reasoning: str
    risks: List[str] = Field(default_factory=list)


class ConvictionStage(BaseModel):
    """One stage of the critic's chain and the conviction it ended with."""
    stage: str
    conviction: float


class ReflectionDetails(BaseModel):
    """The critic's findings. `upheld` is False when it vetoed the call; `critic_confidence` is its own humility-discounted
    number, kept for audit (it never gates a trade). Both are None on rows written before the veto-only contract."""
    biggest_risk: str
    what_proves_us_wrong: str
    bias_check: List[str] = Field(default_factory=list)
    conviction_adjustments: List[ConvictionStage] = Field(default_factory=list)
    final_action: Action
    final_confidence: float
    base_confidence_entering: float
    humility_reduction: float
    critic_confidence: Optional[float] = None
    upheld: Optional[bool] = None


class SignalDetails(BaseModel):
    """Everything an agent recorded about how it reached its call. Every field is optional: which ones are set depends on
    the tier that answered and on which refinement phases ran."""
    tier: Optional[Literal["cot", "compact"]] = None  # the full chain of thought, or the compact fallback prompt
    cot_failure: Optional[str] = None  # why the full prompt failed, when the compact tier answered
    raw_model: Optional[str] = None  # which configured model answered
    edge_confidence: Optional[float] = None
    confluence_score: Optional[int] = None
    risks: Optional[List[str]] = None
    reasoning_chain: Optional[ReasoningChain] = None
    rule_alignment: Optional[Literal["agree", "deviate"]] = None  # computed from the data, not the model's claim
    falsification: Optional[str] = None
    base_confidence: Optional[float] = None  # the confidence before any refinement phase moved it
    pattern_id: Optional[str] = None
    market_context_json: Optional[dict] = None
    adjusted_signal_confidence: Optional[float] = None
    adjustment_reason: Optional[str] = None
    learning: Optional[LearningDetails] = None
    context: Optional[ContextDetails] = None
    reflection: Optional[ReflectionDetails] = None
    learning_skipped: Optional[bool] = None  # a phase that was due but could not run (every model failed)
    context_skipped: Optional[bool] = None
    reflection_skipped: Optional[bool] = None


class AgentSignal(BaseModel):
    """One agent's opinion: BUY/SELL/HOLD, confidence 0-1 and one-sentence reasoning; `degraded` marks a fail-safe HOLD.

    `details` holds the richer reasoning some agents return. It is populated by the agent code, never read from the
    model's JSON, and is persisted with the decision for audit/replay."""
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    degraded: bool = False  # set only by our fail-safe HOLD, never by a model
    details: Optional[SignalDetails] = None
    raw_model: Optional[str] = None  # which configured model actually answered, stamped client-side by the LLM client

    def to_json_dict(self) -> dict:
        """The audit-table shape: every top-level field as before (nulls kept); inside `details` only the sections and fields
        that were set, while a section that ran keeps all its keys (a null inside it is data, e.g. "no VIX that day")."""
        out = self.model_dump(mode="json", exclude={"details"})
        out["details"] = ({k: v for k, v in self.details.model_dump(mode="json").items() if v is not None}
                          if self.details is not None else None)
        return out
