"""Re-examine a stored decision: what the model was shown, what each agent said, and whether the rules reproduce the result.

The AI's own answer cannot be regenerated (models are not deterministic), but everything around it can: the exact prompt
is rebuilt from the stored indicator inputs, and the combination of the stored signals is recomputed."""
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from src.agents.agents import TechnicalAgent
from src.agents.base import ADVISOR, LEAD
from src.data.indicators import Snapshot
from src.database import DecisionRecord
from src.engine.strategy import combine_signals
from src.llm import Signal

DEFAULT_ROLES = {"technical": LEAD, "sentiment": ADVISOR}


@dataclass(frozen=True)
class Replay:
    """A stored decision re-examined: inputs, prompt, signals and whether the rules reproduce it."""
    decision_id: int
    symbol: str
    inputs: Optional[Snapshot]
    prompt: Optional[str]
    signals: Dict[str, Optional[Signal]]
    stored_action: str
    stored_confidence: float
    recomputed_action: Optional[str]
    rule_based: bool
    reproduced: Optional[bool]


def replay(record: DecisionRecord, roles: Mapping[str, str] = DEFAULT_ROLES) -> Replay:
    """`reproduced` is True/False when the stored inputs allow a check, None when they were not recorded (older rows)."""
    snap = Snapshot(**json.loads(record.snapshot_json)) if record.snapshot_json else None
    signals = {n: Signal(**s) if s else None for n, s in json.loads(record.signals_json or "{}").items()}
    rule_based = record.reasoning.startswith("trend exit:")
    if rule_based:
        recomputed = "SELL" if snap is not None and snap.price < snap.ma200 else None  # a deterministic override, not an AI call
    else:
        recomputed = combine_signals(signals, roles).action if signals else None
    return Replay(
        decision_id=record.id, symbol=record.symbol, inputs=snap,
        prompt=TechnicalAgent.build_prompt(snap) if snap is not None else None, signals=signals,
        stored_action=record.final_action, stored_confidence=record.final_confidence, recomputed_action=recomputed,
        rule_based=rule_based, reproduced=None if recomputed is None else recomputed == record.final_action,
    )
