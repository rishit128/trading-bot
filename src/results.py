"""Result types shared between the workflow and the pipeline."""
from dataclasses import dataclass
from typing import Optional

from src.engine.risk_engine import RiskDecision
from src.engine.strategy import Decision


@dataclass(frozen=True)
class SymbolResult:
    """What happened to one stock in one cycle."""
    symbol: str
    action: str
    confidence: float
    risk: Optional[RiskDecision]
    order_status: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class Verdict:
    """One stock's decided-and-recorded call, on its way to execution: what was decided, the risk engine's answer, the price
    it was sized at, and the audit row that now records it."""
    decision: Decision
    risk: RiskDecision
    price: float
    decision_id: int
