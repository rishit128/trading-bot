"""Result types shared between the workflow and the pipeline."""
from dataclasses import dataclass
from typing import Optional

from src.engine.risk_engine import RiskDecision


@dataclass(frozen=True)
class SymbolResult:
    """What happened to one stock in one cycle."""
    symbol: str
    action: str
    confidence: float
    risk: Optional[RiskDecision]
    order_status: Optional[str]
    error: Optional[str] = None
