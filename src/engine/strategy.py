"""Combining agent signals into one decision by role."""
from dataclasses import dataclass
from typing import Mapping, Optional

from src.agents.base import ADVISOR, LEAD
from src.llm import Signal


@dataclass(frozen=True)
class Decision:
    """The combined call: action, confidence and the reasoning."""
    action: str
    confidence: float
    reasoning: str


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def combine_signals(signals: Mapping[str, Optional[Signal]], roles: Mapping[str, str]) -> Decision:
    """Merge any number of agents. Leads must be unanimous on a direction; advisors can only confirm or veto.

    Missing (None) or neutral (HOLD) advisors are ignored. A lead that produced nothing, or leads that disagree,
    mean no trade. With one lead and one advisor this is the original technical-leads/sentiment-vetoes rule."""
    leads = {n: s for n, s in signals.items() if roles.get(n) == LEAD}
    if not leads:
        return Decision("HOLD", 0.0, "no lead agent configured")
    if any(s is None for s in leads.values()):
        return Decision("HOLD", 0.0, "a lead agent produced no signal")

    label = "+".join(leads)
    actions = {s.action for s in leads.values()}
    said = " | ".join(f"{n} {s.action}: {s.reasoning}" for n, s in leads.items())
    if len(actions) > 1:
        return Decision("HOLD", 0.0, f"lead agents disagree: {said}")
    action = next(iter(actions))
    if action == "HOLD":
        why = " | ".join(s.reasoning for s in leads.values())
        return Decision("HOLD", _mean(s.confidence for s in leads.values()), f"{label} HOLD: {why}")

    advisors = {n: s for n, s in signals.items()
                if roles.get(n) != LEAD and s is not None and s.action != "HOLD"}
    for n, s in advisors.items():
        if s.action != action:
            return Decision("HOLD", 0.0, f"conflict: {label} {action} vs {n} {s.action}")

    confidences = [s.confidence for s in leads.values()] + [s.confidence for s in advisors.values()]
    reasons = " | ".join([s.reasoning for s in leads.values()] + [s.reasoning for s in advisors.values()])
    if advisors:
        return Decision(action, _mean(confidences), f"{label} and {' and '.join(advisors)} agree {action}: {reasons}")
    return Decision(action, _mean(confidences), f"{label} {action}: {reasons}")


def combine(technical: Signal, sentiment: Optional[Signal]) -> Decision:
    """The original two-agent rule, kept as a convenience wrapper."""
    return combine_signals({"technical": technical, "sentiment": sentiment},
                           {"technical": LEAD, "sentiment": ADVISOR})
