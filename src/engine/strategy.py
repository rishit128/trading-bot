"""Combining agent signals into one decision by role."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Optional

from src.engine.enums import Action, DecisionSource, Role
from src.engine.rules import trend_broken
from src.engine.agent_signal import AgentSignal

if TYPE_CHECKING:  # only for the annotation: the engine must not import the data layer at run time
    from src.data.indicators import Snapshot

TREND_EXIT_PREFIX = "trend exit:"  # how a rule-based exit announces itself in the recorded reasoning (read back by replay)


@dataclass(frozen=True)
class Decision:
    """The combined call: action, confidence, the reasoning, and who made it (the agents, or a rule that overrides them)."""
    action: Action
    confidence: float
    reasoning: str
    source: DecisionSource = DecisionSource.AGENTS


def trend_exit_decision(snapshot: "Snapshot", held_qty: int, enabled: bool = True) -> Optional[Decision]:
    """The deterministic exit: a held stock whose last completed close is below its 200-day average is sold, whatever the
    AI says. Like the risk engine this is plain code, not an opinion. None when the rule does not apply."""
    if enabled and held_qty > 0 and trend_broken(snapshot.price, snapshot.ma200):
        return Decision(Action.SELL, 1.0, f"{TREND_EXIT_PREFIX} last close {snapshot.price:,.2f} is below the 200-day average "
                                          f"{snapshot.ma200:,.2f}", DecisionSource.TREND_EXIT)
    return None


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def combine_signals(signals: Mapping[str, Optional[AgentSignal]], roles: Mapping[str, str]) -> Decision:
    """Merge any number of agents. Leads must be unanimous on a direction; advisors can only confirm or veto.

    Missing (None) or neutral (HOLD) advisors are ignored. A lead that produced nothing, or leads that disagree,
    mean no trade. With one lead and one advisor this is the original technical-leads/sentiment-vetoes rule."""
    lead_names = [n for n in signals if roles.get(n) == Role.LEAD]
    if not lead_names:
        return Decision(Action.HOLD, 0.0, "no lead agent configured")
    leads: Dict[str, AgentSignal] = {}
    for n in lead_names:
        lead = signals[n]
        if lead is None:
            return Decision(Action.HOLD, 0.0, "a lead agent produced no signal")
        leads[n] = lead

    label = "+".join(leads)
    actions = {s.action for s in leads.values()}
    said = " | ".join(f"{n} {s.action}: {s.reasoning}" for n, s in leads.items())
    if len(actions) > 1:
        return Decision(Action.HOLD, 0.0, f"lead agents disagree: {said}")
    action = Action(next(iter(actions)))
    if action == Action.HOLD:
        why = " | ".join(s.reasoning for s in leads.values())
        return Decision(Action.HOLD, _mean(s.confidence for s in leads.values()), f"{label} HOLD: {why}")

    advisors = {n: s for n, s in signals.items()
                if roles.get(n) != Role.LEAD and s is not None and s.action != Action.HOLD}
    for n, s in advisors.items():
        if s.action != action:
            return Decision(Action.HOLD, 0.0, f"conflict: {label} {action} vs {n} {s.action}")

    confidences = [s.confidence for s in leads.values()] + [s.confidence for s in advisors.values()]
    reasons = " | ".join([s.reasoning for s in leads.values()] + [s.reasoning for s in advisors.values()])
    if advisors:
        return Decision(action, _mean(confidences), f"{label} and {' and '.join(advisors)} agree {action}: {reasons}")
    return Decision(action, _mean(confidences), f"{label} {action}: {reasons}")


def combine(technical: AgentSignal, sentiment: Optional[AgentSignal]) -> Decision:
    """The original two-agent rule, kept as a convenience wrapper."""
    return combine_signals({"technical": technical, "sentiment": sentiment},
                           {"technical": Role.LEAD, "sentiment": Role.ADVISOR})
