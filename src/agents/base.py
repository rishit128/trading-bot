"""The contract every AI agent follows, so new agents plug into the LangGraph workflow without graph changes.

An agent has a unique `name` and a `role`:
  * LEAD    - proposes the trade. All lead agents must agree on a direction or the bot stands down.
  * ADVISOR - can only confirm (raising confidence) or veto (opposing a lead). Neutral/absent advisors are ignored.
`analyze` returns a AgentSignal, or None when the agent has nothing to say (e.g. no news)."""
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from src.data.indicators import Snapshot
from src.engine.enums import Action, Role
from src.engine.agent_signal import AgentSignal

LEAD = Role.LEAD  # short names for the two roles, as the agents declare them
ADVISOR = Role.ADVISOR


def fail_safe_hold(reason: str) -> AgentSignal:
    """The signal an agent returns when it cannot form a view: a HOLD flagged `degraded` so the pipeline can tell a real
    call from a stand-down (an AI outage must never look like a decision)."""
    return AgentSignal(action=Action.HOLD, confidence=0.0, reasoning=reason, degraded=True)


@dataclass(frozen=True)
class AgentContext:
    """What an agent is given: the symbol, its indicator snapshot, and a lazy headline fetcher."""
    symbol: str
    snapshot: Snapshot
    # Lazy, so only agents that need news pay for fetching it (and they fetch it in parallel with the others).
    headlines: Callable[[], Sequence[str]] = lambda: []
    # Lazy market context (index regime / VIX / relative strength); None when it is unavailable or not wired up.
    market: Callable[[], Optional[object]] = lambda: None


class Agent(Protocol):
    """Contract for any agent: a unique name, a role (lead or advisor) and analyze(ctx)."""
    name: str
    role: Role

    def analyze(self, ctx: AgentContext) -> Optional[AgentSignal]:
        """Return a AgentSignal, or None to abstain."""
        ...
