"""The contract every AI agent follows, so new agents plug into the LangGraph workflow without graph changes.

An agent has a unique `name` and a `role`:
  * LEAD    - proposes the trade. All lead agents must agree on a direction or the bot stands down.
  * ADVISOR - can only confirm (raising confidence) or veto (opposing a lead). Neutral/absent advisors are ignored.
`analyze` returns a Signal, or None when the agent has nothing to say (e.g. no news)."""
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from src.data.indicators import Snapshot
from src.llm import Signal

LEAD = "lead"
ADVISOR = "advisor"


@dataclass(frozen=True)
class AgentContext:
    """What an agent is given: the symbol, its indicator snapshot, and a lazy headline fetcher."""
    symbol: str
    snapshot: Snapshot
    # Lazy, so only agents that need news pay for fetching it (and they fetch it in parallel with the others).
    headlines: Callable[[], Sequence[str]] = lambda: []


class Agent(Protocol):
    """Contract for any agent: a unique name, a role (lead or advisor) and analyze(ctx)."""
    name: str
    role: str

    def analyze(self, ctx: AgentContext) -> Optional[Signal]:
        """Return a Signal, or None to abstain."""
        ...
