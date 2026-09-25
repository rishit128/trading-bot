"""The bot's fixed vocabularies, as enums instead of scattered string literals.

Each member IS its old string (they are `StrEnum`s): `Action.BUY == "BUY"`, they serialise to JSON and SQL unchanged, and
existing rows, prompts and tests keep working. What changes is that a typo (`"BUY "`, `"Hold"`) is now an AttributeError at
import time instead of a silently different branch."""
from enum import StrEnum


class Action(StrEnum):
    """What an agent, the strategy or the risk engine can ask for."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


ACTION_VALUES = [a.value for a in Action]  # the same three words as a list, for the `enum` of every JSON schema sent to the model


class OrderStatus(StrEnum):
    """How an order attempt ended, as recorded in `orders.status`. The first two come from the broker; the rest are the
    pipeline's own outcomes (an order that was never sent has a status too, so every decision leaves an audit row)."""
    FILLED = "filled"
    CONFIRMED = "confirmed"
    DRY_RUN = "DRY_RUN"
    PAUSED = "PAUSED"
    MARKET_CLOSED = "MARKET_CLOSED"
    SKIPPED_COOLDOWN = "SKIPPED_COOLDOWN"
    SKIPPED_OPEN_ORDER = "SKIPPED_OPEN_ORDER"
    FAILED = "FAILED"


# Outcomes that mean "we did not go for this buy": they must not start a re-buy cooldown.
NOT_A_PLACED_BUY = (OrderStatus.SKIPPED_OPEN_ORDER, OrderStatus.SKIPPED_COOLDOWN, OrderStatus.FAILED, OrderStatus.PAUSED,
                    OrderStatus.MARKET_CLOSED)
# Outcomes that need no alert: nothing reached (or was even meant to reach) the broker.
SILENT_STATUSES = (OrderStatus.DRY_RUN, OrderStatus.PAUSED, OrderStatus.MARKET_CLOSED, OrderStatus.SKIPPED_COOLDOWN)


class Role(StrEnum):
    """An agent's part in a decision: a LEAD proposes the trade; an ADVISOR can only confirm or veto it."""
    LEAD = "lead"
    ADVISOR = "advisor"


class DecisionSource(StrEnum):
    """Who produced a decision: the AI agents, or a deterministic rule that overrides them."""
    AGENTS = "agents"
    TREND_EXIT = "trend_exit"
