"""How an LLM call fails: the one exception callers see, and the internal classification that decides what to do next."""
from typing import Optional

TRANSIENT, TRUNCATED, FORMAT, PERMANENT = "transient", "truncated", "format", "permanent"


class LLMUnavailable(Exception):
    """Raised when every configured model failed."""
    pass


class Failure(Exception):
    """One classified failed attempt (internal)."""

    def __init__(self, kind: str, detail: str, retry_after: Optional[float] = None):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.retry_after = kind, detail, retry_after
