"""Retrying a flaky network call, with the rules for what is worth retrying written down once.

Yahoo, NSE and Telegram fail transiently (a dropped connection, a rate limit, an empty reply). A single failure used to
cost a whole cycle for that stock (30 minutes) or a lost alert. What must NOT be retried is a fact about the data: a stale
last bar, "no market data", "need 200 bars" are our own `ValueError`s and asking again returns the same answer."""
import json
import logging
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def is_transient_data_error(exc: BaseException) -> bool:
    """True for failures worth another attempt (network, rate limit, garbled reply); False for a definite answer about the
    data (our own ValueErrors, including StaleDataError). A JSON decode error is a ValueError too, but it means the reply
    was cut off or garbled, so it IS transient."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    return not isinstance(exc, ValueError)


def retry_call(fn: Callable[[], T], *, attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
               retry_if: Callable[[BaseException], bool] = is_transient_data_error,
               sleep: Callable[[float], None] = time.sleep, what: str = "call") -> T:
    """Call `fn()`; if it raises something `retry_if` accepts, wait `delay * backoff**(n-1)` seconds and try again, up to
    `attempts` calls in all. The last failure is re-raised unchanged, so callers keep their existing error handling."""
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not retry_if(e):
                raise
            wait = delay * backoff ** (attempt - 1)
            log.warning("%s failed (attempt %d/%d), retrying in %.1fs: %s: %s", what, attempt, attempts, wait,
                        type(e).__name__, str(e)[:100])
            sleep(wait)
    raise AssertionError("unreachable")  # the loop always returns or raises
