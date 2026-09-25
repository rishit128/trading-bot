"""Running cycles: once, or in a crash-safe loop while the market is open."""
import logging
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)


def format_result(r) -> str:
    """One printable line for a stock's result."""
    risk = r.risk.reason if r.risk else "-"
    return f"{r.symbol:6s} {r.action:5s} conf={r.confidence:.2f} order={r.order_status or '-'} | {risk} {r.error or ''}"


def run_cycle(pipeline, out: Callable[[str], None] = print) -> None:
    """Run one cycle and print each result."""
    for r in pipeline.run_once():
        out(format_result(r))


def run_loop(pipeline, minutes: float, sleep: Callable[[float], None] = time.sleep,
             max_cycles: Optional[int] = None, after_cycle: Optional[Callable[[], None]] = None,
             out: Callable[[str], None] = print) -> None:
    """Unattended loop: a failed cycle (network, broker outage) is reported and retried, never fatal."""
    cycles = 0
    while True:
        try:
            if pipeline.broker.is_market_open():
                run_cycle(pipeline, out)
                if after_cycle is not None:
                    after_cycle()
            else:
                out("market closed; waiting")
        except Exception as e:
            log.exception("cycle failed")
            pipeline.notify(f"CYCLE FAILED (will retry in {minutes:g} min): {type(e).__name__}: {e}")
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return
        sleep(minutes * 60)
