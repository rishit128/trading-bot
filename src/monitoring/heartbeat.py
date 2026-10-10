"""A heartbeat file each bot loop rewrites every cycle, and the check a container health check runs against it.

A running process is not a working bot: a loop that is stuck, or one whose every cycle raises (Yahoo down, a bug), looks fine
to Docker. The loop records each cycle's outcome here; `check` calls the bot unhealthy when the file is stale (the loop
stopped) or when the last few cycles all failed.

    python -m src.monitoring.heartbeat check /data/heartbeat-swing 4500     # exit 0 healthy, 1 unhealthy
"""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3  # one failed cycle is a blip (and is alerted on); three in a row means something is broken


class Heartbeat:
    """Writes the loop's latest cycle outcome to `path` (atomically, so a reader never sees half a file)."""

    def __init__(self, path: str, clock: Callable[[], float] = time.time):
        self.path, self.clock, self.failures = Path(path), clock, 0

    @classmethod
    def from_setting(cls, path: str) -> "Heartbeat | None":
        """The heartbeat for a configured path, or None when it is not configured (local runs, tests)."""
        return cls(path) if path else None

    def beat(self, ok: bool, detail: str = "") -> None:
        """Record one cycle. Never raises: a heartbeat that cannot be written must not stop trading."""
        self.failures = 0 if ok else self.failures + 1
        body = {"at": self.clock(), "ok": ok, "failures": self.failures, "detail": detail[:200]}
        temp = self.path.with_name(self.path.name + ".tmp")
        try:
            temp.write_text(json.dumps(body), encoding="utf-8")
            os.replace(temp, self.path)
        except OSError:
            log.warning("could not write the heartbeat file %s", self.path, exc_info=True)


def check(path: str, max_age_seconds: float, clock: Callable[[], float] = time.time) -> tuple[bool, str]:
    """(healthy, why): healthy when the heartbeat is fresh and the latest cycles did not all fail."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        age = clock() - float(body["at"])
        failures = int(body["failures"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return False, f"no readable heartbeat at {path} ({type(e).__name__})"
    if age > max_age_seconds:
        return False, f"the loop has not reported for {age / 60:.0f} minutes (limit {max_age_seconds / 60:.0f}): stuck or stopped"
    if failures >= MAX_CONSECUTIVE_FAILURES:
        return False, f"the last {failures} cycles failed: {body.get('detail', '')}"
    return True, f"last cycle {age:.0f}s ago: {body.get('detail', '')}"


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] != "check":
        print("usage: python -m src.monitoring.heartbeat check PATH MAX_AGE_SECONDS", file=sys.stderr)
        return 2
    healthy, why = check(argv[2], float(argv[3]))
    print(("healthy: " if healthy else "UNHEALTHY: ") + why)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
