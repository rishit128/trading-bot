"""Keeps the record-only tests running without anyone asking: after each trading day's close the intraday bot starts
scripts/shadow_report.py --notify, which sends the day's numbers to Telegram and says when a test reaches its verdict. A bot or machine
that was down at the hour catches up when it returns. The report runs in its own low-priority process, so that it cannot
disturb trading. It only reads the databases and sends messages: a pass switches nothing on."""

import json
import logging
import subprocess  # nosec B404
import sys
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.data.india import IST

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "logs" / "shadow_state.json"
LOG_FILE = ROOT / "logs" / "shadow_watch.log"
RUN_AFTER = time(16, 30)  # the final Angel One bars are in by about 15:35
SESSION_FINISHED = time(15, 40)  # from here the report counts today's session
CHECK_EVERY_S = 300.0
RUN_LIMIT_S = 900.0


def load_state(path: Path = STATE_FILE) -> dict:
    """What was last sent; empty when there is none or it cannot be read (the worst case is one repeated message)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=1))
    temp.replace(path)


def is_due(now: datetime, last_sent: Optional[str], trading_day: bool) -> bool:
    """A trading day, past the hour, and today's message not sent yet."""
    return trading_day and now.time() >= RUN_AFTER and last_sent != now.date().isoformat()


def new_verdicts(before: Dict[str, str], now: Dict[str, str]) -> List[str]:
    """Tests that reached a verdict since the last message. A test that is still running says RUNNING."""
    return [
        f"{name}: {verdict}"
        for name, verdict in now.items()
        if not verdict.startswith("RUNNING") and verdict.split(":")[0] != before.get(name, "RUNNING").split(":")[0]
    ]


def latest_summary(path: Path = STATE_FILE) -> str:
    """What /shadow answers: instant, because it reads the last report and does not work it out again."""
    return load_state(path).get("summary") or "No test report yet: the first comes after the next close, at 16:30."


class ShadowWatch:
    """A background thread in the intraday bot. `run` is the report (replaced in tests)."""

    def __init__(
        self,
        is_trading_day: Callable[[datetime], bool],
        notify: Callable[[str], object],
        run: Optional[Callable[[], bool]] = None,
        state_file: Path = STATE_FILE,
    ):
        self.is_trading_day, self.notify, self.state_file = is_trading_day, notify, state_file
        self.run = run or run_report
        self._tried: Optional[str] = None  # a report that failed is tried once a day, and reported once
        self._stop = threading.Event()

    def check(self, now: datetime) -> bool:
        """Run the report if it is due; True when it ran."""
        day = now.date().isoformat()
        if self._tried == day or not is_due(now, load_state(self.state_file).get("sent_day"), self.is_trading_day(now)):
            return False
        self._tried = day
        if not self.run():
            self.notify(f"Test tracker: today's report failed, see {LOG_FILE.name}. Trading is not affected.")
        return True

    def loop(self) -> None:
        while not self._stop.wait(CHECK_EVERY_S):
            self.tick()

    def tick(self) -> None:
        try:
            self.check(datetime.now(IST))
        except Exception as e:  # a tracker must never take the bot down with it
            log.warning("test tracker: %s: %s", type(e).__name__, e)

    def start(self) -> None:
        def body() -> None:
            self.tick()  # at once, so that a restart after the hour catches up
            self.loop()

        threading.Thread(target=body, name="shadow-watch", daemon=True).start()


def run_report() -> bool:
    command = ["nice", "-n", "19", sys.executable, "-u", str(ROOT / "scripts" / "shadow_report.py"), "--notify"]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a") as out:
            return subprocess.run(command, cwd=ROOT, stdout=out, stderr=out, timeout=RUN_LIMIT_S, check=False).returncode == 0  # nosec B603
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("test tracker could not run: %s", type(e).__name__)
        return False
