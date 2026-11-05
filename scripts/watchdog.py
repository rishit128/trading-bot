"""The watchdog's cron entry: gathers the facts, sends the alerts and restarts a stopped bot (src/ops/watchdog.py decides).

*/2 * * * * cd "<project>" && ./venv/bin/python scripts/watchdog.py >> logs/watchdog.log 2>&1"""

import json
import os
import sqlite3
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ops.watchdog import Facts, decide  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
STATE = ROOT / "logs" / "watchdog_state.json"
BOTS = {"swing": "--loop 30 --live --holdout", "intraday": "--intraday --live"}  # as scripts/start_bots.sh starts them
HEARTBEATS = {"swing": "heartbeat-bot", "intraday": "heartbeat-intraday"}  # named after each bot's log file by start_bots.sh
INTRADAY_DBS = {
    "ORB": "intraday.db",
    "gap-and-go": "gapgo.db",
    "gap-only": "gaponly.db",
    "news": "news.db",
    "broker calls": "broker_calls.db",
}
KEEP_KEYS = 200


def running(arguments: str) -> bool:
    return subprocess.run(["pgrep", "-f", f"venv/bin/python -u main.py {arguments}"], capture_output=True).returncode == 0  # nosec


def is_trading_day(now: datetime, state: dict) -> bool:
    """A session today, from the exchange calendar (slow to load, so read once a day and kept in the state). Every check that
    needs it is in market hours, so outside them the answer is simply no."""
    day = f"{now:%Y-%m-%d}"
    if now.weekday() >= 5 or not 9 <= now.hour < 16:
        return False
    if state.get("calendar", [None])[0] == day:
        return bool(state["calendar"][1])
    try:
        import pandas as pd

        from src.data.india import IndiaClock

        cal, today = IndiaClock().cal, pd.Timestamp(now.date())
        session = today > cal.last_session.tz_localize(None) or bool(cal.is_session(today))
    except Exception as e:  # the calendar is a convenience: a weekday counts as a trading day without it
        print(f"{now:%H:%M} no holiday calendar: {type(e).__name__}")
        session = True
    state["calendar"] = [day, session]
    return session


def _scalar(db: str, query: str):
    path = ROOT / db
    if not path.exists():
        return None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as con:
        return con.execute(query).fetchone()[0]


def research_age(now: datetime) -> Optional[float]:
    at = _scalar("house_calls.db", "select max(fetched_at) from house_calls")
    if not at:
        return None
    return (now - datetime.fromisoformat(at).replace(tzinfo=timezone.utc)).total_seconds()


def heartbeat_age(bot: str) -> Optional[float]:
    try:
        return time.time() - json.loads((ROOT / "logs" / HEARTBEATS[bot]).read_text())["at"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def gather(now: datetime, state: dict) -> Facts:
    trading = is_trading_day(now, state)
    left: Dict[str, int] = {}
    if trading and now.hour == 15:
        left = {name: _scalar(db, "select count(*) from paper_positions") or 0 for name, db in INTRADAY_DBS.items()}
    return Facts(
        now=now,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        uptime_s=float(Path("/proc/uptime").read_text().split()[0]),
        not_running=tuple(bot for bot, arguments in BOTS.items() if not running(arguments)),
        trading_day=trading,
        heartbeat_age_s={bot: heartbeat_age(bot) for bot in BOTS},
        research_age_s=research_age(now) if trading else None,
        left_open=left,
    )


def main() -> None:
    now = datetime.now(IST)
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    plan = decide(gather(now, state), state.get("boot_id"), state.get("sent", []))
    if plan.restart:
        subprocess.run([str(ROOT / "scripts" / "start_bots.sh")], check=False)  # nosec B603
    sent = list(state.get("sent", []))
    if plan.alerts:
        from src.config import load_settings  # noqa: F401  (loads .env)
        from src.monitoring.telegram import Notifier

        notifier = Notifier(os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", ""))
        for key, text in plan.alerts:
            print(f"{now:%Y-%m-%d %H:%M} {text}")
            if notifier.send(text):
                sent.append(key)
    state.update(boot_id=plan.boot_id, sent=sent[-KEEP_KEYS:])
    STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
