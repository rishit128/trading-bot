"""What the watchdog tells the owner, from facts gathered by scripts/watchdog.py (cron, every 2 minutes, outside the bots, so it
still works when they hang or die). Pure, so every case can be tested. It restarts a bot that is not running, says when the
machine restarted, and warns about a loop that stopped reporting, a research watch nobody started and positions still open
after the square-off. It never trades. The bots keep their own alerts; this covers what a dead or stuck process cannot say."""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

BOOT_GRACE_S = 180.0  # @reboot waits 60 seconds before starting the bots: do not race it
MARKET_OPEN, SQUARE_OFF_CHECK, CHECK_UNTIL = time(9, 15), time(15, 20), time(15, 45)
HEARTBEAT_FROM = time(9, 25)  # a loop reports after its first pass of the session
HEARTBEAT_LIMITS_S = {"intraday": 20 * 60.0, "swing": 75 * 60.0}  # steps every 5 minutes; every 30 plus the cycle's own length
RESEARCH_REMINDERS = (time(9, 20), time(11, 0))  # the OTP is the owner's, so the watch can only be asked for
RESEARCH_UNTIL = time(15, 0)
RESEARCH_STALE_S = 30 * 60.0  # the bots stop relying on calls older than this (src.data.house_calls.MAX_AGE)


@dataclass(frozen=True)
class Facts:
    now: datetime  # IST
    boot_id: str
    uptime_s: float
    not_running: Tuple[str, ...] = ()
    trading_day: bool = False
    heartbeat_age_s: Dict[str, Optional[float]] = field(default_factory=dict)  # None: no heartbeat file yet
    research_age_s: Optional[float] = None  # None: never read
    left_open: Dict[str, int] = field(default_factory=dict)  # intraday account -> open positions


@dataclass(frozen=True)
class Plan:
    restart: bool
    alerts: List[Tuple[str, str]]  # (key, text): a key with a date goes out once that day
    boot_id: str


def _clock(at: datetime) -> str:
    return f"{at:%H:%M}"


def decide(facts: Facts, last_boot_id: Optional[str], sent: List[str]) -> Plan:
    """What to do now. `last_boot_id` and `sent` (the keys already delivered) are the state kept between runs."""
    now, day, alerts = facts.now, f"{facts.now:%Y-%m-%d}", []
    t = now.time()
    in_session = facts.trading_day and MARKET_OPEN <= t < CHECK_UNTIL

    def once(name: str, text: str) -> None:
        key = f"{day} {name}"
        if key not in sent:
            alerts.append((key, text))

    if last_boot_id and last_boot_id != facts.boot_id:
        alerts.append(
            (
                f"boot {facts.boot_id}",
                f"⚠️ The machine restarted (found at {_clock(now)}, up {facts.uptime_s / 60:.0f} minutes)."
                + (" Entries may have been missed, and the research watch ended." if in_session else ""),
            )
        )
    restart = bool(facts.not_running) and facts.uptime_s >= BOOT_GRACE_S
    if restart:
        alerts.append(
            (f"restart {now:%Y-%m-%d %H:%M}", f"⚠️ Not running at {_clock(now)}: {', '.join(facts.not_running)}. Restarting it.")
        )
    if in_session and t >= HEARTBEAT_FROM:
        for bot, limit in HEARTBEAT_LIMITS_S.items():
            age = facts.heartbeat_age_s.get(bot)
            if age is not None and age > limit and bot not in facts.not_running:
                once(f"stuck {bot}", f"⚠️ The {bot} loop has not reported for {age / 60:.0f} minutes: it may be stuck.")
    stale = facts.research_age_s is None or facts.research_age_s > RESEARCH_STALE_S
    if facts.trading_day and stale and t < RESEARCH_UNTIL:
        for i, due in enumerate(RESEARCH_REMINDERS):
            if t >= due:
                once(f"research {i}", "📡 No broker research calls are coming in. Send /research_start (it asks for the OTP).")
    if facts.trading_day and SQUARE_OFF_CHECK <= t < CHECK_UNTIL:
        left = {name: n for name, n in facts.left_open.items() if n}
        if left:
            shown = ", ".join(f"{name} {n}" for name, n in left.items())
            once("square-off", f"⚠️ Still open after the 15:15 square-off: {shown}. Is the intraday engine running?")
    return Plan(restart, alerts, facts.boot_id)
