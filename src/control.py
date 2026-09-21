"""Runtime switches shared through the database: the kill switch and the drawdown-baseline reset."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import sessionmaker

from src.database import ControlFlag

PAUSED = "paused"
PEAK_SINCE = "peak_since"


class Control:
    """Kill switch shared between the trading loop and the Telegram listener via the database."""

    def __init__(self, session_factory: sessionmaker):
        self.sessions = session_factory

    def is_paused(self) -> bool:
        """True when trading is paused (Telegram /pause)."""
        with self.sessions() as s:
            flag = s.get(ControlFlag, PAUSED)
            return flag is not None and flag.value == "1"

    def set_paused(self, paused: bool) -> None:
        """Pause or resume order placement."""
        with self.sessions() as s:
            s.merge(ControlFlag(key=PAUSED, value="1" if paused else "0"))
            s.commit()

    def rebase_peak(self, now: Optional[datetime] = None) -> None:
        """Start measuring peak equity (and so drawdown) from now. A human decision: it clears a -20% halt."""
        with self.sessions() as s:
            s.merge(ControlFlag(key=PEAK_SINCE, value=(now or datetime.now(timezone.utc)).isoformat()))
            s.commit()

    def peak_since(self) -> Optional[datetime]:
        """When the peak-equity baseline was last reset, or None."""
        with self.sessions() as s:
            flag = s.get(ControlFlag, PEAK_SINCE)
        return datetime.fromisoformat(flag.value) if flag is not None else None
