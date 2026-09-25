"""The learning loop's daily chore: label the outcomes that have matured, once a day, without ever hurting trading."""
import logging
from datetime import date, datetime, timezone
from typing import Callable, Optional

import pandas as pd

from src.control import Control
from src.learning.outcomes import HORIZONS, LabelSummary, label_outcomes

log = logging.getLogger(__name__)

LABELLED_ON = "outcomes_labelled_on"  # the control flag holding the date of the last successful run


class DailyOutcomeLabelling:
    """Callable for the run loop's after-cycle hook. Runs `label_outcomes` at most once per calendar day; a failure is
    logged and retried at the next cycle, and it can never raise into (or stop) the trading loop."""

    def __init__(self, sessions, control: Control, fetch_bars: Callable[[str], Optional[pd.DataFrame]], stop_pct: float,
                 notify: Optional[Callable[[str], object]] = None, horizons=HORIZONS,
                 today: Callable[[], date] = lambda: datetime.now(timezone.utc).date()):
        self.sessions, self.control, self.fetch_bars = sessions, control, fetch_bars
        self.stop_pct, self.notify, self.horizons, self.today = stop_pct, notify, horizons, today

    def __call__(self) -> Optional[LabelSummary]:
        day = self.today()
        try:
            if self.control.get_flag(LABELLED_ON) == day.isoformat():
                return None
            summary = label_outcomes(self.sessions, self.fetch_bars, self.horizons, self.stop_pct, today=day)
            self.control.set_flag(LABELLED_ON, day.isoformat())
        except Exception:
            log.exception("outcome labelling failed; will retry at the next cycle")
            return None
        log.info("outcome labelling: %d new outcomes (%d already done, %d not yet matured, %d without data)", summary.labelled,
                 summary.already_done, summary.not_mature, summary.no_data)
        return summary
