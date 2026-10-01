"""The learning loop's chores, none of which may ever hurt trading: label matured outcomes once a day, and once a week tell
the owner what has been learned so far (above all: do the AI's calls beat the plain rule it sits on?)."""
import logging
from datetime import date, datetime, timezone
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import func, select

from src.control import Control
from src.database import DecisionRecord, HoldoutMark, OutcomeRecord
from src.learning.mistakes import ai_vs_rule, breakdown, fmt, load_judged
from src.learning.outcomes import HORIZONS, LabelSummary, label_outcomes

log = logging.getLogger(__name__)

LABELLED_ON = "outcomes_labelled_on"  # the control flag holding the date of the last successful run
REPORTED_WEEK = "learning_report_week"  # the ISO week (YYYY-Www) the last weekly report was sent for


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


def learning_report(sessions, horizon: int = 20, min_samples: int = 30) -> str:
    """What the bot has learned so far, in plain text: how much is labelled, whether the AI's calls beat the plain rule's
    (net of costs, `horizon`-session outcomes, each group judged only once it has `min_samples`), and the holdout."""
    with sessions() as s:
        decided = s.scalar(select(func.count()).select_from(
            select(DecisionRecord.symbol, DecisionRecord.bar_date).where(DecisionRecord.bar_date.isnot(None)).distinct().subquery()))
        labelled = dict(s.execute(select(OutcomeRecord.horizon, func.count()).group_by(OutcomeRecord.horizon)).all())
        mark = s.scalar(select(HoldoutMark).order_by(HoldoutMark.id.desc()).limit(1))
    lines = ["Weekly learning report",
             f"Stocks decided (stock-sessions): {decided or 0}",
             "Outcomes labelled: " + ", ".join(f"{h} sessions {labelled.get(h, 0)}" for h in HORIZONS),
             f"AI vs the plain entry rule ({horizon}-session outcome, net of costs):"]
    groups = [g for g in breakdown(load_judged(sessions, horizon), ai_vs_rule, min_samples) if g.label != "n/a"]
    lines += [f"  {fmt(g)}" for g in groups] or ["  nothing matured yet"]
    lines.append("  The AI adds value only if 'rule BUY, AI bought' beats 'rule BUY, AI passed'.")
    if mark is not None:
        lines.append(f"Holdout reserve (the recorded decisions replayed on their own): equity {mark.equity:,.0f} "
                     f"({mark.return_pct:+.2%}), {mark.decisions} decisions, {mark.closed_trades} closed trades")
    return "\n".join(lines)


class WeeklyLearningReport:
    """Callable for the run loop's after-cycle hook: sends `learning_report` to Telegram once per ISO week. A failure is
    logged and retried next cycle; it never raises into the trading loop."""

    def __init__(self, sessions, control: Control, notify: Optional[Callable[[str], object]], horizon: int = 20,
                 min_samples: int = 30, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date()):
        self.sessions, self.control, self.notify = sessions, control, notify
        self.horizon, self.min_samples, self.today = horizon, min_samples, today

    def __call__(self) -> Optional[str]:
        if self.notify is None:
            return None
        year, week, _ = self.today().isocalendar()
        tag = f"{year}-W{week:02d}"
        try:
            if self.control.get_flag(REPORTED_WEEK) == tag:
                return None
            text = learning_report(self.sessions, self.horizon, self.min_samples)
            self.notify(text)
            self.control.set_flag(REPORTED_WEEK, tag)
        except Exception:
            log.exception("weekly learning report failed; will retry at the next cycle")
            return None
        return text
