"""The learning loop's chores, none of which may ever hurt trading: label matured outcomes once a day, and once a week tell
the owner what has been learned so far (above all: do the AI's calls beat the plain rule it sits on?)."""

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
from sqlalchemy import func, select

from src.control import Control
from src.database import DecisionRecord, HoldoutMark, OutcomeRecord
from src.learning.mistakes import (
    Judged,
    ai_vs_rule,
    breakdown,
    fmt,
    group_stats,
    load_judged,
    rule_action_from_snapshot,
    rule_buy_passes,
)
from src.learning.outcomes import HORIZONS, LabelSummary, label_outcomes

log = logging.getLogger(__name__)


class Replay(Protocol):
    """What the weekly report needs of the holdout reserve (src.ops.holdout): a rule-versus-AI comparison, or None."""

    def compare(self) -> Optional[dict]: ...


LABELLED_ON = "outcomes_labelled_on"  # the control flag holding the date of the last successful run
REPORTED_WEEK = "learning_report_week"  # the ISO week (YYYY-Www) the last weekly report was sent for
FALLBACK_ALERTED = "fallback_alert_day"  # the date the owner was last told that fallback models were answering
FALLBACK_LIMIT, FALLBACK_WINDOW_DAYS, FALLBACK_MIN_CALLS = 0.20, 7, 20
MIN_VETOES = 60  # matured critic vetoes needed before the critic is judged


class DailyOutcomeLabelling:
    """Callable for the run loop's after-cycle hook. Runs `label_outcomes` at most once per calendar day; a failure is
    logged and retried at the next cycle, and it can never raise into (or stop) the trading loop."""

    def __init__(
        self,
        sessions,
        control: Control,
        fetch_bars: Callable[[str], Optional[pd.DataFrame]],
        stop_pct: float,
        notify: Optional[Callable[[str], object]] = None,
        horizons=HORIZONS,
        today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
    ):
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
        log.info(
            "outcome labelling: %d new outcomes (%d already done, %d not yet matured, %d without data)",
            summary.labelled,
            summary.already_done,
            summary.not_mature,
            summary.no_data,
        )
        return summary


def learning_report(
    sessions,
    horizon: int = 20,
    min_samples: int = 30,
    make_reserve: Optional[Callable[[], Replay]] = None,
    primary_model: Optional[str] = None,
) -> str:
    """What the bot has learned so far, in plain text: how much is labelled, whether the AI's calls beat the plain rule's
    (net of costs, `horizon`-session outcomes, each group judged only once it has `min_samples`), and the holdout."""
    with sessions() as s:
        decided = s.scalar(
            select(func.count()).select_from(
                select(DecisionRecord.symbol, DecisionRecord.bar_date)
                .where(DecisionRecord.bar_date.isnot(None))
                .distinct()
                .subquery()
            )
        )
        labelled = dict(s.execute(select(OutcomeRecord.horizon, func.count()).group_by(OutcomeRecord.horizon)).all())
        mark = s.scalar(select(HoldoutMark).order_by(HoldoutMark.id.desc()).limit(1))
    lines = [
        "Weekly learning report",
        f"Stocks decided (stock-sessions): {decided or 0}",
        "Outcomes labelled: " + ", ".join(f"{h} sessions {labelled.get(h, 0)}" for h in HORIZONS),
        f"AI vs the plain entry rule ({horizon}-session outcome, net of costs):",
    ]
    judged = load_judged(sessions, horizon)
    groups = [g for g in breakdown(judged, ai_vs_rule, min_samples, horizon) if g.label != "n/a"]
    lines += [f"  {fmt(g)}" for g in groups] or ["  nothing matured yet"]
    passes = [j for j in judged if ai_vs_rule(j) == "rule BUY, AI passed"]
    if passes:
        lines.append("  Of the rule BUYs it passed on:")
        lines += [f"    {fmt(g)}" for g in breakdown(passes, rule_buy_passes, min_samples, horizon)]
    lines.append("  The AI adds value only if 'rule BUY, AI bought' beats 'rule BUY, AI passed'.")
    lines += critic_lines(judged, horizon, min_samples)
    lines += model_lines(sessions, judged, horizon, min_samples, primary_model)
    if make_reserve is not None:
        lines += reserve_lines(make_reserve)
    lines.append(research_gate_line(sessions))
    if mark is not None:
        lines.append(
            f"Holdout reserve (the recorded decisions replayed on their own): equity {mark.equity:,.0f} "
            f"({mark.return_pct:+.2%}), {mark.decisions} decisions, {mark.closed_trades} closed trades"
        )
    return "\n".join(lines)


def critic_lines(judged: Sequence[Judged], horizon: int, min_samples: int) -> List[str]:
    """Does the reflection critic earn its keep? Its vetoed BUYs against the BUYs it let through (both on rule BUYs). After
    MIN_VETOES matured vetoes: if the vetoed group did not do worse than the upheld one (by more than their combined 95%
    intervals, which are over dates), the critic avoided nothing and LLM_REFLECT can go off. Only advice: nothing is switched."""
    vetoed = [j for j in judged if ai_vs_rule(j) == "rule BUY, AI passed" and j.vetoed]
    upheld = [j for j in judged if ai_vs_rule(j) == "rule BUY, AI bought"]
    out = group_stats("critic vetoed", vetoed, min_samples, horizon), group_stats("critic upheld", upheld, min_samples, horizon)
    lines = [f"  {fmt(g)}" for g in out]
    veto, kept = out
    if len(vetoed) < MIN_VETOES:
        return lines + [f"  Critic verdict after {MIN_VETOES} matured vetoes (have {len(vetoed)}): not yet."]
    if not (veto.sufficient and kept.sufficient and veto.mean_net is not None and kept.mean_net is not None):
        return lines + ["  Critic verdict: not enough independent observations yet."]
    worse = veto.mean_net < kept.mean_net - math.hypot(veto.ci95 or 0.0, kept.ci95 or 0.0)
    return lines + [
        "  Critic verdict: the vetoed BUYs did worse, so the critic is avoiding losses: keep LLM_REFLECT on."
        if worse
        else "  Critic verdict: the vetoed BUYs did not do worse than the upheld ones: LLM_REFLECT can be switched off."
    ]


def fallback_share(sessions, primary_model: str, since: datetime) -> Tuple[float, int]:
    """(share of the AI's answers since `since` that came from a model other than the primary one, the number of answers)."""
    with sessions() as s:
        models = s.scalars(
            select(DecisionRecord.raw_model).where(DecisionRecord.raw_model.isnot(None), DecisionRecord.created_at >= since)
        ).all()
    return (sum(m != primary_model for m in models) / len(models) if models else 0.0), len(models)


def model_lines(sessions, judged: Sequence[Judged], horizon: int, min_samples: int, primary_model: Optional[str]) -> List[str]:
    """Which model answered, and how its BUYs did (an evaluation window should have one primary model: results mixed from
    several are not one test), plus how much of the last week fallback models answered."""
    lines = ["  AI BUYs by the model that answered:"]
    buys = [j for j in judged if j.action == "BUY"]
    lines += [f"    {fmt(g)}" for g in breakdown(buys, lambda j: j.model or "unknown", min_samples, horizon)] or [
        "    nothing matured yet"
    ]
    if primary_model:
        share, calls = fallback_share(sessions, primary_model, datetime.now(timezone.utc) - timedelta(days=FALLBACK_WINDOW_DAYS))
        lines.append(
            f"  Fallback models answered {share:.0%} of {calls} AI calls in the last {FALLBACK_WINDOW_DAYS} days (primary: {primary_model})."
        )
    return lines


def reserve_lines(make_reserve: Callable[[], Replay]) -> List[str]:
    """The AI's recorded BUYs against the plain rule's BUYs, replayed through the same simulator (see HoldoutReserve.compare)."""
    try:
        c = make_reserve().compare()
    except Exception:
        log.exception("rule-only replay failed")
        return ["  Rule-only replay: unavailable this week (see the log)."]
    if c is None:
        return []
    return [
        f"  Replay, same simulator, limits and costs: plain rule {c['rule_equity']:,.0f} ({c['rule_trades']} trades) vs the AI's "
        f"BUYs {c['ai_equity']:,.0f} ({c['ai_trades']} trades); AI minus rule {c['mean_diff']:+.3%} a day (t {c['t']:+.1f}, "
        f"{c['days']} paired days)."
    ]


def research_gate_line(sessions) -> str:
    """What a gate on research-only candidates would have changed (record only: nothing is blocked). Counted on the
    latest decision per stock-session whose stock only a broker research call put on the list."""
    latest = (
        select(func.max(DecisionRecord.id))
        .where(DecisionRecord.bar_date.isnot(None))
        .group_by(DecisionRecord.symbol, DecisionRecord.bar_date)
    )
    with sessions() as s:
        rows = s.execute(
            select(DecisionRecord.final_action, DecisionRecord.rule_action, DecisionRecord.snapshot_json, DecisionRecord.screen_verdict)
            .where(DecisionRecord.id.in_(latest), DecisionRecord.candidate_source == "research")
        ).all()  # fmt: skip
    buys = [(rule or rule_action_from_snapshot(snap), screen) for action, rule, snap, screen in rows if action == "BUY"]
    rule_hold = sum(rule == "HOLD" for rule, _ in buys)
    filtered = sum(screen == "filtered out" for _, screen in buys)
    return (
        f"Research-only candidates (recorded, nothing blocked): {len(rows)} stock-sessions, {len(buys)} AI BUYs; a rule "
        f"gate would have turned {rule_hold} of those BUYs into HOLDs, and the screener had filtered out {filtered}"
    )


class WeeklyLearningReport:
    """Callable for the run loop's after-cycle hook: sends `learning_report` to Telegram once per ISO week. A failure is
    logged and retried next cycle; it never raises into the trading loop."""

    def __init__(
        self,
        sessions,
        control: Control,
        notify: Optional[Callable[[str], object]],
        horizon: int = 20,
        min_samples: int = 30,
        today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
        make_reserve: Optional[Callable[[], Replay]] = None,
        primary_model: Optional[str] = None,
    ):
        self.sessions, self.control, self.notify = sessions, control, notify
        self.horizon, self.min_samples, self.today = horizon, min_samples, today
        self.make_reserve, self.primary_model = make_reserve, primary_model  # the reserve is built only when a report is due

    def _check_fallbacks(self) -> None:
        """At most once a day: tell the owner when fallback models answer more than FALLBACK_LIMIT of the AI's calls."""
        day = self.today().isoformat()
        try:
            if self.notify is None or not self.primary_model or self.control.get_flag(FALLBACK_ALERTED) == day:
                return
            since = datetime.combine(self.today() - timedelta(days=FALLBACK_WINDOW_DAYS), datetime.min.time(), timezone.utc)
            share, calls = fallback_share(self.sessions, self.primary_model, since)
            if calls >= FALLBACK_MIN_CALLS and share > FALLBACK_LIMIT:
                self.notify(
                    f"AI MODEL ALERT: fallback models answered {share:.0%} of {calls} AI calls in the last {FALLBACK_WINDOW_DAYS} "
                    f"days (primary {self.primary_model}). The primary model is failing or rate-limited; results mix several models."
                )
                self.control.set_flag(FALLBACK_ALERTED, day)
        except Exception:
            log.exception("fallback check failed")

    def __call__(self) -> Optional[str]:
        if self.notify is None:
            return None
        self._check_fallbacks()
        year, week, _ = self.today().isocalendar()
        tag = f"{year}-W{week:02d}"
        try:
            if self.control.get_flag(REPORTED_WEEK) == tag:
                return None
            text = learning_report(
                self.sessions,
                self.horizon,
                self.min_samples,
                self.make_reserve,
                self.primary_model,
            )
            self.notify(text)
            self.control.set_flag(REPORTED_WEEK, tag)
        except Exception:
            log.exception("weekly learning report failed; will retry at the next cycle")
            return None
        return text
