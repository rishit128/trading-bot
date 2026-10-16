"""The learning loop's chores: the daily outcome labelling and the weekly report that scores the AI against the plain
rule. Both must never hurt trading: failures are logged, retried next cycle and never raised."""

import json
from datetime import date

from sqlalchemy import select

from src.control import Control
from src.database import DecisionRecord, OutcomeRecord, make_session_factory
from src.learning.jobs import LABELLED_ON, DailyOutcomeLabelling


def snap_dict(symbol="X", rsi=60.0, adx=35.0, volume=1000, bar_date="2025-01-06"):
    return {
        "symbol": symbol,
        "price": 100.0,
        "ma50": 98.0,
        "ma200": 95.0,
        "rsi": rsi,
        "volume": volume,
        "avg_volume": 1000,
        "adx": adx,
        "bar_date": bar_date,
    }


def seed(tmp_path, observations, name="m.db"):
    """observations: (symbol, bar_date, snapshot kwargs, gross_return, exit_date) tuples; returns the session factory."""
    sessions = make_session_factory(f"sqlite:///{tmp_path / name}")
    with sessions() as s:
        for symbol, bar_date, kw, gross, exit_date in observations:
            s.add(
                DecisionRecord(
                    symbol=symbol,
                    price=100.0,
                    final_action="BUY",
                    final_confidence=0.7,
                    reasoning="r",
                    risk_approved=False,
                    risk_quantity=0,
                    risk_reason="x",
                    bar_date=bar_date,
                    snapshot_json=json.dumps(snap_dict(symbol=symbol, bar_date=bar_date, **kw)),
                )
            )
            s.add(
                OutcomeRecord(
                    symbol=symbol,
                    bar_date=bar_date,
                    horizon=20,
                    entry_date=bar_date,
                    exit_date=exit_date,
                    entry_price=100.0,
                    exit_price=100 * (1 + gross),
                    gross_return=gross,
                    worst_return=min(0.0, gross),
                    hit_stop=False,
                    stop_pct=0.15,
                )
            )
        s.commit()
    return sessions


# ---------------------------------------------------------------- the daily job
def job(tmp_path, fetch, days):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'j.db'}")
    control, current = Control(sessions), [days[0]]
    return DailyOutcomeLabelling(sessions, control, fetch, 0.15, today=lambda: current[0]), control, current


def test_the_job_runs_once_a_day_and_a_failure_is_retried_and_never_raised(tmp_path):
    calls = []
    d1, d2 = date(2026, 3, 1), date(2026, 3, 2)
    run, control, today = job(tmp_path, None, [d1])
    monkey = {"fail": True}

    import src.learning.jobs as jobs

    real = jobs.label_outcomes

    def fake(*a, **k):
        calls.append(k["today"])
        if monkey["fail"]:
            raise ConnectionError("down")
        return real(*a, **k)

    jobs.label_outcomes = fake
    try:
        assert run() is None and control.get_flag(LABELLED_ON) is None  # failed: not marked done, no exception
        monkey["fail"] = False
        assert run() is not None and control.get_flag(LABELLED_ON) == d1.isoformat()
        assert run() is None and len(calls) == 2  # already done today: no second run
        today[0] = d2
        assert run() is not None and len(calls) == 3  # a new day runs again
    finally:
        jobs.label_outcomes = real


def test_control_flags_round_trip_and_overwrite(tmp_path):
    control = Control(make_session_factory(f"sqlite:///{tmp_path / 'f.db'}"))
    assert control.get_flag("k") is None
    control.set_flag("k", "a")
    control.set_flag("k", "b")
    assert control.get_flag("k") == "b"


# ---------------------------------------------------------------- the weekly learning report
def test_the_weekly_report_is_sent_once_per_week_and_again_the_next_week(tmp_path):
    from src.learning.jobs import WeeklyLearningReport

    sessions = seed(tmp_path, [])
    sent, today = [], [date(2026, 10, 5)]  # a Monday
    report = WeeklyLearningReport(sessions, Control(sessions), sent.append, today=lambda: today[0])
    report()
    today[0] = date(2026, 10, 9)  # same week
    report()
    today[0] = date(2026, 10, 12)  # next week
    report()
    assert len(sent) == 2 and sent[0].startswith("Weekly learning report")


def test_the_report_says_plainly_when_nothing_has_matured(tmp_path):
    from src.learning.jobs import learning_report

    text = learning_report(seed(tmp_path, []))
    assert "nothing matured yet" in text and "Outcomes labelled: 5 sessions 0, 20 sessions 0, 60 sessions 0" in text


def test_the_report_compares_the_ais_calls_with_the_rules_once_enough_have_matured(tmp_path):
    from src.learning.jobs import learning_report

    days = ("2025-01-06", "2025-01-07", "2025-01-08")  # an interval is computed over dates, so it needs more than one
    sessions = seed(tmp_path, [(f"S{i}", days[i % 3], {}, 0.05 + 0.01 * (i % 3), "2025-02-03") for i in range(30)])
    with sessions() as s:
        for d in s.scalars(select(DecisionRecord)):
            d.decision_source, d.final_action, d.rule_action = "agents", "BUY", "BUY"  # the rule said BUY and the AI bought
        s.commit()
    text = learning_report(sessions, horizon=20, min_samples=30)
    line = text.split("rule BUY, AI bought")[1].splitlines()[0]
    assert "n=30" in line and "insufficient" not in line and "30 independent observations on 3 dates" in line


def test_a_failing_report_never_reaches_the_trading_loop_and_is_retried(tmp_path):
    from src.learning.jobs import WeeklyLearningReport

    sessions = seed(tmp_path, [])

    def broken(text):
        raise ConnectionError("telegram down")

    report = WeeklyLearningReport(sessions, Control(sessions), broken, today=lambda: date(2026, 10, 5))
    assert report() is None
    assert Control(sessions).get_flag("learning_report_week") is None  # not marked as sent: next cycle tries again
