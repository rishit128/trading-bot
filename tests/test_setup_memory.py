"""Setup memory (learning from every labelled outcome) and the daily labelling job. Honesty rules are tested as rules."""
import json
from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.control import Control
from src.data.indicators import Snapshot
from src.database import DecisionRecord, OutcomeRecord, make_session_factory
from src.learning.jobs import LABELLED_ON, DailyOutcomeLabelling
from src.learning.setups import ROUND_TRIP_COST, SetupMemory, adx_band, setup_labels, volume_band

START = date(2025, 1, 6)  # a Monday


def snap_dict(symbol="X", rsi=60.0, adx=35.0, volume=1000, bar_date="2025-01-06"):
    return {"symbol": symbol, "price": 100.0, "ma50": 98.0, "ma200": 95.0, "rsi": rsi, "volume": volume, "avg_volume": 1000,
            "adx": adx, "bar_date": bar_date}


def snap(**kw):
    return Snapshot(**snap_dict(**kw))


def seed(tmp_path, observations, name="m.db"):
    """observations: (symbol, bar_date, snapshot kwargs, gross_return, exit_date) tuples; returns the session factory."""
    sessions = make_session_factory(f"sqlite:///{tmp_path / name}")
    with sessions() as s:
        for symbol, bar_date, kw, gross, exit_date in observations:
            s.add(DecisionRecord(symbol=symbol, price=100.0, final_action="BUY", final_confidence=0.7, reasoning="r",
                                 risk_approved=False, risk_quantity=0, risk_reason="x", bar_date=bar_date,
                                 snapshot_json=json.dumps(snap_dict(symbol=symbol, bar_date=bar_date, **kw))))
            s.add(OutcomeRecord(symbol=symbol, bar_date=bar_date, horizon=20, entry_date=bar_date, exit_date=exit_date,
                                entry_price=100.0, exit_price=100 * (1 + gross), gross_return=gross, worst_return=min(0.0, gross),
                                hit_stop=False, stop_pct=0.15))
        s.commit()
    return sessions


def many(n, gross, **kw):
    """n observations on n different stocks, on one session."""
    return [(f"S{i}", "2025-01-06", kw, gross, "2025-02-03") for i in range(n)]


def test_it_stays_silent_below_the_minimum_number_of_observations(tmp_path):
    memory = SetupMemory(seed(tmp_path, many(29, 0.10)), min_samples=30)
    assert memory.stats(snap()) is None


def test_matched_stats_are_net_of_costs_and_carry_the_market_wide_baseline(tmp_path):
    obs = many(30, 0.10) + [(f"L{i}", "2025-01-06", {"rsi": 30.0, "adx": 15.0}, -0.10, "2025-02-03") for i in range(30)]
    stats = SetupMemory(seed(tmp_path, obs), min_samples=30).stats(snap())
    assert stats.scope == "market" and stats.sample_size == 30 and stats.win_rate == 1.0
    assert stats.avg_win_pct == pytest.approx((0.10 - ROUND_TRIP_COST) * 100)
    assert stats.baseline_win_rate == pytest.approx(0.5)  # all comparable setups, not just the matched one
    assert stats.best_holding_days == 20 and stats.confidence_in_pattern == pytest.approx(0.5)


def test_a_gain_smaller_than_the_costs_counts_as_a_loss(tmp_path):
    stats = SetupMemory(seed(tmp_path, many(30, 0.002)), min_samples=30).stats(snap())
    assert stats.win_rate == 0.0 and stats.avg_loss_pct == pytest.approx((0.002 - ROUND_TRIP_COST) * 100)


def test_it_backs_off_to_a_coarser_label_until_enough_observations_match(tmp_path):
    # 20 with high volume + 20 with normal volume, all the same trend/RSI/ADX: the fine label has 20, the coarser has 40
    obs = many(20, 0.05, volume=2000) + [(f"N{i}", "2025-01-06", {"volume": 1000}, 0.05, "2025-02-03") for i in range(20)]
    stats = SetupMemory(seed(tmp_path, obs), min_samples=30).stats(snap(volume=2000))
    assert stats.sample_size == 40 and "no_volume" in stats.matched_on
    coarse = SetupMemory(seed(tmp_path, many(35, 0.05, adx=10.0), "coarse.db"), min_samples=30).stats(snap(adx=35.0))
    assert coarse is not None and "trend_rsi" in coarse.matched_on and coarse.sample_size == 35


def test_a_stock_analysed_repeatedly_in_one_week_counts_once(tmp_path):
    obs = [("AAA", str(START + timedelta(days=d)), {}, 0.05, "2025-03-03") for d in range(5)]  # Mon..Fri, one ISO week
    memory = SetupMemory(seed(tmp_path, obs), min_samples=1)
    assert memory.stats(snap()).sample_size == 1
    other_week = obs + [("AAA", str(START + timedelta(days=7)), {}, 0.05, "2025-03-03")]
    sessions = seed(tmp_path, other_week, "weeks.db")
    assert SetupMemory(sessions, min_samples=1).stats(snap()).sample_size == 2


def test_point_in_time_ignores_outcomes_that_had_not_finished_yet(tmp_path):
    memory = SetupMemory(seed(tmp_path, many(30, 0.10)), min_samples=30)
    assert memory.stats(snap(), as_of=date(2025, 2, 2)) is None       # exit was 2025-02-03: unknown that day
    assert memory.stats(snap(), as_of=date(2025, 2, 3)) is not None


def test_an_unloadable_stored_snapshot_is_skipped_not_guessed(tmp_path):
    sessions = seed(tmp_path, many(30, 0.10))
    with sessions() as s:
        d = s.scalars(select(DecisionRecord)).first()
        d.snapshot_json = json.dumps({"symbol": "X", "obsolete_field": 1})
        s.commit()
    assert SetupMemory(sessions, min_samples=30).stats(snap()) is None  # 29 usable rows left
    assert len(SetupMemory(sessions, min_samples=1).rows()) == 29


def test_only_the_latest_decision_per_stock_session_is_used(tmp_path):
    sessions = seed(tmp_path, [("AAA", "2025-01-06", {}, 0.05, "2025-02-03")])
    with sessions() as s:
        s.add(DecisionRecord(symbol="AAA", price=100.0, final_action="HOLD", final_confidence=0.5, reasoning="r", risk_approved=False,
                             risk_quantity=0, risk_reason="x", bar_date="2025-01-06",
                             snapshot_json=json.dumps(snap_dict(symbol="AAA", bar_date="2025-01-06"))))
        s.commit()
    assert len(SetupMemory(sessions, min_samples=1).rows()) == 1


def test_the_result_is_cached_for_the_ttl_then_refreshed(tmp_path):
    sessions = seed(tmp_path, many(30, 0.10))
    now = [0.0]
    memory = SetupMemory(sessions, min_samples=30, ttl_seconds=600, clock=lambda: now[0])
    assert memory.stats(snap()).sample_size == 30
    with sessions() as s:
        s.query(OutcomeRecord).delete()
        s.commit()
    now[0] = 599
    assert memory.stats(snap()) is not None  # still cached
    now[0] = 601
    assert memory.stats(snap()) is None       # refreshed: the outcomes are gone


def test_bands_and_labels():
    assert [adx_band(v) for v in (None, 19.9, 20, 29.9, 30)] == ["na", "weak", "mid", "mid", "strong"]
    assert [volume_band(v, 1000) for v in (799, 800, 1250, 1251)] == ["low", "normal", "normal", "high"]
    assert volume_band(5, None) == "na" and volume_band(5, 0) == "na"
    labels = setup_labels(snap())
    assert labels["full"].startswith(labels["no_volume"]) and labels["no_volume"].startswith(labels["trend_rsi"])
    with pytest.raises(ValueError):
        SetupMemory(None, min_samples=0)


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
        assert run() is None and control.get_flag(LABELLED_ON) is None       # failed: not marked done, no exception
        monkey["fail"] = False
        assert run() is not None and control.get_flag(LABELLED_ON) == d1.isoformat()
        assert run() is None and len(calls) == 2                             # already done today: no second run
        today[0] = d2
        assert run() is not None and len(calls) == 3                         # a new day runs again
    finally:
        jobs.label_outcomes = real


def test_control_flags_round_trip_and_overwrite(tmp_path):
    control = Control(make_session_factory(f"sqlite:///{tmp_path / 'f.db'}"))
    assert control.get_flag("k") is None
    control.set_flag("k", "a")
    control.set_flag("k", "b")
    assert control.get_flag("k") == "b"


def test_a_break_even_result_is_not_a_win_in_the_baseline_or_the_match(tmp_path):
    stats = SetupMemory(seed(tmp_path, many(30, ROUND_TRIP_COST)), min_samples=30).stats(snap())
    assert stats.win_rate == 0.0 and stats.baseline_win_rate == 0.0


def test_a_horizon_that_is_never_labelled_is_refused_rather_than_left_silent(tmp_path):
    """Only 5, 20 and 60 sessions are labelled; e.g. 40 would make the memory find nothing, forever, without a word."""
    import pytest

    from src.database import make_session_factory

    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    with pytest.raises(ValueError, match="labelled horizons"):
        SetupMemory(sessions, horizon=40)
    assert SetupMemory(sessions, horizon=60).horizon == 60


# ---------------------------------------------------------------- setups rebuilt from price history (LEARNING_SEED)
def add_history(sessions, observations, horizon=20):
    """observations: (symbol, bar_date, gross_return, exit_date) setups rebuilt from history."""
    from src.database import SetupObservationRecord

    with sessions() as s:
        for symbol, bar_date, gross, exit_date in observations:
            s.add(SetupObservationRecord(symbol=symbol, bar_date=bar_date, horizon=horizon, exit_date=exit_date, gross_return=gross,
                                         snapshot_json=json.dumps(snap_dict(symbol=symbol, bar_date=bar_date))))
        s.commit()


def history(n, gross, bar_date="2024-01-08", exit_date="2024-02-05"):
    return [(f"H{i}", bar_date, gross, exit_date) for i in range(n)]


def test_seeded_history_is_ignored_unless_switched_on(tmp_path):
    sessions = seed(tmp_path, [])
    add_history(sessions, history(40, 0.10))
    assert SetupMemory(sessions, min_samples=30).stats(snap()) is None
    stats = SetupMemory(sessions, min_samples=30, use_history=True).stats(snap())
    assert stats.sample_size == 40 and stats.win_rate == 1.0


def test_seeded_history_only_counts_for_the_memory_horizon(tmp_path):
    sessions = seed(tmp_path, [])
    add_history(sessions, history(40, 0.10), horizon=60)
    assert SetupMemory(sessions, horizon=20, min_samples=30, use_history=True).stats(snap()) is None


def test_seeded_history_is_point_in_time_too(tmp_path):
    sessions = seed(tmp_path, [])
    add_history(sessions, history(40, 0.10, exit_date="2024-02-05"))
    memory = SetupMemory(sessions, min_samples=30, use_history=True)
    assert memory.stats(snap(), as_of=date(2024, 2, 4)) is None  # none had finished yet
    assert memory.stats(snap(), as_of=date(2024, 2, 5)).sample_size == 40


def test_the_bots_own_observation_wins_its_stock_week_over_a_rebuilt_one(tmp_path):
    own = [(f"H{i}", "2025-01-06", {}, -0.10, "2025-02-03") for i in range(30)]
    sessions = seed(tmp_path, own)
    add_history(sessions, history(30, 0.10, bar_date="2025-01-07", exit_date="2025-02-04"))  # same stocks, same week
    stats = SetupMemory(sessions, min_samples=30, use_history=True).stats(snap())
    assert stats.sample_size == 30 and stats.win_rate == 0.0  # the real calls, not the rebuilt ones


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

    sessions = seed(tmp_path, [(f"S{i}", "2025-01-06", {}, 0.05, "2025-02-03") for i in range(30)])
    with sessions() as s:
        for d in s.scalars(select(DecisionRecord)):
            d.decision_source, d.rule_alignment = "agents", "agree"  # rule said BUY and the AI bought
        s.commit()
    text = learning_report(sessions, horizon=20, min_samples=30)
    assert "rule BUY, AI bought" in text and "n=30" in text and "insufficient" not in text.split("rule BUY, AI bought")[1][:40]


def test_a_failing_report_never_reaches_the_trading_loop_and_is_retried(tmp_path):
    from src.learning.jobs import WeeklyLearningReport

    sessions = seed(tmp_path, [])

    def broken(text):
        raise ConnectionError("telegram down")

    report = WeeklyLearningReport(sessions, Control(sessions), broken, today=lambda: date(2026, 10, 5))
    assert report() is None
    assert Control(sessions).get_flag("learning_report_week") is None  # not marked as sent: next cycle tries again
