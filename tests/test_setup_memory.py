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
