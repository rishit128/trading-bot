"""Mistake analysis: facts with their uncertainty, never a verdict on too little data."""
import json

import pytest

from src.database import DecisionRecord, OutcomeRecord, make_session_factory
from src.learning.mistakes import (Judged, breakdown, confidence_bucket, group_stats, load_judged, missed_gains, worst_calls)
from src.learning.setups import ROUND_TRIP_COST


def j(ret, action="BUY", conf=0.75, stop=False, symbol="X"):
    return Judged(symbol, "2025-01-06", action, conf, "llm", "why", ret, stop)


def test_a_group_below_the_minimum_is_not_judged():
    g = group_stats("g", [j(0.5)] * 4, 5)
    assert not g.sufficient and g.mean_net is None and g.n == 4


def test_mean_interval_win_and_stop_rates():
    g = group_stats("g", [j(0.10), j(-0.10, stop=True), j(0.10), j(-0.10)], 4)
    assert g.mean_net == pytest.approx(0.0) and g.win_rate == 0.5 and g.stop_rate == 0.25
    assert g.ci95 == pytest.approx(1.96 * (0.04 / 3) ** 0.5 / 2)  # sample sd of four +-10% results


def test_breakdown_groups_by_key_and_confidence_buckets_have_the_right_edges():
    rows = [j(0.1, action="BUY"), j(0.1, action="BUY"), j(0.0, action="HOLD")]
    assert {g.label: g.n for g in breakdown(rows, lambda x: x.action, 1)} == {"BUY": 2, "HOLD": 1}
    assert [confidence_bucket(j(0, conf=c)) for c in (0.69, 0.70, 0.79, 0.80)] == ["<0.70", "0.70-0.80", "0.70-0.80", ">=0.80"]


def test_worst_buys_and_missed_gains_pick_the_right_side():
    rows = [j(-0.3, symbol="A"), j(0.2, symbol="B"), j(-0.1, symbol="C"), j(0.5, "HOLD", symbol="D"), j(-0.9, "HOLD", symbol="E")]
    assert [x.symbol for x in worst_calls(rows, "BUY", 2)] == ["A", "C"]
    assert [x.symbol for x in missed_gains(rows, 1)] == ["D"]


def test_load_judged_uses_the_latest_decision_and_subtracts_costs(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    with sessions() as s:
        for action in ("BUY", "HOLD"):  # the same session decided twice: the later call stands
            s.add(DecisionRecord(symbol="AAA", price=1.0, final_action=action, final_confidence=0.6, reasoning="r", risk_approved=False,
                                 risk_quantity=0, risk_reason="x", bar_date="2025-01-06", snapshot_json=json.dumps({})))
        s.add(OutcomeRecord(symbol="AAA", bar_date="2025-01-06", horizon=20, entry_date="2025-01-07", exit_date="2025-02-04",
                            entry_price=100.0, exit_price=110.0, gross_return=0.10, worst_return=-0.02, hit_stop=False, stop_pct=0.15))
        s.commit()
    (row,) = load_judged(sessions, 20)
    assert row.action == "HOLD" and row.net_return == pytest.approx(0.10 - ROUND_TRIP_COST)
    assert load_judged(sessions, 5) == []


def test_a_veto_is_recognised_from_the_reasoning_and_model_fields_are_loaded(tmp_path):
    assert Judged("X", "d", "HOLD", 0.5, None, "base [reflection vetoed to HOLD: risk]", 0.0, False).vetoed
    assert not Judged("X", "d", "BUY", 0.7, None, "base [reflection upheld: fine]", 0.0, False).vetoed
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'v.db'}")
    with sessions() as s:
        s.add(DecisionRecord(symbol="AAA", price=1.0, final_action="BUY", final_confidence=0.6, reasoning="r", risk_approved=False,
                             risk_quantity=0, risk_reason="x", bar_date="2025-01-06", raw_model="m1", rule_alignment="agree"))
        s.add(OutcomeRecord(symbol="AAA", bar_date="2025-01-06", horizon=20, entry_date="a", exit_date="b", entry_price=1.0,
                            exit_price=1.0, gross_return=0.0, worst_return=0.0, hit_stop=False, stop_pct=0.15))
        s.commit()
    (row,) = load_judged(sessions, 20)
    assert row.model == "m1" and row.rule_alignment == "agree"
