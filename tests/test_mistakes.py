"""Mistake analysis: facts with their uncertainty, never a verdict on too little data."""

import json

import pytest

from src.database import DecisionRecord, OutcomeRecord, make_session_factory
from src.engine.costs import ROUND_TRIP_COST
from src.learning.mistakes import Judged, breakdown, confidence_bucket, group_stats, load_judged, missed_gains, worst_calls


def j(ret, action="BUY", conf=0.75, stop=False, symbol="X"):
    return Judged(symbol, "2025-01-06", action, conf, "llm", "why", ret, stop)


def test_a_group_below_the_minimum_is_not_judged():
    g = group_stats("g", [j(0.5)] * 4, 5)
    assert not g.sufficient and g.mean_net is None and g.n == 4


def on(day, ret, symbol, stop=False, action="BUY"):
    return Judged(symbol, f"2025-01-{day:02d}", action, 0.75, "agents", "why", ret, stop)


def test_mean_interval_win_and_stop_rates():
    rows = [on(6, 0.10, "A"), on(7, -0.10, "B", stop=True), on(8, 0.10, "C"), on(9, -0.10, "D")]
    g = group_stats("g", rows, 4)
    assert g.mean_net == pytest.approx(0.0) and g.win_rate == 0.5 and g.stop_rate == 0.25
    assert g.ci95 == pytest.approx(1.96 * (0.04 / 3) ** 0.5 / 2)  # four dates, one +-10% result each
    assert (g.used, g.dates) == (4, 4)


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
            s.add(
                DecisionRecord(
                    symbol="AAA",
                    price=1.0,
                    final_action=action,
                    final_confidence=0.6,
                    reasoning="r",
                    risk_approved=False,
                    risk_quantity=0,
                    risk_reason="x",
                    bar_date="2025-01-06",
                    snapshot_json=json.dumps({}),
                )
            )
        s.add(
            OutcomeRecord(
                symbol="AAA",
                bar_date="2025-01-06",
                horizon=20,
                entry_date="2025-01-07",
                exit_date="2025-02-04",
                entry_price=100.0,
                exit_price=110.0,
                gross_return=0.10,
                worst_return=-0.02,
                hit_stop=False,
                stop_pct=0.15,
            )
        )
        s.commit()
    (row,) = load_judged(sessions, 20)
    assert row.action == "HOLD" and row.net_return == pytest.approx(0.10 - ROUND_TRIP_COST)
    assert load_judged(sessions, 5) == []


def test_a_veto_is_recognised_from_the_reasoning_and_model_fields_are_loaded(tmp_path):
    assert Judged("X", "d", "HOLD", 0.5, None, "base [reflection vetoed to HOLD: risk]", 0.0, False).vetoed
    assert not Judged("X", "d", "BUY", 0.7, None, "base [reflection upheld: fine]", 0.0, False).vetoed
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'v.db'}")
    with sessions() as s:
        s.add(
            DecisionRecord(
                symbol="AAA",
                price=1.0,
                final_action="BUY",
                final_confidence=0.6,
                reasoning="r",
                risk_approved=False,
                risk_quantity=0,
                risk_reason="x",
                bar_date="2025-01-06",
                raw_model="m1",
                rule_alignment="agree",
            )
        )
        s.add(
            OutcomeRecord(
                symbol="AAA",
                bar_date="2025-01-06",
                horizon=20,
                entry_date="a",
                exit_date="b",
                entry_price=1.0,
                exit_price=1.0,
                gross_return=0.0,
                worst_return=0.0,
                hit_stop=False,
                stop_pct=0.15,
            )
        )
        s.commit()
    (row,) = load_judged(sessions, 20)
    assert row.model == "m1" and row.rule_alignment == "agree"


def test_ai_vs_rule_classifies_by_the_rules_own_verdict():
    from src.learning.mistakes import Judged, ai_vs_rule

    def j(action, rule, source="agents"):
        return Judged("X", "2026-09-01", action, 0.7, source, "r", 0.0, False, rule_action=rule)

    assert ai_vs_rule(j("BUY", "BUY")) == "rule BUY, AI bought"
    assert ai_vs_rule(j("HOLD", "BUY")) == "rule BUY, AI passed"
    assert ai_vs_rule(j("SELL", "BUY")) == "rule BUY, AI passed"
    assert ai_vs_rule(j("BUY", "HOLD")) == "rule passed, AI bought"
    assert ai_vs_rule(j("HOLD", "HOLD")) == "both passed"
    assert ai_vs_rule(j("SELL", "HOLD", source="trend_exit")) == "n/a"  # a rule-based exit is not an AI call
    assert ai_vs_rule(j("HOLD", None)) == "n/a"  # the rule's verdict is unknown (no snapshot)


def test_a_critic_vetoed_rule_buy_is_a_pass_on_a_rule_buy_not_both_passed():
    """The bug: the analyst agreed with the rule's BUY, the critic turned it into a HOLD, and the old classification (from
    the analyst's pre-critic alignment) filed it as 'both passed'."""
    from src.learning.mistakes import Judged, ai_vs_rule, rule_buy_passes

    vetoed = Judged("X", "2026-09-01", "HOLD", 0.3, "agents", "BUY [reflection vetoed to HOLD: risk]", 0.0, False,
                    rule_alignment="agree", rule_action="BUY")  # fmt: skip
    analyst = Judged(
        "Y", "2026-09-01", "HOLD", 0.6, "agents", "no catalyst", 0.0, False, rule_alignment="deviate", rule_action="BUY"
    )
    assert ai_vs_rule(vetoed) == "rule BUY, AI passed" and rule_buy_passes(vetoed) == "critic vetoed the BUY"
    assert ai_vs_rule(analyst) == "rule BUY, AI passed" and rule_buy_passes(analyst) == "analyst passed"


def test_older_rows_get_the_rules_verdict_from_their_snapshot():
    from src.learning.mistakes import rule_action_from_snapshot

    up = {"price": 110.0, "ma50": 100.0, "ma200": 90.0, "rsi": 55.0}
    assert rule_action_from_snapshot(json.dumps(up)) == "BUY"
    assert rule_action_from_snapshot(json.dumps({**up, "rsi": 75.0})) == "HOLD"  # overbought
    assert rule_action_from_snapshot(json.dumps({**up, "price": 95.0})) == "HOLD"  # below MA50
    assert rule_action_from_snapshot(json.dumps({"price": 1.0})) is None and rule_action_from_snapshot(None) is None


def test_the_backfill_agrees_with_the_recorded_alignment_on_every_call_the_critic_left_alone():
    """Acceptance check for the backfill: where the critic did not veto, the analyst's alignment and its action pin down the
    rule's verdict, and the snapshot recomputation must give the same answer."""
    from src.learning.mistakes import rule_action_from_snapshot

    cases = [  # (snapshot, final action, recorded alignment)
        ({"price": 110.0, "ma50": 100.0, "ma200": 90.0, "rsi": 55.0}, "BUY", "agree"),
        ({"price": 110.0, "ma50": 100.0, "ma200": 90.0, "rsi": 55.0}, "HOLD", "deviate"),
        ({"price": 95.0, "ma50": 100.0, "ma200": 90.0, "rsi": 55.0}, "HOLD", "agree"),
        ({"price": 95.0, "ma50": 100.0, "ma200": 90.0, "rsi": 55.0}, "BUY", "deviate"),
    ]
    for snapshot, action, alignment in cases:
        implied = action if alignment == "agree" else ("HOLD" if action == "BUY" else "BUY")
        assert rule_action_from_snapshot(json.dumps(snapshot)) == implied


def test_overlapping_outcomes_of_one_stock_count_once_per_horizon_block():
    from src.learning.mistakes import independent_sample

    days = [f"2025-02-{d:02d}" for d in range(1, 26)]  # 25 sessions
    rows = [Judged("AAA", d, "BUY", 0.7, "agents", "r", 0.01, False) for d in days]
    rows += [Judged("BBB", d, "BUY", 0.7, "agents", "r", 0.02, False) for d in days[:3]]
    kept = independent_sample(rows, horizon=20)
    assert [(k.symbol, k.bar_date) for k in kept] == [("AAA", days[0]), ("BBB", days[0]), ("AAA", days[20])]
    g = group_stats("g", rows, 2, horizon=20)
    assert (g.n, g.used, g.dates) == (28, 3, 2)  # 28 rows were 3 independent observations on 2 dates


def test_the_report_line_states_the_dates_behind_each_interval():
    from src.learning.mistakes import fmt

    rows = [on(6, 0.10, "A"), on(7, -0.10, "B"), on(8, 0.10, "C"), on(9, -0.10, "D")]
    assert "4 independent observations on 4 dates" in fmt(group_stats("g", rows, 4))
    assert "insufficient data (1 independent observations on 1 dates)" in fmt(group_stats("g", [on(6, 0.1, "A")] * 3, 2))


def test_every_decision_records_the_rules_own_verdict(tmp_path):
    """Written by the pipeline at decision time, so the report never has to infer it from the analyst's answer."""
    from sqlalchemy import select

    from tests.test_llm_and_pipeline import make_pipeline

    pipe, _, sessions = make_pipeline(tmp_path, technical_action="HOLD")
    pipe.run_once()
    with sessions() as s:
        rows = s.execute(select(DecisionRecord.rule_action, DecisionRecord.snapshot_json)).all()
    from src.learning.mistakes import rule_action_from_snapshot

    assert rows and all(rule in ("BUY", "HOLD") and rule == rule_action_from_snapshot(snap) for rule, snap in rows)
