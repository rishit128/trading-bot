"""`AgentSignal.details` is typed: what is written to the audit table keeps a stable shape, and old rows (including those
from the removed decision-memory and market-context phases) still load."""
import json

import pytest

from src.agents.technical import TechnicalAgent
from src.engine.agent_signal import AgentSignal, SignalDetails
from src.pipeline import details_columns
from tests.test_agent_analysis import COMPANY, COT_GOOD, MARKET, REFLECT_OK, context, llm

REAL_TOP_LEVEL = {"action", "confidence", "degraded", "details", "raw_model", "reasoning"}
REAL_REFLECTION_KEYS = {"base_confidence_entering", "bias_check", "biggest_risk", "conviction_adjustments", "critic_confidence",
                        "final_action", "final_confidence", "humility_reduction", "upheld", "what_proves_us_wrong"}


def full_signal():
    return TechnicalAgent(llm([COT_GOOD, REFLECT_OK])).analyze(context(MARKET, COMPANY))


def test_the_stored_json_has_a_stable_shape():
    stored = full_signal().to_json_dict()
    assert set(stored) == REAL_TOP_LEVEL
    details = stored["details"]
    assert set(details["reflection"]) == REAL_REFLECTION_KEYS
    assert set(details["reasoning_chain"]) == {"trend", "overbought", "volume", "company", "market"}
    assert details["company_json"]["pe"] == pytest.approx(19.5) and details["market_context_json"]["vix"] == 13.0
    assert details["reflection"]["conviction_adjustments"][0] == {"stage": "step1_technical", "conviction": pytest.approx(0.75)}
    assert None not in details.values()  # unset sections are absent, not stored as nulls
    json.dumps(stored)  # and it is plain JSON


def test_a_signal_survives_the_round_trip_through_the_audit_table():
    original = full_signal()
    restored = AgentSignal(**json.loads(json.dumps(original.to_json_dict())))  # what replay does
    assert restored.action == original.action and restored.confidence == original.confidence
    assert restored.details == original.details


def test_a_real_stored_row_from_before_this_change_still_loads():
    """A trimmed copy of a live decision written by the old agent (learning/context sections, pattern_id, a future key)."""
    legacy = {"action": "BUY", "confidence": 0.6, "reasoning": "r", "degraded": False, "raw_model": None, "details": {
        "tier": "cot", "raw_model": "nvidia/nemotron-3-super-120b-a12b:free", "edge_confidence": 0.5, "confluence_score": 4,
        "risks": ["gap"], "reasoning_chain": {"trend": "t", "overbought": "o", "volume": "v"}, "rule_alignment": "agree",
        "falsification": "a close below 240", "base_confidence": 0.6, "pattern_id": "P>MA50+MA50>MA200|RSI50",
        "learning": {"win_rate": 0.4}, "context": {"regime": "bull"},
        "adjusted_signal_confidence": 0.6, "adjustment_reason": "self-critique cut confidence to 0.60",
        "reflection": {"biggest_risk": "b", "what_proves_us_wrong": "w", "bias_check": ["anchoring"],
                       "conviction_adjustments": [{"stage": "step1_technical", "conviction": 0.6}],
                       "final_action": "BUY", "final_confidence": 0.6, "base_confidence_entering": 0.6,
                       "humility_reduction": 0.1, "critic_confidence": 0.4, "upheld": True},
        "a_key_from_the_future": 1}}
    signal = AgentSignal(**legacy)
    assert signal.details.tier == "cot" and signal.details.reflection.upheld is True
    assert signal.details.reasoning_chain.company is None and signal.details.company_json is None
    assert signal.action == "BUY" and signal.details.raw_model.startswith("nvidia")


def test_invalid_values_are_rejected_instead_of_silently_accepted():
    with pytest.raises(ValueError):
        SignalDetails(tier="turbo")  # not one of the two tiers
    with pytest.raises(ValueError):
        SignalDetails(rule_alignment="maybe")


def test_the_decision_columns_are_derived_from_the_typed_details():
    cols = details_columns(full_signal().details)
    assert cols["base_confidence"] == pytest.approx(0.75)
    assert json.loads(cols["risks_json"]) and json.loads(cols["reasoning_chain_json"])["company"].startswith("Sales")
    assert json.loads(cols["market_context_json"])["index_above_ma200"] is True
    assert json.loads(cols["bias_check"]) == ["confirmation bias", "anchoring to the trend"]
    assert [c["stage"] for c in json.loads(cols["conviction_adjustments"])][0] == "step1_technical"
    assert cols["rule_alignment"] in ("agree", "deviate")
    assert details_columns(None) == {}
    assert details_columns(SignalDetails())["risks_json"] is None and details_columns(SignalDetails())["bias_check"] is None


def test_a_vetoed_call_records_the_veto_and_its_reason_and_takes_the_critics_confidence():
    veto = json.dumps({**json.loads(REFLECT_OK), "action": "HOLD"})
    signal = TechnicalAgent(llm([COT_GOOD, veto])).analyze(context())
    assert signal.action == "HOLD" and signal.confidence == pytest.approx(0.5)  # the critic's humility-discounted number
    assert signal.details.reflection.upheld is False and signal.details.reflection.critic_confidence == pytest.approx(0.5)
    assert signal.details.adjustment_reason == "reflection vetoed the call: a gap-down on results"
    assert "vetoed to HOLD" in signal.reasoning
