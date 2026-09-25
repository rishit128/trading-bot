"""`AgentSignal.details` is typed now; what is written to the audit table must look exactly as it did when it was a dict, and
old rows must still load."""
import json

import pytest

from src.agents.technical import TechnicalAgent
from src.engine.agent_signal import AgentSignal, SignalDetails

from src.pipeline import details_columns

from tests.test_learning_context_reflect import (COT_GOOD, CONTEXT_OK, LEARN_KEEP, REFLECT_OK, agent_context, llm, make_stats)
from src.data.market_context import MarketContext

# The keys of a real stored decision (taken from trading.db, 2026-09-25): the shape the analysis scripts depend on.
REAL_TOP_LEVEL = {"action", "confidence", "degraded", "details", "raw_model", "reasoning"}
REAL_REFLECTION_KEYS = {"base_confidence_entering", "bias_check", "biggest_risk", "conviction_adjustments", "critic_confidence",
                        "final_action", "final_confidence", "humility_reduction", "upheld", "what_proves_us_wrong"}
REAL_CONTEXT_KEYS = {"diversification_score", "earnings_risk", "index_above_ma200", "index_rsi", "macro_support", "reasoning",
                     "regime", "risks", "sector_support", "vix", "vix_percentile", "vs_index_6m"}
REAL_LEARNING_KEYS = {"adjusted_action", "adjusted_confidence", "pattern_reliability", "sample_size", "win_rate", "avg_win_pct",
                      "avg_loss_pct", "profit_factor", "best_holding_days", "worst_holding_days", "confidence_in_pattern",
                      "reason", "what_could_break"}


def full_signal():
    stressed = MarketContext(index_above_ma200=True, index_rsi=55.0, vix=28.0, vix_percentile=0.9)
    agent = TechnicalAgent(llm([COT_GOOD, LEARN_KEEP, CONTEXT_OK, REFLECT_OK]), history_fn=lambda s: make_stats(12, 0.6))
    return agent.analyze(agent_context(market=stressed))


def test_the_stored_json_has_the_same_shape_as_the_old_dict():
    stored = full_signal().to_json_dict()
    assert set(stored) == REAL_TOP_LEVEL
    details = stored["details"]
    assert set(details["reflection"]) == REAL_REFLECTION_KEYS
    assert set(details["context"]) == REAL_CONTEXT_KEYS
    assert set(details["learning"]) == REAL_LEARNING_KEYS
    assert set(details["reasoning_chain"]) == {"trend", "overbought", "volume"}
    assert details["reflection"]["conviction_adjustments"][0] == {"stage": "step1_technical", "conviction": pytest.approx(0.6)}
    assert None not in details.values()  # unset sections are absent, exactly as before, not stored as nulls
    assert details["context"]["vs_index_6m"] is None  # ... but a section that ran keeps all its keys, nulls included
    json.dumps(stored)  # and it is plain JSON


def test_a_signal_survives_the_round_trip_through_the_audit_table():
    original = full_signal()
    restored = AgentSignal(**json.loads(json.dumps(original.to_json_dict())))  # what replay does
    assert restored.action == original.action and restored.confidence == original.confidence
    assert restored.details == original.details


def test_a_real_stored_row_from_before_this_change_still_loads():
    """A trimmed copy of a live decision (with a key the model does not know, as a future version might add)."""
    legacy = {"action": "BUY", "confidence": 0.6, "reasoning": "r", "degraded": False, "raw_model": None, "details": {
        "tier": "cot", "raw_model": "nvidia/nemotron-3-super-120b-a12b:free", "edge_confidence": 0.5, "confluence_score": 4,
        "risks": ["gap"], "reasoning_chain": {"trend": "t", "overbought": "o", "volume": "v"}, "rule_alignment": "agree",
        "falsification": "a close below 240", "base_confidence": 0.6, "pattern_id": "P>MA50+MA50>MA200|RSI50",
        "adjusted_signal_confidence": 0.6, "adjustment_reason": "self-critique cut confidence to 0.60",
        "reflection": {"biggest_risk": "b", "what_proves_us_wrong": "w", "bias_check": ["anchoring"],
                       "conviction_adjustments": [{"stage": "step1_technical", "conviction": 0.6}],
                       "final_action": "BUY", "final_confidence": 0.6, "base_confidence_entering": 0.6,
                       "humility_reduction": 0.1, "critic_confidence": 0.4, "upheld": True},
        "a_key_from_the_future": 1}}
    signal = AgentSignal(**legacy)
    assert signal.details.tier == "cot" and signal.details.reflection.upheld is True and signal.details.learning is None
    assert signal.action == "BUY" and signal.details.raw_model.startswith("nvidia")


def test_partial_sections_are_rejected_instead_of_silently_accepted():
    with pytest.raises(ValueError):
        SignalDetails(learning={"win_rate": 0.45})  # the old dict would have taken this; the typed model demands the rest
    with pytest.raises(ValueError):
        SignalDetails(tier="turbo")  # not one of the two tiers
    with pytest.raises(ValueError):
        SignalDetails(rule_alignment="maybe")


def test_the_decision_columns_are_derived_from_the_typed_details():
    cols = details_columns(full_signal().details)
    assert cols["base_confidence"] == pytest.approx(0.75) and cols["context_regime"] == "risk_off"
    assert cols["historical_sample_size"] == 12 and cols["historical_win_rate"] == pytest.approx(0.6)
    assert json.loads(cols["risks_json"]) and json.loads(cols["reasoning_chain_json"]).keys() == {"trend", "overbought", "volume"}
    assert json.loads(cols["bias_check"]) == ["confirmation bias", "anchoring to the trend"]
    assert [c["stage"] for c in json.loads(cols["conviction_adjustments"])][0] == "step1_technical"
    assert cols["pattern_id"].startswith("P>MA50") and cols["rule_alignment"] in ("agree", "deviate")
    assert details_columns(None) == {}
    assert details_columns(SignalDetails())["risks_json"] is None and details_columns(SignalDetails())["bias_check"] is None


def test_the_recorded_reason_says_what_reflection_actually_did():
    upheld = TechnicalAgent(llm([COT_GOOD, REFLECT_OK]), use_learning=False, use_context=False).analyze(agent_context())
    assert upheld.details.adjustment_reason == "reflection upheld the call unchanged"


def test_a_vetoed_call_records_the_veto_and_its_reason_and_takes_the_critics_confidence():
    veto = json.dumps({**json.loads(REFLECT_OK), "action": "HOLD"})
    signal = TechnicalAgent(llm([COT_GOOD, veto]), use_learning=False, use_context=False).analyze(agent_context())
    assert signal.action == "HOLD" and signal.confidence == pytest.approx(0.5)  # the critic's humility-discounted number
    assert signal.details.reflection.upheld is False and signal.details.reflection.critic_confidence == pytest.approx(0.5)
    assert signal.details.adjustment_reason == "reflection vetoed the call: a gap-down on earnings"
    assert "vetoed to HOLD" in signal.reasoning
