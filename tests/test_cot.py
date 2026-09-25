"""Chain-of-thought: the structured schema, the client's generic structured call, prompt construction,
agent fail-safes, and that the reasoning chain is persisted and replay-able."""
import json

import pandas as pd
import pytest
from sqlalchemy import select

from src.agents.technical import AdvancedSignal, COT_SCHEMA, TechnicalAgent
from src.agents.base import ADVISOR, LEAD, AgentContext
from src.config import RiskLimits, Settings, load_settings
from src.data.indicators import Snapshot, build_snapshot
from src.database import DecisionRecord, make_session_factory
from src.engine.ports import Fill
from src.engine.risk_engine import Portfolio
from src.llm import LLMClient, LLMUnavailable
from src.pipeline import TradingPipeline
from src.ops.replay import replay
from tests.test_llm_and_pipeline import FakeOpenAI, StubAgent, api_error

SNAP = Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000)
CTX = AgentContext("AAPL", SNAP)

COT_GOOD = json.dumps({
    "action": "BUY", "confidence": 0.75, "edge_confidence": 0.6,
    "step1_trend": "Uptrend; price above MA50 and MA50 above MA200.",
    "step2_overbought": "RSI 55; not overbought.",
    "step3_volume": "Volume 1.3x average; confirms the move.",
    "step4_confluence": 8,
    "step5_risks": ["Earnings next week", "Sector rotation"],
    "final_reasoning": "Clear uptrend, volume confirms, not overbought.",
})


def test_cot_agent_parses_into_signal_with_details():
    agent = TechnicalAgent(LLMClient(["a"], client=FakeOpenAI({"a": COT_GOOD})))
    s = agent.analyze(CTX)
    assert s.action == "BUY" and s.confidence == 0.75
    assert s.reasoning == "Clear uptrend, volume confirms, not overbought."
    assert s.details.edge_confidence == 0.6 and s.details.confluence_score == 8
    assert s.details.reasoning_chain.trend.startswith("Uptrend")
    assert s.details.risks == ["Earnings next week", "Sector rotation"]


def test_cot_agent_fails_safe_to_hold_without_details():
    agent = TechnicalAgent(LLMClient(["a"], client=FakeOpenAI({"a": api_error()})))
    s = agent.analyze(CTX)
    assert s.action == "HOLD" and s.degraded is True and s.details is None


def test_cot_prompt_has_the_five_steps_and_volume_ratio():
    p = TechnicalAgent.build_prompt(SNAP)
    for token in ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5", "RSI(14): 60.0", "n/a"):
        assert token in p, token
    with_avg = Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000, avg_volume=800)
    assert "1.25x" in TechnicalAgent.build_prompt(with_avg)


def test_simple_prompt_mode_used_when_cot_disabled():
    text = json.dumps({"action": "BUY", "confidence": 0.8, "reasoning": "uptrend"})
    fake = FakeOpenAI({"a": text})
    agent = TechnicalAgent(LLMClient(["a"], client=fake), use_cot=False)
    s = agent.analyze(CTX)
    assert s.action == "BUY" and s.details is None
    sent = fake.calls[0][1]["messages"][0]["content"]
    assert "STEP 1" not in sent and "50-day MA" in sent


def test_cot_schema_is_strict_and_complete():
    props = COT_SCHEMA["schema"]["properties"]
    required = set(COT_SCHEMA["schema"]["required"])
    assert COT_SCHEMA["schema"]["additionalProperties"] is False
    assert required == set(props)
    assert props["action"]["enum"] == ["BUY", "SELL", "HOLD"]
    assert props["step4_confluence"]["maximum"] == 10
    assert props["edge_confidence"]["minimum"] == 0 and props["edge_confidence"]["maximum"] == 1


def test_structured_call_sends_the_schema_and_caches_identical_prompts():
    fake = FakeOpenAI({"a": COT_GOOD})
    llm = LLMClient(["a"], client=fake, cache_ttl_seconds=3600)
    first = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    second = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    assert first is second  # served from cache, not a second call
    assert len(fake.calls) == 1
    schema_sent = fake.calls[0][1]["response_format"]["json_schema"]
    assert schema_sent["name"] == "trading_signal_v1" and schema_sent["strict"] is True


def test_structured_call_falls_back_to_the_next_model():
    fake = FakeOpenAI({"a": api_error(), "b": COT_GOOD})
    out = LLMClient(["a", "b"], client=fake).structured_call("p", AdvancedSignal, COT_SCHEMA)
    assert isinstance(out, AdvancedSignal) and out.action == "BUY"


def test_structured_call_raises_when_every_model_fails():
    fake = FakeOpenAI({"a": "garbage", "b": json.dumps({"action": "BUY", "confidence": 0.5})})
    with pytest.raises(LLMUnavailable):
        LLMClient(["a", "b"], client=fake).structured_call("p", AdvancedSignal, COT_SCHEMA)


def sample_snapshot(sym: str) -> Snapshot:
    return Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000)


def make_details_pipeline(tmp_path):
    """A pipeline whose lead agent returns a chain-of-thought AgentSignal with full details."""
    details = {"edge_confidence": 0.6, "confluence_score": 8, "risks": ["earnings next week"],
               "reasoning_chain": {"trend": "uptrend above MA50", "overbought": "RSI 55 not overbought",
                                   "volume": "volume confirms"}}
    tech = StubAgent(lambda s: Signal_stub(details), "technical", LEAD)
    sent = StubAgent(lambda sym, heads: None, "sentiment", ADVISOR)
    sessions = make_session_factory(f"sqlite:///{tmp_path / 't.db'}")

    class FakeBroker:
        def portfolio(self):
            return Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)

        def is_market_open(self):
            return True

        def has_open_order(self, symbol):
            return False

        def buy_with_bracket(self, *a):
            return Fill("x", "accepted")

        def sell(self, *a):
            return Fill("y", "accepted")

    pipe = TradingPipeline(Settings(watchlist=("AAPL",), dry_run=True, risk=RiskLimits()),
                           [tech, sent], FakeBroker(), sessions, sample_snapshot, lambda sym: [])
    return pipe, sessions


def Signal_stub(details):
    from src.llm import AgentSignal
    return AgentSignal(action="BUY", confidence=0.75, reasoning="r", details=details)


def test_cot_details_are_persisted_and_replayed(tmp_path):
    pipe, sessions = make_details_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        assert row.confluence_score == 8
        assert row.edge_confidence == 0.6
        assert json.loads(row.risks_json) == ["earnings next week"]
        assert json.loads(row.reasoning_chain_json)["trend"].startswith("uptrend")
        session_result = replay(row)
    assert session_result.signals["technical"].details.confluence_score == 8
    assert "STEP 1" in session_result.prompt
    assert session_result.reproduced is True


def test_cot_columns_are_null_for_a_signal_without_details(tmp_path):
    pipe, sessions = make_details_pipeline(tmp_path)
    pipe.agents["technical"] = StubAgent(lambda s: Signal_stub(None), "technical", LEAD)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord).order_by(DecisionRecord.id.desc()))
    assert row.confluence_score is None and row.edge_confidence is None
    assert row.risks_json is None and row.reasoning_chain_json is None


def test_snapshot_records_avg_volume():
    bars = pd.DataFrame({"Close": [float(i) for i in range(1, 251)],
                         "Volume": [1000.0] * 230 + [2000.0] * 20})
    s = build_snapshot("X", bars)
    assert s.avg_volume == 2000 and s.volume == 2000


def test_llm_cot_env_flag(monkeypatch):
    monkeypatch.delenv("LLM_COT", raising=False)
    assert load_settings().llm_cot is True
    monkeypatch.setenv("LLM_COT", "false")
    assert load_settings().llm_cot is False