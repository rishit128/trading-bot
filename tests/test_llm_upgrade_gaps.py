"""Focused tests for the LLM-upgrade gaps called out in the review doc that earlier tests do not cover:

  * phase 1: the one-shot LLM retry on parse/validation failure (and that transport/dead-model failures are not retried)
  * phase 2: the confidence cap helper, the pattern label, `find_similar_patterns` [] + limit, `analyze_pattern` mapping
  * phase 4: the client-side clamp that stops a stage from raising conviction, and the fixed humility tail
  * cross: the `LLM_MAX_ADJUST` setting, and that both new indexes exist on fresh and pre-existing databases"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.agents.agents import (AdvancedSignal, COT_SCHEMA, ReflectionChain, ReflectionStage, TechnicalAgent,
                               mechanical_action, mechanical_baseline)
from src.agents.base import LEAD
from src.agents.history import MAX_MATCHES, PatternMatch, _pattern_id, analyze_pattern, find_similar_patterns
from src.config import Settings, load_settings
from src.data.indicators import Snapshot
from src.database import DecisionRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.llm import LLMClient, LLMUnavailable, RETRY_HINT, Signal
from src.pipeline import TradingPipeline
from src.replay import replay
from src.versions import VERSIONS
from tests.test_cot import COT_GOOD
from tests.test_learning_context_reflect import SNAP, _add_decision, _add_trade
from tests.test_llm_and_pipeline import FakeBroker, FakeOpenAI, StubAgent

SNAP_RSI45 = Snapshot("AAPL", 100.0, 98.0, 95.0, 45.0, 1000)
DOWN = Snapshot("AAPL", 90.0, 92.0, 95.0, 55.0, 1000)  # price below MA50, MA50 below MA200


def _completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class RetryClient:
    """First call per model is 'on_fail', every later call is 'on_retry'; records messages for inspection."""

    def __init__(self, on_fail, on_retry):
        self.calls = []
        self.on_fail, self.on_retry = on_fail, on_retry
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **kw):
        self.calls.append((model, kw))
        used_up = sum(1 for m, _ in self.calls if m == model) > 1
        return _completion(self.on_retry if used_up else self.on_fail)


def _ask_one(fail_content, retry_content, model="a"):
    client = RetryClient(fail_content, retry_content)
    llm = LLMClient([model], client=client)
    sig = llm.signal("p")
    return sig, client.calls


# ------------------------------------------------------------------ phase 1: retry-once, and where retry does NOT apply
def test_unparseable_json_is_retried_once_then_used():
    sig, calls = _ask_one("not json", json.dumps({"action": "BUY", "confidence": 0.8, "reasoning": "ok"}))
    assert sig.action == "BUY" and len(calls) == 2  # one retry, never more
    assert RETRY_HINT.split(" ")[0] in calls[1][1]["messages"][-1]["content"]


def test_validation_failure_is_retried_once_then_used():
    # Valid JSON that fails the Signal schema (action outside the enum): also a retry, not a model skip.
    sig, calls = _ask_one(json.dumps({"action": "TO THE MOON", "confidence": 0.8, "reasoning": "r"}),
                          json.dumps({"action": "SELL", "confidence": 0.4, "reasoning": "down"}))
    assert sig.action == "SELL" and len(calls) == 2


def test_two_bad_answers_fail_the_model_and_raise():
    client = RetryClient("broken", "also broken")
    with pytest.raises(LLMUnavailable):
        LLMClient(["a"], client=client).signal("p")
    assert len(client.calls) == 2  # retried exactly once, then given up


def test_transient_errors_are_retried_then_the_next_model_and_missing_models_are_skipped():
    import httpx
    import openai

    class NoRetryClient:
        def __init__(self, behaviour):
            self.calls = []
            self.behaviour = behaviour
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, **kw):
            self.calls.append(model)
            raise self.behaviour

    api = NoRetryClient(openai.APIConnectionError(request=httpx.Request("POST", "http://x")))
    with pytest.raises(LLMUnavailable):
        LLMClient(["a", "b"], client=api).signal("p")
    assert api.calls == ["a"] * 3 + ["b"] * 3  # transient: the first try plus two backed-off retries, then the next model

    gone = NoRetryClient(openai.NotFoundError(
        "gone", response=httpx.Response(404, request=httpx.Request("POST", "http://x")), body=None))
    llm = LLMClient(["a"], client=gone)
    with pytest.raises(LLMUnavailable):
        llm.signal("p")
    assert gone.calls == ["a"]  # a missing model is skipped, never retried
    assert "a" in llm.dead  # and skipped for the rest of this session


# ------------------------------------------------------------------ phase 2: cap, label, pattern lookup
def test_clamp_adjustment_bounds_and_cap():
    clamp = TechnicalAgent._clamp_adjustment
    assert clamp(0.75, 0.7, 0.15) == pytest.approx(0.7)  # inside the cap: applied as-is
    assert clamp(0.75, 0.5, 0.15) == pytest.approx(0.6)  # a -0.25 request is capped at -0.15
    assert clamp(0.75, 1.0, 0.15) == pytest.approx(0.9)  # a +0.25 request is capped at +0.15
    assert clamp(0.05, 0.0, 0.15) == pytest.approx(0.0)  # the [0,1] floor still applies
    assert clamp(0.95, 1.0, 0.15) == pytest.approx(1.0)  # the [0,1] ceiling still applies


def test_pattern_id_label_is_stable_and_shape_keyed():
    assert _pattern_id(SNAP) == "P>MA50+MA50>MA200|RSI60"
    assert _pattern_id(SNAP_RSI45) == "P>MA50+MA50>MA200|RSI40"  # RSI decade bucket, not the raw value
    assert _pattern_id(DOWN) == "P<=MA50+MA50<=MA200|RSI50"


def test_find_similar_patterns_returns_empty_without_closed_trades(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    with sessions() as s:
        _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, 60.0, now - timedelta(days=10))
        s.commit()  # a decision exists but nothing closed: nothing to learn from
    assert find_similar_patterns(sessions, "AAPL", (95.0, 105.0), (True, True), (55.0, 65.0), now=now) == []


def test_find_similar_patterns_respects_limit_and_orders_by_quality(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    with sessions() as s:
        for i in range(MAX_MATCHES + 5):
            opened = now - timedelta(days=30 + i)
            _add_trade(s, "AAPL", opened, held_days=5, entry=100.0, exit=101.0)
            _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, 60.0, opened)
        s.commit()
    matches = find_similar_patterns(sessions, "AAPL", (95.0, 105.0), (True, True), (55.0, 65.0), now=now)
    assert len(matches) == MAX_MATCHES  # the sample is capped so one boom period cannot dominate
    assert all(isinstance(m, PatternMatch) and m.pattern_id.startswith("P>MA50") for m in matches)


def test_analyze_pattern_maps_matches_into_stats():
    matches = [
        PatternMatch("P1", "AAPL", now := datetime(2026, 9, 1, tzinfo=timezone.utc), 100.0, 108.0, 4, 8.0, 0.5),
        PatternMatch("P1", "AAPL", now, 100.0, 96.0, 6, -4.0, 1.0),
    ]
    st = analyze_pattern(matches)
    assert st is not None and st.sample_size == 2 and st.win_rate == pytest.approx(0.5)
    assert st.profit_factor == pytest.approx(2.0) and st.best_holding_days == 6 and st.worst_holding_days == 4


# ------------------------------------------------------------------ phase 4: monotonic clamp + humility
def test_reflection_never_lets_a_stage_raise_conviction():
    agent = TechnicalAgent(llm=None)
    base = SimpleNamespace(confidence=0.75)
    rising = ReflectionChain(action="BUY", confidence=0.9,
                             step2_reflection=ReflectionStage(conviction=0.9, reason="overconfident"),
                             step3_fundamental=ReflectionStage(conviction=0.85, reason="sure"),
                             step4_macro=ReflectionStage(conviction=0.8, reason="uptrend"),
                             step5_integration=ReflectionStage(conviction=0.8, reason="still sure"),
                             step6_risk=ReflectionStage(conviction=0.8, reason="very sure"),
                             biggest_risk="x", what_proves_us_wrong="y", bias_check=[])
    steps, final = agent._apply_reflection(base, rising)
    # Every later stage that tried to go above 0.75 is pinned to the previous conviction (never up).
    assert [c for _, c, _ in steps] == pytest.approx([0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
    assert final == pytest.approx(0.65)  # 0.75 minus the 0.10 humility, even though the model wanted 0.8


def test_reflection_keeps_model_drops_and_applies_humility_once():
    agent = TechnicalAgent(llm=None)
    falling = ReflectionChain(action="SELL", confidence=0.3,
                              step2_reflection=ReflectionStage(conviction=0.7, reason="r2"),
                              step3_fundamental=ReflectionStage(conviction=0.6, reason="r3"),
                              step4_macro=ReflectionStage(conviction=0.5, reason="r4"),
                              step5_integration=ReflectionStage(conviction=0.5, reason="r5"),
                              step6_risk=ReflectionStage(conviction=0.4, reason="r6"),
                              biggest_risk="x", what_proves_us_wrong="y", bias_check=[])
    steps, final = agent._apply_reflection(SimpleNamespace(confidence=0.75), falling)
    assert [c for _, c, _ in steps] == pytest.approx([0.75, 0.7, 0.6, 0.5, 0.5, 0.4])
    assert final == pytest.approx(0.3)  # 0.4 - 0.10, exactly the one humility discount


# ------------------------------------------------------------------ cross: setting + database indexes
def test_llm_max_adjust_default_env_and_validation(monkeypatch):
    monkeypatch.delenv("LLM_MAX_ADJUST", raising=False)
    assert load_settings().llm_max_adjust == pytest.approx(0.15)
    monkeypatch.setenv("LLM_MAX_ADJUST", "0.25")
    assert load_settings().llm_max_adjust == pytest.approx(0.25)
    monkeypatch.setenv("LLM_MAX_ADJUST", "1.5")
    with pytest.raises(ValueError):
        load_settings()
    assert TechnicalAgent(llm=None).max_adjustment == pytest.approx(0.15)
    assert TechnicalAgent(llm=None, max_adjustment=0.05).max_adjustment == pytest.approx(0.05)


def _index_names(path):
    con = sqlite3.connect(path)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    return names


def test_indexes_are_created_for_fresh_and_pre_existing_databases(tmp_path):
    fresh = tmp_path / "fresh.db"
    make_session_factory(f"sqlite:///{fresh}")
    for idx in ("ix_decisions_confluence", "ix_decisions_pattern", "ix_decisions_raw_model"):
        assert idx in _index_names(fresh)

    old = tmp_path / "old.db"
    con = sqlite3.connect(old)
    con.execute("""CREATE TABLE decisions (id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), price FLOAT,
        technical_action VARCHAR(8) NOT NULL, technical_confidence FLOAT NOT NULL, sentiment_action VARCHAR(8),
        sentiment_confidence FLOAT, final_action VARCHAR(8), final_confidence FLOAT, reasoning TEXT,
        risk_approved BOOLEAN, risk_quantity INTEGER, risk_reason TEXT, confluence_score FLOAT)""")
    con.commit()
    con.close()
    make_session_factory(f"sqlite:///{old}")  # upgrades the table and must add the indexes too
    assert "ix_decisions_confluence" in _index_names(old)
    assert "ix_decisions_pattern" in _index_names(old)
    assert "ix_decisions_raw_model" in _index_names(old)


# ------------------------------------------------------------------ AI accountability: the mechanical baseline + who answered
def test_mechanical_action_captures_the_filter_rule():
    assert mechanical_action(SNAP) == "BUY"  # price > MA50 > MA200, RSI 60
    assert mechanical_action(SNAP_RSI45) == "BUY"  # sub-70 RSI is still a BUY
    assert mechanical_action(DOWN) == "HOLD"  # trend broken
    overbought = Snapshot("AAPL", 100.0, 98.0, 95.0, 72.0, 1000)
    assert mechanical_action(overbought) == "HOLD"
    assert mechanical_baseline(SNAP).startswith("BUY (") and "filter" in mechanical_baseline(SNAP)
    assert mechanical_baseline(DOWN).startswith("HOLD (")


def test_cot_prompt_declares_mechanical_baseline_and_rule_check():
    p = TechnicalAgent.build_prompt(SNAP)
    assert "MECHANICAL BASELINE" in p and mechanical_baseline(SNAP)[:3] in p
    assert "STEP 6 (rule check)" in p and "rule_alignment" in p and "falsification" in p
    assert "deliberately overriding" in p and "MECHANICAL BASELINE (the deterministic" in p


def test_advanced_signal_carries_alignment_falsification_and_answering_model():
    body = json.loads(COT_GOOD)
    body.update({"rule_alignment": "deviate",
                 "falsification": "a close below MA50 would invalidate the trend"})
    fake = FakeOpenAI({"a": json.dumps(body)})
    llm = LLMClient(["a"], client=fake)
    advanced = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    assert advanced.rule_alignment == "deviate"
    assert advanced.raw_model == "a" and llm.last_model == "a"
    sig = advanced.to_signal()
    assert sig.details["rule_alignment"] == "deviate"
    assert sig.details["falsification"].startswith("a close below")
    assert sig.details["raw_model"] == "a"


def test_old_payloads_without_the_new_fields_still_parse_and_are_stamped():
    llm = LLMClient(["a"], client=FakeOpenAI({"a": COT_GOOD}))  # COT_GOOD predates rule_alignment/falsification
    advanced = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    assert advanced.rule_alignment is None and advanced.falsification is None
    assert advanced.raw_model == "a" and llm.last_model == "a"


def test_raw_model_survives_the_cache_hit():
    fake = FakeOpenAI({"a": COT_GOOD})
    llm = LLMClient(["a"], client=fake, cache_ttl_seconds=3600)
    first = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    second = llm.structured_call("p", AdvancedSignal, COT_SCHEMA)
    assert first is second and len(fake.calls) == 1
    assert second.raw_model == "a" and llm.last_model == "a"  # cache restores the original answering model


def test_pipeline_persists_raw_model_rule_alignment_and_falsification(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'a.db'}")
    sig = Signal(action="BUY", confidence=0.6, reasoning="r", details={
        "raw_model": "deepseek/deepseek-r1", "rule_alignment": "deviate",
        "falsification": "a close below 98 breaks MA50 support",
        "edge_confidence": 0.5, "confluence_score": 7, "risks": ["x"],
        "reasoning_chain": {"trend": "t", "overbought": "o", "volume": "v"},
    })
    pipe = TradingPipeline(
        Settings(watchlist=("AAPL",), dry_run=True),
        [StubAgent(lambda s: sig, "technical", LEAD)],
        FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
        lambda sym: SNAP, lambda sym: [],
    )
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
    assert row.raw_model == "deepseek/deepseek-r1"
    assert row.rule_alignment == "deviate"
    assert row.falsification == "a close below 98 breaks MA50 support"


# ------------------------------------------------------------------ version stamps (P0: attribution without guessing)
def test_version_columns_are_added_to_pre_existing_databases(tmp_path):
    old = tmp_path / "versioned.db"
    con = sqlite3.connect(old)
    con.execute("""CREATE TABLE decisions (id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), price FLOAT,
        final_action VARCHAR(8), final_confidence FLOAT, reasoning TEXT, risk_approved BOOLEAN,
        risk_quantity INTEGER, risk_reason TEXT, confluence_score FLOAT)""")
    con.commit()
    con.close()
    make_session_factory(f"sqlite:///{old}")
    con = sqlite3.connect(old)
    columns = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    con.close()
    assert {"prompt_version", "feature_version", "strategy_version", "universe_size"} <= columns


def test_pipeline_stamps_versions_and_replay_exposes_them(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'v.db'}")
    sig = Signal(action="HOLD", confidence=0.6, reasoning="r")
    pipe = TradingPipeline(
        Settings(watchlist=("AAPL",), dry_run=True),
        [StubAgent(lambda s: sig, "technical", LEAD)],
        FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
        lambda sym: SNAP, lambda sym: [],
    )
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
    assert row.prompt_version == VERSIONS["prompt"]
    assert row.feature_version == VERSIONS["feature"]
    assert row.strategy_version == VERSIONS["strategy"]
    assert replay(row).versions == {"strategy": VERSIONS["strategy"], "prompt": VERSIONS["prompt"],
                                    "feature": VERSIONS["feature"]}


def test_pipeline_stamps_the_per_decision_universe_size(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'u.db'}")
    sig = Signal(action="HOLD", confidence=0.6, reasoning="r")
    pipe = TradingPipeline(
        Settings(watchlist=("AAPL", "MSFT"), dry_run=True),
        [StubAgent(lambda s: sig, "technical", LEAD)],
        FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
        lambda sym: SNAP, lambda sym: [],
    )
    pipe.run_once()
    with sessions() as s:
        rows = s.scalars(select(DecisionRecord)).all()
    assert [r.universe_size for r in rows] == [2, 2]  # one snapshot per decision, taken at decision time


# ------------------------------------------------------------------ P0: the survivorship-free research universe (A1c)
def test_point_in_time_masks_a_dropped_member_even_while_its_price_data_continues():
    import pandas as pd

    from src.research.data import point_in_time

    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    close = pd.DataFrame({"A": [100 + i for i in range(8)],
                          "B": [200 + 2 * i for i in range(8)]}, index=dates)
    volume = close.copy()
    close_pit, volume_pit = point_in_time(close, volume,
                                          membership={"2024-01-01": ["A", "B"], "2024-01-05": ["B"]})
    assert close_pit.loc[: "2024-01-04", "A"].notna().all()  # A was a member at the start
    assert close_pit.loc["2024-01-05":, "A"].isna().all()  # and is masked as soon as it leaves, no matter the price data
    assert close_pit["B"].notna().all()


def test_point_in_time_earns_nothing_for_a_dropped_member_after_the_drop():
    import pandas as pd

    from src.research import engine
    from src.research.data import point_in_time

    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    close = pd.DataFrame({"A": [100 + i for i in range(8)],
                          "STAY": [200 + i for i in range(8)]}, index=dates)
    volume = close.copy()
    close_pit, _ = point_in_time(close, volume,
                                 membership={"2024-01-01": ["A", "STAY"], "2024-01-05": ["STAY"]})
    weights = pd.DataFrame({"A": 0.5, "STAY": 0.5}, index=dates)
    pit = engine.portfolio_returns(close_pit, weights)  # NaN is the mask: the engine must not bridge or hold it
    plain = engine.portfolio_returns(close, weights)
    assert pit["exposure"].iloc[6] == pytest.approx(0.5)  # A is gone from the portfolio after the drop...
    assert plain["exposure"].iloc[6] == pytest.approx(1.0)  # ...while a survivorship-biased run still holds it
