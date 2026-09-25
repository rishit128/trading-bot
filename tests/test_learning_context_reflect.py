"""Phases 2-4: the history lookup (decision memory), the market-context provider, the lead agent's refinement
pipeline (learn / context / reflect), the settings flags, and that every phase is persisted with the decision."""
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select

from src.agents.technical import TechnicalAgent
from src.agents.base import ADVISOR, LEAD
from src.agents.history import PatternStats, history_stats, pattern_stats
from src.config import Settings, load_settings
from src.data.indicators import Snapshot, build_snapshot
from src.data.market_context import MarketContext, MarketContextProvider
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.engine.agent_signal import ContextDetails, LearningDetails, ReasoningChain, SignalDetails
from src.llm import LLMClient, AgentSignal
from src.pipeline import TradingPipeline
from tests.test_llm_and_pipeline import FakeBroker, StubAgent

SNAP = Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000)


def SNAP_with_momentum():
    return Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000, momentum_6m=0.10)


def agent_context(market=None):
    from src.agents.base import AgentContext

    return AgentContext("AAPL", SNAP, market=lambda: market)


COT_GOOD = json.dumps({
    "action": "BUY", "confidence": 0.75, "edge_confidence": 0.6,
    "step1_trend": "Uptrend; price above MA50 and MA50 above MA200.",
    "step2_overbought": "RSI 55; not overbought.",
    "step3_volume": "Volume 1.3x average; confirms the move.",
    "step4_confluence": 8,
    "step5_risks": ["Earnings next week", "Sector rotation"],
    "final_reasoning": "Clear uptrend, volume confirms, not overbought.",
})
LEARN_HOLD = json.dumps({"action": "HOLD", "confidence": 0.3, "pattern_reliability": "no",
                         "reason_for_adjustment": "similar setups lost money on average",
                         "what_could_break": "an earnings gap against the pattern"})
LEARN_KEEP = json.dumps({"action": "BUY", "confidence": 0.7, "pattern_reliability": "maybe",
                         "reason_for_adjustment": "similar setups won slightly more than half the time",
                         "what_could_break": "a broad-market sell-off"})
CONTEXT_OK = json.dumps({"action": "BUY", "confidence": 0.6, "macro_support": "neutral",
                         "sector_support": "neutral", "earnings_risk": False, "diversification_score": 5,
                         "context_reasoning": "regime is stressed but the call survives", "key_context_risks": ["correction"]})
REFLECT_OK = json.dumps({"action": "BUY", "confidence": 0.5,
                         "step2_reflection": {"conviction": 0.7, "reason": "the trend may be extended"},
                         "step3_fundamental": {"conviction": 0.7, "reason": "no fundamental data available; unchanged"},
                         "step4_macro": {"conviction": 0.65, "reason": "a stressed regime warrants a haircut"},
                         "step5_integration": {"conviction": 0.6, "reason": "the case survives but is thinner"},
                         "step6_risk": {"conviction": 0.6, "reason": "the case is decent but I cut confidence to 0.5"},
                         "biggest_risk": "a gap-down on earnings",
                         "what_proves_us_wrong": "a weekly close below the 200-day average",
                         "bias_check": ["confirmation bias", "anchoring to the trend"]})
GOOD = json.dumps({"action": "BUY", "confidence": 0.8, "reasoning": "uptrend"})


class ScriptedClient:
    """One response per call, in order, for model 'a'; records the calls. A list entry may be an exception."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **kw):
        self.calls.append((model, kw))
        position = sum(1 for m, _ in self.calls if m == model)
        behaviour = self.script[min(position - 1, len(self.script) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=behaviour))])


def llm(script, **kw):
    return LLMClient(["a"], client=ScriptedClient(script), **kw)


def make_stats(sample=12, win_rate=0.5):
    return PatternStats(sample_size=sample, win_rate=win_rate, avg_win_pct=8.0, avg_loss_pct=-4.0,
                        profit_factor=1.0, best_holding_days=30, worst_holding_days=3, confidence_in_pattern=0.4)


# ------------------------------------------------------------------ pattern stats / history lookup
def test_pattern_stats_math():
    st = pattern_stats([(10.0, 3), (-5.0, 7), (5.0, 2)])
    assert st.sample_size == 3 and st.win_rate == pytest.approx(2 / 3)
    assert st.avg_win_pct == pytest.approx(7.5) and st.avg_loss_pct == pytest.approx(-5.0)
    assert st.profit_factor == pytest.approx(1.5)  # average winner / average loser (7.5 / 5.0), per the review doc
    assert st.best_holding_days == 7 and st.worst_holding_days == 2
    assert st.confidence_in_pattern < 1.0


def test_pattern_stats_handles_no_losses_and_none():
    assert pattern_stats([(10.0, 3)]).profit_factor is None
    assert pattern_stats([]) is None


def _add_decision(s, sym, action, price, ma50, ma200, rsi, when):
    snap = {"symbol": sym, "price": price, "ma50": ma50, "ma200": ma200, "rsi": rsi, "volume": 1000,
            "bar_date": str(when.date())}
    s.add(DecisionRecord(symbol=sym, price=price, technical_action="BUY", technical_confidence=0.7,
                         final_action=action, final_confidence=0.7, reasoning="r",
                         risk_approved=True, risk_quantity=10, risk_reason="x",
                         signals_json='{"technical": null}', snapshot_json=json.dumps(snap), created_at=when))


def _add_trade(s, sym, opened, held_days, entry, exit):
    s.add(PaperTradeRecord(symbol=sym, qty=10, entry_price=entry, exit_price=exit, opened_at=opened,
                           closed_at=opened + timedelta(days=held_days), reason="TREND_EXIT",
                           fees=0.0, net_pnl=(exit - entry) * 10))


def test_history_stats_matches_similar_closed_trade(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    with sessions() as s:
        _add_trade(s, "AAPL", now - timedelta(days=30), held_days=12, entry=100.0, exit=105.0)
        _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, 60.0, now - timedelta(days=30))
        s.commit()
    st = history_stats(sessions, SNAP, now=now)
    assert st is not None and st.sample_size == 1 and st.win_rate == 1.0
    assert st.avg_win_pct == pytest.approx(5.0) and st.profit_factor is None  # return (105/100-1)*100


def test_history_stats_ignores_dissimilar_setups(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    opened = now - timedelta(days=30)
    with sessions() as s:
        _add_trade(s, "AAPL", opened, held_days=12, entry=100.0, exit=90.0)
        _add_decision(s, "AAPL", "BUY", 95.0, 100.0, 95.0, 30.0, opened)  # broken trend shape + RSI 30 away
        s.commit()
    assert history_stats(sessions, SNAP, now=now) is None


def test_history_stats_ranks_closest_rsi_and_windows_older_than_lookback(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    # RSI 42 is now outside the ±5-point tolerance, so it no longer counts as "similar".
    with sessions() as s:
        for i, (rsi, held, exit_price) in enumerate([(60, 10, 103.0), (42, 20, 101.0), (60, 5, 95.0)]):
            opened = now - timedelta(days=40 + i * 3)
            _add_trade(s, "AAPL", opened, held_days=held, entry=100.0, exit=exit_price)
            _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, rsi, opened)
        stale = now - timedelta(days=400)  # outside the 365-day lookback
        _add_trade(s, "AAPL", stale, held_days=5, entry=100.0, exit=110.0)
        _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, 60.0, stale)
        s.commit()
    st = history_stats(sessions, SNAP, now=now)
    assert st is not None and st.sample_size == 2  # the 400-day-old trade and the RSI-42 one are ignored
    assert st.win_rate == pytest.approx(0.5)  # the 95 exit loses, the 103 exit wins


def test_history_stats_none_without_closed_trades(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    with sessions() as s:
        _add_decision(s, "AAPL", "BUY", 100.0, 98.0, 95.0, 60.0, now - timedelta(days=10))
        s.commit()
    assert history_stats(sessions, SNAP, now=now) is None


# ------------------------------------------------------------------ market context provider
def _series(n_points, start_value, step):
    idx = pd.date_range(end=date(2026, 9, 18), periods=n_points, freq="B")
    return pd.Series([start_value + i * step for i in range(n_points)], index=idx)


NIFTY = _series(400, 100.0, 0.5)
VIX = _series(252, 12.0, 0.02)


def market_provider(tmp_path, download):
    return MarketContextProvider(cache_dir=tmp_path, download=download, today=lambda: date(2026, 9, 21))


def test_market_regime_labels_and_notable():
    calm = MarketContext(index_above_ma200=True, index_rsi=60.0, vix_percentile=0.3)
    assert calm.regime_label == "bull" and calm.is_notable() is False
    stress = MarketContext(index_above_ma200=True, vix_percentile=0.9)
    assert stress.regime_label == "risk_off" and stress.is_notable() is True
    bear = MarketContext(index_above_ma200=False, vix_percentile=0.1)
    assert bear.regime_label == "risk_off" and bear.is_notable() is True
    euphoric = MarketContext(index_above_ma200=True, index_rsi=80.0, vix_percentile=0.2)
    assert euphoric.regime_label == "bull" and euphoric.is_notable() is True


def test_provider_builds_context_and_fetches_index_once(tmp_path):
    calls = []

    def download(tickers, start, end):  # matches the yfinance signature used by the data layer
        calls.append(tickers[0])
        if tickers[0] == "^NSEI":
            return pd.DataFrame({"Close": NIFTY})
        return pd.DataFrame({"Close": VIX})

    p = market_provider(tmp_path, download)
    mom = SNAP_with_momentum()
    m1 = p.context("AAPL", mom)
    m2 = p.context("TCS", mom)
    assert m1 is not None and m2 == m1
    assert m1.index_above_ma200 is True and m1.index_rsi >= 90  # monotonic series, RSI pegged high
    assert m1.vix == pytest.approx(float(VIX.iloc[-1])) and m1.vix_percentile == pytest.approx(1.0)
    assert m1.vs_index_6m == pytest.approx(mom.momentum_6m - (float(NIFTY.iloc[-1]) / float(NIFTY.iloc[-127]) - 1))
    # the index series is memoised: one download per ticker regardless of how many stocks ask for context
    assert calls.count("^NSEI") == 1 and calls.count("^INDIAVIX") == 1


def test_provider_returns_none_when_index_unavailable(tmp_path):
    def download(tickers, start, end):
        raise RuntimeError("no index data")

    p = market_provider(tmp_path, download)
    assert p.context("AAPL", SNAP) is None


def test_snapshot_computes_momentum_and_replay_backfills_none():
    bars = pd.DataFrame({"Close": [float(i) for i in range(1, 251)], "Volume": [1000.0] * 250})
    s = build_snapshot("X", bars)
    assert s.momentum_6m == pytest.approx(250.0 / 124.0 - 1)
    older = Snapshot("X", 100.0, 98.0, 95.0, 60.0, 1000)  # pre-phase-2 rows: field defaults to None
    assert older.momentum_6m is None


# ------------------------------------------------------------------ agent refinement pipeline
def test_learning_phase_adjusts_and_records():
    agent = TechnicalAgent(llm([COT_GOOD, LEARN_HOLD]), use_learning=True, use_context=False, use_reflect=False,
                           history_fn=lambda s: make_stats(12, 0.4))
    s = agent.analyze(agent_context())
    assert s.action == "HOLD" and s.confidence == pytest.approx(0.6, abs=1e-4)  # 0.3 request capped at -0.15
    assert s.degraded is False and s.reasoning.startswith("Clear uptrend")
    assert s.details.base_confidence == pytest.approx(0.75)
    learning = s.details.learning
    assert learning.sample_size == 12 and learning.win_rate == pytest.approx(0.4)
    assert learning.adjusted_action == "HOLD" and learning.adjusted_confidence == pytest.approx(0.6, abs=1e-4)
    assert learning.pattern_reliability == "no" and "lost money" in learning.reason
    assert s.details.context is None
    assert s.details.pattern_id.startswith("P>MA50+MA50>MA200|RSI")  # stamped from the snapshot
    assert s.details.adjusted_signal_confidence == pytest.approx(0.6, abs=1e-4)
    assert "lost money" in s.details.adjustment_reason


def test_learning_skipped_without_history_or_sample(tmp_path):
    agent = TechnicalAgent(llm([COT_GOOD]), use_learning=True, use_context=False, use_reflect=False,
                           history_fn=lambda s: None)
    s = agent.analyze(agent_context())
    assert s.action == "BUY" and s.details.base_confidence == pytest.approx(0.75)
    agent2 = TechnicalAgent(llm([COT_GOOD]), use_learning=True, use_context=False, use_reflect=False,
                            history_fn=lambda s: make_stats(2, 0.0), min_pattern_sample=3)
    s2 = agent2.analyze(agent_context())
    assert s2.action == "BUY" and s2.details.learning is None
    assert len(agent2.llm.client.calls) == 1  # a 2-trade record is not evidence, so no second call


def test_learning_failure_keeps_the_base_call():
    agent = TechnicalAgent(llm([COT_GOOD, "not json"]), use_learning=True, use_context=False, use_reflect=False,
                           history_fn=lambda s: make_stats(40, 0.2))
    s = agent.analyze(agent_context())
    assert s.action == "BUY" and s.confidence == pytest.approx(0.75) and s.degraded is False
    assert s.details.learning is None and s.details.base_confidence == pytest.approx(0.75)


def test_context_phase_fires_only_on_a_notable_regime():
    calm = MarketContext(index_above_ma200=True, index_rsi=58.0, vix=13.0, vix_percentile=0.3)
    agent = TechnicalAgent(llm([COT_GOOD]), use_learning=False, use_context=True, use_reflect=False)
    s = agent.analyze(agent_context(market=calm))
    assert s.action == "BUY" and len(agent.llm.client.calls) == 1  # calm regime: no extra call

    stressed = MarketContext(index_above_ma200=True, index_rsi=55.0, vix=28.0, vix_percentile=0.9)
    agent2 = TechnicalAgent(llm([COT_GOOD, CONTEXT_OK]), use_learning=False, use_context=True, use_reflect=False)
    s2 = agent2.analyze(agent_context(market=stressed))
    assert s2.action == "BUY" and s2.confidence == pytest.approx(0.6)
    assert s2.details.context.regime == "risk_off"
    assert s2.details.context.macro_support == "neutral"


def test_reflection_is_a_veto_only_and_skips_holds():
    agent = TechnicalAgent(llm([COT_GOOD, REFLECT_OK]), use_learning=False, use_context=False, use_reflect=True)
    s = agent.analyze(agent_context())
    assert s.action == "BUY" and s.confidence == pytest.approx(0.75)  # upheld: confidence untouched
    reflection = s.details.reflection
    assert reflection.final_action == "BUY" and reflection.biggest_risk == "a gap-down on earnings"
    assert reflection.what_proves_us_wrong == "a weekly close below the 200-day average"
    assert reflection.bias_check == ["confirmation bias", "anchoring to the trend"]
    adjustments = reflection.conviction_adjustments
    # Review doc 4.2: conviction only ever stays put or falls across the six stages, then humility applies.
    assert [a.stage for a in adjustments] == ["step1_technical", "step2_reflection", "step3_fundamental",
                                                 "step4_macro", "step5_integration", "step6_risk"]
    assert [a.conviction for a in adjustments] == pytest.approx([0.75, 0.7, 0.7, 0.65, 0.6, 0.6])
    assert reflection.critic_confidence == pytest.approx(0.5)  # 0.6 (step6) minus the 0.10 humility, audit only
    assert reflection.final_confidence == pytest.approx(0.75) and reflection.upheld is True
    assert reflection.humility_reduction == pytest.approx(0.10)
    assert s.reasoning.endswith("(critic conviction 0.50)]")

    agent2 = TechnicalAgent(llm([COT_GOOD, LEARN_HOLD]), use_learning=True, use_context=False, use_reflect=True,
                            history_fn=lambda s: make_stats(12, 0.3))
    s2 = agent2.analyze(agent_context())
    assert s2.action == "HOLD" and len(agent2.llm.client.calls) == 2  # learning already stood down: no reflection


def test_all_phases_chain_into_details_and_final():
    stressed = MarketContext(index_above_ma200=True, index_rsi=55.0, vix=28.0, vix_percentile=0.9)
    agent = TechnicalAgent(llm([COT_GOOD, LEARN_KEEP, CONTEXT_OK, REFLECT_OK]),
                           history_fn=lambda s: make_stats(12, 0.6))
    s = agent.analyze(agent_context(market=stressed))
    assert s.action == "BUY" and s.confidence == pytest.approx(0.6)  # context cut 0.75 to 0.60; reflection upheld it
    assert s.details.base_confidence == pytest.approx(0.75)
    assert s.details.learning.adjusted_action == "BUY"
    assert s.details.context.regime == "risk_off"
    assert s.details.reflection.final_confidence == pytest.approx(0.6)
    assert len(agent.llm.client.calls) == 4  # CoT + learning + context + reflection


def test_first_call_failure_still_degrades_to_hold_even_with_phases():
    from src.llm import LLMUnavailable

    client = ScriptedClient([LLMUnavailable("all down")])
    agent = TechnicalAgent(LLMClient(["a"], client=client), history_fn=lambda s: make_stats(50, 0.1))
    s = agent.analyze(agent_context(market=MarketContext(vix_percentile=0.95)))
    assert s.action == "HOLD" and s.degraded is True and s.details is None
    assert len(client.calls) == 2  # the chain-of-thought call and its compact fallback; no later phase is reached


def test_simple_mode_signal_stays_without_details_when_phases_disabled():
    agent = TechnicalAgent(llm([GOOD]), use_cot=False, use_learning=False, use_context=False, use_reflect=False)
    s = agent.analyze(agent_context())
    assert s.action == "BUY" and s.details is None


# ------------------------------------------------------------------ pipeline persistence + context wiring
def detailed_signal():
    return AgentSignal(action="BUY", confidence=0.5, reasoning="r", details=SignalDetails(
        base_confidence=0.75, edge_confidence=0.6, confluence_score=8, risks=["earnings"],
        reasoning_chain=ReasoningChain(trend="uptrend", overbought="no", volume="yes"),
        learning=LearningDetails(adjusted_action="BUY", adjusted_confidence=0.5, pattern_reliability="yes", sample_size=11,
                                 win_rate=0.45, avg_win_pct=6.0, avg_loss_pct=-3.0, profit_factor=2.0, best_holding_days=30,
                                 worst_holding_days=3, confidence_in_pattern=0.4, reason="held up", what_could_break="a gap"),
        context=ContextDetails(regime="risk_off", macro_support="no", sector_support="neutral", earnings_risk=False,
                               diversification_score=5, reasoning="index below its 200-day", risks=["broad sell-off"])))


def test_pipeline_persists_phase_columns(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    pipe = TradingPipeline(
        Settings(watchlist=("AAPL",), dry_run=True),
        [StubAgent(lambda s: detailed_signal(), "technical", LEAD),
         StubAgent(lambda sym, heads: None, "sentiment", ADVISOR)],
        FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
        lambda sym: SNAP, lambda sym: [],
    )
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
    assert row.base_confidence == pytest.approx(0.75)
    assert row.context_regime == "risk_off"
    assert row.historical_win_rate == pytest.approx(0.45) and row.historical_sample_size == 11


def test_pipeline_feeds_market_context_to_agents(tmp_path):
    seen = []

    class MarketAwareAgent:
        name, role = "technical", LEAD

        def analyze(self, ctx):
            seen.append(ctx.market())
            return AgentSignal(action="HOLD", confidence=0.5, reasoning="r")

    sessions = make_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    ag = MarketAwareAgent()
    pipe = TradingPipeline(
        Settings(watchlist=("AAPL",), dry_run=True),
        [ag], FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions,
        lambda sym: SNAP, lambda sym: [],
        market_context_fn=lambda snap: MarketContext(index_above_ma200=False),
    )
    pipe.run_once()
    assert seen == [MarketContext(index_above_ma200=False)]


# ------------------------------------------------------------------ settings flags
def test_phase_flags_default_on_and_env_toggle(monkeypatch):
    for name in ("LLM_LEARNING", "LLM_CONTEXT", "LLM_REFLECT"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.llm_learning is True and s.llm_context is True and s.llm_reflect is True
    monkeypatch.setenv("LLM_LEARNING", "false")
    monkeypatch.setenv("LLM_CONTEXT", "false")
    monkeypatch.setenv("LLM_REFLECT", "false")
    s = load_settings()
    assert s.llm_learning is False and s.llm_context is False and s.llm_reflect is False