"""P0 B1: the phase ablation over one window (DET / +LLM / +LEARNING / +CONTEXT / +REFLECTION / FULL).

The hole being closed: the review asked for a single-window comparison of each incremental capability end to end.
Tests here pin that the framework exists, that every phase feeds the *same* simulator, and that the one source of
leakage phase 2 could have - closed trades from after the analysis date - is filtered out of the lookups."""
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.history import history_stats
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory
from src.data.indicators import build_snapshot
from src.llm import LLMClient
from src.research.ablation import DET, PHASES, run_ablation


# ------------------------------------------------------------------ deterministic LLM stubs
class ScriptedClient:
    """Responds based on prompt content (order-independent), so phases with variable call counts stay aligned."""

    def __init__(self, router):
        self.router = router
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **kw):
        self.calls.append((model, kw))
        prompt = kw["messages"][0]["content"]
        behaviour = self.router(prompt)
        if isinstance(behaviour, Exception):
            raise behaviour
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=behaviour))])


def route(prompt, cot, learn=None, context=None, reflect=None):
    if reflect is not None and "self-critique" in prompt:
        return reflect
    if learn is not None and "similar setups" in prompt:
        return learn
    if context is not None and "market regime" in prompt:
        return context
    return cot


def llm(router):
    return LLMClient(["a"], client=ScriptedClient(router))


PLOT = {}
COT_GOOD = json.dumps({
    "action": "BUY", "confidence": 0.75, "edge_confidence": 0.6,
    "step1_trend": "Uptrend; price above MA50 and MA50 above MA200.",
    "step2_overbought": "RSI 55; not overbought.",
    "step3_volume": "Volume 1.3x average; confirms the move.",
    "step4_confluence": 8,
    "step5_risks": ["Earnings next week", "Sector rotation"],
    "final_reasoning": "Clear uptrend, volume confirms, not overbought.",
    "rule_alignment": "agree",
    "falsification": "Close below MA200",
})
LEARN_HOLD = json.dumps({"action": "HOLD", "confidence": 0.3, "pattern_reliability": "no",
                         "reason_for_adjustment": "similar setups lost money on average",
                         "what_could_break": "an earnings gap against the pattern"})
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


@dataclass(frozen=True)
class FakeMarket:
    """A minimal phase-3 context object, so the ablation can distinguish the context phase without a live feed."""
    regime_label: str = "rangebound"
    index_above_ma200: bool = True
    index_price: float = 22000.0
    index_ma200: float = 21000.0
    index_rsi: float = 55.0
    vs_index_6m: float = 0.02
    index_ytd: float = 0.05
    vix: float = 14.0
    vix_percentile: float = 0.3
    sector_trend: str = "bullish"
    earnings_days_until: int = None
    beta_6m: float = 1.0
    correlation_with_index: float = 0.7

    def is_notable(self) -> bool:
        return True


def _trending_bars(days=280, seed=11):
    rng = np.random.default_rng(seed)
    close = 100.0 * (rng.normal(0.0009, 0.015, days) + 1).cumprod()  # gentle drift: stacked MAs, RSI around 55
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.DataFrame({"Open": [100.0] + list(close[:-1]), "High": close * 1.008, "Low": close * 0.992,
                         "Close": close, "Volume": 10_000}, index=idx)


BARS = {"A": _trending_bars()}
DATES = list(BARS["A"].index[200:])


def _stored_snap(snap):
    d = asdict(snap)
    d["bar_date"] = str(d["bar_date"])
    return json.dumps(d)


def _matched_history(sessions, snap, now, closed_after=False):
    """Three closed trades matching `snap`, either before `now` (learning should see them) or after."""
    stored = _stored_snap(snap)
    opened = now - timedelta(days=60)
    with sessions() as s:
        d = DecisionRecord(symbol="A", price=snap.price, final_action="BUY", final_confidence=0.9,
                           reasoning="r", risk_approved=True, risk_quantity=100, risk_reason="ok",
                           created_at=opened, snapshot_json=stored)
        s.add(d)
        s.flush()
        for i, day in enumerate((50, 45, 40)):
            s.add(PaperTradeRecord(symbol="A", qty=10, entry_price=snap.price, exit_price=snap.price * 1.01,
                                   opened_at=opened - timedelta(days=i),  # distinct opens, still within MIN_OPEN_GAP
                                   closed_at=now - timedelta(days=day) if not closed_after
                                   else now + timedelta(days=i + 1),
                                   reason="TARGET", fees=0.0, net_pnl=10.0))
        s.commit()


def test_ablation_runs_all_six_phases_and_differs_by_capability():
    def phased_llm(phase):
        if phase == "+LLM":
            return llm(lambda p: route(p, COT_GOOD))
        if phase == "+LEARNING":
            return llm(lambda p: route(p, COT_GOOD, learn=LEARN_HOLD))
        if phase == "+CONTEXT":
            return llm(lambda p: route(p, COT_GOOD, context=CONTEXT_OK))
        if phase == "+REFLECTION":
            return llm(lambda p: route(p, COT_GOOD, reflect=REFLECT_OK))
        if phase == "FULL":
            return llm(lambda p: route(p, COT_GOOD, learn=LEARN_HOLD, context=CONTEXT_OK, reflect=REFLECT_OK))
        raise AssertionError(phase)

    result = run_ablation("A", BARS, DATES, llm=phased_llm, market_fn=lambda _now: FakeMarket())
    assert set(result.reports) == set(PHASES)
    for phase in PHASES:
        r = result.reports[phase]
        assert r.skipped is False, phase
        assert r.decisions == len(DATES), phase
        assert r.degraded_share == 0.0, phase
        assert "total_return" in r.curve_metrics and "max_drawdown" in r.curve_metrics, phase
        assert "trades" in r.trade_metrics, phase
        assert r.action_counts, phase
    at = result.reports  # the base BUY carries 0.75; each capability must move confidence/action or nothing changed
    assert at["+LLM"].mean_confidence == pytest.approx(0.75)
    assert at["+LLM"].action_counts["BUY"] == len(DATES)
    assert at["+CONTEXT"].mean_confidence == pytest.approx(0.60)
    assert at["+REFLECTION"].mean_confidence == pytest.approx(0.75)  # reflection is a veto only: an upheld call keeps its confidence
    assert at["+LLM"].mean_confidence > at["+CONTEXT"].mean_confidence


def test_offline_ablation_reports_mechanical_only():
    result = run_ablation("A", BARS, DATES, llm=None)
    assert result.reports[DET].skipped is False
    for phase in PHASES:
        if phase != DET:
            assert result.reports[phase].skipped is True


def test_learning_phase_uses_point_in_time_closed_trades():
    sessions = make_session_factory("sqlite://")
    snap = build_snapshot("A", BARS["A"].loc[:DATES[3]])
    now = pd.Timestamp(DATES[3]).tz_localize("UTC").ceil("D").to_pydatetime()

    _matched_history(sessions, snap, now, closed_after=False)
    stats = history_stats(sessions, snap, now=now)
    assert stats is not None and stats.sample_size == 3

    future = make_session_factory("sqlite://")
    _matched_history(future, snap, now, closed_after=True)
    assert history_stats(future, snap, now=now) is None, "learning must not see trades closed after the analysis date"


def test_learning_phase_downgrades_when_history_says_losses():
    sessions = None  # wiring is proven via the history hook; PIT filtering of the DB hook is pinned separately

    def make_stats(_sample=12, win_rate=0.4):
        from src.agents.history import PatternStats
        return PatternStats(sample_size=_sample, win_rate=win_rate, avg_win_pct=8.0, avg_loss_pct=-4.0,
                            profit_factor=1.0, best_holding_days=30, worst_holding_days=3, confidence_in_pattern=0.4)

    def phased_llm(phase):
        if phase in ("+LEARNING", "FULL"):
            return llm(lambda p: route(p, COT_GOOD, learn=LEARN_HOLD, context=CONTEXT_OK, reflect=REFLECT_OK))
        return llm(lambda p: route(p, COT_GOOD))  # irrelevant phases get a plain client; DET never calls the model

    result = run_ablation("A", BARS, DATES, llm=phased_llm, sessions=sessions,
                          history_fn_factory=lambda _now: (lambda sn: make_stats()),
                          market_fn=lambda _now: FakeMarket())
    def fills(phase):  # live exits have no time limit, so an entry may still be open at the end of the window
        r = result.reports[phase]
        return r.trade_metrics.get("trades", 0) + r.open_at_end

    assert fills("+LLM") > 0
    assert fills("+LEARNING") == 0, "the learning veto must suppress the trade"

def test_learning_phase_gets_the_history_hook_when_only_sessions_is_given(monkeypatch):
    """Regression: the sessions-based history function was defined but never passed to the agent."""
    from src.research import ablation

    seen = {}

    class Spy:
        def __init__(self, llm, use_cot=True, history_fn=None, **flags):
            seen["history_fn"] = history_fn

        def analyze(self, ctx):
            from src.llm import Signal
            return Signal(action="HOLD", confidence=0.5, reasoning="x")

    monkeypatch.setattr(ablation, "TechnicalAgent", Spy)
    ablation._agent_signals("A", BARS, [DATES[0]], {"use_learning": True, "use_context": False, "use_reflect": False},
                            object(), object(), None, None)
    assert callable(seen["history_fn"])
