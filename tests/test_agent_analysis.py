"""The agent's inputs and review: what the model is shown (chart, 12-month momentum, company data, market situation), the
providers behind the company and market data, the veto-only critic, persistence and exact replay of the prompt."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select

from src.agents.base import LEAD, AgentContext
from src.agents.technical import TechnicalAgent
from src.agents.technical.prompts import company_text, market_text
from src.config import Settings, load_settings
from src.data.fundamentals import Fundamentals, FundamentalsProvider, Quarter, yahoo_fundamentals
from src.data.indicators import Snapshot, build_snapshot
from src.data.market_context import MarketContext, MarketContextProvider
from src.database import DecisionRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.llm import AgentSignal, LLMClient, LLMUnavailable
from src.ops.replay import replay
from src.pipeline import TradingPipeline
from tests.test_llm_and_pipeline import FakeBroker

SNAP = Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000, momentum_12_1=0.85)
MARKET = MarketContext(index_price=25_000.0, index_ma200=24_000.0, index_above_ma200=True, index_rsi=58.0,
                       index_mom_6m=0.06, vix=13.0, vix_percentile=0.3, vs_index_6m=0.40)
COMPANY = Fundamentals(sector="Industrials", industry="Machinery", market_cap=35_000_000_000.0, pe=19.5, debt_to_equity=0.18,
                       profit_margin=0.145, revenue_growth=0.27, earnings_growth=0.64, promoter_holding=0.66,
                       quarters=(Quarter("2026-06-30", 3_473_760_000.0, 566_090_000.0),
                                 Quarter("2026-03-31", 3_389_250_000.0, 511_450_000.0)))

COT_GOOD = json.dumps({
    "action": "BUY", "confidence": 0.75, "edge_confidence": 0.6,
    "step1_trend": "Uptrend; price above MA50 and MA50 above MA200.",
    "step2_overbought": "RSI 55; not overbought.",
    "step3_volume": "Volume 1.3x average; confirms the move.",
    "company_check": "Sales +27% and profit +64% year on year; low debt.",
    "market_check": "Nifty above its 200-day average; VIX calm.",
    "step4_confluence": 8, "step5_risks": ["Earnings miss", "Sector rotation", "Market pullback"],
    "final_reasoning": "Strong momentum backed by growing results.", "rule_alignment": "agree",
    "falsification": "a close below the 50-day average",
})
REFLECT_OK = json.dumps({"action": "BUY", "confidence": 0.5,
                         "step2_reflection": {"conviction": 0.7, "reason": "the trend may be extended"},
                         "step3_fundamental": {"conviction": 0.7, "reason": "results support the call"},
                         "step4_macro": {"conviction": 0.65, "reason": "a calm market"},
                         "step5_integration": {"conviction": 0.6, "reason": "the case survives"},
                         "step6_risk": {"conviction": 0.6, "reason": "the biggest risk is a results miss"},
                         "biggest_risk": "a gap-down on results",
                         "what_proves_us_wrong": "a weekly close below the 200-day average",
                         "bias_check": ["confirmation bias", "anchoring to the trend"]})
GOOD = json.dumps({"action": "BUY", "confidence": 0.8, "reasoning": "uptrend"})


class ScriptedClient:
    """One response per call, in order; records the calls. A list entry may be an exception."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **kw):
        self.calls.append((model, kw))
        behaviour = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=behaviour))])


def llm(script):
    return LLMClient(["a"], client=ScriptedClient(script))


def context(market=None, company=None):
    return AgentContext("AAPL", SNAP, market=lambda: market, company=lambda: company)


def first_prompt(agent):
    return agent.llm.client.calls[0][1]["messages"][0]["content"]


# ------------------------------------------------------------------ what the model is shown
def test_the_prompt_shows_why_the_stock_was_picked_the_company_and_the_market():
    agent = TechnicalAgent(llm([COT_GOOD]), use_reflect=False)
    agent.analyze(context(MARKET, COMPANY))
    prompt = first_prompt(agent)
    assert "12-month momentum" in prompt and "+85%" in prompt
    assert "sales +27.0%, profit +64.0%" in prompt and "debt/equity 0.18" in prompt and "promoters/insiders hold 66.0%" in prompt
    assert "2026-06 347.4 / 56.6" in prompt  # newest quarter, Rs crore
    assert "Nifty 50 25,000, above its 200-day average 24,000" in prompt and "India VIX 13.0" in prompt
    assert "a weak market on its own" in prompt  # the market is not a veto: the 9-year test said so


def test_missing_company_or_market_data_is_said_plainly_never_invented():
    agent = TechnicalAgent(llm([COT_GOOD]), use_reflect=False)
    agent.analyze(context())
    prompt = first_prompt(agent)
    assert "COMPANY: no company data available" in prompt and "MARKET: no market data available" in prompt


def test_a_failing_data_source_means_no_data_not_a_failed_analysis():
    def broken():
        raise ConnectionError("yahoo down")

    agent = TechnicalAgent(llm([COT_GOOD]), use_reflect=False)
    signal = agent.analyze(AgentContext("AAPL", SNAP, market=broken, company=broken))
    assert signal.action == "BUY" and signal.degraded is False
    assert "no company data available" in first_prompt(agent)


def test_the_company_and_market_checks_are_recorded_with_the_data_they_were_based_on():
    signal = TechnicalAgent(llm([COT_GOOD]), use_reflect=False).analyze(context(MARKET, COMPANY))
    chain = signal.details.reasoning_chain
    assert chain.company.startswith("Sales +27%") and chain.market.startswith("Nifty above")
    assert signal.details.company_json["revenue_growth"] == pytest.approx(0.27)
    assert signal.details.market_context_json["index_above_ma200"] is True


def test_the_text_helpers_accept_the_stored_dict_as_well_as_the_live_object():
    import dataclasses

    assert company_text(COMPANY) == company_text(json.loads(json.dumps(dataclasses.asdict(COMPANY))))
    assert market_text(MARKET) == market_text(dataclasses.asdict(MARKET))


# ------------------------------------------------------------------ review (veto-only critic)
def test_reflection_is_a_veto_only_and_sees_the_same_company_and_market_data():
    agent = TechnicalAgent(llm([COT_GOOD, REFLECT_OK]))
    s = agent.analyze(context(MARKET, COMPANY))
    assert s.action == "BUY" and s.confidence == pytest.approx(0.75)  # upheld: confidence untouched
    reflection = s.details.reflection
    assert [a.conviction for a in reflection.conviction_adjustments] == pytest.approx([0.75, 0.7, 0.7, 0.65, 0.6, 0.6])
    assert reflection.critic_confidence == pytest.approx(0.5) and reflection.upheld is True  # 0.6 - 0.10, audit only
    assert s.details.adjustment_reason == "reflection upheld the call unchanged"
    critic_prompt = agent.llm.client.calls[1][1]["messages"][0]["content"]
    assert "sales +27.0%" in critic_prompt and "above its 200-day average" in critic_prompt


def test_reflection_can_veto_and_is_skipped_for_a_hold():
    veto = json.loads(REFLECT_OK)
    veto["action"] = "HOLD"
    s = TechnicalAgent(llm([COT_GOOD, json.dumps(veto)])).analyze(context())
    assert s.action == "HOLD" and s.details.reflection.upheld is False
    hold = json.loads(COT_GOOD)
    hold["action"] = "HOLD"
    agent = TechnicalAgent(llm([json.dumps(hold)]))
    assert agent.analyze(context()).action == "HOLD" and len(agent.llm.client.calls) == 1


def test_a_first_call_failure_degrades_to_hold_and_nothing_else_runs():
    client = ScriptedClient([LLMUnavailable("all down")])
    s = TechnicalAgent(LLMClient(["a"], client=client)).analyze(context())
    assert s.action == "HOLD" and s.degraded is True and s.details is None
    assert len(client.calls) == 2  # the full analysis and its compact fallback


def test_simple_mode_signal_stays_without_details():
    s = TechnicalAgent(llm([GOOD]), use_cot=False, use_reflect=False).analyze(context(MARKET, COMPANY))
    assert s.action == "BUY" and s.details is None


# ------------------------------------------------------------------ company data
def fake_ticker(info, statement):
    return lambda name: SimpleNamespace(info=info, quarterly_income_stmt=statement)


def test_yahoo_fundamentals_reads_ratios_and_the_newest_quarters_first():
    cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")]
    statement = pd.DataFrame([[1.0e9, 3.0e9, 2.0e9], [1.0e8, 3.0e8, float("nan")]],
                             index=["Total Revenue", "Net Income"], columns=cols)
    info = {"sector": "Industrials", "trailingPE": 19.5, "debtToEquity": 17.8, "revenueGrowth": 0.27,
            "heldPercentInsiders": 0.66}
    f = yahoo_fundamentals("X", ticker=fake_ticker(info, statement))
    assert f.pe == 19.5 and f.debt_to_equity == pytest.approx(0.178) and f.promoter_holding == 0.66
    assert [q.period_end for q in f.quarters] == ["2026-06-30", "2026-03-31", "2025-12-31"]
    assert f.quarters[1].net_income is None  # a missing figure stays missing, never zero


def test_a_company_yahoo_knows_nothing_about_is_none():
    assert yahoo_fundamentals("X", ticker=fake_ticker({}, pd.DataFrame())) is None


def test_the_provider_reads_each_company_once_and_retries_a_failure_later():
    calls, now = [], [0.0]

    def fetch(symbol):
        calls.append(symbol)
        if symbol == "BAD":
            raise ConnectionError("down")
        return COMPANY

    provider = FundamentalsProvider(fetch, clock=lambda: now[0])
    assert provider("A") is COMPANY and provider("A") is COMPANY and provider("BAD") is None and provider("BAD") is None
    assert calls == ["A", "BAD"]
    now[0] = 1801.0  # half an hour later the failure is retried; the good answer is still fresh
    provider("A"), provider("BAD")
    assert calls == ["A", "BAD", "BAD"]


# ------------------------------------------------------------------ market context provider
def _series(n_points, start_value, step):
    idx = pd.date_range(end=date(2026, 9, 18), periods=n_points, freq="B")
    return pd.Series([start_value + i * step for i in range(n_points)], index=idx)


NIFTY = _series(400, 100.0, 0.5)
VIX = _series(252, 12.0, 0.02)


def test_provider_builds_context_and_fetches_index_once(tmp_path):
    calls = []

    def download(tickers, start, end):
        calls.append(tickers[0])
        return pd.DataFrame({"Close": NIFTY if tickers[0] == "^NSEI" else VIX})

    p = MarketContextProvider(cache_dir=tmp_path, download=download, today=lambda: date(2026, 9, 21))
    snap = Snapshot("AAPL", 100.0, 98.0, 95.0, 60.0, 1000, momentum_6m=0.10)
    m1, m2 = p.context("AAPL", snap), p.context("TCS", snap)
    assert m1 is not None and m2 == m1
    assert m1.index_above_ma200 is True and m1.index_rsi >= 90
    assert m1.vix == pytest.approx(float(VIX.iloc[-1])) and m1.vix_percentile == pytest.approx(1.0)
    assert m1.vs_index_6m == pytest.approx(0.10 - (float(NIFTY.iloc[-1]) / float(NIFTY.iloc[-127]) - 1))
    assert calls.count("^NSEI") == 1 and calls.count("^INDIAVIX") == 1


def test_provider_returns_none_when_index_unavailable(tmp_path):
    def download(tickers, start, end):
        raise RuntimeError("no index data")

    p = MarketContextProvider(cache_dir=tmp_path, download=download, today=lambda: date(2026, 9, 21))
    assert p.context("AAPL", SNAP) is None


def test_snapshot_computes_six_month_and_twelve_minus_one_month_momentum():
    closes = [float(i) for i in range(1, 301)]
    s = build_snapshot("X", pd.DataFrame({"Close": closes, "Volume": [1000.0] * 300}))
    assert s.momentum_6m == pytest.approx(300.0 / 174.0 - 1)
    assert s.momentum_12_1 == pytest.approx(closes[-22] / closes[-253] - 1)  # the scanner's formula
    short = build_snapshot("X", pd.DataFrame({"Close": closes[:250], "Volume": [1000.0] * 250}))
    assert short.momentum_12_1 is None  # under a year of bars: not guessed


# ------------------------------------------------------------------ pipeline wiring, persistence, replay
def test_the_pipeline_hands_agents_the_company_and_market_data_and_replay_rebuilds_the_exact_prompt(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    agent = TechnicalAgent(llm([COT_GOOD]), use_reflect=False)
    pipe = TradingPipeline(Settings(watchlist=("AAPL",), dry_run=True), [agent],
                           FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0)), sessions, lambda sym: SNAP,
                           market_context_fn=lambda snap: MARKET, fundamentals_fn=lambda symbol: COMPANY)
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(DecisionRecord))
        assert row.base_confidence == pytest.approx(0.75) and json.loads(row.market_context_json)["vix"] == 13.0
        assert json.loads(row.reasoning_chain_json)["company"].startswith("Sales +27%")
        assert replay(row).prompt == first_prompt(agent)


def test_reflect_and_cot_flags_default_on_and_toggle(monkeypatch):
    for name in ("LLM_COT", "LLM_REFLECT"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.llm_cot is True and s.llm_reflect is True
    monkeypatch.setenv("LLM_REFLECT", "false")
    assert load_settings().llm_reflect is False


def test_a_stored_signal_from_before_this_change_still_loads():
    old = {"action": "BUY", "confidence": 0.7, "reasoning": "r",
           "details": {"tier": "cot", "pattern_id": "P>MA50+MA50>MA200|RSI60",
                       "learning": {"adjusted_action": "BUY"}, "context": {"regime": "bull"},
                       "reasoning_chain": {"trend": "t", "overbought": "o", "volume": "v"}}}
    signal = AgentSignal(**old)
    assert signal.details.reasoning_chain.company is None and signal.details.tier == "cot"


def test_the_pipeline_lead_agent_role_is_unchanged():
    assert TechnicalAgent.role == LEAD and TechnicalAgent.name == "technical"


def test_the_market_data_cache_goes_where_market_cache_dir_says(tmp_path, monkeypatch):
    """The bot user cannot write to /app in the container; the cache must follow MARKET_CACHE_DIR (the /data volume)."""
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path / "cache"))

    def download(tickers, start, end):
        return pd.DataFrame({"Close": NIFTY if tickers[0] == "^NSEI" else VIX})

    p = MarketContextProvider(download=download, today=lambda: date(2026, 9, 21))
    assert p.context("AAPL", SNAP) is not None
    assert (tmp_path / "cache").is_dir() and any((tmp_path / "cache").iterdir())


def test_both_docker_images_point_the_market_cache_at_the_writable_data_volume():
    from pathlib import Path

    for name in ("Dockerfile", "Dockerfile.aws"):
        assert "ENV MARKET_CACHE_DIR=/data/cache" in Path(name).read_text(encoding="utf-8"), name
