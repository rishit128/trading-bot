import sqlite3
import threading
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import select

from src.agents.base import ADVISOR, LEAD
from src.data.india import IndiaClock, IntradayFeed
from src.data.market_data import fetch_snapshot
from src.database import DecisionRecord, make_session_factory
from src.engine.strategy import combine_signals
from src.llm import LLMClient, LLMUnavailable, AgentSignal
from tests.test_llm_and_pipeline import GOOD, FakeOpenAI, api_error

UTC = timezone.utc


def sig(action, conf=0.8, reasoning="r"):
    return AgentSignal(action=action, confidence=conf, reasoning=reasoning)


ROLES = {"technical": LEAD, "trend": LEAD, "sentiment": ADVISOR, "macro": ADVISOR, "fundamentals": ADVISOR}


# ---------------------------------------------------------------- combining any number of agents
def test_no_lead_agent_means_no_trade():
    d = combine_signals({"sentiment": sig("BUY")}, ROLES)
    assert d.action == "HOLD" and "no lead" in d.reasoning


def test_a_lead_that_produced_nothing_blocks_the_trade():
    assert combine_signals({"technical": None}, ROLES).action == "HOLD"


def test_multiple_leads_must_be_unanimous():
    assert combine_signals({"technical": sig("BUY"), "trend": sig("BUY")}, ROLES).action == "BUY"
    d = combine_signals({"technical": sig("BUY"), "trend": sig("SELL")}, ROLES)
    assert d.action == "HOLD" and d.confidence == 0.0 and "disagree" in d.reasoning
    assert combine_signals({"technical": sig("BUY"), "trend": sig("HOLD")}, ROLES).action == "HOLD"


def test_all_leads_hold_keeps_mean_confidence():
    d = combine_signals({"technical": sig("HOLD", 0.6), "trend": sig("HOLD", 0.8)}, ROLES)
    assert d.action == "HOLD" and d.confidence == pytest.approx(0.7)


def test_any_single_opposing_advisor_vetoes():
    signals = {"technical": sig("BUY"), "sentiment": sig("BUY"), "macro": sig("SELL"), "fundamentals": None}
    d = combine_signals(signals, ROLES)
    assert d.action == "HOLD" and d.confidence == 0.0 and "macro SELL" in d.reasoning


def test_agreeing_advisors_are_averaged_in_and_neutral_or_missing_ones_ignored():
    signals = {"technical": sig("BUY", 0.9), "sentiment": sig("BUY", 0.6), "macro": sig("HOLD", 0.1), "fundamentals": None}
    d = combine_signals(signals, ROLES)
    assert d.action == "BUY" and d.confidence == pytest.approx(0.75)
    assert "technical and sentiment agree BUY" in d.reasoning


def test_unknown_role_is_treated_as_advisor_never_as_lead():
    assert combine_signals({"technical": sig("BUY"), "mystery": sig("SELL")}, {"technical": LEAD}).action == "HOLD"


def test_single_lead_buy_and_sell_pass_through_unchanged():
    assert combine_signals({"technical": sig("SELL", 0.7)}, ROLES).action == "SELL"


# ---------------------------------------------------------------- LLM answer cache
def cached_client(behaviours, ttl=3600.0, clock=None, **kw):
    fake = FakeOpenAI(behaviours)
    return LLMClient(["a"], client=fake, cache_ttl_seconds=ttl, clock=clock or (lambda: 0.0), **kw), fake


def test_identical_prompt_is_answered_from_cache():
    llm, fake = cached_client({"a": GOOD})
    assert llm.signal("p").action == llm.signal("p").action == "BUY"
    assert len(fake.calls) == 1


def test_different_prompts_are_not_shared():
    llm, fake = cached_client({"a": GOOD})
    llm.signal("p1")
    llm.signal("p2")
    assert len(fake.calls) == 2


def test_cache_is_off_by_default_and_when_ttl_is_zero():
    fake = FakeOpenAI({"a": GOOD})
    plain = LLMClient(["a"], client=fake)
    plain.signal("p")
    plain.signal("p")
    assert len(fake.calls) == 2


def test_cache_entries_expire():
    now = [0.0]
    llm, fake = cached_client({"a": GOOD}, ttl=100.0, clock=lambda: now[0])
    llm.signal("p")
    now[0] = 99.0
    llm.signal("p")
    assert len(fake.calls) == 1
    now[0] = 101.0
    llm.signal("p")
    assert len(fake.calls) == 2


def test_failures_are_never_cached():
    fake = FakeOpenAI({"a": api_error()})
    llm = LLMClient(["a"], client=fake, cache_ttl_seconds=3600.0, clock=lambda: 0.0)
    with pytest.raises(LLMUnavailable):
        llm.signal("p")
    failed_calls = len(fake.calls)  # the failed call itself was retried (transient errors are), but nothing was cached
    fake.behaviours["a"] = GOOD
    assert llm.signal("p").action == "BUY" and len(fake.calls) == failed_calls + 1


def test_cache_is_bounded():
    llm, fake = cached_client({"a": GOOD}, max_cache_entries=2)
    for prompt in ("p1", "p2", "p3"):
        llm.signal(prompt)
    llm.signal("p3")
    assert len(fake.calls) == 3 and len(llm._cache) == 2


def test_cache_is_safe_under_concurrent_use():
    llm, fake = cached_client({"a": GOOD})
    results = []
    threads = [threading.Thread(target=lambda: results.append(llm.signal("p").action)) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == ["BUY"] * 12


# ---------------------------------------------------------------- database migration
def test_older_database_without_signals_column_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE decisions (id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), price FLOAT,
        technical_action VARCHAR(8) NOT NULL, technical_confidence FLOAT NOT NULL, sentiment_action VARCHAR(8),
        sentiment_confidence FLOAT, final_action VARCHAR(8), final_confidence FLOAT, reasoning TEXT,
        risk_approved BOOLEAN, risk_quantity INTEGER, risk_reason TEXT)""")
    con.execute("INSERT INTO decisions (symbol, price, technical_action, technical_confidence, final_action, final_confidence,"
                " reasoning, risk_approved, risk_quantity, risk_reason) VALUES ('OLD', 1.0, 'BUY', 0.5, 'BUY', 0.5, 'r', 1, 1, 'x')")
    con.commit()
    con.close()

    sessions = make_session_factory(f"sqlite:///{path}")
    with sessions() as s:
        s.add(DecisionRecord(symbol="NEW", price=2.0, technical_action="BUY", technical_confidence=0.9, final_action="BUY",
                             final_confidence=0.9, reasoning="r", risk_approved=True, risk_quantity=1, risk_reason="x",
                             signals_json='{"technical": null}'))
        s.commit()
        rows = {r.symbol: r.signals_json for r in s.scalars(select(DecisionRecord))}
    assert rows == {"OLD": None, "NEW": '{"technical": null}'}


def test_opening_an_up_to_date_database_twice_is_harmless(tmp_path):
    url = f"sqlite:///{tmp_path / 'x.db'}"
    make_session_factory(url)
    make_session_factory(url)


# ---------------------------------------------------------------- completed-session snapshots
def daily_bars(n=260, tz=None):
    idx = pd.date_range(end="2026-09-18", periods=n, freq="B", tz=tz)
    close = pd.Series([100.0 + i * 0.1 for i in range(n)], index=idx)
    return pd.DataFrame({"Close": close, "Volume": 1000.0})


def test_snapshot_uses_all_bars_when_no_as_of_is_given():
    bars = daily_bars()
    snap = fetch_snapshot("X", download=lambda *a, **k: bars, today=lambda: date(2026, 9, 21))
    assert snap.price == pytest.approx(float(bars["Close"].iloc[-1]))


def test_snapshot_drops_bars_after_the_last_completed_session():
    bars = daily_bars()
    cutoff = bars.index[-3].date()
    snap = fetch_snapshot("X", as_of=lambda: cutoff, download=lambda *a, **k: bars)
    assert snap.price == pytest.approx(float(bars["Close"].iloc[-3]))


def test_snapshot_as_of_works_with_timezone_aware_indexes():
    bars = daily_bars(tz="Asia/Kolkata")
    cutoff = bars.index[-2].date()
    snap = fetch_snapshot("X", as_of=lambda: cutoff, download=lambda *a, **k: bars)
    assert snap.price == pytest.approx(float(bars["Close"].iloc[-2]))


def test_snapshot_is_identical_on_every_call_within_a_session():
    bars = daily_bars()
    grown = pd.concat([bars, daily_bars(1).set_axis([bars.index[-1] + pd.offsets.BDay(1)])])  # today's forming bar appears
    cutoff = bars.index[-1].date()
    a = fetch_snapshot("X", as_of=lambda: cutoff, download=lambda *a, **k: bars)
    b = fetch_snapshot("X", as_of=lambda: cutoff, download=lambda *a, **k: grown)
    assert a == b


def test_snapshot_keeps_plain_symbol_and_passes_the_yahoo_suffix():
    seen = {}

    def download(ticker, **kw):
        seen["ticker"] = ticker
        return daily_bars()

    snap = fetch_snapshot("RELIANCE", suffix=".NS", download=download, today=lambda: date(2026, 9, 21))
    assert snap.symbol == "RELIANCE" and seen["ticker"] == "RELIANCE.NS"


# ---------------------------------------------------------------- India session logic
CLOCK = IndiaClock()


@pytest.mark.parametrize("now,expected", [
    (datetime(2026, 9, 21, 5, 0, tzinfo=UTC), date(2026, 9, 18)),    # Monday 10:30 IST, market open -> Friday's close
    (datetime(2026, 9, 21, 11, 30, tzinfo=UTC), date(2026, 9, 21)),  # Monday 17:00 IST, after the close -> today's
    (datetime(2026, 9, 20, 8, 0, tzinfo=UTC), date(2026, 9, 18)),    # Sunday -> Friday
    (datetime(2026, 1, 26, 5, 0, tzinfo=UTC), date(2026, 1, 23)),    # Republic Day Monday -> previous Friday
    (datetime(2027, 6, 1, 5, 0, tzinfo=UTC), date(2027, 5, 31)),     # past the calendar: weekday fallback, during hours
    (datetime(2027, 6, 1, 11, 30, tzinfo=UTC), date(2027, 6, 1)),    # past the calendar, after 15:30 IST
    (datetime(2027, 6, 6, 5, 0, tzinfo=UTC), date(2027, 6, 4)),      # past the calendar, Sunday -> Friday
])
def test_last_completed_session(now, expected):
    assert CLOCK.last_completed_session(now) == expected


def five_min_bars():
    idx = pd.date_range("2026-09-18 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [100.0, 101.0, 102.0, 103.0]}, index=idx)


def test_intraday_bars_are_cached_briefly_and_refetched_after_the_ttl():
    calls, now = [], [0.0]

    def download(ticker, **kw):
        calls.append(ticker)
        return five_min_bars()

    feed = IntradayFeed(download=download, ttl_seconds=30.0, clock=lambda: now[0])
    feed.last_price("TCS")
    feed.bars_since("TCS", datetime(2026, 9, 18, tzinfo=UTC))
    feed.last_price("TCS")
    assert calls == ["TCS.NS"]
    feed.last_price("INFY")
    assert calls == ["TCS.NS", "INFY.NS"]
    now[0] = 31.0
    feed.last_price("TCS")
    assert calls == ["TCS.NS", "INFY.NS", "TCS.NS"]


# ---------------------------------------------------------------- config
def test_new_performance_settings(monkeypatch):
    from src.config import load_settings

    for name in ("ANALYSIS_WORKERS", "LLM_CACHE_HOURS"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.analysis_workers == 3 and s.llm_cache_hours == 12.0
    monkeypatch.setenv("ANALYSIS_WORKERS", "6")
    monkeypatch.setenv("LLM_CACHE_HOURS", "0")
    s = load_settings()
    assert s.analysis_workers == 6 and s.llm_cache_hours == 0.0


@pytest.mark.parametrize("name,value", [("ANALYSIS_WORKERS", "0"), ("LLM_CACHE_HOURS", "-1")])
def test_invalid_performance_settings_are_rejected(monkeypatch, name, value):
    from src.config import load_settings

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings()
