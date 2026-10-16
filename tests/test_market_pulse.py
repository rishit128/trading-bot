"""The market pulse: finding what is surging today (stocks and sectors) from the scan's own bars, how the swing pipeline
uses it (extra candidates, the AI's context, a 50-day exit for those positions) and that nothing changes without it."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.agents.base import LEAD
from src.config import Settings
from src.data.indicators import Snapshot
from src.data.market_pulse import (
    HOT_RET_21D,
    MAX_OPEN,
    HotStock,
    MarketPulse,
    Mover,
    hot_stocks,
    mover,
    movers_from_bars,
    pulse_line,
    summary,
    yahoo_sector,
)
from src.data.universe import ScreenConfig, UniverseScreener
from src.database import DecisionRecord, OrderRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.engine.strategy import trend_exit_decision
from src.llm import AgentSignal
from src.pipeline import TradingPipeline
from tests.test_llm_and_pipeline import FakeBroker

DAY = date(2026, 10, 16)
CFG = ScreenConfig(min_price=100.0, min_traded_value=1e7, max_daily_volatility=0.04)


def series(last_month_gain: float, n: int = 120, start: float = 100.0, noise: float = 0.012):
    """Daily closes: a flat base, then `last_month_gain` spread over the last 21 sessions, with day-to-day noise (a
    perfectly smooth climb has an RSI of 100, which the pulse rightly calls too stretched)."""
    path = np.concatenate([np.full(n - 21, start), start * (1 + last_month_gain) ** (np.arange(1, 22) / 21)])
    return path * (1 + np.random.default_rng(7).normal(0, noise, n))


def test_a_stock_up_8_percent_in_a_month_above_its_50_day_average_is_a_mover():
    close = series(0.12)
    volume = np.full(len(close), 1e6)
    idx = pd.date_range(end="2026-10-15", periods=len(close), freq="B")
    m = mover("TECHM", close, volume, idx[-1], idx[-1], 100.0, 1e7, 0.04)
    assert m is not None and m.ret21 >= HOT_RET_21D and m.price > 100


@pytest.mark.parametrize(
    "gain,volume,noise,why",
    [(0.03, 1e6, 0.012, "not up enough"), (0.12, 10.0, 0.012, "too thin"), (0.12, 1e6, 0.0, "RSI past the pulse's line")],
)
def test_other_stocks_are_not_movers(gain, volume, noise, why):
    close = series(gain, noise=noise)
    idx = pd.date_range(end="2026-10-15", periods=len(close), freq="B")
    assert mover("X", close, np.full(len(close), volume), idx[-1], idx[-1], 100.0, 1e7, 0.04) is None, why


def test_a_blow_off_month_is_not_a_trend_to_join():
    close = series(0.60)  # +60% in a month
    idx = pd.date_range(end="2026-10-15", periods=len(close), freq="B")
    assert mover("X", close, np.full(len(close), 1e6), idx[-1], idx[-1], 100.0, 1e7, 0.5) is None


def test_short_history_stale_data_wild_stocks_and_stocks_below_their_50_day_average_are_not_movers():
    close, vol = series(0.12), np.full(120, 1e6)
    last = pd.Timestamp("2026-10-15")
    assert mover("X", close[-30:], vol[-30:], last, last, 100.0, 1e7, 0.04) is None  # not enough bars
    assert mover("X", close, vol, last - pd.Timedelta(days=9), last, 100.0, 1e7, 0.04) is None  # data stopped
    assert mover("X", close, vol, last, last, 100.0, 1e7, 0.0001) is None  # too volatile for the cap
    down = np.concatenate([np.full(100, 200.0), np.linspace(120, 110, 20)])
    assert mover("X", down, np.full(120, 1e6), last, last, 100.0, 1e7, 0.5) is None  # under its 50-day average


def bars_frame(stocks):
    """The screener's long format: MultiIndex (symbol, timestamp), Close and Volume."""
    frames = []
    for symbol, close in stocks.items():
        idx = pd.date_range(end="2026-10-15", periods=len(close), freq="B")
        frames.append(pd.DataFrame({"Close": close, "Volume": 1e6}, index=pd.MultiIndex.from_product([[symbol], idx])))
    return pd.concat(frames)


def test_movers_are_found_in_a_screener_format_frame():
    frame = bars_frame({"UP": series(0.15), "FLAT": series(0.0), "UP2": series(0.1)})
    assert sorted(m.symbol for m in movers_from_bars(frame, CFG)) == ["UP", "UP2"]
    assert movers_from_bars(pd.DataFrame(), CFG) == []


def test_the_screener_keeps_the_movers_of_the_scan_it_already_runs():
    frame = bars_frame({"UP": series(0.15), "FLAT": series(0.0)})
    screener = UniverseScreener(lambda: ["UP", "FLAT"], lambda syms: frame, CFG, today=lambda: DAY)
    assert [m.symbol for m in screener.movers()] == ["UP"]
    assert [m.symbol for m in screener.movers()] == ["UP"]  # the day's scan is reused


SECTORS = {"A": "IT", "B": "IT", "C": "IT", "D": "Banks", "E": "Banks", "F": None}


def movers_of(symbols, price=200.0):
    return [Mover(s, price, 0.10 + i / 100, 0.2, 1e8) for i, s in enumerate(symbols)]


def test_a_sector_is_hot_only_with_three_surging_stocks_and_unknown_sectors_are_ignored():
    hot = hot_stocks(movers_of("ABCDEF"), SECTORS.get)
    assert [h.symbol for h in hot] == ["C", "B", "A"] and {h.sector for h in hot} == {"IT"}  # strongest first
    assert all(h.sector_stocks == 3 for h in hot) and hot[0].sector_avg_ret21 == pytest.approx(0.11)
    assert hot_stocks([], SECTORS.get) == []


def test_the_pulse_is_computed_once_a_day_and_picks_respect_budget_held_and_slots():
    calls, day = [], [DAY]
    asked = []

    def sector(s):
        asked.append(s)
        return SECTORS.get(s)

    movers = movers_of("ABC") + [Mover("D", 5_000.0, 0.5, 0.5, 1e8)]
    pulse = MarketPulse(lambda: (calls.append(1), movers)[1], sector, today=lambda: day[0])
    assert [h.symbol for h in pulse.hot()] == ["C", "B", "A"] and pulse.hot() is pulse.hot() and len(calls) == 1
    assert [h.symbol for h in pulse.picks(1_000.0, [], 5)] == ["C", "B", "A"]
    assert [h.symbol for h in pulse.picks(1_000.0, ["C"], 5)] == ["B", "A"]  # held stocks are not picked again
    assert [h.symbol for h in pulse.picks(100.0, [], 5)] == []  # one share does not fit
    assert [h.symbol for h in pulse.picks(1_000.0, [], 2)] == ["C", "B"] and pulse.picks(1_000.0, [], 0) == []
    day[0] = date(2026, 10, 17)
    pulse.hot()
    assert len(calls) == 2 and asked.count("A") == 1  # a new day re-reads the movers; known sectors are remembered


def test_picks_are_the_most_liquid_hot_stocks_not_the_biggest_spikes():
    movers = [
        Mover("SPIKE", 200.0, 0.30, 0.4, 2e7),
        Mover("LEADER", 200.0, 0.10, 0.2, 9e8),
        Mover("MID", 200.0, 0.2, 0.3, 3e8),
    ]
    pulse = MarketPulse(lambda: movers, lambda s: "IT", today=lambda: DAY)
    assert [h.symbol for h in pulse.picks(1_000.0, [], 5)] == ["LEADER", "MID", "SPIKE"]


def test_the_hot_stock_text_and_json():
    h = HotStock("TECHM", "Technology", 6, 0.111, 0.13, 0.09)
    assert "Technology sector is surging" in pulse_line(h) and "TECHM is +13%" in pulse_line(h)
    assert HotStock.from_dict(json.loads(h.to_json())) == h
    assert summary([h]).startswith("Technology: 1 stocks, avg +11%") and summary([]) == ""


def test_yahoo_sector_never_raises():
    class T:
        def __init__(self, sym):
            self.info = {"sector": "Technology"} if sym.startswith("TCS") else {}

    assert yahoo_sector("TCS", T) == "Technology" and yahoo_sector("ZZZ", T) is None  # no industry: the sector

    class Both:
        def __init__(self, sym):
            self.info = {"sector": "Technology", "industry": "Information Technology Services"}

    assert yahoo_sector("TCS", Both) == "Information Technology Services"  # the finer label wins

    def boom(sym):
        raise RuntimeError("yahoo down")

    assert yahoo_sector("TCS", boom) is None


# ---------------------------------------------------------------- the swing pipeline
class Seeing:
    """A lead agent that records the pulse it was shown and says BUY."""

    name, role = "technical", LEAD

    def __init__(self):
        self.shown = {}

    def analyze(self, ctx):
        self.shown[ctx.symbol] = ctx.pulse()
        return AgentSignal(action="BUY", confidence=0.9, reasoning="t")


def snap(sym, price=250.0, ma50=240.0, ma200=260.0, rsi=74.0):
    return Snapshot(sym, price, ma50, ma200, rsi, 1000, bar_date="2026-10-15")


def pulse_pipeline(tmp_path, symbols="ABC", held=None, enabled=True, snapshots=None, pulse=True, watchlist=("TCS",)):
    agent = Seeing()
    held = held or {}
    broker = FakeBroker(Portfolio(100_000.0, 100_000.0, {s: 1000.0 for s in held}, dict(held), 100_000.0))
    sessions = make_session_factory(f"sqlite:///{tmp_path / 't.db'}")
    told = []
    snaps = snapshots or {}
    pipe = TradingPipeline(
        Settings(watchlist=watchlist, dry_run=True, market_pulse=enabled),
        [agent],
        broker,
        sessions,
        lambda sym: snaps.get(sym) or snap(sym),
        notify=told.append,
        pulse=MarketPulse(lambda: movers_of(symbols), SECTORS.get, today=lambda: DAY) if pulse else None,
    )
    return pipe, agent, sessions, told


def decisions(sessions):
    with sessions() as s:
        return list(s.scalars(select(DecisionRecord).order_by(DecisionRecord.id)))


def bought_on_pulse(sessions, *symbols):
    """Record a filled BUY that came from a pulse candidate for each symbol."""
    with sessions() as s:
        for sym in symbols:
            d = DecisionRecord(symbol=sym, price=1.0, final_action="BUY", final_confidence=0.9, reasoning="x", candidate_source="pulse",
                               risk_approved=True, risk_quantity=1, risk_reason="ok")  # fmt: skip
            s.add(d)
            s.flush()
            s.add(OrderRecord(decision_id=d.id, symbol=sym, side="BUY", quantity=1, status="filled"))
        s.commit()


def test_stocks_of_a_surging_sector_join_the_candidates_and_the_ai_is_shown_the_pulse(tmp_path):
    pipe, agent, sessions, told = pulse_pipeline(tmp_path)
    assert [r.symbol for r in pipe.run_once()] == ["TCS", "C", "B", "A"]
    rows = {d.symbol: d for d in decisions(sessions)}
    assert {s: d.candidate_source for s, d in rows.items()} == {"TCS": "watchlist", "C": "pulse", "B": "pulse", "A": "pulse"}
    assert agent.shown["TCS"] is None and agent.shown["C"].sector == "IT"
    assert HotStock.from_dict(json.loads(rows["C"].pulse_json)) == agent.shown["C"] and rows["TCS"].pulse_json is None
    assert len([m for m in told if m.startswith("🔥 Market pulse")]) == 1 and "IT: 3 stocks" in told[0]
    pipe.run_once()
    assert len([m for m in told if m.startswith("🔥 Market pulse")]) == 1  # told once per day and set of sectors


def test_a_remembered_answer_is_not_reused_once_the_stock_becomes_part_of_a_surge(tmp_path):
    pipe, agent, sessions, _ = pulse_pipeline(tmp_path, pulse=False)
    pipe.run_once()
    pipe.run_once()
    assert len(agent.shown) == 1  # no pulse: the AI asked once for the bar
    agent.shown.clear()
    pipe.pulse = MarketPulse(
        lambda: movers_of("TCS" and ["TCS", "B", "C"]), {"TCS": "IT", "B": "IT", "C": "IT"}.get, today=lambda: DAY
    )
    pipe.run_once()
    assert agent.shown["TCS"] is not None  # asked again, now with the pulse


def test_the_pulse_can_be_switched_off_and_a_failing_pulse_changes_nothing(tmp_path):
    (tmp_path / "off").mkdir()
    pipe, _, _, told = pulse_pipeline(tmp_path / "off", enabled=False)
    assert [r.symbol for r in pipe.run_once()] == ["TCS"] and told == []
    (tmp_path / "failing").mkdir()
    pipe, _, _, told = pulse_pipeline(tmp_path / "failing")
    pipe.pulse = MarketPulse(lambda: 1 / 0, SECTORS.get, today=lambda: DAY)
    assert [r.symbol for r in pipe.run_once()] == ["TCS"] and told == []  # the cycle carries on without the pulse


def test_at_most_three_pulse_positions_are_open_at_once(tmp_path):
    pipe, _, sessions, _ = pulse_pipeline(tmp_path, held={"A": 5, "B": 5}, watchlist=("TCS", "A", "B"))
    bought_on_pulse(sessions, "A", "B")
    symbols = [r.symbol for r in pipe.run_once()]
    assert MAX_OPEN == 3 and sorted(symbols) == ["A", "B", "C", "TCS"]  # two pulse positions open: one more slot, for C


def test_a_position_bought_on_the_pulse_is_sold_on_a_close_below_its_50_day_average(tmp_path):
    below_50 = snap("A", price=230.0, ma50=240.0, ma200=260.0, rsi=50.0)  # still under MA200, now under MA50 too
    above_50 = snap("B", price=250.0, ma50=240.0, ma200=260.0, rsi=50.0)  # under MA200 but above MA50: held
    pipe, _, sessions, _ = pulse_pipeline(
        tmp_path, held={"A": 5, "B": 5}, snapshots={"A": below_50, "B": above_50}, watchlist=("A", "B")
    )
    bought_on_pulse(sessions, "A", "B")
    pipe.run_once()
    rows = {d.symbol: d for d in decisions(sessions)}
    assert (
        rows["A"].final_action == "SELL" and "50-day average" in rows["A"].reasoning and rows["A"].decision_source == "trend_exit"
    )
    assert rows["B"].final_action == "BUY"  # not sold: its line is the 50-day average, which it holds


def test_a_stock_later_bought_by_the_screener_goes_back_to_the_200_day_exit(tmp_path):
    pipe, _, sessions, _ = pulse_pipeline(
        tmp_path, held={"A": 5}, watchlist=("A",), snapshots={"A": snap("A", 250.0, 240.0, 260.0, 50.0)}
    )
    bought_on_pulse(sessions, "A")
    with sessions() as s:  # sold, then bought again as an ordinary screener candidate
        d = DecisionRecord(symbol="A", price=1.0, final_action="BUY", final_confidence=0.9, reasoning="x", candidate_source="screener",
                           risk_approved=True, risk_quantity=1, risk_reason="ok")  # fmt: skip
        s.add(d)
        s.flush()
        s.add(OrderRecord(decision_id=d.id, symbol="A", side="BUY", quantity=1, status="filled"))
        s.commit()
    pipe.run_once()
    assert next(d for d in decisions(sessions) if d.reasoning != "x").final_action == "SELL"  # under MA200: the usual exit


def test_a_screener_position_keeps_the_200_day_exit(tmp_path):
    pipe, _, sessions, _ = pulse_pipeline(tmp_path, held={"TCS": 5}, snapshots={"TCS": snap("TCS", 250.0, 240.0, 260.0, 50.0)})
    pipe.run_once()
    d = next(d for d in decisions(sessions) if d.symbol == "TCS")
    assert d.final_action == "SELL" and "200-day average" in d.reasoning  # never bought on the pulse: unchanged rule


def test_the_trend_exit_line_depends_on_how_the_position_was_bought():
    s = snap("A", price=245.0, ma50=240.0, ma200=260.0)
    assert trend_exit_decision(s, 5).reasoning.endswith("200-day average 260.00")
    assert trend_exit_decision(s, 5, short=True) is None  # above its 50-day average
    assert "50-day average 250.00" in trend_exit_decision(snap("A", 245.0, 250.0, 260.0), 5, short=True).reasoning
