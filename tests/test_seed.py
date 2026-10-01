"""Seeding the decision memory from price history, and the walk-forward gate that decides whether it may be used."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from src.data.universe import ScreenConfig
from src.database import SetupObservationRecord, make_session_factory
from src.research.seed_memory import Observation, gate, rebuild, store, walk_forward, weekly_dates


def test_weekly_dates_are_the_first_session_of_each_week_with_room_on_both_sides():
    index = pd.bdate_range("2026-01-05", periods=30)  # six full weeks, Mondays first
    dates = weekly_dates(index, warmup=5, horizon=5)
    assert all(d.weekday() == 0 for d in dates)
    assert dates[0] == index[5] and dates[-1] <= index[len(index) - 6]


def trending_bars(n=700, start=100.0, daily=0.002, seed=0):
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + daily + rng.normal(0, 0.005, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 2_000_000.0},
                        index=idx)


def test_rebuild_turns_weekly_scanner_picks_into_setups_with_snapshots_and_forward_outcomes():
    bars = {"UP": trending_bars(), "FLAT": trending_bars(daily=0.0, seed=1)}
    cfg = ScreenConfig(max_candidates=5, min_price=10.0, min_traded_value=1e6, max_daily_volatility=0.04)
    obs = rebuild(bars, cfg, horizon=20, stop_pct=0.15, warmup=300)
    assert obs and {o.symbol for o in obs} <= {"UP", "FLAT"}
    o = obs[0]
    assert o.snapshot["bar_date"] == o.bar_date and o.exit_date > o.bar_date and o.horizon == 20
    entry_pos = bars[o.symbol].index.get_loc(pd.Timestamp(o.bar_date)) + 1
    entry, exit_ = bars[o.symbol]["Open"].iloc[entry_pos], bars[o.symbol]["Close"].iloc[entry_pos + 19]
    assert o.gross_return == pytest.approx(exit_ / entry - 1)  # next open in, close `horizon` sessions on


def snapshot(symbol, bar_date, rsi):
    return {"symbol": symbol, "price": 100.0, "ma50": 98.0, "ma200": 95.0, "rsi": rsi, "volume": 1000, "avg_volume": 1000,
            "adx": 35.0, "bar_date": bar_date}


def obs(symbol, bar_date, exit_date, rsi, gross):
    return Observation(symbol, bar_date, 20, exit_date, snapshot(symbol, bar_date, rsi), gross)


def history_with_an_edge(edge=True):
    """Two kinds of setup, 40 stocks each, every week of 2024: RSI 50s and RSI 60s. With `edge`, RSI 60s always gain."""
    out = []
    weeks = pd.date_range("2024-01-01", "2024-12-23", freq="W-MON")
    for w, monday in enumerate(weeks):
        day, done = str(monday.date()), str((monday + pd.Timedelta(days=28)).date())
        for i in range(40):
            good = 0.05 if edge else (0.05 if (i + w) % 2 else -0.05)
            bad = -0.05 if edge else (-0.05 if (i + w) % 2 else 0.05)
            out += [obs(f"G{i}", day, done, 65.0, good), obs(f"B{i}", day, done, 55.0, bad)]
    return out


def run_check(tmp_path, observations):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 's.db'}")
    store(sessions, observations)
    return walk_forward(sessions, observations, horizon=20, min_samples=30, test_from="2024-07-01")


def test_store_writes_the_setups_once_however_often_it_runs(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 's.db'}")
    rows = [obs("A", "2024-01-01", "2024-01-29", 60.0, 0.01)]
    store(sessions, rows)
    store(sessions, rows)
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(SetupObservationRecord)) == 1


def test_a_memory_with_a_real_edge_passes_the_gate(tmp_path):
    groups = run_check(tmp_path, history_with_an_edge(edge=True))
    assert groups["above baseline"] and all(v > 0 for _, v in groups["above baseline"])
    passed, lines = gate(groups)
    assert passed and lines[-1].startswith("GATE: PASSED")


def test_a_memory_rating_noise_fails_the_gate(tmp_path):
    passed, lines = gate(run_check(tmp_path, history_with_an_edge(edge=False)))
    assert not passed and ("FAILED" in lines[-1] or "not enough" in lines[-1])


def test_the_walk_forward_never_lets_a_setup_see_its_own_or_later_outcomes(tmp_path):
    groups = run_check(tmp_path, history_with_an_edge(edge=True)[:160])  # only January: nothing had finished yet
    assert groups["above baseline"] == [] and groups["at or below baseline"] == []


def test_an_observation_survives_a_round_trip_through_its_file_line():
    o = obs("A", "2024-01-01", "2024-01-29", 60.0, 0.0123)
    assert Observation.from_json(o.to_json()) == o
