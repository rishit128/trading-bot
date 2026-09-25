"""P0 E1b: every fill, in every execution path, must leave the position protected or the test fails.

Invariants pinned here so a regression that opens a position without its bracket fails CI:
  * backtest: every closed trade carries the stop/target levels its bracket set off the signal-day close; a STOP exit
    always fills at or below the stop and a TARGET exit at or above the target - across a sweep of random series;
  * paper broker: every buy writes a position row whose stop level sits below the fill and whose target sits above,
    and `unprotected_positions()` reports nothing (the working-stop invariant a real broker must restore)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest import Result, simulate
from src.config import RiskLimits
from src.database import make_session_factory
from src.engine.paper_broker import PaperBroker, india_delivery_fees

LIMITS = RiskLimits(stop_loss_pct=0.08)  # these tests pin exit mechanics at a fixed 8% level, not the default


def _walk(seed, n=400):
    rng = np.random.default_rng(seed)
    close = 100.0 * (rng.normal(0.0005, 0.02, n) + 1).cumprod()  # 2% daily vol: the trend rule fires often
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({"Open": [100.0] + list(close[:-1]), "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 10_000},
                        index=idx)


def _scheduled_signals(index, every=6, sell_from=208):
    """Deterministic trigger stream: a BUY every `every` bars after warmup, a SELL three bars later, so the sweep
    opens and closes positions regardless of whether the trend rule happens to fire that week."""
    out = {}
    dates = list(index)
    for i, d in enumerate(dates):
        if i < 200:
            continue
        if (i - 200) % every == 0:
            out[d] = ("BUY", 0.9)
        elif (i - 200) % every == 3:
            out[d] = ("SELL", 0.9)
    return out


def test_every_trade_in_a_noisy_sweep_remembers_its_bracket():
    all_trades = []
    for seed in (1, 7, 99):
        bars = {"A": _walk(seed)}
        res: Result = simulate(bars, {"A": _scheduled_signals(bars["A"].index)}, LIMITS, start_equity=100_000,
                               max_hold_days=10, fees=india_delivery_fees, slippage=0.0005)
        for t in res.trades:
            assert t.stop is not None and t.target is not None, f"seed {seed}: {t} was opened without a bracket!"
            assert t.stop < t.target
            if t.reason == "STOP":
                assert t.exit_price <= t.stop, f"seed {seed}: STOP filled {t.exit_price:.4f} above its {t.stop:.4f} level"
            if t.reason == "TARGET":
                assert t.exit_price >= t.target, f"seed {seed}: TARGET filled {t.exit_price:.4f} below its {t.target:.4f} level"
        all_trades.extend(res.trades)
    assert len(all_trades) >= 20, f"the sweep produced too few fills to mean anything: {len(all_trades)}"


def test_a_gap_through_the_stop_fills_at_the_bar_open():
    dt = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bars = {"A": pd.DataFrame(
        {"Open": [100.0, 88.0], "High": [100.0, 90.0], "Low": [100.0, 87.0], "Close": [100.5, 89.0], "Volume": [1] * 2},
        index=dt)}
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9)}}
    res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=0.0005)
    assert len(res.trades) == 1 and res.trades[0].reason == "STOP"
    assert res.trades[0].exit_price == pytest.approx(88.0, abs=1e-9)  # the gap open, already below the 92.46 stop
    assert res.trades[0].stop == pytest.approx(92.46, abs=0.005)


def test_an_intraday_cross_of_the_stop_fills_at_the_level():
    dt = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bars = {"A": pd.DataFrame(
        {"Open": [100.0, 95.0], "High": [100.0, 96.0], "Low": [100.0, 91.0], "Close": [100.5, 92.0], "Volume": [1] * 2},
        index=dt)}
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9)}}
    res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=0.0005)
    assert res.trades[0].reason == "STOP"
    assert res.trades[0].exit_price == pytest.approx(92.46, abs=0.005)  # the exact level, no slippage widening


def test_a_gap_through_the_target_fills_at_the_bar_open():
    dt = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bars = {"A": pd.DataFrame(
        {"Open": [100.0, 210.0], "High": [100.0, 211.0], "Low": [100.0, 209.0], "Close": [100.5, 210.0], "Volume": [1] * 2},
        index=dt)}
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9)}}
    res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=0.0005)
    assert res.trades[0].reason == "TARGET"
    assert res.trades[0].exit_price == pytest.approx(210.0, abs=1e-9)  # gap above the 201 target, filled at the open
    assert res.trades[0].target == pytest.approx(201.0, abs=0.005)


def test_the_paper_buy_always_writes_a_position_with_its_stop_level(tmp_path):
    class Feed:
        def last_price(self, symbol):
            return 100.0

        def bars_since(self, symbol, since):
            return pd.DataFrame()

    class Clock:
        def is_open(self, now=None):
            return True

        def today_ist(self, now=None):
            return "2024-01-03"

    sessions = make_session_factory(f"sqlite:///{tmp_path / 's.db'}")
    broker = PaperBroker(sessions, Feed(), Clock(),
                         now_fn=lambda: datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc))
    broker.buy_with_bracket("A", 100, 100.5, LIMITS.stop_loss_pct, LIMITS.take_profit_pct)
    holdings = broker.holdings()
    assert len(holdings) == 1 and holdings[0]["qty"] == 100
    assert holdings[0]["stop"] == pytest.approx(92.46, abs=0.005)  # 8% below the 100.5 signal close
    assert holdings[0]["stop"] < holdings[0]["avg_price"]
    assert holdings[0]["avg_price"] < 100.5 * (1 + LIMITS.take_profit_pct)
    assert broker.unprotected_positions() == []  # the paper broker must never report a naked position