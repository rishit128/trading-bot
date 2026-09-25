import pandas as pd
import pytest

from src.research.backtest import TIME_LIMITED, ExitPolicy, LIVE_EXITS, buy_and_hold_curve, curve_metrics, simulate, trade_metrics
from src.config import RiskLimits

LIM = RiskLimits(max_position_pct=0.05, stop_loss_pct=0.02, take_profit_pct=0.05, min_confidence=0.6)
DAYS = pd.bdate_range("2026-01-05", periods=15)


def make_bars(overrides=None, n=15, price=100.0):
    """Flat 100 bars; overrides maps day index -> dict(Open/High/Low/Close)."""
    rows = []
    for i in range(n):
        row = dict(Open=price, High=price, Low=price, Close=price)
        row.update((overrides or {}).get(i, {}))
        rows.append(row)
    return pd.DataFrame(rows, index=DAYS[:n])


def buy_on(day_idx, conf=0.9):
    return {"X": {DAYS[day_idx]: ("BUY", conf)}}


def run(bars, signals):
    return simulate({"X": bars}, signals, LIM)


def test_no_signals_means_flat_equity_and_no_trades():
    r = run(make_bars(), {})
    assert r.trades == [] and (r.equity == 100_000.0).all()


def test_buy_executes_next_open_sized_by_risk_engine_then_hits_target():
    bars = make_bars({3: dict(High=106.0)})
    [t] = run(bars, buy_on(0)).trades
    assert t.entry_date == DAYS[1] and t.entry_price == 100.0 and t.qty == 50
    assert t.reason == "TARGET" and t.exit_price == pytest.approx(105.0) and t.exit_date == DAYS[3]


def test_stop_hit_intraday_exits_at_stop_price():
    [t] = run(make_bars({2: dict(Low=95.0)}), buy_on(0)).trades
    assert t.reason == "STOP" and t.exit_price == pytest.approx(98.0)


def test_gap_below_stop_exits_at_open_not_stop():
    bars = make_bars({2: dict(Open=90.0, High=90.0, Low=90.0, Close=90.0)})
    [t] = run(bars, buy_on(0)).trades
    assert t.reason == "STOP" and t.exit_price == 90.0


def test_stop_wins_when_stop_and_target_both_touched_same_day():
    [t] = run(make_bars({2: dict(Low=90.0, High=110.0)}), buy_on(0)).trades
    assert t.reason == "STOP"


def test_time_exit_after_max_hold_days_at_close():
    bars = make_bars({10: dict(Close=101.0)})
    [t] = simulate({"X": bars}, buy_on(0), LIM, exits=TIME_LIMITED).trades
    assert t.reason == "TIME" and t.exit_date == DAYS[10] and t.exit_price == 101.0


def test_the_default_exit_policy_is_the_live_one_so_a_backtest_tests_the_running_strategy():
    """Regression: simulate() used to default to a 10-day time exit, so holdout, walk-forward and ablation silently tested a
    strategy that is not the one running. The live policy is now the default and any other must be requested by name."""
    bars = make_bars({10: dict(Close=101.0)}, n=15)
    default = simulate({"X": bars}, buy_on(0), LIM)
    assert default.trades == [] and default.open_at_end == 1  # no time limit: still holding at the end of the data
    assert simulate({"X": bars}, buy_on(0), LIM, exits=LIVE_EXITS).equity.equals(default.equity)
    assert LIVE_EXITS == ExitPolicy(max_hold_days=None, trend_exit=True)
    assert TIME_LIMITED == ExitPolicy(max_hold_days=10, trend_exit=False)


def test_exit_policy_rejects_a_nonsensical_hold_limit():
    with pytest.raises(ValueError, match="max_hold_days"):
        ExitPolicy(max_hold_days=0)
    assert ExitPolicy(max_hold_days=None).max_hold_days is None


def test_sell_signal_exits_next_open():
    signals = {"X": {DAYS[0]: ("BUY", 0.9), DAYS[3]: ("SELL", 0.9)}}
    bars = make_bars({4: dict(Open=101.0, High=101.0, Low=101.0, Close=101.0)})
    [t] = run(bars, signals).trades
    assert t.reason == "SIGNAL" and t.exit_date == DAYS[4] and t.exit_price == 101.0


def test_signal_on_last_day_is_never_executed():
    r = run(make_bars(n=5), buy_on(4))
    assert r.trades == [] and r.open_at_end == 0


def test_low_confidence_buy_is_ignored():
    assert run(make_bars(), buy_on(0, conf=0.5)).trades == []


def test_open_position_is_not_added_to():
    signals = {"X": {DAYS[0]: ("BUY", 0.9), DAYS[1]: ("BUY", 0.9)}}
    r = run(make_bars(n=6), signals)
    assert r.open_at_end == 1 and r.trades == []
    assert r.equity.iloc[-1] == 100_000.0


def test_equity_reflects_open_position_mark_to_market():
    bars = make_bars({2: dict(Close=101.0, High=101.0)})
    r = run(bars, buy_on(0))
    assert r.equity.iloc[2] == pytest.approx(100_000.0 + 50 * 1.0)


def test_metrics():
    curve = pd.Series([100.0, 110.0, 99.0, 121.0])
    m = curve_metrics(curve)
    assert m["total_return"] == pytest.approx(0.21) and m["max_drawdown"] == pytest.approx(-0.10)
    assert trade_metrics([]) == {"trades": 0}


def test_buy_and_hold_equal_weight():
    up = make_bars({14: dict(Close=110.0)})
    flat = make_bars()
    curve = buy_and_hold_curve({"A": up, "B": flat}, 100_000.0)
    assert curve.iloc[-1] == pytest.approx(105_000.0)


def test_total_return_is_measured_from_start_equity_not_first_point():
    curve = pd.Series([110.0, 121.0])  # first point already includes a +10% day-one move
    assert curve_metrics(curve)["total_return"] == pytest.approx(0.10)
    assert curve_metrics(curve, start_equity=100.0)["total_return"] == pytest.approx(0.21)


def test_fee_model_reduces_cash_and_makes_trade_returns_net():
    flat_fee = lambda side, value: 10.0  # noqa: E731
    bars = make_bars({3: dict(High=106.0)})  # target hit at 105
    r = simulate({"X": bars}, buy_on(0), LIM, fees=flat_fee)
    [t] = r.trades
    assert t.fees == 20.0 and t.qty == 50
    assert r.equity.iloc[-1] == pytest.approx(100_000.0 + 50 * 5.0 - 20.0)
    assert t.return_pct == pytest.approx(0.05 - 20.0 / (100.0 * 50))


def test_fees_never_let_cash_go_negative():
    huge_fee = lambda side, value: 100.0  # noqa: E731
    r = simulate({"X": make_bars()}, buy_on(0), LIM, start_equity=5_050.0, fees=huge_fee)
    assert (r.equity > 0).all()


def test_trade_analysis_breaks_pnl_down_by_reason_holding_period_year_and_symbol():
    from src.research.backtest import Trade, analyze_trades

    def trade(sym, entry, exit_, ep, xp, reason, qty=10, fees=0.0):
        return Trade(sym, pd.Timestamp(entry), ep, pd.Timestamp(exit_), xp, qty, reason, fees)

    trades = [
        trade("A", "2024-01-02", "2024-01-03", 100, 98, "STOP"),       # -2%, 1 day
        trade("A", "2024-02-01", "2024-02-02", 100, 98, "STOP"),
        trade("B", "2024-03-01", "2024-03-20", 100, 105, "TARGET"),    # +5%, 19 days
        trade("C", "2025-01-01", "2025-01-05", 100, 101, "TIME", fees=2.0),
    ]
    a = analyze_trades(trades)
    stop, target = a["by_reason"].loc["STOP"], a["by_reason"].loc["TARGET"]
    assert stop["n"] == 2 and stop["win_rate"] == 0 and stop["avg_return"] == pytest.approx(-0.02) and stop["pnl"] == pytest.approx(-40)
    assert target["pnl"] == pytest.approx(50)
    assert a["by_reason"].loc["TIME", "pnl"] == pytest.approx(10 - 2)          # fees are deducted
    assert a["by_holding"].loc["<=2d", "n"] == 2 and a["by_holding"].loc["15-30d", "n"] == 1
    assert a["by_year"].loc[2024, "pnl"] == pytest.approx(10) and a["by_year"].loc[2025, "n"] == 1
    assert list(a["worst_symbols"].index)[0] == "A" and a["median_days_stopped_out"] == 1


def test_trade_analysis_of_no_trades_is_empty():
    from src.research.backtest import analyze_trades

    assert analyze_trades([]) == {}


def test_trend_exit_sells_at_the_next_open_after_a_close_below_the_200_day_average():
    import numpy as np
    import pandas as pd
    from src.config import RiskLimits

    idx = pd.bdate_range("2020-01-01", periods=320)
    close = np.concatenate([np.linspace(100, 150, 260), np.linspace(150, 60, 60)])
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1e6}, index=idx)
    signals = {"A": {idx[250]: ("BUY", 0.9)}}
    limits = RiskLimits(stop_loss_pct=0.99)  # keep the protective stop out of the way
    held = simulate({"A": df}, signals, limits, exits=ExitPolicy(max_hold_days=None, trend_exit=False))
    assert held.open_at_end == 1  # no time exit, no trend exit: still holding at the end
    out = simulate({"A": df}, signals, limits, exits=LIVE_EXITS)
    assert out.open_at_end == 0 and out.trades[0].reason == "SIGNAL"
    ma200 = df["Close"].rolling(200).mean()
    broke = df.index[(df["Close"] < ma200) & (df.index > idx[250])][0]
    assert out.trades[0].exit_date == df.index[df.index.get_loc(broke) + 1]


def test_simulator_resumes_buying_after_the_drawdown_pause_but_not_when_permanent():
    import dataclasses
    import numpy as np
    from src.config import RiskLimits

    idx = pd.bdate_range("2024-01-01", periods=120)
    crash = np.concatenate([np.full(10, 100.0), np.linspace(100, 60, 10), np.full(100, 60.0)])

    def frame(close):
        df = pd.DataFrame({"Close": close, "High": close, "Low": close, "Volume": 1e6}, index=idx)
        df["Open"] = df["Close"].shift(1).fillna(close[0])
        return df

    bars = {"X": frame(crash), "Y": frame(np.full(120, 100.0))}
    sig = {"X": {idx[2]: ("BUY", 0.9)}, "Y": {idx[80]: ("BUY", 0.9)}}  # Y is bought long after X's crash tripped -20%
    base = RiskLimits(max_position_pct=0.5, min_position_pct=0.5, max_portfolio_exposure_pct=1.0, stop_loss_pct=0.99,
                      max_drawdown_pct=0.20, max_open_positions=5)
    # X at 50% of equity falling 40% is a 20%+ drawdown, so the halt starts around day 20
    permanent = simulate(bars, sig, dataclasses.replace(base, drawdown_pause_days=0))
    paused = simulate(bars, sig, dataclasses.replace(base, drawdown_pause_days=30))
    assert permanent.open_at_end == 1  # the day-80 BUY of Y was refused by the standing halt
    assert paused.open_at_end == 2     # 30 days later the peak was rebased and Y was bought


def test_every_research_entry_point_simulates_with_the_live_exit_policy(monkeypatch, tmp_path):
    """holdout, walk-forward, ablation and the live-config backtest must not quietly pick their own exits."""
    import numpy as np
    from src.research import backtest as backtest_module
    from src.config import RiskLimits
    from src.ops.holdout import HoldoutReserve
    from src.research import ablation, live_backtest, walk_forward
    from src.data.universe import ScreenConfig
    from src.database import make_session_factory

    seen = []
    real = backtest_module.simulate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("exits", LIVE_EXITS))  # absent means "the default", which is the live policy
        return real(*args, **kwargs)

    for module in (walk_forward, ablation, live_backtest):
        monkeypatch.setattr(module, "simulate", spy)
    import src.ops.holdout as holdout_module
    monkeypatch.setattr(holdout_module, "simulate", spy)

    idx = pd.bdate_range("2023-01-02", periods=320)
    close = pd.Series(np.linspace(100, 160, 320) + np.sin(np.arange(320)), index=idx)
    bars = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx)
    walk_forward.run_walk_forward("A", {"A": bars}, list(idx[250:]), n_windows=2)
    ablation.run_ablation("A", {"A": bars}, list(idx[250:]))
    live_backtest.run_live_config({"A": bars}, RiskLimits(), ScreenConfig(min_price=1, min_traded_value=1), 20_000.0,
                                  dates=list(idx[250:]))
    from datetime import datetime, timezone
    from src.database import DecisionRecord

    sessions = make_session_factory(f"sqlite:///{tmp_path / 'h.db'}")
    with sessions() as s:  # one approved BUY on a day the bars cover, so the reserve really replays something
        s.add(DecisionRecord(symbol="A", price=100.0, final_action="BUY", final_confidence=0.8, reasoning="r",
                             risk_approved=True, risk_quantity=10, risk_reason="ok",
                             created_at=datetime(2023, 9, 14, 5, 30, tzinfo=timezone.utc)))
        s.commit()
    before = len(seen)
    HoldoutReserve(sessions, lambda symbol: bars, initial_cash=100_000.0).mark()
    assert len(seen) == before + 1, "the holdout reserve never reached the simulator, so this test would prove nothing about it"
    assert len(seen) >= 4 and all(policy == LIVE_EXITS for policy in seen)
