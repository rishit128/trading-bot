import pandas as pd
import pytest

from src.backtest import buy_and_hold_curve, curve_metrics, simulate, trade_metrics
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
    [t] = run(make_bars({10: dict(Close=101.0)}), buy_on(0)).trades
    assert t.reason == "TIME" and t.exit_date == DAYS[10] and t.exit_price == 101.0


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
    from src.backtest import Trade, analyze_trades

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
    from src.backtest import analyze_trades

    assert analyze_trades([]) == {}
