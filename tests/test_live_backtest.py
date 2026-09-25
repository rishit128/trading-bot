"""The live-configuration backtest must run the live pieces (scanner filter, ranking, affordability, risk engine, exits)."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import LIVE_PARITY, simulate
from src.config import RiskLimits
from src.data.universe import (Candidate, ScreenConfig, UniverseScreener, screen_bars, select_candidates)
from src.engine.paper_broker import india_delivery_fees
from src.research.live_backtest import (make_buy_source, ranked_candidates, run_live_config, summarize_run)

CFG = ScreenConfig(max_candidates=5, min_price=1.0, min_traded_value=1.0, max_daily_volatility=0.5)
N = 330
IDX = pd.bdate_range("2023-01-02", periods=N)


def bars_for(seed, slope, base=100.0, n=N, idx=None):
    """A noisy uptrend as OHLCV; the seed keeps every run identical."""
    rng = np.random.default_rng(seed)
    idx = IDX[-n:] if idx is None else idx
    close = base * (1 + slope * np.arange(n)) + np.cumsum(rng.normal(0, 0.006 * base, n))
    close = pd.Series(close, index=idx)
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 1_000_000.0}, index=idx)


UNIVERSE = {"AAA": bars_for(1, 0.002), "BBB": bars_for(2, 0.002), "CCC": bars_for(3, 0.002, base=5000.0)}


def wide(col):
    return pd.DataFrame({s: df[col] for s, df in UNIVERSE.items()}).sort_index()


def test_backtest_scanner_matches_the_live_scanner_on_the_same_bars():
    close, volume = wide("Close"), wide("Volume")
    day = IDX[-1]
    ranked = ranked_candidates(close, volume, [day], CFG)[day]
    long = pd.concat({s: df.loc[:day][["Close", "Volume"]] for s, df in UNIVERSE.items()}, names=["symbol", "timestamp"])
    live = screen_bars(long, CFG)
    assert {c.symbol for c in ranked} == {c.symbol for c in live} and ranked  # at least one stock must pass to mean anything
    by = {c.symbol: c for c in live}
    for c in ranked:
        assert c.price == pytest.approx(by[c.symbol].price) and c.momentum_12_1 == pytest.approx(by[c.symbol].momentum_12_1)
    assert [c.momentum_12_1 for c in ranked] == sorted((c.momentum_12_1 for c in ranked), reverse=True)


def test_unaffordable_stocks_are_dropped_in_live_and_backtest_alike():
    ranked = [Candidate("CHEAP", 500.0, 0, 50, 1e9, 0.01, 0.9), Candidate("PRICEY", 9000.0, 0, 50, 1e9, 0.01, 1.5)]
    assert [c.symbol for c in select_candidates(ranked, 5)] == ["CHEAP", "PRICEY"]  # no rule, no filter
    assert [c.symbol for c in select_candidates(ranked, 5, equity=20_000, affordable_pct=0.10)] == ["CHEAP"]
    assert [c.symbol for c in select_candidates(ranked, 1, equity=1_000_000, affordable_pct=0.05)] == ["CHEAP"]  # top-n cut

    cfg = ScreenConfig(max_candidates=5, affordable_pct=0.10)
    source = make_buy_source({IDX[0]: ranked}, cfg, confidence=0.8)
    from src.engine.risk_engine import Portfolio
    assert source(IDX[0], Portfolio(cash=20_000, equity=20_000)) == [("CHEAP", 0.8)]


def test_screener_applies_affordability_to_the_symbols_it_hands_the_pipeline():
    from datetime import date
    from src.engine.risk_engine import Portfolio

    long = pd.concat({s: df[["Close", "Volume"]] for s, df in UNIVERSE.items()}, names=["symbol", "timestamp"])
    cfg = ScreenConfig(max_candidates=5, min_price=1.0, min_traded_value=1.0, max_daily_volatility=0.5, affordable_pct=0.10)
    screener = UniverseScreener(lambda: list(UNIVERSE), lambda syms: long, cfg, today=lambda: date(2026, 1, 1))
    small = screener.symbols_for(Portfolio(cash=20_000, equity=20_000))
    assert "CCC" not in small and "AAA" in small  # CCC costs ~Rs 5,000+ > 10% of Rs 20,000
    assert "CCC" in screener.symbols_for(Portfolio(cash=1_000_000, equity=1_000_000))


def _two_stock_bars():
    idx = pd.bdate_range("2024-01-01", periods=10)
    flat = lambda p: pd.DataFrame({"Open": p, "High": p, "Low": p, "Close": p, "Volume": 1e6}, index=idx)
    return {"AAA": flat(100.0), "ZZZ": flat(100.0)}, idx


def test_ordered_buy_source_fills_the_best_ranked_stock_not_the_alphabetical_one():
    bars, idx = _two_stock_bars()
    limits = RiskLimits(max_open_positions=1, min_position_pct=0.5, max_position_pct=0.5, max_portfolio_exposure_pct=0.8)
    by_signal = simulate(bars, {"AAA": {idx[0]: ("BUY", 0.9)}, "ZZZ": {idx[0]: ("BUY", 0.9)}}, limits, **LIVE_PARITY)
    assert [t for t in by_signal.trades] == [] and by_signal.open_at_end == 1  # alphabetical: one slot goes to AAA
    ranked = simulate(bars, {}, limits, buy_source=lambda day, pf: [("ZZZ", 0.9), ("AAA", 0.9)] if day == idx[0] else [],
                      **LIVE_PARITY)
    close = simulate(bars, {}, limits, buy_source=lambda day, pf: [("ZZZ", 0.9), ("AAA", 0.9)] if day == idx[0] else [],
                     max_hold_days=1)
    assert ranked.open_at_end == 1 and [t.symbol for t in close.trades] == ["ZZZ"]  # rank order wins the only slot


def test_a_short_history_does_not_truncate_the_calendar_of_the_others():
    idx = pd.bdate_range("2024-01-01", periods=30)
    full = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1e6}, index=idx)
    late = full.iloc[20:]
    result = simulate({"OLD": full, "NEW": late}, {"OLD": {idx[0]: ("BUY", 0.9)}}, RiskLimits(), max_hold_days=100)
    assert len(result.equity) == 30 and result.open_at_end == 1  # the old calendar is kept (intersection was 10 days)
    late_buy = simulate({"OLD": full, "NEW": late}, {"NEW": {idx[25]: ("BUY", 0.9)}}, RiskLimits(), max_hold_days=100)
    assert late_buy.open_at_end == 1


def test_live_config_run_uses_live_limits_costs_and_reports_expectancy():
    close, volume = wide("Close"), wide("Volume")
    dates = list(IDX[260:])
    limits = RiskLimits(max_open_positions=2, min_position_pct=0.10, max_position_pct=0.25, stop_loss_pct=0.08)
    cfg = ScreenConfig(max_candidates=5, min_price=1.0, min_traded_value=1.0, max_daily_volatility=0.5, affordable_pct=0.10)
    run = run_live_config(UNIVERSE, limits, cfg, 20_000.0, dates=dates)
    s = summarize_run(run)
    assert set(s) >= {"total_return", "cagr", "sharpe", "max_drawdown", "trades", "win_rate", "expectancy_pct",
                      "profit_factor", "fees", "exits"}
    held = run.result.open_at_end + s["trades"]
    assert held >= 1
    assert "CCC" not in {t.symbol for t in run.result.trades}  # ~Rs 5,000+/share can't be bought with Rs 20,000 at 10%
    # fixed sale charge is really charged: any closed trade paid at least the Rs 15.93 DP fee
    assert all(t.fees >= 15.93 for t in run.result.trades)
    assert run.result.equity.iloc[0] == pytest.approx(20_000.0, rel=0.05)
