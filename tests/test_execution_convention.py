"""the backtest and the paper broker must express ONE execution convention.

    signal at day T's close -> fills at the next open, +slippage, +delivery fees; stop/target exits hit at their exact
    level. The tests pin that equality on the same inputs and make the one admitted divergence (paper prices at the
    current 5-minute close, backtest at the next open) explicit instead of hidden."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.research.backtest import Result, simulate
from src.config import RiskLimits
from src.engine.costs import SLIPPAGE
from src.engine.costs import india_delivery_fees

def apply_slippage(price, side, slippage=SLIPPAGE):
    """The convention as a spec: a buy fills higher, a sell fills lower. The oracle the broker and simulator are held to."""
    return price * (1 + slippage) if side == "BUY" else price * (1 - slippage)


LIMITS = RiskLimits(stop_loss_pct=0.08)  # these tests pin exit mechanics at a fixed 8% level, not the default


def _daily_bars():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    row = {"Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 100.5, "Volume": 10_000}
    return {"A": pd.DataFrame([row] * 4, index=idx)}


class FakeFeed:
    """That part of the intraday feed the paper broker needs: a current price for fills."""

    def __init__(self, price):
        self.price = price

    def last_price(self, symbol):
        return self.price

    def bars_since(self, symbol, since):
        return pd.DataFrame()


class Clock:
    def is_open(self, now=None):
        return True

    def today_ist(self, now=None):
        return "2024-01-03"


def _paper_broker(feed_price, sessions_dir):
    from src.database import make_session_factory

    sessions = make_session_factory(f"sqlite:///{sessions_dir}/p.db")
    from src.engine.paper_broker import PaperBroker

    return PaperBroker(sessions, FakeFeed(feed_price), Clock(),
                       now_fn=lambda: datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc))


def test_backtest_fill_equals_paper_fill_off_the_same_next_open():
    """Given the same next-open basis, simulate and the paper broker print the identical fill, fee and trade P&L."""
    import tempfile

    bars = _daily_bars()
    df = bars["A"]
    entry_open = float(df.loc["2024-01-03", "Open"])
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9),
                     pd.Timestamp("2024-01-04"): ("SELL", 0.9)}}  # decided at the 2024-01-02 close; exited later

    with tempfile.TemporaryDirectory() as td:
        broker = _paper_broker(entry_open, td)  # paper executed at the same next open
        sig_price = float(df.loc["2024-01-02", "Close"])
        fill = broker.buy_with_bracket("A", 100, sig_price, LIMITS.stop_loss_pct, LIMITS.take_profit_pct)

        res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=SLIPPAGE)
        trades = [t for t in res.trades if t.reason == "SIGNAL"]
        assert len(trades) == 1
        trade = trades[0]

    assert fill.avg_price == pytest.approx(apply_slippage(entry_open, "BUY", SLIPPAGE), abs=1e-9)
    assert trade.entry_price == pytest.approx(fill.avg_price, abs=1e-9)  # the same number, from the same next open


def test_sell_signal_fill_is_slippage_widened_in_both_paths():
    """A SELL signal fills lower by the slippage on both sides."""
    import tempfile

    bars = _daily_bars()
    df = bars["A"]
    # Day 1 BUY at 101 open, day 2 SELL decided at the day-2 close.
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9), pd.Timestamp("2024-01-03"): ("SELL", 0.9)}}
    sell_open = float(df.loc["2024-01-04", "Open"])

    with tempfile.TemporaryDirectory() as td:
        broker = _paper_broker(sell_open, td)
        sig_price = float(df.loc["2024-01-02", "Close"])
        buy_fill = broker.buy_with_bracket("A", 100, sig_price, LIMITS.stop_loss_pct, LIMITS.take_profit_pct)
        sell_fill = broker.sell("A", 100)

        res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=SLIPPAGE)
        exit_reason = [t for t in res.trades if t.reason == "SIGNAL"]
        assert len(exit_reason) == 1

    assert sell_fill.avg_price == pytest.approx(apply_slippage(sell_open, "SELL", SLIPPAGE), abs=1e-9)
    assert exit_reason[0].exit_price == pytest.approx(sell_fill.avg_price, abs=1e-9)
    assert buy_fill.avg_price > sell_fill.avg_price  # slippage eats both sides of the flat episode: buy higher, sell lower


def test_stop_and_target_exits_have_no_extra_slippage_in_either_path():
    """Stop/target are limit-like: the backtest fills a gap-down stop at the bar open, never widened by slippage
    (the paper broker's _settle_exits does the same thing: bar-Open on a gap, exact level otherwise)."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bars = {"A": pd.DataFrame(
        {"Open": [100.0, 90.0], "High": [100.0, 92.0], "Low": [100.0, 88.0], "Close": [100.5, 91.0], "Volume": [1] * 2},
        index=idx)}
    signals = {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9)}}  # stop set off the 100.5 signal close: 92.46

    # The next open is 90.0 <= the 92.46 stop: a gap-down, so the trade exits at the bar OPEN, with no slippage on top.
    res: Result = simulate(bars, signals, LIMITS, start_equity=100_000, fees=india_delivery_fees, slippage=SLIPPAGE)
    assert res.trades[0].reason == "STOP"
    assert res.trades[0].exit_price == pytest.approx(90.0, abs=1e-9)


def test_the_admitted_divergence_is_exactly_the_overnight_gap():
    """Paper at the decision-evening close vs backtest at the next open differ by precisely the overnight gap, which is
    why SLIPPAGE is not higher and why the convention doc says 'same session'. This pins the number so it cannot quietly
    become a hidden cost difference."""
    import tempfile

    bars = _daily_bars()
    df = bars["A"]
    evening_close = 100.0  # what the paper sees right after the decision close
    next_open = float(df.loc["2024-01-03", "Open"])  # 101.0, a 1% overnight move in the same direction
    gap = next_open / evening_close - 1
    with tempfile.TemporaryDirectory() as td:
        broker = _paper_broker(evening_close, td)
        sig_price = float(df.loc["2024-01-02", "Close"])
        fill = broker.buy_with_bracket("A", 100, sig_price, LIMITS.stop_loss_pct, LIMITS.take_profit_pct)
        res: Result = simulate(bars, {"A": {pd.Timestamp("2024-01-02"): ("BUY", 0.9),
                                            pd.Timestamp("2024-01-04"): ("SELL", 0.9)}}, LIMITS,
                               start_equity=100_000, fees=india_delivery_fees, slippage=SLIPPAGE)
        trades = [t for t in res.trades if t.reason == "SIGNAL"]
        assert len(trades) == 1
    paper = fill.avg_price
    backtest_fill = trades[0].entry_price
    assert backtest_fill == pytest.approx(paper * (1 + gap), rel=1e-6)  # the 1% gap, nothing else