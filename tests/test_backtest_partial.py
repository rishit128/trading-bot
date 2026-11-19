"""Research only: the simulator can book half a position's profit at a target and let the rest run."""

import pandas as pd
import pytest

from src.config import RiskLimits
from src.engine.enums import Action
from src.research.backtest import simulate


def bars(rows):
    index = pd.bdate_range("2026-01-05", periods=len(rows))
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"]).assign(Volume=1_000)


def run(rows, **kw):
    frame = bars(rows)
    signals = {"AAA": {frame.index[0]: (Action.BUY, 1.0)}}
    return simulate({"AAA": frame}, signals, RiskLimits(), trend_exit=False, **kw)


RISE = [(100, 100, 100, 100), (100, 101, 99, 100), (105, 135, 104, 132), (132, 133, 120, 125)]


def test_half_the_shares_are_sold_at_the_target_and_the_rest_stays_open():
    res = run(RISE, partial_at=0.30)
    (trade,) = res.trades
    assert trade.reason == "PARTIAL" and trade.exit_price == pytest.approx(130.0)
    assert trade.qty == 25 and res.open_at_end == 1  # 5% of 100,000 at 100: 50 shares, 25 sold
    assert res.equity.iloc[-1] == pytest.approx(100_000 + 25 * 30 + 25 * 25)


def test_the_half_is_sold_only_once_and_at_the_open_when_it_gaps_above():
    gap = RISE[:2] + [(140, 142, 138, 140), (150, 160, 149, 155)]
    res = run(gap, partial_at=0.30)
    assert [(t.reason, t.exit_price) for t in res.trades] == [("PARTIAL", 140.0)]


def test_without_the_option_nothing_is_sold():
    assert run(RISE).trades == []
