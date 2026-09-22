import pytest

from src.config import RiskLimits
from src.engine.risk_engine import Portfolio, RiskEngine

# min_position_pct == max_position_pct here: a flat 5% regardless of confidence, so the existing quantity assertions
# below (written before confidence-scaled sizing existed) still hold. Scaling itself is tested separately below.
LIM = RiskLimits(min_position_pct=0.05, max_position_pct=0.05, max_portfolio_exposure_pct=0.80, max_daily_loss_pct=0.02,
                 max_drawdown_pct=0.20, min_confidence=0.6)
ENGINE = RiskEngine(LIM)


def pf(**kw):
    base = dict(cash=100_000.0, equity=100_000.0, positions={}, position_qty={},
                start_of_day_equity=100_000.0, peak_equity=100_000.0)
    base.update(kw)
    return Portfolio(**base)


def test_buy_is_capped_at_position_limit():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 150.0, pf())
    assert d.approved and d.quantity == 33  # 5% of 100k = 5000 -> 33 shares


def test_position_limit_counts_existing_holding_at_its_own_value():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 150.0, pf(positions={"AAPL": 4000.0}, position_qty={"AAPL": 20}))
    assert d.approved and d.quantity == 6  # only $1000 of room left


def test_buy_rejected_when_position_full():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 150.0, pf(positions={"AAPL": 5000.0}, position_qty={"AAPL": 33}))
    assert not d.approved and d.quantity == 0


def test_exposure_uses_dollar_values_not_share_counts():
    # Original plan summed share counts and multiplied by the new symbol's price. 79k deployed elsewhere:
    positions = {"MSFT": 40_000.0, "GOOGL": 39_000.0}
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 100.0, pf(positions=positions))
    assert d.approved and d.quantity == 10  # only $1000 of the 80% exposure budget remains


def test_exposure_limit_blocks_buy():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 100.0, pf(positions={"MSFT": 80_000.0}))
    assert not d.approved


def test_buy_limited_by_cash():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 100.0, pf(cash=250.0))
    assert d.approved and d.quantity == 2


def test_low_confidence_rejected():
    assert not ENGINE.evaluate("BUY", 0.59, "AAPL", 100.0, pf()).approved


def test_hold_never_trades():
    assert not ENGINE.evaluate("HOLD", 1.0, "AAPL", 100.0, pf()).approved


def test_daily_loss_limit_halts_buys():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 100.0, pf(equity=97_900.0, cash=97_900.0))
    assert not d.approved and "daily loss" in d.reason


def test_drawdown_limit_halts_buys():
    d = ENGINE.evaluate("BUY", 0.9, "AAPL", 100.0, pf(equity=79_000.0, cash=79_000.0, start_of_day_equity=79_000.0))
    assert not d.approved and "drawdown" in d.reason


def test_sell_closes_full_position_and_ignores_daily_loss():
    d = ENGINE.evaluate("SELL", 0.9, "AAPL", 100.0,
                        pf(equity=90_000.0, positions={"AAPL": 3000.0}, position_qty={"AAPL": 30}))
    assert d.approved and d.quantity == 30


def test_sell_without_position_rejected():
    d = ENGINE.evaluate("SELL", 0.9, "AAPL", 100.0, pf())
    assert not d.approved and "shorting" in d.reason


def test_invalid_price_rejected():
    assert not ENGINE.evaluate("BUY", 0.9, "AAPL", 0.0, pf()).approved


def test_halt_reason_kinds_are_stable_and_none_when_healthy():
    assert ENGINE.halt_reason(pf()) is None
    kind, _ = ENGINE.halt_reason(pf(equity=97_000.0, cash=97_000.0))
    assert kind == "daily_loss"
    kind2, _ = ENGINE.halt_reason(pf(equity=96_000.0, cash=96_000.0))
    assert kind2 == "daily_loss"  # the message changes with the percentage, the kind must not
    kind3, _ = ENGINE.halt_reason(pf(equity=79_000.0, cash=79_000.0, start_of_day_equity=79_000.0))
    assert kind3 == "drawdown"


def test_new_symbol_rejected_when_max_open_positions_reached():
    limits = RiskLimits(max_open_positions=2)
    held = {"A": 1000.0, "B": 1000.0}
    p = pf(positions=held, position_qty={"A": 10, "B": 10})
    d = RiskEngine(limits).evaluate("BUY", 0.9, "C", 100.0, p)
    assert not d.approved and "max open positions" in d.reason


def test_existing_position_can_still_be_topped_up_at_max_open_positions():
    limits = RiskLimits(max_open_positions=2)
    p = pf(positions={"A": 1000.0, "B": 1000.0}, position_qty={"A": 10, "B": 10})
    assert RiskEngine(limits).evaluate("BUY", 0.9, "A", 100.0, p).approved


def test_sells_are_never_blocked_by_max_open_positions():
    limits = RiskLimits(max_open_positions=1)
    p = pf(positions={"A": 1000.0, "B": 1000.0}, position_qty={"A": 10, "B": 10})
    assert RiskEngine(limits).evaluate("SELL", 0.9, "A", 100.0, p).approved


# ---------------------------------------------------------------- confidence-scaled position size
SCALED = RiskEngine(RiskLimits(min_position_pct=0.10, max_position_pct=0.25, max_portfolio_exposure_pct=0.80,
                               max_daily_loss_pct=0.02, max_drawdown_pct=0.20, min_confidence=0.60))


def test_position_pct_is_the_floor_at_the_minimum_confidence():
    assert SCALED.position_pct(0.60) == 0.10


def test_position_pct_is_the_ceiling_at_full_confidence():
    assert SCALED.position_pct(1.0) == 0.25


def test_position_pct_scales_linearly_in_between():
    assert SCALED.position_pct(0.80) == pytest.approx(0.175)  # halfway from 0.60 to 1.0 -> halfway from 10% to 25%


def test_a_higher_confidence_signal_gets_a_bigger_position_than_a_lower_one():
    weak = SCALED.evaluate("BUY", 0.61, "AAA", 100.0, pf())
    strong = SCALED.evaluate("BUY", 0.99, "AAA", 100.0, pf())
    assert weak.approved and strong.approved and strong.quantity > weak.quantity


def test_approval_reason_reports_the_sizing_percent_and_confidence_used():
    d = SCALED.evaluate("BUY", 1.0, "AAA", 100.0, pf())
    assert "sized at 25.0% for confidence 1.00" in d.reason


def test_scaling_never_exceeds_the_exposure_or_cash_limits():
    d = SCALED.evaluate("BUY", 1.0, "AAA", 100.0, pf(cash=1_000.0))  # 25% of equity would be 25,000, cash caps it
    assert d.approved and d.quantity == 10
