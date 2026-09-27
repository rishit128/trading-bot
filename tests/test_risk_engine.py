"""The deterministic risk gate: example-based checks below, then property-style tests over thousands of seeded random
accounts (the risk engine is the only thing standing between an AI's opinion and an order, so its guarantees are
checked broadly, not just on a few examples)."""
import random
from datetime import date, timedelta

import pytest

from src.config import RiskLimits
from src.engine.costs import india_delivery_fees
from src.engine.risk_engine import Portfolio, RiskEngine, drawdown_pause

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


# ---------------------------------------------------------------- drawdown pause (cool-off instead of a permanent halt)
D0 = date(2026, 1, 1)


def test_drawdown_pause_starts_the_clock_only_while_the_limit_is_breached():
    assert drawdown_pause(100.0, 95.0, None, D0, 0.20, 30) == (100.0, None)           # only -5%: nothing pending
    assert drawdown_pause(100.0, 79.0, None, D0, 0.20, 30) == (100.0, D0)              # -21%: halt begins today
    assert drawdown_pause(100.0, 79.0, D0, D0 + timedelta(days=29), 0.20, 30) == (100.0, D0)  # still cooling off


def test_drawdown_pause_rebases_the_peak_after_the_cool_off_and_resets_on_recovery():
    assert drawdown_pause(100.0, 79.0, D0, D0 + timedelta(days=30), 0.20, 30) == (79.0, None)
    assert drawdown_pause(100.0, 90.0, D0, D0 + timedelta(days=5), 0.20, 30) == (100.0, None)  # recovered: clock cleared


def test_drawdown_pause_zero_days_keeps_the_permanent_halt():
    assert drawdown_pause(100.0, 50.0, D0, D0 + timedelta(days=999), 0.20, 0) == (100.0, D0)


def test_risk_limits_reject_a_negative_pause():
    with pytest.raises(ValueError, match="drawdown_pause_days"):
        RiskLimits(drawdown_pause_days=-1)


# ---------------------------------------------------------------- fee drag: tiny positions are not worth their fixed charges
def _engine(fee_drag=0.04, fees=india_delivery_fees):
    return RiskEngine(RiskLimits(min_position_pct=0.05, max_position_pct=0.05, max_fee_drag_pct=fee_drag), fees=fees)


def test_a_one_share_position_whose_fixed_sale_charge_dwarfs_it_is_refused():
    squeezed = Portfolio(cash=400, equity=20_000)  # only Rs 400 of cash/exposure room left: one Rs 363 share fits
    tiny = _engine().evaluate("BUY", 0.8, "TINY", 363.0, squeezed)  # the Rs 15.93 sale charge alone is ~4.4%
    assert not tiny.approved and "fees would cost" in tiny.reason
    room = Portfolio(cash=20_000, equity=20_000)
    fine = _engine().evaluate("BUY", 0.8, "FINE", 950.0, room)  # Rs 950: fees ~1.9%
    assert fine.approved and fine.quantity == 1


def test_the_fee_rule_can_be_turned_off_or_is_skipped_without_a_fee_schedule():
    squeezed = Portfolio(cash=400, equity=20_000)
    assert _engine(fee_drag=0.0).evaluate("BUY", 0.8, "TINY", 363.0, squeezed).approved
    assert _engine(fees=None).evaluate("BUY", 0.8, "TINY", 363.0, squeezed).approved


def test_the_fee_rule_never_blocks_a_sell():
    held = Portfolio(cash=0, equity=20_000, positions={"TINY": 363.0}, position_qty={"TINY": 1})
    assert _engine().evaluate("SELL", 1.0, "TINY", 363.0, held).approved  # you must always be able to exit


def test_risk_limits_validate_the_fee_cap_and_the_pipeline_uses_the_brokers_fee_schedule(tmp_path):
    from tests.test_llm_and_pipeline import FakeBroker, make_pipeline

    with pytest.raises(ValueError, match="max_fee_drag_pct"):
        RiskLimits(max_fee_drag_pct=1.0)
    broker = FakeBroker(Portfolio(20_000.0, 20_000.0, {}, {}, 20_000.0))
    broker.fees = india_delivery_fees
    pipe, _, _ = make_pipeline(tmp_path, broker=broker)
    assert pipe.risk.fees is india_delivery_fees


# ==================================================================================================================
# Property-style tests over thousands of seeded random accounts (see the module docstring).
CASES = 4000


def random_limits(rng):
    max_pos = rng.choice([0.03, 0.05, 0.1, 0.25])
    min_pos = rng.choice([max_pos, max_pos / 2, max_pos / 4])
    return RiskLimits(min_position_pct=min_pos, max_position_pct=max_pos,
                      max_portfolio_exposure_pct=rng.choice([0.5, 0.8, 1.0]), max_daily_loss_pct=rng.choice([0.02, 0.05]),
                      max_drawdown_pct=rng.choice([0.1, 0.2, 0.3]), min_confidence=rng.choice([0.5, 0.6, 0.7]),
                      max_open_positions=rng.choice([1, 3, 5, 10]), max_fee_drag_pct=rng.choice([0.0, 0.04]))


def random_portfolio(rng):
    equity = rng.uniform(5_000, 2_000_000)
    names = [f"S{i}" for i in range(rng.randint(0, 9))]
    values = {n: equity * rng.uniform(0.005, 0.12) for n in names}
    invested = sum(values.values())
    cash = max(0.0, equity - invested) * rng.choice([1.0, 0.8, 0.5, 0.03, 0.01])  # some accounts are nearly out of cash
    prices = {n: equity * rng.uniform(0.0005, 0.05) for n in names}
    qty = {n: max(1, int(values[n] / prices[n])) for n in names}
    sod = equity * rng.choice([1.0, 1.0, 1.03, 0.99, 1.1])
    peak = equity * rng.choice([1.0, 1.0, 1.05, 1.15, 1.4])
    return Portfolio(cash=cash, equity=equity, positions=values, position_qty=qty, start_of_day_equity=sod, peak_equity=peak)


def scenarios(seed):
    rng = random.Random(seed)
    for _ in range(CASES):
        limits, p = random_limits(rng), random_portfolio(rng)
        symbol = rng.choice(list(p.positions) + ["NEW1", "NEW2"])
        price = p.equity * rng.uniform(0.0003, 0.06)  # from a fraction of a percent to a share worth 6% of the account
        yield limits, p, symbol, price, rng.uniform(0.3, 1.0), rng.choice(["BUY", "BUY", "BUY", "SELL", "HOLD"])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_an_approved_buy_never_breaks_a_limit(seed):
    approved = 0
    for limits, p, symbol, price, conf, action in scenarios(seed):
        engine = RiskEngine(limits, fees=india_delivery_fees)
        d = engine.evaluate(action, conf, symbol, price, p)
        if action != "BUY":
            continue
        if not d.approved:
            assert d.quantity == 0
            continue
        approved += 1
        value = d.quantity * price
        existing = p.positions.get(symbol, 0.0)
        assert d.quantity >= 1
        assert conf >= limits.min_confidence
        assert value <= p.cash + 1e-6                                                       # never spends money it lacks
        assert existing + value <= p.equity * engine.position_pct(conf) + price            # size follows confidence
        assert existing + value <= p.equity * limits.max_position_pct + price              # ... and the position cap
        assert sum(p.positions.values()) + value <= p.equity * limits.max_portfolio_exposure_pct + price
        assert symbol in p.positions or len(p.positions) < limits.max_open_positions       # never a slot beyond the cap
        assert engine.halt_reason(p) is None                                                # never buys while halted
        if limits.max_fee_drag_pct > 0:
            drag = (india_delivery_fees("BUY", value) + india_delivery_fees("SELL", value)) / value
            assert drag <= limits.max_fee_drag_pct + 1e-9                                   # never a fee-dominated position
    assert approved > 150  # the generator really produces approvable cases; a vacuous pass would prove nothing


@pytest.mark.parametrize("seed", [4, 5])
def test_sells_only_close_what_is_held_and_holds_never_trade(seed):
    sells = 0
    for limits, p, symbol, price, conf, action in scenarios(seed):
        d = RiskEngine(limits).evaluate(action, conf, symbol, price, p)
        if action == "HOLD":
            assert not d.approved and d.quantity == 0
        elif action == "SELL":
            held = p.position_qty.get(symbol, 0)
            if conf < limits.min_confidence:
                assert not d.approved
            elif d.approved:
                sells += 1
                assert held > 0 and d.quantity == held  # a full close of a real position, never a short
            else:
                assert held == 0 or conf < limits.min_confidence
    assert sells > 100


def test_position_size_never_shrinks_as_confidence_grows_and_stays_within_its_bounds():
    rng = random.Random(6)
    for _ in range(500):
        limits = random_limits(rng)
        engine, last = RiskEngine(limits), -1.0
        for conf in [i / 100 for i in range(0, 101)]:
            pct = engine.position_pct(conf)
            assert limits.min_position_pct - 1e-12 <= pct <= limits.max_position_pct + 1e-12
            assert pct >= last - 1e-12
            last = pct


def test_a_halt_is_reported_exactly_when_a_limit_is_breached():
    rng = random.Random(7)
    for _ in range(CASES):
        limits, p = random_limits(rng), random_portfolio(rng)
        halt = RiskEngine(limits).halt_reason(p)
        day_loss = (p.start_of_day_equity - p.equity) / p.start_of_day_equity
        drawdown = (p.peak_equity - p.equity) / p.peak_equity
        breached = day_loss >= limits.max_daily_loss_pct or drawdown >= limits.max_drawdown_pct
        assert (halt is not None) == breached
        if halt:
            assert halt[0] in ("daily_loss", "drawdown")


def test_the_drawdown_pause_only_ever_rebases_downward_after_the_full_cool_off():
    rng, today = random.Random(8), date(2026, 1, 1)
    for _ in range(CASES):
        peak, equity = rng.uniform(1_000, 2e6), rng.uniform(500, 2e6)
        limit, pause = rng.choice([0.1, 0.2, 0.3]), rng.choice([0, 7, 30])
        since = rng.choice([None, today - timedelta(days=rng.randint(0, 60))])
        new_peak, new_since = drawdown_pause(peak, equity, since, today, limit, pause)
        in_drawdown = (peak - equity) / peak >= limit
        if pause <= 0:
            assert (new_peak, new_since) == (peak, since)
        elif not in_drawdown:
            assert new_peak == peak and new_since is None      # recovered: nothing pending, peak untouched
        elif new_peak != peak:
            assert new_peak == equity and new_since is None    # rebased to today's equity only ...
            assert since is not None and (today - since).days >= pause  # ... after a full cool-off
        else:
            assert new_since is not None                       # still cooling off, and the clock is running
