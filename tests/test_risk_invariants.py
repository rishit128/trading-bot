"""Property-style tests for the deterministic risk gate. The risk engine is the only thing standing between an AI's
opinion and an order, so its guarantees are checked over thousands of seeded random accounts, not just a few examples."""
import random
from datetime import date, timedelta

import pytest

from src.config import RiskLimits
from src.engine.costs import india_delivery_fees
from src.engine.risk_engine import Portfolio, RiskEngine, drawdown_pause

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
        limits, pf = random_limits(rng), random_portfolio(rng)
        symbol = rng.choice(list(pf.positions) + ["NEW1", "NEW2"])
        price = pf.equity * rng.uniform(0.0003, 0.06)  # from a fraction of a percent to a share worth 6% of the account
        yield limits, pf, symbol, price, rng.uniform(0.3, 1.0), rng.choice(["BUY", "BUY", "BUY", "SELL", "HOLD"])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_an_approved_buy_never_breaks_a_limit(seed):
    approved = 0
    for limits, pf, symbol, price, conf, action in scenarios(seed):
        engine = RiskEngine(limits, fees=india_delivery_fees)
        d = engine.evaluate(action, conf, symbol, price, pf)
        if action != "BUY":
            continue
        if not d.approved:
            assert d.quantity == 0
            continue
        approved += 1
        value = d.quantity * price
        existing = pf.positions.get(symbol, 0.0)
        assert d.quantity >= 1
        assert conf >= limits.min_confidence
        assert value <= pf.cash + 1e-6                                                      # never spends money it lacks
        assert existing + value <= pf.equity * engine.position_pct(conf) + price            # size follows confidence
        assert existing + value <= pf.equity * limits.max_position_pct + price             # ... and the position cap
        assert sum(pf.positions.values()) + value <= pf.equity * limits.max_portfolio_exposure_pct + price
        assert symbol in pf.positions or len(pf.positions) < limits.max_open_positions     # never a slot beyond the cap
        assert engine.halt_reason(pf) is None                                               # never buys while halted
        if limits.max_fee_drag_pct > 0:
            drag = (india_delivery_fees("BUY", value) + india_delivery_fees("SELL", value)) / value
            assert drag <= limits.max_fee_drag_pct + 1e-9                                   # never a fee-dominated position
    assert approved > 150  # the generator really produces approvable cases; a vacuous pass would prove nothing


@pytest.mark.parametrize("seed", [4, 5])
def test_sells_only_close_what_is_held_and_holds_never_trade(seed):
    sells = 0
    for limits, pf, symbol, price, conf, action in scenarios(seed):
        d = RiskEngine(limits).evaluate(action, conf, symbol, price, pf)
        if action == "HOLD":
            assert not d.approved and d.quantity == 0
        elif action == "SELL":
            held = pf.position_qty.get(symbol, 0)
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
        limits, pf = random_limits(rng), random_portfolio(rng)
        halt = RiskEngine(limits).halt_reason(pf)
        day_loss = (pf.start_of_day_equity - pf.equity) / pf.start_of_day_equity
        drawdown = (pf.peak_equity - pf.equity) / pf.peak_equity
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
