"""The shared strategy rules: boundary behaviour, scalar/vector agreement, and that every consumer really uses them."""

import numpy as np
import pandas as pd
import pytest

from src.engine import rules
from src.engine.strategy import TREND_EXIT_PREFIX, trend_exit_decision
from src.data.indicators import Snapshot


def snap(price=100.0, ma50=98.0, ma200=95.0, rsi=60.0):
    return Snapshot("X", price, ma50, ma200, rsi, 1000)


@pytest.mark.parametrize("args,expected", [
    ((100, 98, 95, 60), True),
    ((100, 98, 95, 69.99), True),
    ((100, 98, 95, 70.0), False),   # exactly overbought is out
    ((100, 100, 95, 60), False),    # price equal to MA50 is not "above"
    ((100, 98, 98, 60), False),     # MA50 equal to MA200 is not a bullish stack
    ((97, 98, 95, 60), False),      # below MA50
    ((100, 94, 95, 60), False),     # MA50 below MA200
])
def test_the_entry_filter_boundaries(args, expected):
    assert rules.entry_filter(*args) is expected


@pytest.mark.parametrize("price,ma200,expected", [(94.99, 95.0, True), (95.0, 95.0, False), (100.0, 95.0, False)])
def test_the_trend_exit_boundary(price, ma200, expected):
    assert rules.trend_broken(price, ma200) is expected


def test_scalar_and_vector_forms_agree_on_random_data():
    rng = np.random.default_rng(3)
    close, ma50, ma200 = (rng.uniform(80, 120, 5000) for _ in range(3))
    rsi = rng.uniform(20, 90, 5000)
    for ties in (close, ma50):  # force exact ties so the strict/non-strict edges are exercised
        ties[::50] = ma200[::50]
    mask = rules.entry_filter_mask(close, ma50, ma200, rsi)
    exit_mask = rules.trend_broken_mask(close, ma200)
    for i in range(5000):
        assert bool(mask[i]) == rules.entry_filter(close[i], ma50[i], ma200[i], rsi[i])
        assert bool(exit_mask[i]) == rules.trend_broken(close[i], ma200[i])
    series = rules.entry_filter_mask(pd.Series(close), pd.Series(ma50), pd.Series(ma200), pd.Series(rsi))
    assert list(series) == list(mask)  # pandas in, pandas out, same answers


def test_nan_inputs_never_pass_the_entry_filter_or_trigger_the_exit():
    nan = float("nan")
    assert not rules.entry_filter(nan, 98, 95, 60) and not rules.entry_filter(100, nan, 95, 60)
    assert not rules.trend_broken(nan, 95) and not rules.trend_broken(100, nan)


def test_the_trend_exit_decision_only_applies_to_a_held_stock_below_its_200_day_average():
    below = snap(price=90.0, ma50=92.0, ma200=95.0)
    d = trend_exit_decision(below, held_qty=100)
    assert d.action == "SELL" and d.confidence == 1.0 and d.reasoning.startswith(TREND_EXIT_PREFIX)
    assert "90.00" in d.reasoning and "95.00" in d.reasoning
    assert trend_exit_decision(below, held_qty=0) is None              # nothing to sell
    assert trend_exit_decision(below, held_qty=100, enabled=False) is None
    assert trend_exit_decision(snap(), held_qty=100) is None            # trend intact


def test_every_consumer_of_the_rules_uses_this_module_not_its_own_copy():
    """Structural guard: the literal comparisons must not creep back into other modules."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    banned = [r"price\s*>\s*(?:snap\.|s\.)?ma50", r"ma50\s*>\s*(?:snap\.|s\.)?ma200", r"(?:price|close)\s*<\s*(?:snap\.|s\.)?ma200",
              r"rsi\s*<\s*70", r"rsi\s*>=\s*70"]
    offenders = []
    for path in list((root / "src").rglob("*.py")) + [root / "main.py"]:
        if path.name == "rules.py" or "indicators" in path.name:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#")[0]
            if any(re.search(pattern, code) for pattern in banned) and not code.strip().startswith(('"', "'", "f\"", "f'")):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()[:90]}")
    assert not offenders, "the entry/exit rule is written out again outside src/engine/rules.py:\n" + "\n".join(offenders)
