import math

import pytest

from src.research.significance import (
    deflated_sharpe,
    deflated_t,
    detectable_effect,
    expected_max_null,
    units_needed,
)


def test_expected_max_matches_the_published_table():
    # ml4t-deflated-sharpe: E[max] in units of sd is about 1.6 for 10 trials, 2.3 for 50, 2.5 for 100, 3.1 for 500
    assert expected_max_null(1) == 0.0
    for n, want in ((10, 1.57), (50, 2.25), (100, 2.53), (500, 3.04)):
        assert expected_max_null(n) == pytest.approx(want, abs=0.03)
    assert expected_max_null(10, sd=0.1) == pytest.approx(0.1 * expected_max_null(10))


def test_one_trial_is_a_plain_one_sided_test():
    d = deflated_t(1.645, 1)
    assert d.expected_max == 0.0 and d.probability == pytest.approx(0.95, abs=1e-3)


def test_more_trials_lower_the_probability():
    assert deflated_t(2.0, 1).probability > deflated_t(2.0, 10).probability > deflated_t(2.0, 60).probability


def test_deflated_sharpe_uses_the_spread_of_the_estimates():
    # per-trade Sharpe 0.1 on 1,000 trades: strong alone; the best of 50 variants spread by 0.03 still clears ~0.07
    alone = deflated_sharpe(0.1, 1000, 1, 0.03)
    searched = deflated_sharpe(0.1, 1000, 50, 0.03)
    assert alone.probability > 0.99 and searched.probability < alone.probability
    assert searched.expected_max == pytest.approx(0.03 * expected_max_null(50))
    # fat tails widen the standard error, so the same Sharpe is less convincing
    assert deflated_sharpe(0.1, 1000, 50, 0.03, kurtosis=10).probability < searched.probability


def test_deflated_sharpe_rejects_bad_input():
    with pytest.raises(ValueError):
        deflated_sharpe(0.1, 1, 10, 0.03)
    with pytest.raises(ValueError):
        expected_max_null(0)


def test_sample_size_and_detectable_effect_are_inverse():
    n = units_needed(0.0025, 0.01)  # +0.25% when single dates spread by 1%: (2.486 * 4)^2 = 98.9 -> 99
    assert n == 99
    assert detectable_effect(n, 0.01) <= 0.0025 < detectable_effect(n - 1, 0.01)
    assert math.isclose(detectable_effect(100, 0.01, alpha=0.05, power=0.5), 1.6449 * 0.001, rel_tol=1e-3)
    with pytest.raises(ValueError):
        units_needed(0, 0.01)
