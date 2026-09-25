"""The feature study's machinery: no lookahead, correct forward returns, and it finds a planted signal but not noise."""
import numpy as np
import pandas as pd
import pytest

from src.learning.setups import ROUND_TRIP_COST
from src.research.feature_study import BucketResult, build_observations, features, forward_net, study


def make_bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(900, 1100, n).astype(float)}, index=idx)


def test_features_use_only_data_up_to_the_session():
    bars = make_bars()
    full, cut = features(bars), features(bars.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut, check_exact=False)  # the future cannot change the past


def test_forward_return_enters_next_open_exits_close_and_subtracts_cost():
    bars = make_bars(30)
    bars["Open"] = 100.0
    bars["Close"] = 100.0
    bars.iloc[10, bars.columns.get_loc("Close")] = 50.0    # analysed session's own close: irrelevant
    bars.iloc[25, bars.columns.get_loc("Close")] = 110.0   # 15 sessions after index 10
    out = forward_net(bars, 15)
    assert out.iloc[10] == pytest.approx(0.10 - ROUND_TRIP_COST) and np.isnan(out.iloc[-1])


def universe(signal, n_stocks=40, seed=1):
    """Stocks whose 6m momentum feature does (signal) or does not predict the next 20 sessions."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_stocks):
        drift = rng.normal(0, 0.002)
        n = 1500
        r = rng.normal(0, 0.01, n)
        if signal:
            r += drift  # persistent drift: a stock that has been rising keeps rising
        close = 100 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2016-01-01", periods=n)
        out[f"S{i}"] = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                                     "Volume": np.full(n, 1000.0)}, index=idx)
    return out


def test_it_finds_a_planted_persistent_signal_and_not_pure_noise():
    planted = study(build_observations(universe(True)), "2019-12-31", feature_names=("momentum_6m",))
    top = [r for r in planted if r.bucket == 5][0]
    assert top.train_mean > 0 and top.test_mean > 0 and top.candidate
    noise = study(build_observations(universe(False)), "2019-12-31", feature_names=("momentum_6m",))
    assert not any(r.candidate for r in noise)


def test_the_candidate_rule_needs_both_periods():
    ok = BucketResult("f", 5, 0, 1, 10, 0.01, 3.5, 10, 0.01, 3.5)
    assert ok.candidate
    assert not BucketResult("f", 5, 0, 1, 10, 0.01, 3.5, 10, 0.01, 2.9).candidate   # weak test t
    assert not BucketResult("f", 5, 0, 1, 10, 0.01, 3.5, 10, -0.01, 3.5).candidate  # wrong sign in test
    assert not BucketResult("f", 5, 0, 1, 10, -0.01, 3.5, 10, 0.01, 3.5).candidate  # wrong sign in train
