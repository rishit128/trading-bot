"""P0 C2: confidence calibration buckets and reliability analysis are pure functions of outcome pairs.

The strategy claims its confidence predicts outcomes; these tests pin the aggregator so the live
before/after ruler measures the model, not the statistician. Perfect calibration must yield ECE~0 and
a degenerate Brier decomposition, while a confidence that rises as outcomes fall must fail the
monotone check and inflate ECE - and low-sample buckets must never be quoted as numbers."""
from src.research.calibration import (
    MIN_SAMPLE,
    bucket_profiles,
    calibration_metrics,
    monotone_in_confidence,
    sufficient_buckets,
)


def _perfect_pairs():
    """Two buckets where per-bucket realized win rate equals per-bucket mean confidence."""
    pairs = [(0.3, 1)] * 3 + [(0.3, 0)] * 7  # bucket [0.0, 0.5): 30% wins, mean conf 0.3
    pairs += [(0.7, 1)] * 7 + [(0.7, 0)] * 3  # bucket [0.5, 1.01): 70% wins, mean conf 0.7
    return pairs


def _inverted_pairs():
    """Confidence rises across buckets while outcomes fall: the worst-calibrated reverse case."""
    pairs = [(0.7, 1)] * 7 + [(0.7, 0)] * 3  # bucket [0.7, 0.8): 70% wins
    pairs += [(0.9, 1)] * 2 + [(0.9, 0)] * 8  # bucket [0.9, 1.01): 20% wins
    return pairs


def test_bucket_boundaries_are_half_open():
    pairs = [(0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1), (0.9, 1), (0.99, 1)]
    prof = bucket_profiles(pairs)
    edges = [p.lo for p in prof]
    assert edges == [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert [p.n for p in prof] == [1, 1, 1, 1, 1, 2]  # 0.9 and 0.99 share the top bucket
    assert prof[0].hi == 0.5  # 0.4 falls in the first bucket
    assert prof[4].lo == 0.8 and prof[5].lo == 0.9  # 0.8 and 0.9 each open their own bucket


def test_perfect_calibration_is_ece_zero_and_degenerate_brier():
    pairs = _perfect_pairs()
    m = calibration_metrics(pairs)
    assert len(m["profiles"]) == 2
    assert m["ece"] < 1e-9
    assert m["reliability"] < 1e-9  # no overconfidence component at all
    assert m["monotone"] is True
    assert abs(m["brier"] - (m["uncertainty"] - m["resolution"])) < 1e-9  # Brier = U - R when reliable
    assert m["check"] is True
    assert abs(m["brier"] - 0.21) < 1e-9  # 0.25 uncertainty - 0.04 resolution


def test_inverted_confidence_fails_monotone_and_lifts_ece():
    pairs = _inverted_pairs()
    m = calibration_metrics(pairs)
    assert m["monotone"] is False  # realized rate falls from 0.7 to 0.2 as confidence rises
    assert abs(m["ece"] - 0.35) < 1e-12  # (10*0 + 10*0.7) / 20
    assert m["reliability"] > 0.1
    assert m["check"] is True


def test_small_buckets_are_insufficient_and_excluded_from_ece():
    pairs = [(0.65, 1)] * 3 + [(0.9, 1)] * 7 + [(0.9, 0)] * 3
    prof = bucket_profiles(pairs)
    tiny = next(p for p in prof if p.lo == 0.6)
    decent = next(p for p in prof if p.lo == 0.9)
    assert tiny.n == 3 and tiny.insufficient
    assert sufficient_buckets([tiny]) == [] and sufficient_buckets([decent])[0] is decent
    full = calibration_metrics(pairs)
    # only the n=10 bucket is quoted into the ECE: |0.9 - 0.7| and 0.2 is the whole (weighted) error
    assert abs(full["ece"] - 0.2) < 1e-12


def test_empty_input_cannot_crash_or_fabricate():
    m = calibration_metrics([])
    assert m["ece"] == 0.0
    assert m["base_rate"] == 0.0
    assert m["monotone"] is True  # vacuous for no buckets
    assert monotone_in_confidence([]) is True
    assert MIN_SAMPLE == 5