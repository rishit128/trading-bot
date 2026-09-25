"""Confidence calibration buckets and reliability analysis.

Calibration asks a strict question of the confidence score: when the model says 70%, does it win 70% of
the time? Buckets make that judgment sample-aware - a bucket with fewer than `MIN_SAMPLE` outcomes is
"insufficient" rather than a quoted number - and the aggregated ECE, the Brier-score decomposition and
the monotone check give the before/after ruler in `scripts/analyze_calibration.py` a single number to
track. Everything here is a pure function of (confidence, outcome) pairs so the same code powers tests,
backtests and the live paper database.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

MIN_SAMPLE = 5

# Bucket edges for approved-BUY confidence. The first zip starts at 0.0 (anything below min_confidence
# never trades, but must still be classifiable); the 1.01 sentinel closes the [0.9, 1.01) top bucket.
DEFAULT_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


@dataclass(frozen=True)
class BucketProfile:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    realized_rate: float  # frequency of wins in the bucket
    abs_error: float  # |realized_rate - mean_confidence|

    @property
    def insufficient(self) -> bool:
        return self.n < MIN_SAMPLE


def _edges(edges):
    return tuple(edges)


def bucket_profiles(pairs, edges: tuple = DEFAULT_EDGES):
    """Group (confidence, win=0/1) pairs by confidence edge.

    Every pair lands in exactly one bucket: `lo <= confidence < hi`. Buckets with no pairs are
    skipped, so the caller never sees an empty phantom bucket."""
    profiles = []
    edges_t = _edges(edges)
    for lo, hi in zip((0.0,) + edges_t[:-1], edges_t):
        in_bucket = [(c, b) for c, b in pairs if lo <= c < hi]
        if not in_bucket:
            continue
        realized = mean(b for _, b in in_bucket)
        mean_conf = mean(c for c, _ in in_bucket)
        profiles.append(BucketProfile(lo, hi, len(in_bucket), mean_conf, realized,
                                      abs(realized - mean_conf)))
    return profiles


def sufficient_buckets(profiles, min_sample: int = MIN_SAMPLE):
    """Buckets with enough outcomes to be quoted."""
    return [p for p in profiles if p.n >= min_sample]


def expected_calibration_error(profiles, min_sample: int = MIN_SAMPLE) -> float:
    """Sample-weighted mean |realized - mean_confidence| over buckets with enough sample.

    0.0 means perfectly calibrated; a confidence that means nothing sits near the historical
    dispersion of win rates. Untrustworthy micro-buckets are excluded so one lucky trade can't move it."""
    good = sufficient_buckets(profiles, min_sample)
    if not good:
        return 0.0
    total = sum(p.n for p in good)
    return sum(p.n * p.abs_error for p in good) / total


def brier_score(pairs) -> float:
    """Mean squared error between stated confidence and binary outcome (lower is better)."""
    return mean((c - b) ** 2 for c, b in pairs) if pairs else 0.0


def brier_decomposition(pairs, edges: tuple = DEFAULT_EDGES):
    """(reliability, resolution, uncertainty) of the Brier score.

    Brier == reliability + uncertainty - resolution. A calibrated-but-useless model has zero
    reliability (no overconfidence) and zero resolution (confidence adds nothing over the base rate)."""
    if not pairs:
        return 0.0, 0.0, 0.0
    base = mean(b for _, b in pairs)
    uncertainty = base * (1 - base)
    reliability = resolution = 0.0
    n = len(pairs)
    edges_t = _edges(edges)
    for lo, hi in zip((0.0,) + edges_t[:-1], edges_t):
        in_bucket = [(c, b) for c, b in pairs if lo <= c < hi]
        if not in_bucket:
            continue
        nk = len(in_bucket)
        ok = mean(b for _, b in in_bucket)
        avg_conf = mean(c for c, _ in in_bucket)
        reliability += (nk / n) * (avg_conf - ok) ** 2
        resolution += (nk / n) * (ok - base) ** 2
    return reliability, resolution, uncertainty


def monotone_in_confidence(pairs, edges: tuple = DEFAULT_EDGES, min_sample: int = MIN_SAMPLE) -> bool:
    """Whether per-bucket realized win rate rises with confidence across sufficient buckets.

    `buckets()` may have skipped buckets in between; this checks the buckets that exist, in order."""
    good = sufficient_buckets(bucket_profiles(pairs, edges), min_sample)
    return all(r.realized_rate <= n.realized_rate for r, n in zip(good, good[1:]))


def calibration_metrics(pairs, edges: tuple = DEFAULT_EDGES, min_sample: int = MIN_SAMPLE) -> dict:
    """One dict with everything a caller needs for a reliability line, or for a test assertion."""
    profiles = bucket_profiles(pairs, edges)
    reliability, resolution, uncertainty = brier_decomposition(pairs, edges)
    brier = brier_score(pairs)
    return {
        "profiles": profiles,
        "ece": expected_calibration_error(profiles, min_sample),
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "monotone": monotone_in_confidence(pairs, edges, min_sample),
        "base_rate": mean(b for _, b in pairs) if pairs else 0.0,
        "check": abs(brier - (reliability + uncertainty - resolution)) < 1e-9,
    }