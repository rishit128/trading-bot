"""Data-honesty checks run at fetch time (P0). The indicators assume one continuous, comparably-scaled price series;
a split/bonus/rights event in raw history breaks that assumption (MA/RSI/ATR see a fake -50% move). Yahoo's
`auto_adjust=True` restates the whole history, so a correct daily feed should have NO split-like overnight gaps. Any
bar that still shows one (or leaves it ambiguous with a real crash) is flagged for the caller instead of being fed to
the indicators silently.

A real one-day crash also gaps down, but it comes with an expanding intraday range; a split restates the symbol's
price scale, so the new bar's own high-low range is ordinary. A bar is flagged only when both hold: the overnight
close-to-close move exceeds `threshold` AND the bar's own range is NOT unusually wide."""
import logging

import pandas as pd

log = logging.getLogger(__name__)

SPLIT_LOOKALIKE_RATIO = 0.30  # an overnight -30% / +43% close-to-close that is not a crash
RANGE_EXPANSION_MULT = 3.0  # the bar's high-low may not exceed this times the 20-day median range


def split_like_gaps(df: pd.DataFrame, threshold: float = SPLIT_LOOKALIKE_RATIO) -> pd.DataFrame:
    """DataFrame of the same shape with `is_split_like` (bool) and `overnight_ret` (fraction, NaN elsewhere)."""
    close = df["Close"].astype(float)
    ret = close / close.shift(1) - 1
    hits = ret.abs() >= threshold
    if len(df) >= 2 and not hits.any():
        return pd.DataFrame({"is_split_like": False, "overnight_ret": float("nan")}, index=df.index)
    hi, lo = df["High"].astype(float), df["Low"].astype(float)
    rng = hi - lo
    med = rng.rolling(20, min_periods=5).median().shift(1)
    wide = rng > RANGE_EXPANSION_MULT * med
    flag = hits & ~wide
    if not flag.any():
        return pd.DataFrame({"is_split_like": False, "overnight_ret": float("nan")}, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["is_split_like"] = flag
    out["overnight_ret"] = ret.where(flag)
    return out


def check_bars_honesty(df: pd.DataFrame) -> list:
    """Human-readable findings for a fetched daily frame; the fetcher logs these. Pure report, no mutation."""
    issues = []
    if df is None or df.empty:
        return issues
    if not df.index.is_monotonic_increasing:
        issues.append("index is not monotonic (bars misordered)")
    dupes = int(df.index.duplicated().sum())
    if dupes:
        issues.append(f"{dupes} duplicate timestamps")
    if (df["Volume"].astype(float) <= 0).any():
        issues.append("zero or negative volume bars present")
    for idx, row in split_like_gaps(df).iterrows():
        if row["is_split_like"]:
            issues.append(f"{idx.date()} split-like overnight {row['overnight_ret']:+.1%}; "
                          "corporate action or unadjusted mix - indicators for this range are unreliable")
    return issues