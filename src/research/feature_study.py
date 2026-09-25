"""Do simple indicator buckets predict forward returns, out of sample? A report-only study; it adopts nothing.

PRE-DECLARED before any result was seen. Observations: every 5th session per stock (limits overlap), feature values known at
that session's close, outcome = entry at the next open, exit at the close `horizon` sessions later, net of the round-trip
cost, measured EXCESS over the average of all stocks on the same date (so a rising market is not an edge). Bucket edges
(quintiles) come from the TRAIN period only. A bucket is a CANDIDATE only if all hold:
  1. train (<= split): mean excess > 0 and |t| >= 3
  2. test (> split):   mean excess > 0 and t >= 3, same bucket, edges frozen
t-statistics are computed over per-DATE average excess returns (stocks on one date move together, so they are not
independent). The bar is 3 rather than 2 because 6 features x 5 buckets = 30 comparisons are looked at. A candidate is a
lead to test properly, never an instruction to change the bot.

Honest limits: today's Nifty 500 members only (survivorship bias flatters winners), adjusted daily bars, costs approximate."""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.learning.setups import ROUND_TRIP_COST

FEATURES = ("rsi", "adx", "atr_pct", "volume_ratio", "dist_ma50", "momentum_6m")
STEP = 5
T_BAR = 3.0


def _wilder(x: pd.Series, n: int = 14) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(bars: pd.DataFrame) -> pd.DataFrame:
    """The six features for every session of one stock, using only data up to and including that session."""
    c, h, lo, v = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    delta = c.diff()
    rsi = 100 - 100 / (1 + _wilder(delta.clip(lower=0)) / _wilder((-delta).clip(lower=0)).replace(0, np.nan))
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr)
    up, down = h.diff(), -lo.diff()
    plus = _wilder(pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=c.index)) / atr * 100
    minus = _wilder(pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=c.index)) / atr * 100
    adx = _wilder(((plus - minus).abs() / (plus + minus).replace(0, np.nan)) * 100)
    return pd.DataFrame({"rsi": rsi, "adx": adx, "atr_pct": atr / c * 100, "volume_ratio": v / v.rolling(20).mean().replace(0, np.nan),
                         "dist_ma50": (c / c.rolling(50).mean() - 1) * 100, "momentum_6m": c / c.shift(126) - 1}, index=c.index)


def forward_net(bars: pd.DataFrame, horizon: int, cost: float = ROUND_TRIP_COST) -> pd.Series:
    """Per session: entry at the next open, exit at the close `horizon` sessions later, net of cost (NaN if unknown)."""
    return bars["Close"].shift(-horizon) / bars["Open"].shift(-1) - 1 - cost


def build_observations(universe: Dict[str, pd.DataFrame], horizon: int = 20, step: int = STEP) -> pd.DataFrame:
    """Long table: date, symbol, the features, net return and the same-date excess over the universe average."""
    parts: List[pd.DataFrame] = []
    for symbol, bars in universe.items():
        if len(bars) < 260:
            continue
        frame = features(bars)
        frame["net"] = forward_net(bars, horizon)
        frame["symbol"] = symbol
        parts.append(frame.iloc[::step].dropna())
    if not parts:
        return pd.DataFrame()
    obs = pd.concat(parts)
    obs.index.name = "date"
    obs = obs.reset_index()
    obs["date"] = pd.to_datetime(obs["date"]).dt.tz_localize(None)
    obs["excess"] = obs["net"] - obs.groupby("date")["net"].transform("mean")
    return obs


@dataclass(frozen=True)
class BucketResult:
    feature: str
    bucket: int
    low: float
    high: float
    train_n: int
    train_mean: float
    train_t: float
    test_n: int
    test_mean: float
    test_t: float

    @property
    def candidate(self) -> bool:
        return self.train_mean > 0 and self.train_t >= T_BAR and self.test_mean > 0 and self.test_t >= T_BAR


def _stat(frame: pd.DataFrame) -> tuple:
    """(n, mean excess, t over per-date means) of a set of observations."""
    if frame.empty:
        return 0, 0.0, 0.0
    per_date = frame.groupby("date")["excess"].mean()
    if len(per_date) < 2 or per_date.std() == 0:
        return len(frame), float(frame["excess"].mean()), 0.0
    return len(frame), float(frame["excess"].mean()), float(per_date.mean() / (per_date.std() / np.sqrt(len(per_date))))


def study(obs: pd.DataFrame, split: str, buckets: int = 5, feature_names=FEATURES) -> List[BucketResult]:
    """Quintile-bucket each feature (edges from the train period) and score every bucket in train and test."""
    cut = pd.Timestamp(split)
    train, test = obs[obs["date"] <= cut], obs[obs["date"] > cut]
    results: List[BucketResult] = []
    for name in feature_names:
        edges = np.unique(np.quantile(train[name], np.linspace(0, 1, buckets + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            pick = lambda f: f[(f[name] > lo) & (f[name] <= hi)] if b else f[f[name] <= hi]  # noqa: E731
            (tn, tm, tt), (sn, sm, st) = _stat(pick(train)), _stat(pick(test))
            results.append(BucketResult(name, b + 1, float(lo), float(hi), tn, tm, tt, sn, sm, st))
    return results
