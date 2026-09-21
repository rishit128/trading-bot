"""Candidate signals, each returning target weights (dates x symbols, decided at that date's close).

The parameters are the standard textbook/literature values and are NOT tuned on the data; that is the point of fixing
them in advance. Every candidate uses a liquidity filter so it only holds stocks that could really be traded."""
import numpy as np
import pandas as pd

from src.research.engine import price_returns


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def _hold_between_rebalances(weights: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep each month-end target until the next month end."""
    weights = weights.copy()
    weights.loc[~index.isin(month_end_dates(index))] = np.nan
    return weights.ffill().fillna(0.0)


def liquid_mask(close: pd.DataFrame, volume: pd.DataFrame, min_traded_value: float, window: int = 60) -> pd.DataFrame:
    """True where a stock's 60-day average traded value meets the minimum."""
    return ((close * volume).rolling(window, min_periods=window).mean() >= min_traded_value) & close.notna()


def _rebalanced(close: pd.DataFrame, score: pd.DataFrame, eligible: pd.DataFrame, top_n: int, ascending: bool) -> pd.DataFrame:
    """At each month end hold the top_n eligible stocks by score, equal weight, until the next month end."""
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for d in month_end_dates(close.index):
        s = score.loc[d].where(eligible.loc[d]).dropna()
        if len(s) < top_n:
            continue
        chosen = s.sort_values(ascending=ascending).index[:top_n]
        weights.loc[d, chosen] = 1.0 / top_n
    return _hold_between_rebalances(weights, close.index)


def xs_momentum(close, volume, min_traded_value, lookback=252, skip=21, top_n=20):
    """Cross-sectional 12-1 momentum: best 12-month return excluding the latest month, rebalanced monthly."""
    score = close.shift(skip) / close.shift(lookback) - 1
    return _rebalanced(close, score, liquid_mask(close, volume, min_traded_value) & score.notna(), top_n, ascending=False)


def low_volatility(close, volume, min_traded_value, lookback=252, top_n=20):
    """Lowest 12-month volatility among liquid stocks, rebalanced monthly."""
    score = price_returns(close).rolling(lookback, min_periods=lookback).std()
    return _rebalanced(close, score, liquid_mask(close, volume, min_traded_value) & score.notna(), top_n, ascending=True)


def stateful(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """Boolean holding matrix: enter when `enter`, keep until `exit_`."""
    e, x = enter.to_numpy(dtype=bool), exit_.to_numpy(dtype=bool)
    held = np.zeros_like(e)
    for t in range(len(e)):
        prev = held[t - 1] if t else np.zeros(e.shape[1], dtype=bool)
        held[t] = (prev & ~x[t]) | (~prev & e[t])
    return pd.DataFrame(held, index=enter.index, columns=enter.columns)


def _capped_equal_weight(held: pd.DataFrame, cap: int) -> pd.DataFrame:
    count = held.sum(axis=1).clip(lower=1)
    per = np.minimum(1.0 / cap, 1.0 / count)
    return held.astype(float).mul(per, axis=0)


def donchian_breakout(close, volume, min_traded_value, entry=55, exit_=20, cap=20):
    """Classic trend following: buy a new 55-day high, sell a new 20-day low; at most `cap` equal-weight positions."""
    hi = close.rolling(entry).max().shift(1)
    lo = close.rolling(exit_).min().shift(1)
    ok = liquid_mask(close, volume, min_traded_value)
    return _capped_equal_weight(stateful((close > hi) & ok, close < lo), cap)


def rsi_series(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder RSI for every column."""
    delta = close.diff()  # NaN stays NaN; no implicit forward fill
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)  # no losses at all -> RS is inf -> RSI 100


def rsi2_mean_reversion(close, volume, min_traded_value, cap=20):
    """Buy short-term oversold (RSI(2) < 10) dips inside a long-term uptrend (above the 200-day); sell when RSI(2) > 60."""
    rsi2, ma200 = rsi_series(close, 2), close.rolling(200).mean()
    ok = liquid_mask(close, volume, min_traded_value)
    return _capped_equal_weight(stateful((rsi2 < 10) & (close > ma200) & ok, rsi2 > 60), cap)


def trend_rule(close, volume, min_traded_value, cap=20):
    """The bot's current entry rule without stops: price > MA50 > MA200 and RSI(14) < 70; exit below the MA200."""
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    ok = liquid_mask(close, volume, min_traded_value)
    enter = (close > ma50) & (ma50 > ma200) & (rsi_series(close, 14) < 70) & ok
    return _capped_equal_weight(stateful(enter, close < ma200), cap)


def equal_weight_universe(close, volume, min_traded_value):
    """Benchmark: hold every liquid stock equally, rebalanced monthly."""
    ok = liquid_mask(close, volume, min_traded_value)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for d in month_end_dates(close.index):
        names = ok.loc[d][ok.loc[d]].index
        if len(names):
            weights.loc[d, names] = 1.0 / len(names)
    return _hold_between_rebalances(weights, close.index)


def index_trend_timing(index_close: pd.Series, window: int = 200) -> pd.DataFrame:
    """Hold the index while it is above its 200-day average, otherwise cash."""
    return pd.DataFrame({"INDEX": (index_close > index_close.rolling(window).mean()).astype(float)})


# ---- Round 2 candidates (declared before running; literature-default parameters, not tuned) ----
def regime_filter(weights: pd.DataFrame, index_close: pd.Series, window: int = 200) -> pd.DataFrame:
    """Hold the given weights only while the index is above its 200-day average; otherwise go to cash."""
    on = (index_close > index_close.rolling(window).mean()).astype(float).reindex(weights.index).fillna(0.0)
    return weights.mul(on, axis=0)


def xs_momentum_6m(close, volume, min_traded_value, top_n=20):
    """6-1 month momentum (shorter memory than 12-1), top 20 by return, rebalanced monthly."""
    return xs_momentum(close, volume, min_traded_value, lookback=126, skip=21, top_n=top_n)


def trend_rule_beating_index(close, volume, min_traded_value, index_close: pd.Series, lookback=126, cap=20):
    """The current trend rule, but a stock may only be bought if it beat the Nifty over the last 6 months."""
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    idx = index_close.reindex(close.index).ffill()
    beats = (close / close.shift(lookback) - 1).gt(idx / idx.shift(lookback) - 1, axis=0)
    ok = liquid_mask(close, volume, min_traded_value)
    enter = (close > ma50) & (ma50 > ma200) & (rsi_series(close, 14) < 70) & beats & ok
    return _capped_equal_weight(stateful(enter, close < ma200), cap)
