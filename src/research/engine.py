"""Vectorised portfolio engine for signal research.

Timing is conservative and free of lookahead. A weight decided from data up to the close of day t is executed at the
close of day t + exec_lag (default 1, i.e. the next day) and first earns the return of the day after that. Costs are
charged on the execution day, on turnover (sum of absolute weight changes)."""
import math
from typing import Dict

import pandas as pd

TRADING_DAYS = 252


def price_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily returns. A stock's own missing days are bridged (at most a week) so the gap return lands on the day it
    resumes trading; days before it existed stay NaN."""
    return close.ffill(limit=5).pct_change(fill_method=None)


def portfolio_returns(close: pd.DataFrame, weights: pd.DataFrame, cost_per_side: float = 0.0012,
                      exec_lag: int = 1) -> pd.DataFrame:
    """close: dates x symbols. weights: target weights decided at each date's close (NaN = 0). Returns gross/net daily
    returns plus turnover and exposure."""
    r = price_returns(close).fillna(0.0)
    w = weights.reindex(close.index).fillna(0.0).where(close.notna(), 0.0)
    held = w.shift(1 + exec_lag).fillna(0.0)          # what is actually owned during each day
    traded = w.shift(exec_lag).fillna(0.0)             # target that gets executed at each day's close
    turnover = traded.diff().abs().sum(axis=1)
    turnover.iloc[0] = traded.iloc[0].abs().sum()
    gross = (held * r).sum(axis=1)
    return pd.DataFrame({"gross": gross, "net": gross - turnover * cost_per_side, "turnover": turnover,
                         "exposure": held.abs().sum(axis=1)})


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compound daily returns into an equity curve."""
    return (1.0 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    years = len(returns) / TRADING_DAYS
    total = equity_curve(returns).iloc[-1]
    return float(total ** (1 / years) - 1) if years > 0 and total > 0 else -1.0


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free rate 0)."""
    sd = returns.std()
    # Floating-point noise makes a constant series' std ~1e-19, not 0; treat that as no variance.
    return float(returns.mean() / sd * math.sqrt(TRADING_DAYS)) if sd and sd > 1e-12 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall."""
    curve = equity_curve(returns)
    return float((curve / curve.cummax() - 1).min())


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Return for each calendar year."""
    return (1.0 + returns).groupby(pd.DatetimeIndex(returns.index).year).prod() - 1.0


def summarize(result: pd.DataFrame, column: str = "net") -> Dict[str, float]:
    """Headline statistics: CAGR, Sharpe, drawdown, turnover, exposure and the share of positive years."""
    r = result[column]
    yearly = yearly_returns(r)
    return {
        "cagr": cagr(r), "sharpe": sharpe(r), "max_drawdown": max_drawdown(r),
        "turnover_per_year": float(result["turnover"].sum() / (len(r) / TRADING_DAYS)),
        "avg_exposure": float(result["exposure"].mean()),
        "pct_years_positive": float((yearly > 0).mean()), "years": int(len(yearly)),
    }


def slice_period(result: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Restrict a result to a date range."""
    out = result
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    return out


def benchmark_returns(close: pd.Series) -> pd.DataFrame:
    """Buy and hold a single price series, zero costs."""
    r = close.pct_change(fill_method=None).fillna(0.0)
    return pd.DataFrame({"gross": r, "net": r, "turnover": 0.0, "exposure": 1.0}, index=close.index)

