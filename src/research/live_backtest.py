"""Backtest of the LIVE configuration: the same scanner, the same risk engine and limits, the same exits and costs, on the
same account size as the paper account.

Earlier research used a large account (Rs 10 lakh, 10 positions of 5%) and a percent-of-equity engine. That says nothing
about a Rs 20,000 account with 5 positions of 10-25%, where whole shares, a fixed Rs 15.93 charge per sale and the
inability to buy a Rs 9,000 stock all matter. This module runs the actual pipeline pieces instead:

  * entry candidates:  `screen_symbol` (the scanner's own filter) + 12-1 momentum ranking, every decision day,
    with the affordability rule the live scanner applies (`select_candidates`);
  * sizing / limits:   `RiskEngine` with the live `RiskLimits` (confidence-scaled size, position and exposure caps,
    daily-loss and drawdown halts);
  * exits:             8% protective stop + the MA200 trend exit, no time limit (`LIVE_PARITY`);
  * costs:             the paper broker's delivery fee schedule and slippage.

The one stand-in is the AI: it is replaced by a constant approval confidence (`confidence`), because a backtest cannot
replay a live LLM. Whether the AI adds anything is a separate question (ablation)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtest import LIVE_PARITY, Result, curve_metrics, simulate
from src.config import RiskLimits
from src.data.indicators import MIN_BARS
from src.data.universe import (MOMENTUM_BARS, Candidate, ScreenConfig, above_median_delivery, rank, screen_arrays,
                               select_candidates)
from src.engine.convention import SLIPPAGE
from src.engine.paper_broker import india_delivery_fees

LIVE_LOOKBACK_DAYS = 400  # calendar days of bars the live scanner downloads (`yf_bar_fetcher`); RSI depends on the window
DEFAULT_CONFIDENCE = 0.75  # stand-in for the AI: about the mean confidence of the live approved BUYs


def ranked_candidates(close: pd.DataFrame, volume: pd.DataFrame, dates: Sequence[pd.Timestamp], cfg: ScreenConfig,
                      delivery_avg: Optional[pd.DataFrame] = None) -> Dict[pd.Timestamp, List[Candidate]]:
    """For each decision date, every stock passing the live scanner, best 12-1 momentum first (no top-N cut yet).

    Uses only bars up to and including the date, in the same 400-calendar-day window the live scanner downloads.
    `delivery_avg` (see `delivery_average`) applies the delivery filter on the dates it covers, and is skipped on the
    dates it does not, exactly as the live scanner fails open."""
    out: Dict[pd.Timestamp, List[Candidate]] = {}
    index, symbols = close.index, list(close.columns)
    C, V = close.to_numpy(dtype=float), volume.reindex(index=index, columns=close.columns).to_numpy(dtype=float)
    min_bars = max(MIN_BARS, MOMENTUM_BARS)
    for d in dates:
        end = index.get_loc(d) + 1
        start = index.searchsorted(d - pd.Timedelta(days=LIVE_LOOKBACK_DAYS))
        c_win, v_win, idx_win = C[start:end], V[start:end], index[start:end]
        valid = ~np.isnan(c_win)
        found = []
        for j in np.flatnonzero(valid.sum(axis=0) >= min_bars):  # cheap length test before touching any column
            rows = np.flatnonzero(valid[:, j])
            candidate = screen_arrays(symbols[j], c_win[rows, j], np.nan_to_num(v_win[rows, j]), idx_win[rows[-1]], d, cfg)
            if candidate is not None:
                found.append(candidate)
        if delivery_avg is not None:
            keep = above_median_delivery(delivery_avg, d.date())
            if keep is not None:
                found = [c for c in found if c.symbol in keep]
        out[d] = rank(found, len(found))
    return out


def make_buy_source(ranked: Dict[pd.Timestamp, List[Candidate]], cfg: ScreenConfig,
                    confidence: float = DEFAULT_CONFIDENCE):
    """The entry feed for `simulate`: the day's affordable top candidates in rank order, each at the stand-in confidence."""
    def source(signal_day, portfolio):
        picks = select_candidates(ranked.get(signal_day, []), cfg.max_candidates, portfolio.equity, cfg.affordable_pct)
        return [(c.symbol, confidence) for c in picks]

    return source


@dataclass(frozen=True)
class LiveRun:
    """One live-configuration backtest and how to read it."""
    result: Result
    capital: float
    limits: RiskLimits


def run_live_config(bars: Dict[str, pd.DataFrame], limits: RiskLimits, cfg: ScreenConfig, capital: float,
                    dates: Optional[Sequence[pd.Timestamp]] = None, confidence: float = DEFAULT_CONFIDENCE,
                    delivery_avg: Optional[pd.DataFrame] = None,
                    fees: Callable[[str, float], float] = india_delivery_fees, slippage: float = SLIPPAGE,
                    ranked: Optional[Dict[pd.Timestamp, List[Candidate]]] = None,
                    start: Optional[pd.Timestamp] = None) -> LiveRun:
    """Run the live entry/exit/sizing/cost rules over `bars` (symbol -> OHLCV frame) from `capital`.

    `ranked` lets several account sizes reuse one (slow) scan, since the ranking does not depend on the account.
    `start` reports the equity curve from that date on (bars before it are only history for the 200-day average)."""
    if ranked is None:
        close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
        volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
        dates = list(close.index if dates is None else dates)
        ranked = ranked_candidates(close, volume, dates, cfg, delivery_avg)
    result = simulate(bars, {}, limits, start_equity=capital, fees=fees, slippage=slippage,
                      buy_source=make_buy_source(ranked, cfg, confidence), **LIVE_PARITY)
    if start is not None:
        result = Result(result.equity.loc[start:], [t for t in result.trades if t.entry_date >= start], result.open_at_end)
    return LiveRun(result, capital, limits)


def summarize_run(run: LiveRun) -> dict:
    """Headline numbers of one run: return, CAGR, risk, and trade-level expectancy net of fees."""
    r, capital = run.result, run.capital
    m = curve_metrics(r.equity, capital)
    years = max(len(r.equity) / 252, 1e-9)
    total = 1 + m["total_return"]
    pnl = [(t.exit_price - t.entry_price) * t.qty - t.fees for t in r.trades]
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    trades = len(r.trades)
    return {
        **m, "cagr": float(total ** (1 / years) - 1) if total > 0 else -1.0,
        "trades": trades, "open_at_end": r.open_at_end,
        "win_rate": len(wins) / trades if trades else 0.0,
        "expectancy_pct": float(np.mean([t.return_pct for t in r.trades])) if trades else 0.0,
        "profit_factor": (sum(wins) / -sum(losses)) if wins and losses and sum(losses) < 0 else None,
        "fees": float(sum(t.fees for t in r.trades)),
        "fees_pct_of_capital": float(sum(t.fees for t in r.trades) / capital),
        "exits": {k: sum(1 for t in r.trades if t.reason == k) for k in ("STOP", "TARGET", "SIGNAL", "TIME")},
    }


def research_setup(limits: RiskLimits) -> RiskLimits:
    """The account the earlier research tested: 10 positions of a flat 5% (the size of the account is set separately)."""
    return replace(limits, max_open_positions=10, min_position_pct=0.05, max_position_pct=0.05)


def equal_weight_basket(bars: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp,
                        capital: float) -> pd.Series:
    """Daily-rebalanced equal-weight holding of every stock in the same universe (same survivorship bias as the runs)."""
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index().loc[start:end]
    ret = close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    return capital * (1 + ret).cumprod()
