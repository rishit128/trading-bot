"""Swing sizing at the real fees (2026-11-13, see STRATEGY.md swing-v7): old 15 x 8.8% against fewer, larger positions.
In the live account 25% sits in the core ETFs, so 6 x 10.8% of equity is 6 x 14.4% of the stock part."""

import dataclasses
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import pandas as pd

from src.app.wiring import screen_config
from src.config import load_settings
from src.data.price_history import load_ohlc_nse
from src.data.universe import MOMENTUM_BARS
from src.research.live_backtest import DEFAULT_CONFIDENCE, ranked_candidates, run_live_config, summarize_run

logging.basicConfig(level=logging.WARNING)
s = load_settings()
cfg0 = screen_config(s)
bars = load_ohlc_nse(10)
close = pd.DataFrame({k: d["Close"] for k, d in bars.items()}).sort_index()
volume = pd.DataFrame({k: d["Volume"] for k, d in bars.items()}).sort_index()
setups = {
    "old 15 x 8.8%, drag 4%": (
        dataclasses.replace(s.risk, max_open_positions=15, min_position_pct=0.088, max_position_pct=0.088, max_fee_drag_pct=0.04),
        0.088,
    ),
    "new 6 x 14.4%, drag 2.5%": (
        dataclasses.replace(s.risk, max_open_positions=6, min_position_pct=0.144, max_position_pct=0.144, max_fee_drag_pct=0.025),
        0.144,
    ),
    "8 x 10.8%, drag 2.5%": (
        dataclasses.replace(s.risk, max_open_positions=8, min_position_pct=0.108, max_position_pct=0.108, max_fee_drag_pct=0.025),
        0.108,
    ),
}
for window, start in (("full", close.index[MOMENTUM_BARS]), ("last 3y", close.index[-1] - pd.DateOffset(years=3))):
    start = close.index[close.index.searchsorted(start)]
    dates = [d for d in close.index if d >= start]
    warm = close.index.searchsorted(start - pd.Timedelta(days=330))
    frames = {k: d.loc[close.index[warm] :] for k, d in bars.items()}
    print(f"\n== {window} from {start.date()} ==", flush=True)
    for label, (limits, pct) in setups.items():
        c = dataclasses.replace(cfg0, affordable_pct=pct)
        r = summarize_run(
            run_live_config(
                frames,
                limits,
                c,
                s.paper_initial_cash,
                ranked=ranked_candidates(close, volume, dates, c, None),
                confidence=DEFAULT_CONFIDENCE,
                start=start,
            )
        )
        print(
            f"{label:26s} return {r['total_return']:+6.0%} CAGR {r['cagr']:+6.1%} Sharpe {r['sharpe']:5.2f} maxDD {r['max_drawdown']:6.1%} trades {r['trades']:4d} win {r['win_rate']:4.0%} fees/capital {r['fees_pct_of_capital']:4.0%}",
            flush=True,
        )
