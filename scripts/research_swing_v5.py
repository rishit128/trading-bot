"""Swing account-level loss limits (pre-declared 2026-10-26; idea from the skills.sh drawdown-circuit-breaker skill, which
halts new risk at 2% a day, 5% a week and 8% a month and after two losing trades).

    python scripts/research_swing_v5.py

The live account halts new buys at a 20% drawdown from its peak, for 30 days, then rebases the peak (the daily 2% limit is
unchanged in every row). Variants change only that limit on the live configuration (15 slots x 8.8% under the 90% cap, 15%
stop, MA200 exit), all NSE stocks, 10 years, the cached candidates of research_swing_v2:
  dd20       live: 20% drawdown, 30 days
  dd15, dd12, dd10, dd8   the same halt at 15%, 12%, 10%, 8%
  dd10_14d, dd10_60d      10% with a 14-day and a 60-day pause
Gate, fixed before running: Sharpe at least 0.05 above dd20's in BOTH the full period and the last 3 years, max drawdown no
worse in both, return not lower than dd20's minus 10% of it in both, both halves of the full period positive, and the
neighbouring thresholds (and, for dd10, the other pauses) also keep Sharpe >= dd20's in both windows (a plateau)."""

import logging
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_swing_v2 import CACHE, LiveRun, make_source, metrics  # noqa: E402
from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402

VARIANTS = {
    "dd20": (0.20, 30), "dd15": (0.15, 30), "dd12": (0.12, 30), "dd10": (0.10, 30), "dd8": (0.08, 30),
    "dd10_14d": (0.10, 14), "dd10_60d": (0.10, 60),
}
NEIGHBOURS = {"dd15": ["dd12"], "dd12": ["dd15", "dd10"], "dd10": ["dd12", "dd8", "dd10_14d", "dd10_60d"], "dd8": ["dd10"]}


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim, capital = screen_config(settings), settings.risk, settings.paper_initial_cash
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    bear = pd.Series(False, index=close.index)
    start_all = close.index[MOMENTUM_BARS]
    windows = (("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3)))
    out = {}
    for wname, w0 in windows:
        w0 = close.index[close.index.searchsorted(w0)]
        warm = close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))]
        sub = {s: df.loc[warm:] for s, df in bars.items()}
        for name, (dd, pause) in VARIANTS.items():
            src, lo, hi = make_source(ranked, cfg, bear)
            limits = replace(lim, min_position_pct=lo, max_position_pct=hi, max_drawdown_pct=dd, drawdown_pause_days=pause)
            res = simulate(sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                           cash_yield=0.0, cash_yield_from=w0)
            res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
            out[(wname, name)] = metrics(LiveRun(res, capital, limits), capital, w0)
        print(wname, "done", flush=True)
    for wname, _ in windows:
        print(f"\n== {wname} ==\n{'variant':10s}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'half1':>8}{'half2':>8}")
        for name in VARIANTS:
            s = out[(wname, name)]
            print(f"{name:10s}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}{s['trades']:7d}"
                  f"{s['win_rate']:5.0%}{s['h1']:+8.1%}{s['h2']:+8.1%}")

    def beats(k, margin):
        ok = True
        for w, _ in windows:
            b, v = out[(w, "dd20")], out[(w, k)]
            ok &= v["sharpe"] >= b["sharpe"] + margin and v["max_drawdown"] >= b["max_drawdown"]
            ok &= v["total_return"] >= b["total_return"] - 0.10 * abs(b["total_return"])
        v = out[("full", k)]
        return ok and v["h1"] > 0 and v["h2"] > 0

    print("\n== gate vs dd20 ==")
    for k in VARIANTS:
        if k == "dd20":
            continue
        plateau = all(beats(n, 0.0) for n in NEIGHBOURS.get(k, []))
        print(f"{k:10s}{'PASS' if beats(k, 0.05) and plateau else 'FAIL'} (own gate {beats(k, 0.05)}, neighbours {plateau})")


if __name__ == "__main__":
    main()
