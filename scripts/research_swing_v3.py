"""Position count sweep for the swing account (pre-declared 2026-10-23, follows research_swing_v2.py where 8 x 11% came closest).

    python scripts/research_swing_v3.py

n positions of 0.90/n each (90% invested), on the cached ranked candidates of v2. Gate, fixed before running: a count is
adopted only if Sharpe >= base's in BOTH the full period and the last 3 years, max drawdown no worse in both, return not
lower than base minus 10% of base's in both, AND both neighbouring counts also have Sharpe >= base's in both windows (a plateau,
not one lucky number). Stops at 15% and everything else as live."""

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
from src.data.price_history import load_index_close, load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402

COUNTS = (5, 6, 8, 10, 12, 15, 20)


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim, capital = screen_config(settings), settings.risk, settings.paper_initial_cash
    bars, nifty = load_ohlc_nse(10), load_index_close("^NSEI", 10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    bear = pd.Series(False, index=close.index)
    start_all = close.index[MOMENTUM_BARS]
    out = {}
    for wname, w0 in (("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3))):
        w0 = close.index[close.index.searchsorted(w0)]
        sub = {s: df.loc[close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))] :] for s, df in bars.items()}
        for n in COUNTS:
            pct = round(0.90 / n, 4)
            c = replace(cfg, affordable_pct=pct)
            src, lo, hi = make_source(ranked, c, bear, base_pct=pct)
            limits = replace(lim, max_open_positions=n, min_position_pct=pct, max_position_pct=pct)
            res = simulate(sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                           cash_yield=0.0, cash_yield_from=w0)
            res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
            out[(wname, n)] = metrics(LiveRun(res, capital, limits), capital, w0)
        print(wname, "done", flush=True)
    for wname in ("full", "last3y"):
        print(f"\n== {wname} ==\n{'n':>3}{'size':>7}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'fees%':>7}{'half1':>8}{'half2':>8}")
        for n in COUNTS:
            s = out[(wname, n)]
            print(f"{n:3d}{0.9 / n:7.1%}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}{s['trades']:7d}"
                  f"{s['win_rate']:5.0%}{s['fees_pct_of_capital']:7.1%}{s['h1']:+8.1%}{s['h2']:+8.1%}")
    print("\n== gate vs the live 15 positions ==")
    for i, n in enumerate(COUNTS):
        if n == 15:
            continue
        def beats(k):
            return all(
                out[(w, k)]["sharpe"] >= out[(w, 15)]["sharpe"]
                and out[(w, k)]["max_drawdown"] >= out[(w, 15)]["max_drawdown"]
                and out[(w, k)]["total_return"] >= out[(w, 15)]["total_return"] - 0.1 * abs(out[(w, 15)]["total_return"])
                for w in ("full", "last3y")
            )
        sharpe_ok = lambda k: all(out[(w, k)]["sharpe"] >= out[(w, 15)]["sharpe"] for w in ("full", "last3y"))  # noqa: E731
        near = [COUNTS[j] for j in (i - 1, i + 1) if 0 <= j < len(COUNTS) and COUNTS[j] != 15]
        print(f"{n:3d}  own gate {'PASS' if beats(n) else 'FAIL'}; neighbours {near} Sharpe>=base: {[sharpe_ok(k) for k in near]}")


if __name__ == "__main__":
    main()
