"""Swing: skip a buy that already ran up a lot in the last month (declared 2026-11-18, before any result; the owner's question).

The owner's point: a stock that is already up a lot over the past month still looks strong to the scanner (12-1 momentum,
price > MA50 > MA200, RSI(14) < 70), but may be due to fall back, so it should not be bought. The live scanner has no direct
1-month limit; only RSI(14) < 70 holds back the most stretched names.

Data: Angel One daily bars for every NSE stock (research_cache/angel_daily_all, Nov 2016 to Nov 2026, splits and bonuses undone
as in scripts/research_short_term.py). Not Yahoo.

Rules (fixed): the live configuration from .env (scanner, 12-1 momentum ranking, risk engine with swing-v7 rounding, sizes, live
exits, paper fees and slippage, AI stand-in confidence 0.75); one entry change per row. A candidate whose close on the signal day
is more than X above its close 21 sessions earlier (about one month) is dropped from the day's list BEFORE the top-N cut, so the
next stock in rank order takes its place, as a live scanner filter would:
  up10/up15/up20/up25/up30   skip when the 1-month gain is above 10% / 15% / 20% / 25% / 30%

Gate (the 26 Oct one, as in research_swing_exits.py): a change passes only if, against the live rule, its Sharpe is at least
0.05 higher and its worst drawdown no worse in BOTH the full period and the last 3 years, its return is within 10% of the
base's in both, both halves of the full period are positive, and its neighbours (up15 for up10 and up20, ...) have a Sharpe at
least the base's in both windows. If several pass, the one with the highest last-3-years Sharpe is taken. Today's listings only:
survivorship bias lifts every row alike, so only differences between rows are read. Also printed (not gated): how the trades
the live rule took but each variant would skip went on to do.

    python scripts/research_swing_runup.py"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_short_term import load  # noqa: E402
from research_swing_exits import halves  # noqa: E402

from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402
from src.research.live_backtest import DEFAULT_CONFIDENCE, LiveRun, make_buy_source, ranked_candidates, summarize_run  # noqa: E402

MONTH = 21
VARIANTS = {"base (live)": None, "up10": 0.10, "up15": 0.15, "up20": 0.20, "up25": 0.25, "up30": 0.30}
FAMILY = ["up10", "up15", "up20", "up25", "up30"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, capital = screen_config(settings), settings.paper_initial_cash
    bars = load()
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    runup = close / close.shift(MONTH) - 1  # per stock on the shared calendar; a missing bar leaves NaN (never skipped)
    print(f"{len(bars)} NSE stocks (Angel One daily), {close.index[0].date()} .. {close.index[-1].date()}")
    stats: dict = {}
    for window, start in (("full", close.index[MOMENTUM_BARS]), ("last 3y", close.index[-1] - pd.DateOffset(years=3))):
        start = close.index[close.index.searchsorted(start)]
        dates = [d for d in close.index if d >= start]
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        frames = {s: df.loc[close.index[warm] :] for s, df in bars.items()}
        ranked = ranked_candidates(close, volume, dates, cfg, None)
        print(f"\n=== {window}: {start.date()} .. {close.index[-1].date()} ===")
        print(
            f"{'entry':13s}{'return':>9}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'win':>6}{'avg':>8}{'half 1':>9}{'half 2':>9}",
            flush=True,
        )
        base_trades = []
        for name, limit in VARIANTS.items():
            if limit is None:
                use = ranked
            else:
                use = {
                    d: [c for c in cands if not (runup.at[d, c.symbol] > limit)]  # NaN compares False: kept
                    for d, cands in ranked.items()
                }
            res = simulate(
                frames,
                {},
                settings.risk,
                start_equity=capital,
                fees=india_delivery_fees,
                slippage=SLIPPAGE,
                buy_source=make_buy_source(use, cfg, DEFAULT_CONFIDENCE),
            )
            res = Result(res.equity.loc[start:], [t for t in res.trades if t.entry_date >= start], res.open_at_end)
            s = summarize_run(LiveRun(res, capital, settings.risk))
            s["h1"], s["h2"] = halves(res.equity)
            s["avg"] = float(np.mean([t.return_pct for t in res.trades])) if res.trades else 0.0
            stats[(name, window)] = s
            if limit is None:
                base_trades = res.trades
            print(
                f"{name:13s}{s['total_return']:+9.0%}{s['sharpe']:8.2f}{s['max_drawdown']:8.1%}{s['trades']:8d}"
                f"{s['win_rate']:6.0%}{s['avg']:+8.2%}{s['h1']:+9.0%}{s['h2']:+9.0%}",
                flush=True,
            )
        # the live rule's own trades, split by the 1-month gain on the day before entry (the signal day)
        rows = []
        for t in base_trades:
            i = close.index.searchsorted(t.entry_date) - 1
            if i >= MONTH:
                rows.append((runup.iloc[i].get(t.symbol, np.nan), t.return_pct))
        df = pd.DataFrame(rows, columns=["runup", "ret"]).dropna()
        print("  live rule's trades by 1-month gain before the buy:   trades   avg return   win")
        for lo, hi in ((-1, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 9)):
            part = df[(df.runup > lo) & (df.runup <= hi)]
            if len(part):
                label = f"{lo:.0%}..{hi:.0%}" if hi < 9 else f"over {lo:.0%}"
                label = "up to 10%" if lo == -1 else label
                print(f"    {label:14s}{len(part):>34d}{part.ret.mean():+12.1%}{(part.ret > 0).mean():7.0%}")

    base, windows = "base (live)", ("full", "last 3y")

    def own(name: str) -> bool:
        return (
            all(
                stats[(name, w)]["sharpe"] >= stats[(base, w)]["sharpe"] + 0.05
                and stats[(name, w)]["max_drawdown"] >= stats[(base, w)]["max_drawdown"]
                and stats[(name, w)]["total_return"]
                >= stats[(base, w)]["total_return"] - 0.10 * abs(stats[(base, w)]["total_return"])
                for w in windows
            )
            and stats[(name, "full")]["h1"] > 0
            and stats[(name, "full")]["h2"] > 0
        )

    def neighbour_ok(name: str) -> bool:
        return all(stats[(name, w)]["sharpe"] >= stats[(base, w)]["sharpe"] for w in windows)

    passed = []
    print("\n== gate against the live rule ==")
    for i, name in enumerate(FAMILY):
        near = [FAMILY[j] for j in (i - 1, i + 1) if 0 <= j < len(FAMILY)]
        ok = own(name) and all(neighbour_ok(n) for n in near)
        print(f"{name:13s}{'PASS' if ok else 'FAIL'} (own gate {own(name)}, neighbours {[n for n in near if neighbour_ok(n)]})")
        if ok:
            passed.append(name)
    best = max(passed, key=lambda n: stats[(n, "last 3y")]["sharpe"]) if passed else None
    print("\nPASSED: " + (f"{', '.join(passed)}; taken: {best}" if best else "none (the live rule stays)"))


if __name__ == "__main__":
    main()
