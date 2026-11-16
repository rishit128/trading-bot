"""Swing exits at the account's real fees (declared 2026-11-16, before any result; the owner's question).

The owner's point: holdings that were well up (CUPID +46%, CHENNPETRO +6% then -4%) are not sold, and give gains back. The
live exits are only a 15% stop below the entry, the close under the 200-day average, and a +100% target. The 26 Oct exit test
(trailing stops, break-even locks; none passed) charged the old zero-brokerage plan; the account pays 0.3% a side and a Rs 23.60
DP charge per sale, which makes every extra sale dearer, so it is run again with tighter trails and profit targets added.

Rules (fixed): the live configuration from .env (scanner, 12-1 momentum ranking, risk engine with swing-v7 rounding, sizes,
paper fees and slippage, AI stand-in confidence 0.75) on every NSE stock in the 10-year cache; one exit change per row:
  trail10/15/20/25/30  the stop trails that far below the highest high since entry (never below the 15% stop)
  lock10/lock20        the stop moves up to the entry price once the high is 10% / 20% above it
  tp20/tp30/tp50       sell at a +20% / +30% / +50% target (the live target is +100%)

Gate (the 26 Oct one, word for word): a change passes only if, against the live exits, its Sharpe is at least 0.05 higher and
its worst drawdown no worse in BOTH the full period and the last 3 years, its return is within 10% of the base's in both, both
halves of the full period are positive, and its neighbours in the same family (trail15 for trail10 and trail20, ...; tp30 for
tp20 and tp50, ...) have a Sharpe at least the base's in both windows. If several pass, the one with the highest last-3-years
Sharpe is taken. Today's listings only: survivorship bias lifts every row alike, so only differences between rows are read.

    python scripts/research_swing_exits.py"""

import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402
from src.research.live_backtest import DEFAULT_CONFIDENCE, LiveRun, make_buy_source, ranked_candidates, summarize_run  # noqa: E402

VARIANTS = {
    "base (live)": {},
    "trail10": {"trail_pct": 0.10},
    "trail15": {"trail_pct": 0.15},
    "trail20": {"trail_pct": 0.20},
    "trail25": {"trail_pct": 0.25},
    "trail30": {"trail_pct": 0.30},
    "lock10": {"lock_at": 0.10},
    "lock20": {"lock_at": 0.20},
    "tp20": {"take_profit_pct": 0.20},
    "tp30": {"take_profit_pct": 0.30},
    "tp50": {"take_profit_pct": 0.50},
}
FAMILIES = [["trail10", "trail15", "trail20", "trail25", "trail30"], ["lock10", "lock20"], ["tp20", "tp30", "tp50"]]


def halves(equity: pd.Series) -> tuple:
    middle = equity.index[len(equity) // 2]
    first, second = equity.loc[:middle], equity.loc[middle:]
    return float(first.iloc[-1] / first.iloc[0] - 1), float(second.iloc[-1] / second.iloc[0] - 1)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, capital = screen_config(settings), settings.paper_initial_cash
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    print(
        f"{len(bars)} NSE stocks to {close.index[-1].date()}; live target +{settings.risk.take_profit_pct:.0%}, stop -{settings.risk.stop_loss_pct:.0%}"
    )
    stats: dict = {}
    for window, start in (("full", close.index[MOMENTUM_BARS]), ("last 3y", close.index[-1] - pd.DateOffset(years=3))):
        start = close.index[close.index.searchsorted(start)]
        dates = [d for d in close.index if d >= start]
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        frames = {s: df.loc[close.index[warm] :] for s, df in bars.items()}
        ranked = ranked_candidates(close, volume, dates, cfg, None)
        print(f"\n=== {window}: {start.date()} .. {close.index[-1].date()} ===")
        print(
            f"{'exit':13s}{'return':>9}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'win':>6}{'avg':>8}{'half 1':>9}{'half 2':>9}",
            flush=True,
        )
        for name, change in VARIANTS.items():
            limits = settings.risk
            kw = dict(change)
            if "take_profit_pct" in kw:
                limits = dataclasses.replace(limits, take_profit_pct=kw.pop("take_profit_pct"))
            res = simulate(
                frames,
                {},
                limits,
                start_equity=capital,
                fees=india_delivery_fees,
                slippage=SLIPPAGE,
                buy_source=make_buy_source(ranked, cfg, DEFAULT_CONFIDENCE),
                **kw,
            )
            res = Result(res.equity.loc[start:], [t for t in res.trades if t.entry_date >= start], res.open_at_end)
            s = summarize_run(LiveRun(res, capital, limits))
            s["h1"], s["h2"] = halves(res.equity)
            s["avg"] = float(np.mean([t.return_pct for t in res.trades])) if res.trades else 0.0
            stats[(name, window)] = s
            print(
                f"{name:13s}{s['total_return']:+9.0%}{s['sharpe']:8.2f}{s['max_drawdown']:8.1%}{s['trades']:8d}"
                f"{s['win_rate']:6.0%}{s['avg']:+8.2%}{s['h1']:+9.0%}{s['h2']:+9.0%}",
                flush=True,
            )

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
    print("\n== gate against the live exits ==")
    for family in FAMILIES:
        for i, name in enumerate(family):
            near = [family[j] for j in (i - 1, i + 1) if 0 <= j < len(family)]
            ok = own(name) and all(neighbour_ok(n) for n in near)
            print(
                f"{name:13s}{'PASS' if ok else 'FAIL'} (own gate {own(name)}, neighbours {[n for n in near if neighbour_ok(n)]})"
            )
            if ok:
                passed.append(name)
    best = max(passed, key=lambda n: stats[(n, "last 3y")]["sharpe"]) if passed else None
    print("\nPASSED: " + (f"{', '.join(passed)}; taken: {best}" if best else "none (the live exits stay)"))


if __name__ == "__main__":
    main()
