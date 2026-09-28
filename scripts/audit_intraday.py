"""Does the intraday opening-range breakout have an edge? A reproducible audit on the archived 5-minute bars (fill it with
scripts/intraday_archive.py first). Read-only: it reports and changes nothing.

  1. does each filter (VWAP, volume, range width) add anything?
  2. does the signal beat a RANDOM entry time with the same stop, target and exits?  <- the test that matters
  3. how different are the results if only each bar's open is examined (the paper broker's old behaviour)?
  4. slices by entry time, volume surge, opening gap and market breadth, and what an account of this size can even trade

    python scripts/audit_intraday.py [--capital 20000] [--seeds 5]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import open_bar_archive  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.research.intraday_audit import (HEADER, describe, evaluate, first_breakout, random_entry, stock_days,  # noqa: E402
                                         time_bucket)


def main():
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capital", type=float, default=settings.paper_initial_cash, help="account size for the tradability split")
    ap.add_argument("--seeds", type=int, default=5, help="random-entry repetitions pooled for the null test")
    args = ap.parse_args()
    data = open_bar_archive().load()
    if not data:
        sys.exit("the archive is empty: run scripts/intraday_archive.py refresh (or import-cache)")
    days = stock_days(data)
    print(f"{len(days)} stock-sessions, {len(data)} stocks, {min(d.date for d in days)}..{max(d.date for d in days)}")

    print("\n== 1. does each filter add anything? (full-bar exits) ==\n" + HEADER)
    no_filters = {"use_vwap": False, "use_volume": False, "use_range": False}
    variants = {"the live rule (range + VWAP + volume)": {}, "without VWAP": {"use_vwap": False},
                "without volume": {"use_volume": False}, "without the range filter": {"use_range": False},
                "plain breakout, no filters": no_filters}
    trades = {}
    for name, kw in variants.items():
        trades[name] = evaluate(days, lambda d, kw=kw: first_breakout(d, **kw))
        print(describe(name, trades[name]))
    live = trades["the live rule (range + VWAP + volume)"]

    print("\n== 2. signal vs a RANDOM entry time (same stop, target, exits) ==\n" + HEADER)
    random_trades = pd.concat([evaluate(days, lambda d, r=np.random.default_rng(seed): random_entry(d, r))
                               for seed in range(args.seeds)])
    print(describe(f"random entry ({args.seeds} seeds pooled)", random_trades))
    print(describe("the live signal", live))
    gap = live["net"].mean() - random_trades["net"].mean()
    print(f"-> the signal's net edge over random entry: {gap:+.3%} per trade "
          "(an edge needs this to be clearly above the ± noise)")

    print("\n== 3. paper-broker shortcut: only each bar's open examined vs full bars ==\n" + HEADER)
    print(describe("full-bar exits (the backtest)", live))
    print(describe("open-only exits (old paper broker)", evaluate(days, first_breakout, mode="open_only")))

    print("\n== 4. slices of the live rule ==\n" + HEADER)
    live = live.assign(bucket=live["t"].map(time_bucket))
    for k, g in live.groupby("bucket"):
        print(describe(f"entry {k}", g))
    for lo, hi, name in ((1.5, 2, "volume 1.5-2x"), (2, 3, "volume 2-3x"), (3, 1e9, "volume 3x or more")):
        print(describe(name, live[(live.volume_ratio >= lo) & (live.volume_ratio < hi)]))
    print(describe("stock gapped up >0.3%", live[live.gap > .003]))
    print(describe("stock gapped down <-0.3%", live[live.gap < -.003]))
    print(describe("market breadth > 60% up", live[live.breadth > .6]))
    print(describe("market breadth < 40% up", live[live.breadth < .4]))
    cap = args.capital * settings.intraday_max_position_pct
    fits = live.price <= cap
    print(f"\nwith Rs {args.capital:,.0f} and a {settings.intraday_max_position_pct:.0%} position cap (Rs {cap:,.0f}): "
          f"{fits.mean():.0%} of these signals can be bought, {1 - fits.mean():.0%} cost more than one position may")
    print(describe("  tradable", live[fits]))
    print(describe("  too dear for the account", live[~fits]))


if __name__ == "__main__":
    main()
