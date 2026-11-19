"""Is the 60% "stocks up" line still right now that orb-v8 counts the whole market? (declared 2026-11-16 before any number seen)

The breakout account refuses entries unless at least 60% of stocks are up on the day (ORB_MIN_BREADTH). That line was set on the
8-year test's stocks (research_cache/angel5m, about 200 large NSE stocks); since orb-v8 (2026-11-16) the live count is every NSE
equity Angel One quotes, small caps included, which may be up less often. If so the same 60% is a stricter rule than the one tested.

Method (fixed): daily close-to-close breadth from Angel One daily bars (research_cache/angel_daily_all), the last 3 years, for
  A the tested set (the angel5m stocks), B today's 500-stock live list (intraday.db intraday_universe), C every stock with a bar.
Decision (fixed): the whole-market line that lets through the same share of days as A >= 60% (quantile match over the 3 years),
rounded to the nearest 1%. It replaces 60% only if it differs by 5 points or more; otherwise 60% stays. Close-to-close breadth is
a stand-in for the intraday count; the agreement of A and C (days both above / both below) is printed so it can be judged.

    python scripts/research_breadth_calibration.py"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LINE, MIN_SHIFT = 0.60, 0.05


def main() -> None:
    daily = ROOT / "research_cache" / "angel_daily_all"
    closes = {}
    for p in daily.glob("*.pkl"):
        df = pd.read_pickle(p)
        if len(df):
            closes[p.stem] = df["Close"].astype(float)
    close = pd.DataFrame(closes).sort_index()
    close = close[close.index >= close.index[-1] - pd.DateOffset(years=3)]
    change = close.pct_change(fill_method=None).iloc[1:]  # a stock with no bar that day is not counted
    traded = change.notna()
    tested = sorted(p.stem for p in (ROOT / "research_cache" / "angel5m").glob("*.pkl"))
    with sqlite3.connect(ROOT / "intraday.db") as db:
        row = db.execute("select symbols_json from intraday_universe order by id desc limit 1").fetchone()
    live = json.loads(row[0]) if row else []

    def breadth(cols):
        cols = [c for c in cols if c in change.columns]
        ch = change[cols]
        return (ch > 0).sum(axis=1) / traded[cols].sum(axis=1), len(cols)

    a, na = breadth(tested)
    b, nb = breadth(live)
    c, nc = breadth(change.columns)
    print(f"{change.index[0].date()} .. {change.index[-1].date()}, {len(change)} sessions")
    print(f"A tested set {na} stocks, B live list {nb}, C whole market {nc}")
    print("\nshare of stocks up on the day:   median   25%    75%   days >= 60%")
    for name, s in (("A tested set", a), ("B live list", b), ("C whole market", c)):
        print(f"  {name:16s}{s.median():8.0%}{s.quantile(0.25):7.0%}{s.quantile(0.75):7.0%}{(s >= LINE).mean():10.0%}")
    share = (a >= LINE).mean()
    matched = float(np.quantile(c, 1 - share))
    rounded = round(matched, 2)
    agree = ((a >= LINE) == (c >= rounded)).mean()
    agree60 = ((a >= LINE) == (c >= LINE)).mean()
    print(f"\nA >= 60% on {share:.0%} of days; the whole-market level with the same share: {matched:.1%} -> {rounded:.0%}")
    print(f"days classified alike with A at 60%: whole market at 60% {agree60:.0%}, at {rounded:.0%} {agree:.0%}")
    gap = (c - a).median()
    print(f"median whole-market minus tested-set breadth: {gap:+.1%}")
    last = pd.DataFrame({"A": a, "B": b, "C": c}).tail(5)
    print("\nlast sessions:\n" + last.map(lambda v: f"{v:.0%}").to_string())
    if round(abs(rounded - LINE) * 100) >= round(MIN_SHIFT * 100):  # whole points: 0.55 - 0.60 is 5, not 4.999
        print(f"\nDECISION: the whole-market line becomes {rounded:.0%} (60% on the tested set lets through the same days)")
    else:
        print(f"\nDECISION: 60% stays (the matched whole-market line {rounded:.0%} is within {MIN_SHIFT:.0%} of it)")


if __name__ == "__main__":
    main()
