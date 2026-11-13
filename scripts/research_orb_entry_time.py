"""ORB entries after midday (declared 2026-11-12, before any result; the owner's question).

The owner's point: orb-v6 buys a first breakout as late as 14:00, when little of the session is left before the 15:15 square-off.
Why a late entry might be worse: the stop (the 09:15 bar's low) and the costs are the same as a morning entry's, but the time for
the move to pay for them is one to three hours instead of six.

Data: the 8-year Angel One 5-minute cache (research_cache/angel5m, 200 stocks, 2018-01-01 on). Today's stock list, so survivorship
lifts both groups alike; only the difference between them is read.

Rule (fixed): the live orb-v6 signal (`src.intraday.strategy.first_breakout`, 09:15 bar as the range, entries 09:20 to 14:00) on
every stock-day whose 09:15 bar exists. Bought at the next bar's open (the signal bar's close if that bar is missing), stop at the
09:15 bar's low (filled at the stop, or at the bar's open if it opens below), else sold at the 15:15 open. Every signal is taken:
no 5-position limit. Net = gross less this account's intraday fees on Rs 5,000 and 0.10% slippage a round trip.

One hypothesis, one cut: entries at or after 12:00 against entries before 12:00. The last entry time moves from 14:00 to 12:00
only if ALL hold:
  1. the late group's mean net a trade (mean of daily means) is below zero;
  2. on dates holding both groups, late minus early (daily means) is below zero with t <= -2.0;
  3. that difference is below zero in both halves of the period by date, and from 2024-11-20 (the recent market).
The hourly table is printed for information only; no other cut-off is chosen from it.

    python scripts/research_orb_entry_time.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_new_intraday import EXIT_SLOT, FEES, LAST_SLOT, RECENT_FROM, load  # noqa: E402
from src.engine.costs import SLIPPAGE  # noqa: E402
from src.intraday.strategy import ORB_ENTRY_FROM, ORB_RANGE_BARS, first_breakout  # noqa: E402

CUT = 12 * 60  # minutes since midnight
COST = FEES + 2 * SLIPPAGE


def trades() -> pd.DataFrame:
    names, days, bars, _, _ = load()
    o, h, l, c, v = (bars[k].astype(float) for k in range(5))
    all_minutes = 555 + 5 * np.arange(o.shape[-1])
    rows = []
    for s in range(len(names)):
        for d in range(len(days)):
            have = ~np.isnan(c[s, d])
            if not have[0] or have.sum() < 10:
                continue
            slots = np.flatnonzero(have)
            hit = first_breakout(
                h[s, d, slots],
                l[s, d, slots],
                c[s, d, slots],
                np.nan_to_num(v[s, d, slots]),
                all_minutes[slots],
                range_bars=ORB_RANGE_BARS,
                entry_from=ORB_ENTRY_FROM,
            )
            if hit is None:
                continue
            signal = int(slots[hit.index])
            buy = signal + 1
            entry = o[s, d, buy] if not np.isnan(o[s, d, buy]) else c[s, d, signal]
            end = o[s, d, EXIT_SLOT] if not np.isnan(o[s, d, EXIT_SLOT]) else c[s, d, LAST_SLOT]
            if np.isnan(end):
                continue
            exit_price = end
            for k in range(buy, EXIT_SLOT):
                if not np.isnan(l[s, d, k]) and l[s, d, k] <= hit.range_low:
                    exit_price = min(o[s, d, k], hit.range_low) if not np.isnan(o[s, d, k]) else hit.range_low
                    break
            rows.append((days[d], int(all_minutes[signal]), exit_price / entry - 1 - COST, exit_price <= hit.range_low))
    return pd.DataFrame(rows, columns=["day", "minute", "net", "stopped"])


def daily_t(per_day: pd.Series) -> float:
    return float(per_day.mean() / (per_day.std() / np.sqrt(len(per_day)))) if len(per_day) > 1 else float("nan")


def main() -> None:
    frame = trades()
    frame["late"] = frame["minute"] >= CUT
    print(f"{len(frame):,} orb-v6 signals on {frame['day'].nunique():,} dates; costs {COST:.3%} a round trip\n")
    print(f"{'signal bar':12s}{'trades':>9}{'gross':>9}{'net':>9}{'t (date)':>10}{'stopped':>9}{'won':>7}")
    hour = frame["minute"] // 60
    groups = [(f"{k:02d}:00-{k:02d}:55", frame[hour == k]) for k in sorted(hour.unique())]
    groups += [("before 12:00", frame[~frame["late"]]), ("12:00 on", frame[frame["late"]]), ("all", frame)]
    for label, part in groups:
        per_day = part.groupby("day")["net"].mean()
        print(
            f"{label:12s}{len(part):9,}{part['net'].mean() + COST:+9.3%}{per_day.mean():+9.3%}{daily_t(per_day):+10.1f}"
            f"{part['stopped'].mean():9.0%}{(part['net'] > 0).mean():7.0%}"
        )

    by_day = frame.groupby(["day", "late"])["net"].mean().unstack().dropna()
    diff = by_day[True] - by_day[False]
    middle = diff.index[len(diff) // 2]
    parts = {"all": diff, "1st half": diff[:middle], "2nd half": diff[middle:], "from 2024-11-20": diff[RECENT_FROM:]}
    print("\nlate minus early, on dates holding both:")
    for label, part in parts.items():
        print(f"  {label:16s}{len(part):6,} dates{part.mean():+9.3%}  t {daily_t(part):+.1f}")
    late_net = frame[frame["late"]].groupby("day")["net"].mean().mean()
    passed = late_net < 0 and daily_t(diff) <= -2.0 and all(part.mean() < 0 for part in parts.values())
    print("\n" + ("PASSED: the last entry moves to 12:00" if passed else "FAILED: the last entry stays at 14:00"))


if __name__ == "__main__":
    main()
