"""Angel One against Yahoo: for each session, how many bars Angel One has, how soon after its close each was complete, how
closely its closes and volumes match the Yahoo archive's, and how many sessions are clean so far (ANGEL_BARS=auto switches the
engines to Angel One at 5).

    python scripts/angel_report.py [--days 5]

Run it after 16:00 IST, once the intraday engine has archived the day's bars."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import intraday_db_url  # noqa: E402
from src.data.angel import BAR  # noqa: E402
from src.data.angel_bars import CLOSE_SHARE_MIN, COVERAGE_MIN, MIN_CLEAN_DAYS, agreement, clean_days  # noqa: E402
from src.database import make_session_factory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=5, help="latest sessions to show")
    args = ap.parse_args()
    engine = create_engine(intraday_db_url())
    angel = pd.read_sql("select * from angel_bars", engine, parse_dates=["ts", "seen_at"])
    yahoo = pd.read_sql("select * from intraday_bars where source is null or source != 'angel'", engine, parse_dates=["ts"])
    if angel.empty:
        sys.exit("no Angel One bars yet: the stream runs inside `python main.py --intraday` during market hours")
    for day in sorted(angel["day"].unique())[-args.days :]:
        a, y = angel[angel["day"] == day], yahoo[yahoo["day"] == day]
        lag = (a["seen_at"] - (a["ts"] + pd.Timedelta(BAR))).dt.total_seconds()
        lag = lag[lag < 600]  # bars fetched afterwards from the candles (a restart, the first day) say nothing about the stream
        print(
            f"\n{day}: Angel One {len(a)} bars on {a['symbol'].nunique()} stocks; Yahoo {len(y)} bars on {y['symbol'].nunique()}"
        )
        print(f"  Angel One bar complete {lag.median():.1f}s after its close (95th percentile {lag.quantile(0.95):.1f}s)")
        if y.empty:
            print("  no Yahoo bars archived for this day (the engine archives them after the close)")
            continue
        stats = agreement(a, y)
        both = a.merge(y, on=["symbol", "ts"], suffixes=("_a", "_y"))
        volume = (both["volume_a"] / both["volume_y"].where(both["volume_y"] > 0)).dropna()
        print(f"  {stats['coverage']:.1%} of Yahoo's bars also in Angel One (need {COVERAGE_MIN:.0%})")
        print(f"  closes within 0.1% on {stats['close_agrees']:.1%} of the shared bars (need {CLOSE_SHARE_MIN:.0%})")
        print(f"  volume, Angel One / Yahoo: median {volume.median():.2f}")
    clean = clean_days(make_session_factory(intraday_db_url()))
    print(f"\nclean sessions: {len(clean)} of {MIN_CLEAN_DAYS} needed for ANGEL_BARS=auto to use Angel One")


if __name__ == "__main__":
    main()
