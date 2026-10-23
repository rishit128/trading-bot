"""Score the broker's research calls on the price (src.data.house_call_score): per kind of call, how many hit the target, the stop
or their time limit, and the average return after costs. Only calls read while still open, within a day of being published, count.

    python scripts/score_house_calls.py"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import house_calls_db_url  # noqa: E402
from src.data.house_call_score import Call, score_call  # noqa: E402
from src.data.intraday_bars import fetch_5m_bars  # noqa: E402
from src.database import HouseCallEvent, make_session_factory  # noqa: E402


def calls_first_seen_open():
    """(the calls read while still open within a day of being published, the number of distinct calls read)."""
    with make_session_factory(house_calls_db_url())() as s:
        events = list(s.scalars(select(HouseCallEvent).order_by(HouseCallEvent.first_seen)))
    first = {}
    for e in events:
        first.setdefault((e.symbol, e.kind, e.side, e.published), e)
    calls = [
        Call(
            e.symbol,
            e.kind,
            e.side,
            pd.Timestamp(e.published, tz="UTC").to_pydatetime(),
            e.entry_low,
            e.entry_high,
            e.stop,
            e.target,
            pd.Timestamp(e.valid_to).date() if e.valid_to else None,
        )
        for e in first.values()
        if e.is_open and e.first_seen - e.published <= timedelta(days=1)
    ]
    return calls, len(first)


def main():
    calls, total = calls_first_seen_open()
    print(
        f"{total} distinct calls read, {len(calls)} read while still open within a day of being published (the rest are skipped: once a call is closed only the winners stay listed, and a call first read days late has part of its outcome already known)"
    )
    bars = fetch_5m_bars(sorted({c.symbol for c in calls}), "1mo")
    rows = []
    for c in calls:
        outcome = score_call(c, bars[c.symbol]) if c.symbol in bars else None
        rows.append(
            {
                "symbol": c.symbol,
                "kind": c.kind,
                "side": c.side,
                "published": c.published,
                **(outcome.__dict__ if outcome else {"result": "NO_DATA"}),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={"gross": "{:+.2%}".format, "net": "{:+.2%}".format}, na_rep="-"))
    done = df[df["result"].isin(["TARGET", "STOP", "TIME"])]
    print("\nresolved calls by kind and side (net of the 0.21% round-trip cost):")
    for (kind, side), g in done.groupby(["kind", "side"]):
        print(
            f"  {kind:<12} {side:<5} n={len(g):<3} target {(g.result == 'TARGET').mean():.0%}  stop {(g.result == 'STOP').mean():.0%}"
            f"  mean net {g.net.mean():+.2%}  | stop and target ignored: direction right {(g.drift > 0).mean():.0%}, mean {g.drift.mean():+.2%}"
        )
    print("\nA verdict needs at least 100 resolved calls of a kind; fewer than that is only a first look.")


if __name__ == "__main__":
    main()
