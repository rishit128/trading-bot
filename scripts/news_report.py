"""The news-catalyst engine's gate report (STRATEGY.md, News-Catalyst v1, gate 1): does the AI read news correctly?

Over every material filing measured so far (one per stock and session: the same news is often filed several times): the move from the filing to the close in the direction the AI predicted (a
"down" filing counts its fall as a gain), against the 0.21% round-trip cost. The gate passes with at least 100 material
filings, a mean above the cost, and t >= 2 (by date: filings on one day move together). Also shown: the same for the
filings the entry rule fired on and those it did not, every filing's label counts, and a random sample of AI labels with
their links for the hand spot-check (30 by default).

    python scripts/news_report.py [--sample 30] [--since 2026-10-22]"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import news_db_url  # noqa: E402
from src.database import NewsFilingRecord, make_session_factory  # noqa: E402

COST = 0.0021
MIN_FILINGS, T_BAR = 100, 2.0


def clustered_t(frame: pd.DataFrame, column: str) -> float:
    per_day = frame.groupby("session")[column].mean()
    if len(per_day) < 2 or per_day.std() == 0:
        return 0.0
    return float((per_day.mean() - COST) / (per_day.std() / np.sqrt(len(per_day))))


def line(name: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return f"{name:<34} none yet"
    return (
        f"{name:<34}{len(frame):>6}{frame['session'].nunique():>7}{frame['signed_close'].mean():>+10.3%}"
        f"{frame['signed_120m'].mean():>+10.3%}{clustered_t(frame, 'signed_close'):>8.1f}{(frame['signed_close'] > COST).mean():>7.0%}"
    )


def ai_filter_check(df: pd.DataFrame) -> None:
    """Is the AI a useful filter (meta-label)? The rule only acts on what the AI calls material, so the AI earns its place
    if those filings move more, and more in its direction, than the ones it set aside. One row per stock and session."""
    moved = df[df["move_close"].notna()].sort_values("published")
    groups = {
        "material": moved[moved["material"] == True],  # noqa: E712
        "labelled, not material": moved[(moved["label_status"] == "LABELLED") & (moved["material"] != True)],  # noqa: E712
        "routine (no AI call)": moved[moved["label_status"] == "ROUTINE"],
    }
    groups = {name: g.drop_duplicates(["symbol", "session"]) for name, g in groups.items()}
    print("\ndoes the AI pick the filings that move? (|move| to the close; 'in AI direction' only where it gave one)")
    print(f"{'':<26}{'n':>6}{'|close|':>10}{'|120 min|':>11}{'in AI direction':>17}{'right side':>12}")
    for name, g in groups.items():
        if g.empty:
            print(f"{name:<26} none yet")
            continue
        signed = g["signed_close"].dropna()
        direction = f"{signed.mean():>+17.3%}{(signed > 0).mean():>12.0%}" if len(signed) else f"{'-':>17}{'-':>12}"
        print(f"{name:<26}{len(g):>6}{g['move_close'].abs().mean():>10.3%}{g['move_120m'].abs().mean():>11.3%}{direction}")
    a, b = groups["material"]["move_close"].abs(), groups["labelled, not material"]["move_close"].abs()
    if len(a) >= 2 and len(b) >= 2:
        t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"material minus not material, |close|: {a.mean() - b.mean():+.3%} (t {t:+.1f}, by filing, not by date)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=30, help="AI labels to list for the hand spot-check")
    ap.add_argument("--since", help="only sessions from this date (YYYY-MM-DD)")
    args = ap.parse_args()
    with make_session_factory(news_db_url())() as s:
        rows = list(s.scalars(select(NewsFilingRecord)))
    if not rows:
        sys.exit("no filings recorded yet: the news engine runs inside `python main.py --intraday`")
    df = pd.DataFrame([{c.name: getattr(r, c.name) for c in NewsFilingRecord.__table__.columns} for r in rows])
    if args.since:
        df = df[df["session"] >= args.since]
    print(f"{len(df)} filings on {df['session'].nunique()} sessions ({df['session'].min()} .. {df['session'].max()})")
    print("label status:", ", ".join(f"{k} {v}" for k, v in df["label_status"].value_counts().items()))
    labelled = df[df["label_status"] == "LABELLED"]
    if len(labelled):
        print("event types:", ", ".join(f"{k} {v}" for k, v in labelled["event_type"].value_counts().items()))
    sign = df["direction"].map({"up": 1.0, "down": -1.0})
    for col in ("close", "120m"):
        df[f"signed_{col}"] = sign * df[f"move_{col}"]
    material = df[(df["material"] == True) & df["signed_close"].notna()]  # noqa: E712
    # A company often files the same news several times (results, press release, presentation): one per stock and session
    material = material.sort_values("published").drop_duplicates(["symbol", "session"])
    print(f"\nmoves in the AI's direction, cost {COST:.2%} (t is by date, of the mean less the cost)")
    print(f"{'':<34}{'n':>6}{'days':>7}{'close':>10}{'120 min':>10}{'t':>8}{'> cost':>7}")
    print(line("material filings", material))
    print(line("  up", material[material["direction"] == "up"]))
    print(line("  down (logged, not traded)", material[material["direction"] == "down"]))
    print(line("  rule fired", material[material["rule_status"] == "FIRED"]))
    print(line("  rule did not fire", material[material["rule_status"] != "FIRED"]))
    n, t = len(material), clustered_t(material, "signed_close") if len(material) else 0.0
    mean = material["signed_close"].mean() if n else float("nan")
    passed = n >= MIN_FILINGS and mean > COST and t >= T_BAR
    print(
        f"\nGATE 1: {'PASS' if passed else 'not yet'} ({n} of {MIN_FILINGS} material filings, mean {mean:+.3%} against "
        f"{COST:.2%}, t {t:.1f} of {T_BAR})"
    )
    ai_filter_check(df)
    pool = labelled.to_dict("records")
    random.Random(0).shuffle(pool)
    print(f"\nAI labels to check by hand ({min(args.sample, len(pool))} of {len(pool)}):")
    for r in pool[: args.sample]:
        numbers = json.loads(r["numbers_json"] or "{}").get("as_written", "")
        print(
            f"- {r['session']} {r['symbol']}: {r['event_type']} {r['direction']} surprise {r['surprise']} "
            f"material={r['material']} | {numbers[:120]}\n    {r['subject'][:110]}\n    {r['link']}"
        )


if __name__ == "__main__":
    main()
