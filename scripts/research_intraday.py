"""Phase 2 of the intraday plan: do any pre-declared ideas beat their costs? Runs the live rule (as a control) and five
hypotheses against random-entry / random-side nulls on the archived 5-minute bars, and applies the criteria written down in
src/research/intraday_hypotheses.py BEFORE any result was seen. If nothing qualifies, the kill criterion applies.

Read-only. Fill the archive first (scripts/intraday_archive.py); rerun as it grows: a longer archive can detect smaller edges.

    python scripts/research_intraday.py [--seeds 5]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import open_bar_archive  # noqa: E402
from src.research import intraday_hypotheses as hyp  # noqa: E402
from src.research.intraday_audit import stock_days  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=hyp.NULL_SEEDS, help="random-null repetitions per hypothesis")
    args = ap.parse_args()
    bars = open_bar_archive().load()
    if not bars:
        sys.exit("the archive is empty: run scripts/intraday_archive.py refresh (or import-cache)")
    days = stock_days(bars)
    print(f"{len(days)} stock-sessions, {len(bars)} stocks, {min(d.date for d in days)}..{max(d.date for d in days)}; "
          f"criteria: >= {hyp.MIN_TRADES} trades, net t >= {hyp.T_BAR}, both halves > 0, 2x slippage > 0, "
          f"beats the null (t >= {hyp.T_BAR})\n")
    for h in [hyp.CONTROL] + hyp.HYPOTHESES:
        print(f"{h.name}: {h.idea}")
    print()
    print(hyp.report(hyp.run(days, [hyp.CONTROL] + hyp.HYPOTHESES, seeds=args.seeds)))


if __name__ == "__main__":
    main()
