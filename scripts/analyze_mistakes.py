"""Where was the bot right and where wrong? Every analysed stock-session with a matured outcome, judged on what the market
then did (entry next open, exit `horizon` sessions later, net of costs). Read-only: it reports, it never tunes anything.

Run scripts/label_outcomes.py first (the bot also does it once a day).

    python scripts/analyze_mistakes.py [--horizon 20] [--min-samples 30]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.learning.mistakes import (breakdown, confidence_bucket, fmt, group_stats, load_judged, missed_gains,  # noqa: E402
                                   worst_calls)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, default=None, help="sessions to measure over (default: LEARNING_HORIZON)")
    ap.add_argument("--min-samples", type=int, default=None, help="smallest group that is judged (default: LEARNING_MIN_SAMPLES)")
    args = ap.parse_args()
    settings = load_settings()
    horizon, floor = args.horizon or settings.learning_horizon, args.min_samples or settings.learning_min_samples
    judged = load_judged(make_session_factory(settings.database_url), horizon)
    print(f"{len(judged)} decisions with a matured {horizon}-session outcome (groups under {floor} are not judged)")
    if not judged:
        print("nothing to judge yet: outcomes appear once decisions are old enough. Run scripts/label_outcomes.py.")
        return
    print("\n=== by the call the bot made (a BUY should beat the rest) ===")
    for g in breakdown(judged, lambda j: j.action, floor):
        print(fmt(g))
    print(fmt(group_stats("ALL decisions", judged, floor)))
    print("\n=== BUYs by confidence (higher confidence should do better; if not, confidence is not calibrated) ===")
    for g in breakdown([j for j in judged if j.action == "BUY"], confidence_bucket, floor):
        print(fmt(g))
    print("\n=== by who made the call ===")
    for g in breakdown(judged, lambda j: j.source or "unknown", floor):
        print(fmt(g))
    print("\n=== did the reflection critic help? (a veto is right when the vetoed stocks do WORSE than the calls it upheld) ===")
    print("(only calls the lead agent wanted to BUY reach the critic; both groups below were run through it)")
    for g in breakdown([j for j in judged if "[reflection " in j.reasoning], lambda j: "vetoed" if j.vetoed else "upheld", floor):
        print(fmt(g))
    print("\n=== by model that answered ===")
    for g in breakdown(judged, lambda j: j.model or "unknown", floor):
        print(fmt(g))
    print("\n=== by agreement with the mechanical rule ===")
    for g in breakdown(judged, lambda j: j.rule_alignment or "unknown", floor):
        print(fmt(g))
    print("\n=== worst BUYs (mistakes of commission) ===")
    for j in worst_calls(judged, "BUY"):
        print(f"{j.bar_date} {j.symbol:<12} {j.net_return:+.1%} conf {j.confidence:.2f} stop {'yes' if j.hit_stop else 'no '} | {j.reasoning[:110]}")
    print("\n=== biggest gains the bot did not buy (mistakes of omission) ===")
    for j in missed_gains(judged):
        print(f"{j.bar_date} {j.symbol:<12} {j.net_return:+.1%} was {j.action} conf {j.confidence:.2f} | {j.reasoning[:110]}")


if __name__ == "__main__":
    main()
