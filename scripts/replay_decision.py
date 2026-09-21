"""Re-examine stored decisions: the exact prompt the AI saw, what each agent said, and whether the rules reproduce the result.

  python scripts/replay_decision.py 42        one decision by id
  python scripts/replay_decision.py --last 5  the five most recent decisions"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.database import DecisionRecord, make_session_factory  # noqa: E402
from src.replay import replay  # noqa: E402


def show(r):
    print(f"--- decision {r.decision_id}: {r.symbol}  stored {r.stored_action} (conf {r.stored_confidence:.2f})")
    if r.inputs is None:
        print("    (no stored inputs: recorded before replay support)")
    else:
        print(f"    inputs: {r.inputs}")
    for name, sig in r.signals.items():
        print(f"    {name}: " + (f"{sig.action} {sig.confidence:.2f} - {sig.reasoning}" if sig else "no signal"))
    verdict = {True: "REPRODUCED", False: "DIFFERS", None: "cannot check"}[r.reproduced]
    kind = "deterministic trend exit" if r.rule_based else "recomputed from the stored signals"
    print(f"    rules {kind}: {r.recomputed_action} -> {verdict}")
    if r.prompt:
        print("    prompt shown to the AI:\n      " + r.prompt.replace("\n", "\n      "))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model reasoning contains non-ASCII characters
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?", type=int)
    ap.add_argument("--last", type=int, default=0)
    args = ap.parse_args()
    if not args.id and not args.last:
        ap.error("give a decision id or --last N")
    sessions = make_session_factory(load_settings().database_url)
    with sessions() as s:
        if args.id:
            rows = [s.get(DecisionRecord, args.id)]
        else:
            rows = list(s.scalars(select(DecisionRecord).order_by(DecisionRecord.id.desc()).limit(args.last)))
        if not rows or rows[0] is None:
            sys.exit("no such decision")
        for row in rows:
            show(replay(row))


if __name__ == "__main__":
    main()
