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
        if sig is None:
            print(f"    {name}: no signal")
            continue
        print(f"    {name}: {sig.action} {sig.confidence:.2f} - {sig.reasoning}")
        if sig.details:
            d = sig.details
            chain = d.get("reasoning_chain") or {}
            for step, text in (("trend", chain.get("trend")), ("overbought", chain.get("overbought")),
                               ("volume", chain.get("volume"))):
                if text:
                    print(f"      {step}: {text}")
            print(f"      confluence: {d.get('confluence_score')} / edge confidence: {d.get('edge_confidence'):.2f}"
                  if d.get("edge_confidence") is not None else
                  f"      confluence: {d.get('confluence_score')}")
            risks = d.get("risks")
            if risks:
                print(f"      risks: {' | '.join(risks)}")
            if d.get("pattern_id"):
                print(f"      pattern: {d['pattern_id']} | adjusted confidence: "
                      f"{d.get('adjusted_signal_confidence')} | reason: {d.get('adjustment_reason')}")
            learning = d.get("learning")
            if learning:
                pf = f"{learning['profit_factor']:.2f}" if learning.get("profit_factor") is not None else "n/a"
                print(f"      phase 2 (history): {learning['sample_size']} similar closed trades, win rate "
                      f"{learning['win_rate']:.0%}, avg win {learning['avg_win_pct']:+.1f}%, avg loss "
                      f"{learning['avg_loss_pct']:+.1f}%, profit factor {pf} -> {learning['adjusted_action']} "
                      f"(conf {learning['adjusted_confidence']:.2f}, {learning['pattern_reliability']})")
                if learning.get("reason"):
                    print(f"        {learning['reason']}")
            context = d.get("context")
            if context:
                extra = f", support={context.get('macro_support')}" if context.get("macro_support") else ""
                ctx_extra = f", sector={context.get('sector_support')}, earnings_risk={context.get('earnings_risk')}, " \
                            f"diversification={context.get('diversification_score')}" if context.get("sector_support") else ""
                print(f"      phase 3 (market): regime {context.get('regime')}{extra}{ctx_extra} -> risks: "
                      f"{' | '.join(context.get('risks') or [])}")
            reflection = d.get("reflection")
            if reflection:
                commitments = reflection.get("conviction_adjustments") or []
                trail = " -> ".join(f"{c['stage']}={c['conviction']:.2f}" for c in commitments)
                print(f"      phase 4 (reflection): conviction {trail} (final {reflection['final_action']} "
                      f"conf {reflection['final_confidence']:.2f})")
                print(f"        biggest risk: {reflection['biggest_risk']} | exit if: "
                      f"{reflection['what_proves_us_wrong']}")
                bias = reflection.get("bias_check")
                if bias:
                    print(f"        biases checked: {' | '.join(bias)}")
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
