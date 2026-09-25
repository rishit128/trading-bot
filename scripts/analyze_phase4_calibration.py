"""Phase 4 calibration: does the six-stage self-critique behave as designed on real decisions?

Checks the enforced contract (review doc 4.2), which is deterministic and checkable on every stored
decision: each stage only keeps or lowers conviction, the chain ends with the fixed humility discount,
the biggest-risk / what-proves-wrong / bias fields were filled, and the final confidence never exceeds
the self-critiqued step 6. It also reports the confidence distribution before/after reflection.

It does NOT fabricate an outcome verdict: whether a humbler call wins more often needs closed trades,
and the number is reported honestly (or reported as missing).

    python scripts/analyze_phase4_calibration.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory  # noqa: E402

MIN_SAMPLE = 5


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    with sessions() as s:
        q = select(DecisionRecord).where(DecisionRecord.conviction_adjustments.isnot(None))
        rows = list(s.scalars(q.order_by(DecisionRecord.id)))
        total = s.scalar(select(func.count(DecisionRecord.id)))
        trades = list(s.scalars(select(PaperTradeRecord)))

    print(f"decisions with a stored reflection chain: {len(rows)} of {total}")
    if not rows:
        print("no reflection chain has been recorded yet: the phase only runs on non-HOLD calls from the earlier")
        print("phases, so it is quiet until actionable signals occur.")
        return

    violations = 0
    humility_checks = 0
    for d in rows:
        try:
            stages = json.loads(d.conviction_adjustments or "[]")
        except (ValueError, json.JSONDecodeError):
            violations += 1
            continue
        prev = None
        for stage in stages:
            conv = stage.get("conviction")
            if prev is not None and (conv is None or conv > prev + 1e-9):
                violations += 1  # a later stage raised conviction: breaks the monotonic contract
            if conv is not None:
                prev = conv
        if prev is not None:
            humility_checks += 1
            if (d.final_confidence or 0.0) > prev - 0.099:
                violations += 1  # the humility discount (0.10) was not applied

    print(f"\nmonotonic + humility contract violations: {violations} of {len(rows)} decisions checked")
    if violations:
        print("  FIX NEEDED: some stored chains do not follow 'conviction only ever falls, minus 0.10 humility'.")
    else:
        print("  all stored chains obey: stage conviction never rises, and final confidence <= last stage - 0.10.")

    filled_risk = sum(1 for d in rows if d.biggest_risk and d.what_proves_us_wrong)
    filled_bias = sum(1 for d in rows if d.bias_check)
    print(f"biggest-risk + exit-if filled: {filled_risk} | bias_check filled: {filled_bias} of {len(rows)}")

    deltas, base = [], 0
    for d in rows:
        if d.base_confidence is not None and d.final_confidence is not None:
            deltas.append(d.base_confidence - d.final_confidence)
        if d.base_confidence is not None:
            base += 1
    if deltas:
        print(f"\nconfidence reductions from base to final across {len(deltas)} decisions: "
              f"avg {sum(deltas)/len(deltas):+.3f} | absorbs the reflection phase's effect on average")
        print("  (review doc expects the self-critique to trim, typically by 0.15 - 0.40 when it finds real holes)")
        print(f"  base confidence recorded on {base} reflected decisions")
    else:
        print("\nno base->final deltas computable yet (reflection ran without a stored base confidence).")

    print(f"\nclosed trades so far: {len(trades)}")
    if not trades:
        print("no closed trades yet - 'humbler calls win more' cannot be measured. This is an outcome claim that only")
        print("paper-trading history will answer; nothing here invents it.")


if __name__ == "__main__":
    main()
