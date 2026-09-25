"""Reflection check: does the reflection step behave as designed on real decisions?

Reflection is a VETO ONLY: the critic may turn a call into a HOLD/SELL, but it does not shave the confidence of a call it
upholds (its monotone, humility-discounted conviction is stored as `critic_confidence` for audit and never gates a trade).
On every stored decision this checks that contract: each stage of the critic's chain only keeps or lowers conviction, the
critic's number is the last stage minus the 0.10 humility discount, an upheld call keeps its confidence, and a vetoed one
took the critic's confidence. Decisions recorded before the veto-only change carried the discount in the final confidence and are
counted separately as legacy. It also reports the veto rate.

It does NOT fabricate an outcome verdict: whether vetoed calls would have lost needs closed trades, and the number is
reported honestly (or reported as missing).

    python scripts/analyze_reflection.py"""
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

    violations, legacy, upheld_n, vetoed_n = 0, 0, 0, 0
    for d in rows:
        try:
            stages = json.loads(d.conviction_adjustments or "[]")
            reflection = ((json.loads(d.signals_json or "{}").get("technical") or {}).get("details") or {}).get("reflection") or {}
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
        if "upheld" not in reflection:
            legacy += 1  # pre-veto decisions: the humility discount was applied to the final confidence itself
            if prev is not None and (d.final_confidence or 0.0) > prev - 0.099:
                violations += 1
            continue
        if prev is not None and abs(reflection.get("critic_confidence", -1) - max(0.0, prev - 0.10)) > 1e-3:
            violations += 1  # the audit number is not last stage minus humility
        if reflection["upheld"]:
            upheld_n += 1
            entering = reflection.get("base_confidence_entering")
            if entering is not None and abs((d.final_confidence or 0.0) - entering) > 1e-3:
                violations += 1  # an upheld call must keep its confidence
        else:
            vetoed_n += 1
            if abs((d.final_confidence or 0.0) - reflection.get("critic_confidence", -1)) > 1e-3:
                violations += 1  # a vetoed call takes the critic's own (humility-discounted) confidence

    print(f"\nreflection contract violations: {violations} of {len(rows)} decisions checked "
          f"({legacy} legacy pre-veto, {upheld_n} upheld, {vetoed_n} vetoed)")
    if upheld_n + vetoed_n:
        print(f"  veto rate on current-contract decisions: {vetoed_n / (upheld_n + vetoed_n):.0%}")
    if violations:
        print("  FIX NEEDED: some stored chains break the contract described at the top of this script.")
    else:
        print("  all stored chains obey the contract.")

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
        print(f"\nconfidence change from base to final across {len(deltas)} decisions: avg {sum(deltas)/len(deltas):+.3f}")
        print("  (legacy decisions carry the old humility discount; current ones change only when a phase before reflection did)")
        print(f"  base confidence recorded on {base} reflected decisions")
    else:
        print("\nno base->final deltas computable yet (reflection ran without a stored base confidence).")

    print(f"\nclosed trades so far: {len(trades)}")
    if not trades:
        print("no closed trades yet - 'humbler calls win more' cannot be measured. This is an outcome claim that only")
        print("paper-trading history will answer; nothing here invents it.")


if __name__ == "__main__":
    main()
