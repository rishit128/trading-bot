"""Decision-memory effectiveness: does the historical-pattern adjustment actually predict outcomes?

For each stored BUY decision that named a pattern, this matches the eventual closed paper trade(s)
it opened and tallies them per pattern, then reports the realised win rate per pattern together with
the learning phase's own verdict on it (pattern_reliability yes/maybe/no and confidence_in_pattern)
when that verdict was stored. Where the outcome loop has no matches yet it says so honestly.

The review doc's acceptance criteria ("pattern win rate > 40% when acted on", "reliability decisions
must separate outcomes") can only be judged from accumulated closed trades - this is the honest ruler.

    python scripts/analyze_learning_effectiveness.py"""
import json
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory  # noqa: E402

MIN_SAMPLE = 5


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _learning_details(decision) -> dict:
    try:
        tech = (json.loads(decision.signals_json or "{}").get("technical") or {}).get("details") or {}
        return (tech.get("learning") or {}) if isinstance(tech, dict) else {}
    except (ValueError, json.JSONDecodeError, TypeError):
        return {}


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    with sessions() as s:
        total = s.scalar(select(func.count(DecisionRecord.id)))
        q = select(DecisionRecord).where(DecisionRecord.pattern_id.isnot(None))
        with_patterns = list(s.scalars(q.order_by(DecisionRecord.id)))
        trades = list(s.scalars(select(PaperTradeRecord)))

    print(f"decisions with a stored pattern_id: {len(with_patterns)} of {total}")
    if not with_patterns:
        print("the learning phase has not fired yet (its memory starts empty), so there is nothing to measure.")
        print("this is expected while paper history is still accumulating.")
        return

    # Which decisions did the learning phase actually touch and what did it say?
    verdicts = {}
    adjusted = 0
    for d in with_patterns:
        if d.adjusted_signal_confidence is not None:
            adjusted += 1
        rel = _learning_details(d).get("pattern_reliability")
        if rel:
            verdicts[rel] = verdicts.get(rel, 0) + 1
    print(f"decisions the learning phase adjusted: {adjusted}")
    if verdicts:
        print(f"verdicts recorded: {dict(sorted(verdicts.items()))}")

    # Match each BUY decision to the paper trade it plausibly opened, then group by pattern.
    outcomes = {}  # pattern_id -> list of (return_pct, confidence_in_pattern, reliability)
    for d in with_patterns:
        if d.final_action != "BUY":
            continue
        opened = _aware(d.created_at)
        hits = [t for t in trades if t.symbol == d.symbol
                and abs((_aware(t.opened_at) - opened).total_seconds()) < 86400]
        if not hits:
            continue
        t = max(hits, key=lambda x: _aware(x.opened_at))
        ret = (t.exit_price / t.entry_price - 1) * 100
        learning = _learning_details(d)
        outcomes.setdefault(d.pattern_id, []).append(
            (ret, learning.get("confidence_in_pattern"), learning.get("pattern_reliability")))

    print("\npatterns named, with realised outcomes where a closed trade matched the decision:")
    if not outcomes:
        print("  no BUY decisions have a matching closed trade yet - the outcome loop needs paper history to fill in.")
    for pid, rows in sorted(outcomes.items(), key=lambda kv: -len(kv[1])):
        wins = sum(1 for r, _, _ in rows if r > 0)
        label = f"{len(rows)} trade(s), {wins / len(rows):.0%} win rate" if rows else "-"
        note = "" if len(rows) >= MIN_SAMPLE else f"   (n < {MIN_SAMPLE}: not enough to judge)"
        print(f"  {pid:<40s} {label}{note}")
    if outcomes:
        all_rows = [r for rows in outcomes.values() for r in rows]
        wins = sum(1 for r, _, _ in all_rows if r > 0)
        print(f"\noverall (all patterns): {len(all_rows)} matched trades, {wins / len(all_rows):.0%} win rate")

    print("\nread this as: the learning phase is only useful if a pattern it calls 'yes' wins more often than the")
    print("bot's baseline and one it calls 'no' does worse. A pattern is only judgeable once it has "
          f"{MIN_SAMPLE} closed trades; a 'no' pattern that still wins is a mislabelled pattern, and a 'yes' "
          "pattern under 40% over-trusts history.")


if __name__ == "__main__":
    main()
