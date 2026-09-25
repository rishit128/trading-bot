"""Phase 3 effectiveness: is the market/sector context phase acting, and does it help?

Reports what the context phase has actually judged (regime, macro/sector support, earnings risk,
diversification score), and how often a risk-off or euphoric regime made the bot stand down or cut
confidence. Outcome-based checks (did risk-off decisions win/lose differently?) need accumulated
closed trades; the script prints them honestly when they exist and says so when they don't.

The review doc's downstream criteria (e.g. "risk-off reduces confidence by > 0.10") are judged on
real decisions here rather than fabricated in a unit test.

    python scripts/analyze_context_effectiveness.py"""
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


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    with sessions() as s:
        q = select(DecisionRecord).where(DecisionRecord.context_regime.isnot(None))
        rows = list(s.scalars(q.order_by(DecisionRecord.id)))
        total = s.scalar(select(func.count(DecisionRecord.id)))
        trades = list(s.scalars(select(PaperTradeRecord)))

    print(f"decisions with a stored regime label: {len(rows)} of {total}")
    if not rows:
        print("no context phase has fired yet: the regime was calm (or index data unavailable) on every cycle.")
        print("this is expected in a rolling normal market; the phase only acts on notable regimes.")
        return

    regimes, macro, sector = {}, {}, {}
    cuts = 0
    for d in rows:
        regimes[d.context_regime] = regimes.get(d.context_regime, 0) + 1
        if d.macro_support:
            macro[d.macro_support] = macro.get(d.macro_support, 0) + 1
        if d.sector_support:
            sector[d.sector_support] = sector.get(d.sector_support, 0) + 1
        if d.base_confidence and d.base_confidence - d.final_confidence >= 0.10:
            cuts += 1
    print(f"\nregimes seen: {dict(sorted(regimes.items()))}")
    if macro:
        print(f"macro support verdicts: {dict(sorted(macro.items()))}")
    if sector:
        print(f"sector support verdicts: {dict(sorted(sector.items()))}")
    print(f"earnings-risk-true decisions: {sum(1 for d in rows if d.earnings_risk)}")
    divs = [d.diversification_score for d in rows if d.diversification_score is not None]
    if divs:
        print(f"diversification score distribution: avg {sum(divs)/len(divs):.1f} over {len(divs)}")
    print(f"\ndecisions where the context phase cut confidence by >= 0.10: {cuts} of {len(rows)}")

    lit = [d for d in rows if d.base_confidence is not None and d.final_confidence is not None]
    if lit:
        avg_delta = sum(d.base_confidence - d.final_confidence for d in lit) / len(lit)
        print(f"avg confidence change across notable-regime decisions: {avg_delta:+.3f}")
        print(f"  (review doc asks risk-off to push confidence down > 0.10 on average; judge at n >= {MIN_SAMPLE})")

    print(f"\nclosed trades so far: {len(trades)}")
    if not trades:
        print("no closed trades yet - the outcome verdict ('risk-off decisions do worse/better') cannot be measured.")
        print("keep paper-trading; this column only fills in as regime events are caught live.")


if __name__ == "__main__":
    main()
