"""Phase 0 baseline: how calibrated are the bot's stored decisions so far?

Reads the live database (decisions + orders + closed paper trades) and reports:
  * decision volume and action mix
  * closed-trade win rate and exits (what the bot actually did)
  * calibration: realised win rate vs the stored final_confidence of each BUY that opened a position,
    printed per confidence bucket with the sample size made explicit (a bucket with < 5 trades is
    reported as "insufficient", not as a number).

Roughly half the roadmap's success criteria ("does confidence predict outcome?") can only be answered
from accumulated decisions, never from a unit test. This script is the honest before/after ruler:
Phase 1 adds the reasoning chain to the record but should NOT move these numbers; a later phase claims
to. Re-run it after each phase.

    python scripts/analyze_calibration.py"""
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory  # noqa: E402
from src.research.calibration import (  # noqa: E402
    DEFAULT_EDGES,
    MIN_SAMPLE,
    bucket_profiles,
    expected_calibration_error,
    monotone_in_confidence,
    brier_score,
)

MAYBE_OPEN_WINDOW_SECONDS = 86400  # a decision and its opening fill land within the same day
Pair = tuple  # (confidence, win=0/1)


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    with sessions() as s:
        decisions = s.scalars(select(DecisionRecord).order_by(DecisionRecord.id)).all()
        trades = list(s.scalars(select(PaperTradeRecord)))

    print(f"decisions: {len(decisions)}")
    if decisions:
        mix, approved = {}, 0
        for d in decisions:
            mix[d.final_action] = mix.get(d.final_action, 0) + 1
            if d.final_action == "BUY" and d.risk_approved and d.final_confidence >= settings.risk.min_confidence:
                approved += 1
        print(f"  by action: {dict(sorted(mix.items()))}")
        span = max((d.created_at for d in decisions), default=None)
        print(f"  approved BUYs above min confidence: {approved}")
        if span is not None:
            print(f"  first decision: {_aware(min(d.created_at for d in decisions)).date()}, last: {_aware(span).date()}")

    if trades:
        wins = sum(1 for t in trades if t.net_pnl > 0)
        by_reason = {}
        for t in trades:
            by_reason[t.reason] = by_reason.get(t.reason, 0) + 1
        print(f"closed trades: {len(trades)} | win rate: {len(wins) / len(trades):.1%} "
              f"(break-even ~50% after -8% stop/+1% trend-exit skew) | net P&L: "
              f"{sum(t.net_pnl for t in trades):+,.0f} | exits: {dict(sorted(by_reason.items()))}")
    else:
        print("closed trades: 0 (nothing to calibrate against yet)")

    pairs = []
    for d in decisions:
        if d.final_action != "BUY" or not d.risk_approved:
            continue
        decided_at = _aware(d.created_at)
        hit = [t for t in trades
               if t.symbol == d.symbol and abs((_aware(t.opened_at) - decided_at).total_seconds()) < MAYBE_OPEN_WINDOW_SECONDS]
        if not hit:
            continue
        t = max(hit, key=lambda t: _aware(t.opened_at))  # the fill that this decision actually opened
        profit = t.exit_price / t.entry_price - 1
        pairs.append((d.final_confidence, 1 if profit > 0 else 0))

    print(f"\ncalibration (approved BUYs matched to a closed trade): {len(pairs)} matched")
    if pairs:
        for p in bucket_profiles(pairs, DEFAULT_EDGES):
            note = (f"realised win rate {p.realized_rate:.0%} vs confidence {p.mean_confidence:.2f}"
                    if p.n >= MIN_SAMPLE else "insufficient sample, not quoted")
            print(f"  conf [{p.lo:.2f}, {p.hi:.2f}) n={p.n:<3d} {note}")
        print(f"  ECE (weighted |realized - confidence| over sufficient buckets): "
              f"{expected_calibration_error(bucket_profiles(pairs, DEFAULT_EDGES)):.2%}")
        print(f"  Brier score: {brier_score(pairs):.3f} (random ~ base_rate*(1-base_rate))")
        trend = "rises with confidence" if monotone_in_confidence(pairs, DEFAULT_EDGES) else "does NOT rise with confidence"
        print(f"  realized win rate vs confidence so far: {trend}")
        print("  All sites answer 'does confidence predict outcome?', but act only when every bucket has n >= 5.")
    else:
        print("  no matched trades yet: keep paper-trading, or backtest, before judging calibration.")
    print("\nread this as: a before/after ruler. Phase 1 changes the decision record only; later phases claim to move")
    print("these numbers. Re-run after each phase and act only on samples big enough to mean anything.")


if __name__ == "__main__":
    main()