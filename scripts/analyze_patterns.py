"""Pattern analysis: which entry conditions actually won?

Joins each closed paper trade to the decision that opened it (matched on symbol + the
decision's snapshot within ~1 day of the entry), then reports win rate and average
return split by the indicator values recorded in `decisions.snapshot_json` at entry
time - so the analysis uses exactly what the bot (and the LLM) saw, never hindsight.

Honest gates, like the other scripts: a bucket with fewer than MIN_SAMPLE trades is
reported as insufficient, not as a number; the correlation coefficients are printed
only once there is enough closed history to mean anything.

    python scripts/analyze_patterns.py"""
import json
import sys
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.database import DecisionRecord, PaperTradeRecord, make_session_factory  # noqa: E402

MIN_SAMPLE = 5  # below this a split is "insufficient", not a signal
MATCH_WINDOW = timedelta(hours=24)  # a decision and its opening fill land within a day


@dataclass
class _Entry:
    trade: object
    snap: dict


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _match(decisions, trade) -> dict:
    """The decision whose snapshot best matches the trade's symbol and opening time."""
    opened = _aware(trade.opened_at)
    best, best_delta = None, None
    for d in decisions:
        if d.symbol != trade.symbol:
            continue
        delta = abs((_aware(d.created_at) - opened).total_seconds())
        if delta <= MATCH_WINDOW.total_seconds() and (best_delta is None or delta < best_delta):
            best, best_delta = d, delta
    if best is None or not best.snapshot_json:
        return {}
    try:
        return json.loads(best.snapshot_json)
    except ValueError:
        return {}


def _split(entries, label, key):
    """Win rate / avg return for the sub-group matching `key`; None when the sample is too small."""
    group = [e for e in entries if key(e)]
    if len(group) < MIN_SAMPLE:
        return None
    wins = sum(1 for e in group if e.trade.net_pnl > 0)
    avg = sum(e.trade.net_pnl for e in group) / len(group)
    return len(group), wins / len(group), avg


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    with sessions() as s:
        decisions = list(s.scalars(select(DecisionRecord)))
        trades = list(s.scalars(select(PaperTradeRecord)))

    print("=== Entry-condition analysis ===")
    print(f"decisions: {len(decisions)} | closed trades: {len(trades)}")
    if len(trades) < MIN_SAMPLE:
        print(f"  fewer than {MIN_SAMPLE} closed trades - the win rate is not")
        print("  yet measurable. This script turns real (not backtest) evidence into")
        print("  \"which entries win\", once paper history exists.")
        return

    entries = [_Entry(t, _match(decisions, t)) for t in trades]
    matched = sum(1 for e in entries if e.snap)
    print(f"entries with a stored entry snapshot: {matched}/{len(trades)}")
    print("  (these drive every split)")

    overall_wins = sum(1 for t in trades if t.net_pnl > 0)
    print(f"\noverall win rate: {overall_wins}/{len(trades)} = {overall_wins/len(trades):.0%} | "
          f"net P&L {sum(t.net_pnl for t in trades):+,.0f}")

    with_snap = [e for e in entries if e.snap]
    if len(with_snap) < MIN_SAMPLE:
        print("  too few matched snapshots to split by condition yet.")
        return

    def above_ma50(e):
        return e.snap.get("price", 0) > e.snap.get("ma50", 0)

    def bollinger_tight(e):
        bb = e.snap.get("bb_position")
        return bb is not None and 0.25 <= bb <= 0.75

    def momentum_positive(e):
        return (e.snap.get("momentum_6m") or 0) > 0

    def macd_bullish(e):
        return (e.snap.get("macd_histogram") or 0) > 0

    def volume_up(e):
        return (e.snap.get("volume_trend") or 0) > 0

    print("\nwin rate by entry condition "
          "(>= {0} trades needed, else 'insufficient'):".format(MIN_SAMPLE))
    rows = [("price above MA50", above_ma50),
            ("BB position 25-75% (not stretched)", bollinger_tight),
            ("6-month momentum positive", momentum_positive),
            ("MACD histogram positive", macd_bullish),
            ("volume trend rising", volume_up)]
    for label, key in rows:
        out = _split(with_snap, label, key)
        if out:
            n, wr, avg = out
            print(f"  {label:34s} {wr:.0%} win rate ({n} trades), avg {avg:+,.0f} per trade")
        else:
            print(f"  {label:34s} insufficient (fewer than {MIN_SAMPLE} matching trades)")

    best = _split(with_snap, "above MA50 and not stretched",
                  lambda e: above_ma50(e) and bollinger_tight(e))
    worst = _split(with_snap, "MACD bearish", lambda e: not macd_bullish(e))
    for label, out in (("above MA50 AND BB 25-75%", best), ("MACD histogram negative", worst)):
        if out:
            n, wr, avg = out
            print(f"  {label:34s} {wr:.0%} win rate ({n} trades), avg {avg:+,.0f} per trade")

    print("\nwhich condition separates winners best (needs the most data to mean anything):")
    corr = []
    ma_ratio = lambda e: e.snap.get("price", 0) / e.snap.get("ma50", 1)  # noqa: E731
    for name, key in (("ma_ratio (price/MA50)", ma_ratio),
                      ("rsi", lambda e: e.snap.get("rsi")),
                      ("bb_position", lambda e: e.snap.get("bb_position")),
                      ("volume_trend", lambda e: e.snap.get("volume_trend")),
                      ("adx", lambda e: e.snap.get("adx"))):
        pairs = [(key(e), e.trade.net_pnl) for e in with_snap if key(e) is not None]
        if len(pairs) < MIN_SAMPLE * 3:
            print(f"  {name:28s} insufficient data")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        # Spearman rank correlation to stay robust to outliers in an early sample.
        xs_r = {v: i for i, v in enumerate(sorted(xs))}
        ys_r = {v: i for i, v in enumerate(sorted(ys))}
        rx = [xs_r[v] for v in xs]
        ry = [ys_r[v] for v in ys]
        n = len(pairs)
        cov = sum((a - (n - 1) / 2) * (b - (n - 1) / 2)
                  for a, b in zip(rx, ry))
        denom = n * (n * n - 1) / 12
        corr.append((name, cov / denom if denom else 0.0))
    for name, c in sorted(corr, key=lambda x: -abs(x[1])):
        print(f"  {name:28s} {c:+.3f}")


if __name__ == "__main__":
    main()
