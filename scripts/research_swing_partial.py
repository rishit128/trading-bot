"""Swing: book half the profit at a target, let the rest run (declared 2026-11-16, before any result; the owner's question).

The owner's point: a holding up Rs 300 today may be up Rs 100 tomorrow, and the account waits for gains that come and go.
The 16 Nov exit test showed that selling a whole winner early (trails, locks, +20/30/50% targets) loses the big winners that
carry the account. This tests the middle way: sell HALF the shares once the high reaches a gain, keep the other half on the
live exits (15% stop, close under the 200-day average, +100% target).

Rules (fixed): the live configuration of scripts/research_swing_exits.py (same data, ranking, sizes, real fees and slippage);
one change per row: half20 / half30 / half50 sell half at +20% / +30% / +50% (the open when it gaps above), once a position.

Gate (the 16 Nov one, word for word): a change passes only if, against the live exits, its Sharpe is at least 0.05 higher and
its worst drawdown no worse in BOTH the full period and the last 3 years, its return is within 10% of the base's in both, both
halves of the full period are positive, and its neighbours (half30 for half20 and half50; half20 and half50 for half30) have a
Sharpe at least the base's in both windows. If several pass, the one with the highest last-3-years Sharpe is taken.

    python scripts/research_swing_partial.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_swing_exits as exits  # noqa: E402

exits.VARIANTS = {
    "base (live)": {},
    "half20": {"partial_at": 0.20},
    "half30": {"partial_at": 0.30},
    "half50": {"partial_at": 0.50},
}
exits.FAMILIES = [["half20", "half30", "half50"]]

if __name__ == "__main__":
    exits.main()
