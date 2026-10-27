"""Forward test of each paper account under the frozen protocol (docs/trading/forward-tests.md). Read only.

Per account: the trades made since the freeze under the frozen version only (a trade under another version means the rule
changed and the test restarted), their mean net return and date-clustered t, how many trading dates the test needs to see
the target edge (from the spread of its own per-date results), drift against the backtest's expectation, and, for the
intraday accounts, the share of entries whose signal is gone on the final bars. Then the protocol's verdict:
  RUNNING      fewer dates than needed (or fewer than MIN_DATES): no verdict, nothing to change;
  GO           enough dates, mean net > 0, t >= 2 and both halves > 0: eligible for a small real-money trial;
  REDESIGN     enough dates and not GO: the frozen rule has no forward edge; design a new version (paper keeps running);
  INVESTIGATE  at any time: live is worse than the backtest by z <= -2, or over 10% of verified entries vanish on the
               final bars; run scripts/reconcile_paper_replay.py before reading the result as the rule's.
The thresholds are the protocol's, fixed 2026-10-26 before the first frozen session; change them only with a new protocol.

    python scripts/forward_test_report.py [--account orb|gapgo|gaponly|news|swing]"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.india import IST  # noqa: E402
from src.research.significance import units_needed  # noqa: E402
from src.versions import GAPGO_VERSION, GAPONLY_VERSION, INTRADAY_VERSION, NEWS_VERSION, STRATEGY_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FREEZE = "2026-10-28"  # the first full session under the frozen versions (10-27 opened on the old code)
MIN_DATES, T_GO, DRIFT_Z, VANISH_SHARE = 30, 2.0, -2.0, 0.10


@dataclass(frozen=True)
class Account:
    db: str
    version: str
    target: float  # the net edge per trade the test must be able to see
    expected: Optional[float]  # the backtest's mean net per trade (None: no backtest)
    expected_source: str
    signals: bool  # has intraday_signals (version stamp and final-bar check)


ACCOUNTS = {
    "orb": Account("intraday.db", INTRADAY_VERSION, 0.0025, -0.00225, "8 years Angel One, 116,908 trades", True),
    "gapgo": Account("gapgo.db", GAPGO_VERSION, 0.0025, -0.0006, "8 years Angel One, 1,815 trades", True),
    "gaponly": Account("gaponly.db", GAPONLY_VERSION, 0.0025, -0.00107, "2 years 1-minute data, 2,806 trades", True),
    "news": Account("news.db", NEWS_VERSION, 0.0025, None, "no backtest (needs results history)", True),
    "swing": Account("trading.db", STRATEGY_VERSION, 0.03, 0.081, "10-year live-configuration backtest", False),
}


def _read(db: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return pd.read_sql(query, con)


def _versions(db: Path, account: Account) -> pd.Series:
    """The version each recorded entry since the freeze was made under."""
    if account.signals:
        return _read(db, f"select system_version from intraday_signals where outcome = 'ENTERED' and day >= '{FREEZE}'")[
            "system_version"
        ]
    frame = _read(db, f"select strategy_version from decisions where final_action = 'BUY' and created_at >= '{FREEZE}'")
    return frame["strategy_version"]


def clustered(per_trade: pd.Series, days: pd.Series) -> tuple[float, float, pd.Series]:
    """Mean per trade, t from the per-date means, and the per-date means."""
    per_date = per_trade.groupby(days.values).mean()
    if len(per_date) < 2 or per_date.std(ddof=1) == 0:
        return float(per_trade.mean()), float("nan"), per_date
    return float(per_trade.mean()), float(per_date.mean() / (per_date.std(ddof=1) / np.sqrt(len(per_date)))), per_date


def report(name: str, account: Account) -> None:
    db = ROOT / account.db
    print(f"\n=== {name}: frozen at {account.version} from {FREEZE} ({db.name}) ===")
    trades = _read(db, "select symbol, qty, entry_price, net_pnl, opened_at from paper_trades")
    if len(trades):
        trades["day"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(IST).date().isoformat() for t in trades["opened_at"]]
        trades["net"] = trades["net_pnl"] / (trades["entry_price"] * trades["qty"])
    before = trades[trades["day"] < FREEZE] if len(trades) else trades
    test = trades[trades["day"] >= FREEZE] if len(trades) else trades
    if len(before):
        print(f"before the freeze (history only, not evidence): {len(before)} trades, mean net {before['net'].mean():+.3%}")

    versions = _versions(db, account)
    other = sorted(set(versions.dropna()) - {account.version})
    if other:
        print(
            f"TEST RESTARTED: entries since the freeze under {other}, not {account.version}. Set a new FREEZE for the new rule."
        )
        return

    flags = []
    if account.signals and name != "news":
        check = _read(
            db,
            f"select final_signal from intraday_signals where outcome = 'ENTERED' and day >= '{FREEZE}' "
            "and final_signal is not null",
        )["final_signal"]
        if len(check):
            vanished = float((check == 0).mean())
            print(f"entries whose signal is gone on the final bars: {int((check == 0).sum())} of {len(check)} ({vanished:.0%})")
            if vanished > VANISH_SHARE and len(check) >= 20:
                flags.append(f"{vanished:.0%} of entries vanish on the final bars (limit {VANISH_SHARE:.0%})")

    if test.empty:
        print("no trades under the frozen rule yet: RUNNING")
        return
    mean, t, per_date = clustered(test["net"], test["day"])
    n_dates = len(per_date)
    sd = float(per_date.std(ddof=1)) if n_dates >= 2 else float("nan")
    needed = max(MIN_DATES, units_needed(account.target, sd)) if sd == sd and sd > 0 else MIN_DATES
    print(
        f"{len(test)} trades on {n_dates} dates | mean net {mean:+.3%} a trade, t {t:+.2f} by date | "
        f"needs {needed} dates to see {account.target:+.2%} (80% power)"
    )
    if account.expected is not None and n_dates >= 2 and sd > 0:
        z = (float(per_date.mean()) - account.expected) / (sd / np.sqrt(n_dates))
        print(f"backtest expects {account.expected:+.3%} a trade ({account.expected_source}); live is z {z:+.2f} from it")
        if z <= DRIFT_Z:
            flags.append(f"live is worse than the backtest (z {z:+.2f} <= {DRIFT_Z})")

    if flags:
        verdict = "INVESTIGATE: " + "; ".join(flags) + ". Run scripts/reconcile_paper_replay.py."
    elif n_dates < needed:
        verdict = f"RUNNING ({n_dates} of {needed} dates)"
    else:
        half = n_dates // 2
        halves = per_date.iloc[:half].mean(), per_date.iloc[half:].mean()
        go = mean > 0 and t >= T_GO and min(halves) > 0
        verdict = f"{'GO' if go else 'REDESIGN'} (mean {mean:+.3%}, t {t:+.2f}, halves {halves[0]:+.3%} / {halves[1]:+.3%})"
    print("verdict:", verdict)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--account", choices=sorted(ACCOUNTS), help="one account (default: all)")
    args = ap.parse_args()
    for name in [args.account] if args.account else ACCOUNTS:
        report(name, ACCOUNTS[name])


if __name__ == "__main__":
    main()
