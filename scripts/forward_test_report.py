"""Forward test of each paper account under the frozen protocol (docs/trading/forward-tests.md). Read only.

Per account: the trades made since the freeze under the frozen version only (a trade under another version means the rule
changed and the test restarted), their mean net return and date-clustered t, how many trading dates the test needs to see
the target edge (from the spread of its own per-date results), drift against the backtest's expectation, and, for the
intraday accounts, the share of entries whose signal is gone on the final bars. Every trade is costed from its fill prices at
today's fee schedule (src.engine.costs), not at what was booked: the intraday accounts were booked at a wrong schedule from
2026-11-06 to 2026-11-11 and the swing account at a zero-brokerage plan until 2026-11-11. Then the protocol's verdict:
  RUNNING      fewer dates than needed (or fewer than MIN_DATES): no verdict, nothing to change;
  GO           enough dates, mean net > 0, t >= 2 and both halves > 0: eligible for a small real-money trial;
  REDESIGN     enough dates and not GO: the frozen rule has no forward edge; design a new version (paper keeps running);
  INVESTIGATE  at any time: live is worse than the backtest by z <= -2, or over 10% of verified entries vanish on the
               final bars; run scripts/reconcile_paper_replay.py before reading the result as the rule's.
The thresholds are the protocol's, fixed 2026-10-26 before the first frozen session; change them only with a new protocol.

    python scripts/forward_test_report.py [--account orb|gaponly|news|calls|swing]"""

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
from src.data.price_history import load_index_close  # noqa: E402
from src.engine.costs import india_delivery_fees, intraday_fees, net_return  # noqa: E402
from src.research.significance import units_needed  # noqa: E402
from src.versions import CALLS_VERSION, GAPONLY_VERSION, INTRADAY_VERSION, NEWS_VERSION, STRATEGY_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FREEZE = "2026-11-04"  # the first full session under the frozen versions (the 09:30 feed fix and the trade-to-trade check)
MIN_DATES, T_GO, DRIFT_Z, VANISH_SHARE = 30, 2.0, -2.0, 0.10


@dataclass(frozen=True)
class Account:
    db: str
    version: str
    target: float  # the net edge per trade the test must be able to see
    expected: Optional[float]  # the backtest's mean net per trade (None: no backtest)
    expected_source: str
    signals: bool  # has intraday_signals (version stamp and final-bar check)
    freeze: str = FREEZE  # the first full session under this account's version


ACCOUNTS = {
    "orb": Account(
        "intraday.db",
        INTRADAY_VERSION,
        0.0025,
        -0.0009,
        "orb-v7 on the Angel One bars of 2026, 2,110 signals",
        True,
        "2026-11-13",
    ),
    "gaponly": Account(
        "gaponly.db",
        GAPONLY_VERSION,
        0.0025,
        0.0044,
        "replay of 68 sessions, 151 trades, the sample the rule came from",
        True,
        "2026-11-13",
    ),
    "news": Account("news.db", NEWS_VERSION, 0.0025, None, "no backtest (needs results history)", True),
    "calls": Account(
        "broker_calls.db",
        CALLS_VERSION,
        0.0025,
        -0.0003,
        "first look at 15 BUY calls on saved Angel One bars",
        True,
        "2026-11-13",
    ),
    "swing": Account("trading.db", STRATEGY_VERSION, 0.03, 0.081, "10-year live-configuration backtest", False, "2026-11-16"),
}


def _read(db: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return pd.read_sql(query, con)


def _versions(db: Path, account: Account) -> pd.Series:
    """The version each recorded entry since the freeze was made under."""
    if account.signals:
        return _read(db, f"select system_version from intraday_signals where outcome = 'ENTERED' and day >= '{account.freeze}'")[
            "system_version"
        ]
    frame = _read(db, f"select strategy_version from decisions where final_action = 'BUY' and created_at >= '{account.freeze}'")
    return frame["strategy_version"]


def clustered(per_trade: pd.Series, days: pd.Series) -> tuple[float, float, pd.Series]:
    """Mean per trade, t from the per-date means, and the per-date means."""
    per_date = per_trade.groupby(days.values).mean()
    if len(per_date) < 2 or per_date.std(ddof=1) == 0:
        return float(per_trade.mean()), float("nan"), per_date
    return float(per_trade.mean()), float(per_date.mean() / (per_date.std(ddof=1) / np.sqrt(len(per_date)))), per_date


def against_the_index(db: Path) -> None:
    """The swing account's profit next to what the same money, put in on the same days, would have made in the Nifty 50. Money
    added later (the budget was raised once) shows as the one jump in equity that matches initial_cash less the first equity."""
    equity = _read(db, "select created_at as at, equity from equity_history order by created_at")
    paid_in = _read(db, "select initial_cash from paper_account")["initial_cash"]
    if len(equity) < 2 or paid_in.empty:
        return
    try:
        nifty = load_index_close("^NSEI", 3)
    except Exception as e:  # the report must not fail on a missing index cache
        print(f"account against the Nifty 50: index not available ({type(e).__name__})")
        return
    days = pd.DatetimeIndex(equity["at"]).normalize()
    index = nifty.copy()
    index.index = pd.DatetimeIndex(index.index).tz_localize(None).normalize()
    index = index.loc[: days[-1]]
    if index.loc[days[0] :].empty:
        return

    def level(day: pd.Timestamp) -> float:
        return float(index.loc[:day].iloc[-1]) if len(index.loc[:day]) else float(index.iloc[0])

    first, added = float(equity["equity"].iloc[0]), float(paid_in.iloc[0]) - float(equity["equity"].iloc[0])
    in_index = first * level(days[-1]) / level(days[0])
    step = equity["equity"].diff()
    if added > 0.01 * first and abs(step.max() - added) < 0.2 * added:
        in_index += added * level(days[-1]) / level(days[int(step.idxmax())])
    else:
        added = 0.0
    put_in = first + added
    print(
        f"since {days[0].date()} on Rs {put_in:,.0f} put in: account Rs {equity['equity'].iloc[-1] - put_in:+,.0f} "
        f"({equity['equity'].iloc[-1] / put_in - 1:+.1%}); the same money in the Nifty 50 Rs {in_index - put_in:+,.0f} "
        f"({in_index / put_in - 1:+.1%})"
    )


def report(name: str, account: Account) -> None:
    db = ROOT / account.db
    print(f"\n=== {name}: frozen at {account.version} from {account.freeze} ({db.name}) ===")
    if not db.exists():
        print("no database yet: RUNNING")
        return
    trades = _read(db, "select symbol, qty, entry_price, exit_price, opened_at from paper_trades")
    if len(trades):
        fees = india_delivery_fees if name == "swing" else intraday_fees
        trades["day"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(IST).date().isoformat() for t in trades["opened_at"]]
        trades["net"] = [net_return(t.entry_price, t.exit_price, t.qty, fees) for t in trades.itertuples()]
    if name == "swing":
        against_the_index(db)
    before = trades[trades["day"] < account.freeze] if len(trades) else trades
    test = trades[trades["day"] >= account.freeze] if len(trades) else trades
    if len(before):
        print(f"before the freeze (history only, not evidence): {len(before)} trades, mean net {before['net'].mean():+.3%}")

    versions = _versions(db, account)
    other = sorted(set(versions.dropna()) - {account.version})
    if other:
        print(
            f"TEST RESTARTED: entries since the freeze under {other}, not {account.version}. Set a new freeze for the new rule."
        )
        return

    flags = []
    if account.signals and name != "news":
        check = _read(
            db,
            f"select final_signal from intraday_signals where outcome = 'ENTERED' and day >= '{account.freeze}' "
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
