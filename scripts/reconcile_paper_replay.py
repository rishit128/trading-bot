"""Paper account against a replay of the same sessions: is the gap between them execution or the rule? Read only.

The replay (src/research/intraday_replay.py) runs the live engine and paper broker over the archived, final 5-minute bars.
The paper account traded the same rule live, on the bars as they were served during the session. So:
  * a trade in both, with different prices, measures execution (live prices, timing, bar revisions, slippage);
  * a trade in only one of them measures a different decision: a signal live saw on bars that were later revised (or
    the reverse), or a different account state (cash, open positions, the daily loss halt).
The replay is run one session at a time on that session's live universe IN ITS LIVE ORDER, with that day's house-call
stocks appended as live appends them: the engine takes breakouts in the order the download returns them and stops at the
position limit, so a different order (the replay's default is alphabetical) picks different stocks. Each session starts
from the configured capital (the paper account keeps no equity history); a matched trade's prices are compared as
returns, so the different sizes do not matter (the fees are proportional below Rs 66,000 a side).

Pre-declared (2026-10-26, before the first run; skills: backtest-vs-broker-fill-reconciliation,
order-book-and-intrabar-price-path-modeling). Execution is called MATERIAL if any of:
  E1 matched trades: the median adverse entry difference (paper entry above the replay's) exceeds 0.05%;
  E2 matched trades: the mean net-return difference (paper minus replay) is below -0.05% a trade with |t| >= 2, t from
     the per-date means (trades on one date are not independent);
  E3 decisions: entries in only one of the two exceed 10% of all distinct entries.
Otherwise the paper losses are the rule's, and work belongs on the rule, not the plumbing. Also reported, no threshold:
the share of exits whose bar touched both the stop and the target (the broker books the stop, the pessimistic reading;
5-minute bars cannot say which came first). Too few matched trades (< 30) gives no verdict for that strategy.

    python scripts/reconcile_paper_replay.py [--strategy orb|gapgo|gaponly|all] [--csv logs/reconcile.csv]"""

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import open_bar_archive  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.india import IST  # noqa: E402
from src.research.intraday_replay import ReplayConfig, replay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATABASES = {"orb": "intraday.db", "gapgo": "gapgo.db", "gaponly": "gaponly.db"}
E1_ENTRY, E2_RETURN, E2_T, E3_SHARE, MIN_MATCHED = 0.0005, -0.0005, 2.0, 0.10, 30


def _utc(ts) -> pd.Timestamp:
    """The databases store naive UTC; the replay returns aware UTC."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def live_universes(db: Path) -> Dict[str, list]:
    """The universe each session ran on, in the engine's order (the session's last record)."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = con.execute("select day, symbols_json from intraday_universe order by id").fetchall()
    return {day: json.loads(symbols) for day, symbols in rows}


def load_paper(db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The paper account's closed trades and its ENTERED signals (stop, target, house call, the engine's own check of the
    signal against the final bars), read only."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        trades = pd.read_sql(
            "select symbol, qty, entry_price, exit_price, opened_at, closed_at, reason, net_pnl from paper_trades", con
        )
        signals = pd.read_sql(
            "select day, symbol, bar_time, stop, target, house_call, final_signal from intraday_signals where outcome = 'ENTERED'",
            con,
        )
    return trades, signals


def keyed(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (session date, symbol) with gross and net returns."""
    t = trades.copy()
    t["day"] = [_utc(x).tz_convert(IST).date().isoformat() for x in t["opened_at"]]
    t["gross"] = t["exit_price"] / t["entry_price"] - 1
    t["net"] = t["net_pnl"] / (t["entry_price"] * t["qty"])
    return t.drop_duplicates(["day", "symbol"]).set_index(["day", "symbol"])


def touched_both(bars: Dict[str, pd.DataFrame], symbol: str, opened_at, closed_at, stop, target) -> Optional[bool]:
    """Whether the bar the exit came from reached both the stop and the target (None when it cannot be told)."""
    df = bars.get(symbol)
    if df is None or stop is None or target is None or pd.isna(stop) or pd.isna(target):
        return None
    idx = pd.DatetimeIndex(df.index).tz_convert("UTC")
    start, end = _utc(opened_at), _utc(closed_at)
    window = df[(idx >= start.floor("5min")) & (idx <= end)]
    for _, bar in window.iterrows():
        hit_stop, hit_target = bar["Low"] <= stop, bar["High"] >= target
        if hit_stop or hit_target:
            return bool(hit_stop and hit_target)
    return None


def date_t(diff: pd.Series) -> tuple[float, float, int]:
    """Mean of a per-trade series, and its t from the per-date means."""
    per_date = diff.groupby(level="day").mean()
    n = len(per_date)
    if n < 2 or per_date.std(ddof=1) == 0:
        return float(diff.mean()), float("nan"), n
    return float(diff.mean()), float(per_date.mean() / (per_date.std(ddof=1) / np.sqrt(n))), n


def reconcile(strategy: str, csv: Optional[Path]) -> None:
    db = ROOT / DATABASES[strategy]
    paper_raw, paper_signals = load_paper(db)
    print(f"\n=== {strategy} ({db.name}) ===")
    if paper_raw.empty:
        print("no paper trades")
        return
    paper = keyed(paper_raw)
    trade_days = sorted(paper.index.get_level_values("day").unique())
    first, last = trade_days[0], trade_days[-1]
    universes = live_universes(db)
    # a week earlier too: the gap rules need the previous session's close
    bars = open_bar_archive().load((date.fromisoformat(first) - timedelta(days=10)).isoformat(), last)
    archived = {d.isoformat() for df in bars.values() for d in pd.DatetimeIndex(df.index).tz_convert(IST).date}
    days = sorted(d for d in universes if first <= d <= last and d in archived)  # every session the engine ran
    config = ReplayConfig.from_settings(load_settings())
    print(f"paper: {len(paper)} trades, {first} .. {last}; replaying {len(days)} sessions on each day's live universe")
    calls = paper_signals[paper_signals["house_call"].notna()]
    trades, signals = [], []
    for day in days:
        added = [s for s in calls.loc[calls["day"] == day, "symbol"] if s not in universes[day]]
        universe = [s for s in universes[day] + added if s in bars]
        result = replay({s: bars[s] for s in universe}, config, universe=universe, start=day, end=day, strategy=strategy)
        trades.append(result.trades)
        signals.append(result.signals)
    rep_trades = pd.concat([t for t in trades if len(t)], ignore_index=True) if any(len(t) for t in trades) else pd.DataFrame()
    rep = keyed(rep_trades) if len(rep_trades) else paper.iloc[0:0]
    rep_signals = pd.concat([x for x in signals if len(x)], ignore_index=True) if any(len(x) for x in signals) else pd.DataFrame()

    both = paper.index.intersection(rep.index)
    only_paper, only_replay = paper.index.difference(rep.index), rep.index.difference(paper.index)
    sig = paper_signals.drop_duplicates(["day", "symbol"]).set_index(["day", "symbol"])
    rsig = (
        rep_signals.drop_duplicates(["day", "symbol"], keep="last").set_index(["day", "symbol"])
        if len(rep_signals)
        else pd.DataFrame()
    )

    m = pd.DataFrame(index=both)
    m["entry_adverse"] = paper.loc[both, "entry_price"] / rep.loc[both, "entry_price"] - 1
    m["exit_adverse"] = 1 - paper.loc[both, "exit_price"] / rep.loc[both, "exit_price"]
    m["net_diff"] = paper.loc[both, "net"] - rep.loc[both, "net"]
    m["same_reason"] = paper.loc[both, "reason"] == rep.loc[both, "reason"]
    m["paper_reason"], m["replay_reason"] = paper.loc[both, "reason"], rep.loc[both, "reason"]
    m["entry_minutes_late"] = [
        (_utc(paper.loc[k, "opened_at"]) - _utc(rep.loc[k, "opened_at"])).total_seconds() / 60 for k in both
    ]

    print(f"\nmatched {len(both)} | paper only {len(only_paper)} | replay only {len(only_replay)}")
    print(f"paper  : mean net {paper['net'].mean():+.3%} a trade over {len(paper)}, total Rs {paper['net_pnl'].sum():+,.0f}")
    if len(rep):
        print(f"replay : mean net {rep['net'].mean():+.3%} a trade over {len(rep)}, total Rs {rep['net_pnl'].sum():+,.0f}")

    verdicts = []
    if len(both):
        mean_diff, t, n_dates = date_t(m["net_diff"])
        med_entry = float(m["entry_adverse"].median())
        print("\nmatched trades (paper vs replay; positive 'adverse' = paper did worse):")
        q = m[["entry_adverse", "exit_adverse", "net_diff"]].quantile([0.1, 0.5, 0.9])
        for col in q.columns:
            print(
                f"  {col:14s} p10 {q.loc[0.1, col]:+.3%}  median {q.loc[0.5, col]:+.3%}  p90 {q.loc[0.9, col]:+.3%}  mean {m[col].mean():+.3%}"
            )
        print(f"  net_diff t (by date, {n_dates} dates) {t:+.2f}")
        print(
            f"  same exit reason {m['same_reason'].mean():.0%}; entry later than replay by median {m['entry_minutes_late'].median():+.0f} min"
        )
        mismatch = m[~m["same_reason"]].groupby(["paper_reason", "replay_reason"]).size()
        if len(mismatch):
            print("  exit reason differences (paper -> replay): " + ", ".join(f"{a}->{b} {n}" for (a, b), n in mismatch.items()))
        if len(both) >= MIN_MATCHED:
            verdicts.append(("E1 median adverse entry > 0.05%", med_entry > E1_ENTRY, f"{med_entry:+.3%}"))
            verdicts.append(
                ("E2 mean net diff < -0.05%, |t| >= 2", mean_diff < E2_RETURN and abs(t) >= E2_T, f"{mean_diff:+.3%}, t {t:+.2f}")
            )

    print("\nentries in only one account, and why (from each side's signal records):")
    reasons_p = []
    for k in only_paper:
        s = sig.loc[k] if k in sig.index else None
        if s is not None and isinstance(s.get("house_call"), str) and s.get("house_call"):
            why = "house call (replay has none)"
        elif s is not None and s.get("final_signal") == 0:
            why = "live signal gone on the final bars"
        elif len(rsig) and k in rsig.index:
            why = f"replay saw it: {rsig.loc[k, 'outcome']} {rsig.loc[k, 'reason'] or ''}".strip()
        else:
            why = "replay saw no signal"
        reasons_p.append(why)
    reasons_r = []
    for k in only_replay:
        s = sig.loc[k] if k in sig.index else None
        reasons_r.append("live saw no signal" if s is None else "live saw it but did not enter")
    for label, reasons in (("paper only", reasons_p), ("replay only", reasons_r)):
        for why, n in pd.Series(reasons, dtype=object).value_counts().items():
            print(f"  {label:11s} {n:4d}  {why}")
    if len(only_paper):
        print(
            f"  paper-only trades: mean net {paper.loc[only_paper, 'net'].mean():+.3%}, Rs {paper.loc[only_paper, 'net_pnl'].sum():+,.0f}"
        )
    if len(only_replay):
        print(
            f"  replay-only trades: mean net {rep.loc[only_replay, 'net'].mean():+.3%}, Rs {rep.loc[only_replay, 'net_pnl'].sum():+,.0f}"
        )
    distinct = len(both) + len(only_paper) + len(only_replay)
    share = (len(only_paper) + len(only_replay)) / distinct if distinct else 0.0
    if len(both) >= MIN_MATCHED:
        verdicts.append(("E3 entries in only one > 10%", share > E3_SHARE, f"{share:.0%}"))

    # exits whose bar touched both levels, on the final bars, for the paper trades (stop/target from the signal records)
    amb = []
    for k in paper.index:
        if k in sig.index:
            amb.append(
                touched_both(
                    bars, k[1], paper.loc[k, "opened_at"], paper.loc[k, "closed_at"], sig.loc[k, "stop"], sig.loc[k, "target"]
                )
            )
    told = [a for a in amb if a is not None]
    if told:
        print(
            f"\nexit bar touched both stop and target: {sum(told)} of {len(told)} paper exits at a level ({sum(told) / len(told):.0%})"
        )

    print("\npre-declared checks:")
    if not verdicts:
        print(f"  no verdict: {len(both)} matched trades (< {MIN_MATCHED})")
    for name, hit, value in verdicts:
        print(f"  {name:40s} {value:>18s}  {'MATERIAL' if hit else 'no'}")
    if verdicts:
        material = any(hit for _, hit, _ in verdicts)
        print(f"  => execution {'IS' if material else 'is NOT'} a material part of the paper result")

    if csv is not None:
        out = pd.concat(
            [
                paper.add_prefix("paper_"),
                rep.add_prefix("replay_"),
            ],
            axis=1,
        )
        out.insert(0, "strategy", strategy)
        path = csv.with_name(f"{csv.stem}_{strategy}{csv.suffix}")
        out.to_csv(path)
        print(f"\nper-trade rows: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", choices=[*DATABASES, "all"], default="all")
    ap.add_argument("--csv", type=Path, help="also write per-trade rows (one file per strategy)")
    args = ap.parse_args()
    for strategy in DATABASES if args.strategy == "all" else [args.strategy]:
        reconcile(strategy, args.csv)


if __name__ == "__main__":
    main()
