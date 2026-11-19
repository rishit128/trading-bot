"""Short-term account, round 3: SU7's entries with profit-target exits instead of a fixed 20-session hold (pre-declared 2026-11-19,
before any result; trial ids SV1-SV4; the owner's idea: "instead of days it should work on profit percentage").

Entries exactly as SU7 (scripts/research_short_term2.py): every 4th week's last session, the 10 liquid stocks above their 200-day
average with the highest 126-session return over volatility; bought at the next open. Exit, whichever comes first:
  SV1 +8% target, stop 8%      SV2 +10% target, stop 8%      SV3 +15% target, stop 10%      SV4 +20% target, stop 10%
and at most 60 sessions (sold at that open). The target fills at the target, or at the open when a session opens above it; the
stop at the stop, or at the open when a session opens below it; when one session's range touches both, the stop is assumed.
Data, costs, control and gate: research_short_term.judge, unchanged. SU7 (20 sessions) is printed beside them for comparison.

    python scripts/research_short_term3.py [--record]"""

import csv
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_short_term import RECENT_FROM, TRIALS, Stock, judge, load  # noqa: E402

MAX_HOLD = 60
EXITS = {"SU7 (20 days)": None, "SV1": (0.08, 0.08), "SV2": (0.10, 0.08), "SV3": (0.15, 0.10), "SV4": (0.20, 0.10)}


def exit_at(b: Stock, e: int, target: float, stop: float):
    entry = b.o[e]
    tp, sl = entry * (1 + target), entry * (1 - stop)
    for k in range(e, min(e + MAX_HOLD, b.n)):
        if b.l[k] <= sl:
            return (min(b.o[k], sl) if k > e else sl) / entry - 1, k
        if b.h[k] >= tp:
            return (max(b.o[k], tp) if k > e else tp) / entry - 1, k
    x = e + MAX_HOLD
    return (b.o[x] / entry - 1, x) if x < b.n else None


def main() -> None:
    stocks = load()
    built = {s: Stock(df) for s, df in stocks.items()}
    days = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in built.values()])))

    def matrix(get):
        return pd.DataFrame({s: pd.Series(get(b), index=b.index) for s, b in built.items()}).reindex(days)

    liquid = matrix(lambda b: b.liquid).astype(float).fillna(0.0).astype(bool)
    close = matrix(lambda b: b.c)
    above200 = (close > matrix(lambda b: b.ma200)) & liquid
    ret126 = matrix(lambda b: b.ret126).where(liquid)
    riskadj = ret126 / (close.pct_change(fill_method=None).rolling(126, min_periods=100).std() * math.sqrt(126))
    week = days.to_period("W-SUN").asi8
    last = days[:-1][week[:-1] != week[1:]]
    picks = [(d, list(riskadj.loc[d][above200.loc[d]].dropna().nlargest(10).index)) for k, d in enumerate(last) if k % 4 == 0]

    rows = []
    for name, ex in EXITS.items():
        busy: dict = {}
        for d, syms in picks:
            for s in syms:
                b = built[s]
                i = b.index.searchsorted(d)
                if i + 1 >= b.n or b.index[i] != d or busy.get(s, pd.Timestamp.min) >= d:
                    continue
                e = i + 1
                r = b.trade(e, 20) if ex is None else exit_at(b, e, *ex)
                if r is not None:
                    busy[s] = b.index[r[1]]
                    if not b.broken[e : r[1] + 1].any():
                        rows.append((name, s, b.index[e], b.index[r[1]], r[0]))
    trades = pd.DataFrame(rows, columns=["rule", "sym", "date", "exit", "gross"])
    log_open = np.log(matrix(lambda b: b.o))
    control = {}
    for e, x in set(zip(trades["date"], trades["exit"])):
        control[(e, x)] = float(np.nanmedian((log_open.loc[x] - log_open.loc[e])[liquid.loc[e]].to_numpy()))
    trades["excess"] = trades["gross"] - [math.expm1(control[(e, x)]) for e, x in zip(trades["date"], trades["exit"])]

    with TRIALS.open() as f:
        n_trials = sum(int(r["variants"] or 0) for r in csv.DictReader(f)) + 4
    print(f"{len(stocks)} NSE stocks (Angel One daily), {days[0].date()} .. {days[-1].date()}; {len(picks)} rebalance dates")
    print(f"\n{'rule':15}{'trades':>7}{'days':>6}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}"
          f"{'win':>6}{'excess':>9}{'ex t':>6}{'yrs+':>7}{'recent net':>12}")  # fmt: skip
    results = {}
    for name in EXITS:
        t = trades[trades["rule"] == name]
        r, rr = judge(name, t, n_trials), judge(name, t[t["date"] >= RECENT_FROM], n_trials, recent=True)
        results[name] = (r, rr)
        print(f"{name:15}{r['trades']:>7}{r['hold']:>6.1f}{r['gross']:>+8.3f}%{r['net']:>+8.3f}%{r['t']:>6.1f}{r['h1']:>+9.3f}%"
              f"{r['h2']:>+9.3f}%{r['win']:>5.0f}%{r['excess']:>+8.3f}%{r['ex_t']:>6.1f}{r['years_pos']:>4}/{r['years']:<2}"
              f"{rr.get('net', float('nan')):>+11.3f}%")  # fmt: skip
        if ex := EXITS[name]:
            hit = t["gross"] >= ex[0] - 1e-9
            print(f"{'':15}target reached on {hit.mean():.0%} of trades; stop or 60-day limit on the rest")
    for name, (r, rr) in results.items():
        if EXITS[name]:
            print(f"  {name}: {'PASS' if r['pass'] else 'fails - ' + r['why']}; recent: {'PASS' if rr['pass'] else 'fails'}")
    if "--record" in sys.argv:
        span = f"Angel One daily {days[0].date()} to {days[-1].date()}, {len(stocks)} NSE stocks"
        with TRIALS.open("a", newline="") as f:
            w = csv.writer(f)
            for name, (r, rr) in results.items():
                if EXITS[name]:
                    tp, sl = EXITS[name]
                    w.writerow([date.today().isoformat(), name, 1, "short-term",
                                f"short-term 3: SU7 entries, +{tp:.0%} target / {sl:.0%} stop, max {MAX_HOLD} sessions (owner's idea)",
                                span, r["trades"], f"{r['net']:+.3f}", f"{r['t']:.1f}",
                                "pass" if r["pass"] else ("recent-only pass" if rr["pass"] else "fail"),
                                "scripts/research_short_term3.py"])  # fmt: skip


if __name__ == "__main__":
    main()
