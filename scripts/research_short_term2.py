"""Short-term account, round 2: seven more candidates (pre-declared 2026-11-18, before any result; trial ids SU1-SU7; the owner's
request to research and explore further after ST1-ST7 all failed).

    python scripts/research_short_term2.py            # the table and the verdicts
    python scripts/research_short_term2.py --record   # the same, and the rows appended to the trial register

What round 1 taught. A delivery round trip costs about 1.17% in fees plus 0.10% slippage at Rs 10,000, so a rule must make 1.5-2%
a trade before costs. ST1-ST7 found real excess over the median stock (+0.6-0.8%), but on 3-7 day holds that was too small. So
round 2 looks for bigger moves: (a) market-wide washouts, where whole-market rebounds are large (SU1, SU5); (b) calendar flow
(SU2); (c) Indian price-band (circuit) days, where a stock that closes locked at its limit has demand left over (SU3); (d) quiet,
deeper pull-backs in leaders and first new highs after long bases, held 4-6 weeks (SU4, SU6); (e) one-month risk-adjusted
momentum, the swing account's effect on a short clock (SU7). Not repeated: anything already in the register (dips, gaps, results,
plain breakouts, weekly momentum/reversal, intraday patterns).

Data, liquidity, entry and exit mechanics, costs, the control (median liquid stock over the same window) and the gate are those
of scripts/research_short_term.py, imported unchanged: signal at D's close, entry at D+1's open, stops on each low, Rs 10,000 a
trade, 5 bps slippage a side, one trade per stock per rule at a time, long only, Angel One prices only.

SU1 Market washout rebound. Breadth = share of liquid stocks closing above their 50-day average. D: breadth under 20% and higher
    than the session before. Buy the 10 liquid stocks with the highest 126-session return. Hold 20 sessions; stop 12%.
SU2 Turn of the month. At the close of each month's 4th-last session, buy the 20 most traded liquid stocks above their 200-day
    average; sell at the open 6 sessions after entry (about the 4th session of the new month). No stop.
SU3 Closed at the upper limit. Liquid at D; D's close at least 9.5% above the previous close and within 0.5% of D's high; volume
    at least 3x its 20-session average. Hold 10 sessions; stop 10%.
SU4 Quiet pull-back in a leader. A close at a 252-session closing high within the 20 sessions before D; D's close 8-20% under the
    highest close of those 20 sessions and above the 200-day average; the 5 sessions to D average under 0.7x the 50-session volume;
    D closes above the previous session's high. Stop at the lowest low of the 10 sessions to D (skipped when more than 12% under
    the entry). Hold 25 sessions.
SU5 Leaders in a weak market. On a week's last session with breadth (as SU1) under 50%, up to 10 liquid stocks closing at a
    252-session closing high, highest 126-session return first. Hold 20 sessions; stop 10%.
SU6 First 52-week high after a long base. Liquid; D's close at a 252-session closing high with no such close in the 120 sessions
    before; volume at least 1.5x its 20-session average. Hold 30 sessions; stop 10%.
SU7 Four-week risk-adjusted momentum. Every 4th week's last session, the 10 liquid stocks above their 200-day average with the
    highest 126-session return divided by the 126-session daily volatility. Hold 20 sessions; no stop.

Gate: research_short_term.judge, unchanged (>= 300 trades, net > 0 with date-clustered t >= 2.6, both halves > 0, net > 0 at 2x
slippage, profitable in >= 6 years with 20+ trades, excess > 0 with t >= 2.0; recent route from 2024-11-20). A rule that
cannot reach 300 trades fails on that, as declared. Nothing is re-run with other thresholds after the first result."""

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

from research_short_term import POSITION, RECENT_FROM, SLIPPAGE, TRIALS, Stock, cost, judge, load, net  # noqa: E402

RULES = ("SU1", "SU2", "SU3", "SU4", "SU5", "SU6", "SU7")
DESCRIPTIONS = {
    "SU1": "short-term 2: market washout (breadth over 50-day under 20% and turning up), top 10 by 126-day return, hold 20, stop 12%",
    "SU2": "short-term 2: turn of the month, 20 most traded above the 200-day, 4th-last session to about the 4th session",
    "SU3": "short-term 2: closed at the upper limit (+9.5%, at the high, 3x volume), hold 10, stop 10%",
    "SU4": "short-term 2: quiet 8-20% pull-back in a 52-week-high leader, reversal day, stop at 10-day low, hold 25",
    "SU5": "short-term 2: weekly, breadth under 50%: up to 10 stocks at a 52-week closing high, hold 20, stop 10%",
    "SU6": "short-term 2: first 52-week closing high in 120 sessions on 1.5x volume, hold 30, stop 10%",
    "SU7": "short-term 2: every 4 weeks, top 10 by 126-day return over volatility above the 200-day, hold 20",
}


def per_stock(st: Stock) -> dict:
    """SU3, SU4, SU6: rules that need only the stock's own bars. (entry index, gross, exit index) per rule."""
    c, h, lo, v, n = st.c, st.h, st.l, st.v, st.n
    hi252 = pd.Series(c).rolling(252).max().to_numpy()
    at_high = np.nan_to_num(c >= hi252, nan=0).astype(bool)
    found: dict = {r: [] for r in ("SU3", "SU4", "SU6")}

    def walk(rule, mask, make):
        busy = -1
        for d in np.flatnonzero(mask):
            if d <= busy or d + 1 >= n:
                continue
            r = make(d)
            if r is not None:
                if not st.broken[d + 1 : r[1] + 1].any():
                    found[rule].append((d + 1, r[0], r[1]))
                busy = r[1]

    # SU3
    limit = np.nan_to_num((c / st.cp - 1 >= 0.095) & (c >= 0.995 * h) & (v >= 3 * st.vavg), nan=0).astype(bool)
    walk("SU3", st.liquid & limit, lambda d: st.trade(d + 1, 10, stop=st.o[d + 1] * 0.90))
    # SU4
    recent_high = pd.Series(at_high.astype(float)).rolling(20).max().shift(1).fillna(0).to_numpy() > 0
    top20 = pd.Series(c).rolling(20).max().shift(1).to_numpy()
    depth = 1 - c / top20
    v5, v50 = pd.Series(v).rolling(5).mean().to_numpy(), pd.Series(v).rolling(50).mean().to_numpy()
    turn = np.r_[False, c[1:] > h[:-1]]
    low10 = pd.Series(lo).rolling(10).min().to_numpy()
    pull = np.nan_to_num((depth >= 0.08) & (depth <= 0.20) & (c > st.ma200) & (v5 < 0.7 * v50), nan=0).astype(bool)

    def su4(d):
        e, stop = d + 1, low10[d]
        if not (stop > 0) or st.o[e] / stop - 1 > 0.12:
            return None
        return st.trade(e, 25, stop=stop)

    walk("SU4", st.liquid & recent_high & turn & pull, su4)
    # SU6
    prior = pd.Series(at_high.astype(float)).rolling(120).max().shift(1).fillna(1).to_numpy() > 0
    vol = np.nan_to_num(v >= 1.5 * st.vavg, nan=0).astype(bool)
    walk("SU6", st.liquid & at_high & ~prior & vol, lambda d: st.trade(d + 1, 30, stop=st.o[d + 1] * 0.90))
    return found


def run(stocks):
    built = {s: Stock(df) for s, df in stocks.items()}
    days = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in built.values()])))

    def matrix(get):
        return pd.DataFrame({s: pd.Series(get(b), index=b.index) for s, b in built.items()}).reindex(days)

    liquid = matrix(lambda b: b.liquid).astype(float).fillna(0.0).astype(bool)
    close = matrix(lambda b: b.c)
    above50 = (close > matrix(lambda b: b.ma50)).where(liquid)
    above200 = (close > matrix(lambda b: b.ma200)) & liquid
    breadth = above50.sum(axis=1) / liquid.sum(axis=1).replace(0, np.nan)
    ret126 = matrix(lambda b: b.ret126).where(liquid)
    tv = matrix(lambda b: b.tv).where(liquid)
    daily = close.pct_change(fill_method=None)
    riskadj = ret126 / (daily.rolling(126, min_periods=100).std() * math.sqrt(126))
    at_high = close >= close.rolling(252, min_periods=252).max()

    rows = []
    for s, b in built.items():
        for rule, trades in per_stock(b).items():
            rows += [(rule, s, b.index[e], b.index[x], g) for e, g, x in trades]

    busy: dict = {}

    def buy(rule, s, d, hold, stop=None):
        b = built[s]
        i = b.index.searchsorted(d)
        if i + 1 >= b.n or b.index[i] != d or busy.get((rule, s), pd.Timestamp.min) >= d:
            return
        e = i + 1
        r = b.trade(e, hold, stop=None if stop is None else b.o[e] * (1 - stop))
        if r is not None:
            busy[(rule, s)] = b.index[r[1]]
            if not b.broken[e : r[1] + 1].any():
                rows.append((rule, s, b.index[e], b.index[r[1]], r[0]))

    # SU1
    for i in range(1, len(days)):
        d = days[i]
        if breadth.iloc[i] < 0.20 and breadth.iloc[i] > breadth.iloc[i - 1]:
            for s in ret126.loc[d].dropna().nlargest(10).index:
                buy("SU1", s, d, 20, 0.12)
    # SU2: the 4th-last session of each month
    month = days.to_period("M").asi8
    for m in np.unique(month)[:-1]:
        sessions = days[month == m]
        if len(sessions) >= 8:
            d = sessions[-4]
            for s in tv.loc[d][above200.loc[d]].dropna().nlargest(20).index:
                buy("SU2", s, d, 6)
    # SU5 / SU7: weekly
    week = days.to_period("W-SUN").asi8
    last = days[:-1][week[:-1] != week[1:]]
    for k, d in enumerate(last):
        if breadth.loc[d] < 0.50:
            picks = ret126.loc[d][at_high.loc[d] & liquid.loc[d]].dropna().nlargest(10).index
            for s in picks:
                buy("SU5", s, d, 20, 0.10)
        if k % 4 == 0:
            for s in riskadj.loc[d][above200.loc[d]].dropna().nlargest(10).index:
                buy("SU7", s, d, 20)

    trades = pd.DataFrame(rows, columns=["rule", "sym", "date", "exit", "gross"])
    log_open = np.log(matrix(lambda b: b.o))
    control = {}
    for e, x in set(zip(trades["date"], trades["exit"])):
        control[(e, x)] = float(np.nanmedian((log_open.loc[x] - log_open.loc[e])[liquid.loc[e]].to_numpy()))
    trades["excess"] = trades["gross"] - [math.expm1(control[(e, x)]) for e, x in zip(trades["date"], trades["exit"])]
    return trades, days


def main() -> None:
    stocks = load()
    trades, days = run(stocks)
    print(
        f"{len(stocks)} NSE stocks (Angel One daily), {days[0].date()} .. {days[-1].date()}; delivery fees "
        f"{cost(POSITION) * 100:.3f}% + slippage {2 * SLIPPAGE * 100:.2f}% a round trip on Rs {POSITION:,.0f}",
        flush=True,
    )
    with TRIALS.open() as f:
        n_trials = sum(int(r["variants"] or 0) for r in csv.DictReader(f)) + len(RULES)
    results = [judge(k, trades[trades["rule"] == k], n_trials) for k in RULES]
    print(f"\n{'rule':5}{'trades':>8}{'days':>6}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}{'2x slip':>9}"
          f"{'win':>6}{'excess':>9}{'ex t':>6}{'yrs+':>7}{'defl':>6}{'@25k':>9}")  # fmt: skip
    for r in results:
        if not r["trades"]:
            print(f"{r['rule']:5} no trades")
            continue
        print(f"{r['rule']:5}{r['trades']:>8}{r['hold']:>6.1f}{r['gross']:>+8.3f}%{r['net']:>+8.3f}%{r['t']:>6.1f}{r['h1']:>+9.3f}%"
              f"{r['h2']:>+9.3f}%{r['slip2']:>+8.3f}%{r['win']:>5.0f}%{r['excess']:>+8.3f}%{r['ex_t']:>6.1f}"
              f"{r['years_pos']:>4}/{r['years']:<2}{r['deflated']:>6.2f}{r['n25']:>+8.3f}%")  # fmt: skip
    for r in results:
        print(f"  {r['rule']}: {'PASS' if r['pass'] else 'fails - ' + r['why']}")
    years = sorted(trades["date"].dt.year.unique())
    print("\nNet a trade by calendar year:\nrule " + "".join(f"{y:>9}" for y in years))
    for k in RULES:
        t = trades[trades["rule"] == k]
        cells = [
            f"{net(t.loc[t['date'].dt.year == y, 'gross']).mean() * 100:>+8.2f}%"
            if (t["date"].dt.year == y).any()
            else f"{'-':>9}"
            for y in years
        ]
        print(f"{k:5}" + "".join(cells))
    recent = [judge(k, trades[(trades["rule"] == k) & (trades["date"] >= RECENT_FROM)], n_trials, recent=True) for k in RULES]
    print(f"\nRecent market only, from {RECENT_FROM.date()}:")
    for r in recent:
        if r["trades"]:
            print(f"  {r['rule']}: {r['trades']} trades, net {r['net']:+.3f}%, t {r['t']:.1f}, excess {r['excess']:+.3f}% -> "
                  f"{'PASS' if r['pass'] else 'fails - ' + r['why']}")  # fmt: skip
    passed = sorted([r for r in results if r["pass"]], key=lambda r: -r["t"])
    rec = [r for r in recent if r["pass"] and r["rule"] not in {p["rule"] for p in passed}]
    print("\nCANDIDATES: " + (", ".join(r["rule"] for r in passed) or "none"))
    print("RECENT-ONLY CANDIDATES (paper test only, weaker evidence): " + (", ".join(r["rule"] for r in rec) or "none"))
    trades.to_pickle(ROOT / "logs" / "research_short_term2_trades.pkl")
    if "--record" not in sys.argv:
        print("\n(not recorded: add --record to append these rows to the trial register)")
        return
    span = f"Angel One daily {days[0].date()} to {days[-1].date()}, {len(stocks)} NSE stocks"
    with TRIALS.open("a", newline="") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow([date.today().isoformat(), r["rule"], 1, "short-term", DESCRIPTIONS[r["rule"]], span, r["trades"],
                        f"{r.get('net', float('nan')):+.3f}", f"{r.get('t', float('nan')):.1f}",
                        "pass" if r["pass"] else ("recent-only pass" if r["rule"] in {x["rule"] for x in rec} else "fail"),
                        "scripts/research_short_term2.py"])  # fmt: skip


if __name__ == "__main__":
    main()
