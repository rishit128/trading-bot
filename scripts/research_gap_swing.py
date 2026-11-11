"""Multi-day gap strategies for the gap-and-go account, on 10 years of daily bars (pre-declared 2026-11-09 before any result was
seen; trial ids G1-G4).

    python scripts/research_gap_swing.py            # prints the table and the verdicts
    python scripts/research_gap_swing.py --record   # the same, and appends the rows to the trial register

Why. The live gap-and-go account (intraday, out by 15:15) lost on 37 trades (19 won, 18 lost), as its 8-year test said it would:
gross +0.15% a trade, net -0.06% at 0.21% costs and worse at Angel One's real ~0.37%. Every intraday gap variant tried failed
(weak-gap days, break-even stop, breadth, bigger gaps first). The cost is paid per trade, so this tests the same trigger, a gap
on heavy volume, held for days, where one round trip of costs is spread over a move of several percent. Nothing here uses the
swing account's screener or momentum rank, the news account's filings, the broker's calls, or ORB.

Data. research_cache/ohlc_nse_10y.pkl: Yahoo daily bars, adjusted, 2,317 NSE stocks that exist today, 2016-09 .. 2026-09. Today's
stocks only, which flatters long rules (survivorship); the excess test against the median liquid stock on the same dates is there
to take that out. Volume is not split-adjusted, so a session that opens more than 25% from the previous close is dropped.

Common to every rule. At the signal session D (known at its close): the previous close is Rs 100 or more (on adjusted prices) and
the 20-session average traded value before D is Rs 10 crore or more. Entry at the OPEN of the session after the signal (D+1; for
G3, D+4), never at a price already seen. Rs 5,000 a trade, 5 bps slippage a side, delivery fees as in src.engine.costs
(india_delivery_fees: STT both sides, stamp, DP charge Rs 15.93 a sale). A stop is checked on every session's low from the entry
session on; it fills at the stop, or at the open when a session opens below it. A stop more than 12% away, or at or above the
entry open, means no trade. Exit at the open of session entry+hold. One trade per stock at a time (a signal while a trade in that
stock is open is skipped). Long only. No position limit and no daily cap (each trade is judged alone, as in the earlier tests).

G1 Gap up that holds. Gap (open over previous close) 3% to 15%; volume at least 2x the 20-session average; closes at or above its
   open and in the top 30% of its range; closes above its 50-day average. Hold 10 sessions; stop at D's low.
G2 Rejected gap down. Gap -3% to -12% on the open; previous close above its 200-day average; volume at least 1.5x average; closes
   above its open and in the top half of its range. Hold 5 sessions; stop at D's low. (Opposite thesis: the sellers are absorbed.)
G3 Gap that survives three sessions. G1's gap and volume at D; then D+1 to D+3 never trade at or below D-1's close (the gap is
   not filled) and D+3 closes at or above D's close. Enter D+4's open; hold 10; stop at D-1's close (a filled gap ends the idea).
G4 Gap to a 252-session closing high. Gap 2% to 15%; D closes at or above its highest close of the previous 252 sessions; volume at
   least 2x average; closes at or above its open. Hold 20 sessions; stop at D's low.

Gate, each rule on its own, all of these: at least 300 trades; mean net > 0; date-clustered t of net >= 2.6 (per entry date, then
across dates); both halves of the dates > 0; net > 0 with the slippage doubled; net > 0 in at least 6 calendar years that have 20
or more trades; and the excess over the median liquid stock's return over the same entry-to-exit window, > 0 with date-clustered
t >= 2.0. The deflated t against every trial in the register is printed beside it. Second route, as in N1-N6: the same rules on
sessions from 2024-11-20 only: >= 150 trades, net > 0 with t >= 2.0, both halves > 0, > 0 at twice the slippage, excess > 0 with
t >= 1.5 (a weaker candidate, for a paper test only). Net by calendar year is printed for every rule. A pass makes the rule a
candidate for the gap-and-go account in paper, the owner's decision. A neighbouring variant is never a second chance: nothing here
is re-run with other thresholds. Nothing here changes a live account."""

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.significance import deflated_t  # noqa: E402

CACHE = ROOT / "research_cache" / "ohlc_nse_10y.pkl"
TRIALS = ROOT / "docs" / "trading" / "research-trials.csv"
POSITION, MIN_PRICE, MIN_VALUE, SPLIT, MAX_STOP = 5000.0, 100.0, 1e8, 0.25, 0.12
MIN_TRADES, T_MIN, T_EXCESS, MIN_YEARS, YEAR_TRADES = 300, 2.6, 2.0, 6, 20
RECENT_FROM, RECENT_TRADES, RECENT_T, RECENT_EX_T = pd.Timestamp("2024-11-20"), 150, 2.0, 1.5
FEES = (india_delivery_fees("BUY", POSITION) + india_delivery_fees("SELL", POSITION)) / POSITION
HOLD = {"G1": 10, "G2": 5, "G3": 10, "G4": 20}
DESCRIPTIONS = {
    "G1": "multi-day: gap up 3-15% on 2x volume, closes strong above MA50, hold 10 sessions",
    "G2": "multi-day: gap down 3-12% rejected in an MA200 uptrend, hold 5 sessions",
    "G3": "multi-day: gap up that stays unfilled for 3 sessions, enter D+4, hold 10",
    "G4": "multi-day: gap up 2-15% to a 252-session closing high on 2x volume, hold 20",
}


def net(gross, slip: float = SLIPPAGE):
    return gross - FEES - 2 * slip


def mean_before(a: np.ndarray, w: int) -> np.ndarray:
    """Mean of a[i-w:i] (the w sessions before i), NaN where there are fewer than w."""
    cs, out = np.cumsum(np.insert(a, 0, 0.0)), np.full(len(a), np.nan)
    out[w:] = (cs[w : len(a)] - cs[: len(a) - w]) / w
    return out


def mean_to(a: np.ndarray, w: int) -> np.ndarray:
    """Mean of a[i-w+1..i] (w sessions up to and including i), NaN where there are fewer than w."""
    cs, out = np.cumsum(np.insert(a, 0, 0.0)), np.full(len(a), np.nan)
    out[w - 1 :] = (cs[w:] - cs[: len(a) - w + 1]) / w
    return out


def run_trade(o, lo, e: int, hold: int, stop: float):
    """(gross return, exit index) of a long entered at open e with the stop, or None when no trade is allowed."""
    entry = o[e]
    if not (stop < entry and (entry - stop) / entry <= MAX_STOP) or e + hold >= len(o):
        return None
    for k in range(e, e + hold):
        if lo[k] <= stop:
            fill = stop if k == e else min(o[k], stop)
            return fill / entry - 1, k
    return o[e + hold] / entry - 1, e + hold


def scan_stock(df: pd.DataFrame):
    """[(rule, entry index, gross)] for one stock, with the one-trade-at-a-time rule applied per rule."""
    o, h, lo, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    if n < 300:
        return [], None
    cp = np.r_[np.nan, c[:-1]]
    gap = o / cp - 1
    vavg, tv = mean_before(v, 20), mean_before(c * v, 20)
    ma50, ma200 = mean_to(c, 50), np.r_[np.nan, mean_to(c, 200)[:-1]]
    hi252 = pd.Series(c).rolling(252).max().shift(1).to_numpy()
    rng = h - lo
    pos = np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1.0), np.nan)
    base = (tv >= MIN_VALUE) & (cp >= MIN_PRICE) & (np.abs(gap) < SPLIT) & (vavg > 0)
    up = base & (gap >= 0.03) & (gap <= 0.15) & (v >= 2 * vavg)
    sig = {
        "G1": up & (c >= o) & (pos >= 0.7) & (c > ma50),
        "G2": base & (gap <= -0.03) & (gap >= -0.12) & (cp > ma200) & (v >= 1.5 * vavg) & (c > o) & (pos >= 0.5),
        "G3": up,
        "G4": base & (gap >= 0.02) & (gap <= 0.15) & (c >= hi252) & (v >= 2 * vavg) & (c >= o),
    }
    rows = []
    for rule, mask in sig.items():
        busy = -1
        for d in np.flatnonzero(np.nan_to_num(mask, nan=0).astype(bool)):
            if d <= busy:
                continue
            if rule == "G3":
                e = d + 4
                if e >= n or not (lo[d + 1 : d + 4].min() > cp[d] and c[d + 3] >= c[d]):
                    continue
                stop = cp[d]
            else:
                e, stop = d + 1, lo[d]
            if e >= n:
                continue
            r = run_trade(o, lo, e, HOLD[rule], stop)
            if r is not None:
                rows.append((rule, e, r[0]))
                busy = r[1]
    liquid = pd.Series(tv >= MIN_VALUE, index=df.index)
    return rows, (df["Open"].astype(float), liquid)


def load():
    raw = pd.read_pickle(CACHE)
    rows, opens, liquid = [], {}, {}
    for sym, df in raw.items():
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_index()
        found, extra = scan_stock(df)
        if extra is None:
            continue
        opens[sym], liquid[sym] = extra
        rows += [(rule, df.index[e], sym, g) for rule, e, g in found]
    o, lq = pd.DataFrame(opens), pd.DataFrame(liquid).fillna(False).astype(bool)
    # the same window held by every liquid stock: its median return is what a random entry on that date would have made
    control = {hold: (o.shift(-hold) / o - 1).where(lq).median(axis=1) for hold in set(HOLD.values())}
    return len(opens), o.index, pd.DataFrame(rows, columns=["rule", "date", "sym", "gross"]), control


def tstat(per: pd.Series) -> float:
    return per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 2 and per.std(ddof=1) > 0 else float("nan")


def judge(rule, t: pd.DataFrame, n_trials, recent=False):
    min_trades, t_min, ex_min = (RECENT_TRADES, RECENT_T, RECENT_EX_T) if recent else (MIN_TRADES, T_MIN, T_EXCESS)
    if t.empty:
        return {"rule": rule, "trades": 0, "pass": False, "why": "no trades"}
    t = t.assign(net=net(t["gross"]), net2=net(t["gross"], 2 * SLIPPAGE))
    per = t.groupby("date")["net"].mean().sort_index()
    half = len(per) // 2
    ex = t.dropna(subset=["excess"]).groupby("date")["excess"].mean()
    years = t.groupby(t["date"].dt.year).agg(n=("net", "size"), m=("net", "mean"))
    years = years[years["n"] >= YEAR_TRADES]
    out = {
        "rule": rule, "trades": len(t), "dates": len(per), "gross": t["gross"].mean() * 100, "net": t["net"].mean() * 100,
        "t": tstat(per), "h1": per.iloc[:half].mean() * 100, "h2": per.iloc[half:].mean() * 100, "slip2": t["net2"].mean() * 100,
        "win": (t["net"] > 0).mean() * 100, "excess": t["excess"].mean() * 100, "ex_t": tstat(ex),
        "years_pos": int((years["m"] > 0).sum()), "years": len(years),
        "avg_win": t.loc[t["net"] > 0, "net"].mean() * 100, "avg_loss": t.loc[t["net"] <= 0, "net"].mean() * 100,
    }  # fmt: skip
    why = []
    if out["trades"] < min_trades:
        why.append(f"{out['trades']} trades < {min_trades}")
    if not (out["net"] > 0 and out["t"] >= t_min):
        why.append(f"net {out['net']:+.3f}% t {out['t']:.1f}")
    if not (out["h1"] > 0 and out["h2"] > 0):
        why.append(f"halves {out['h1']:+.3f}/{out['h2']:+.3f}")
    if out["slip2"] <= 0:
        why.append(f"2x slip {out['slip2']:+.3f}%")
    if not recent and out["years_pos"] < MIN_YEARS:
        why.append(f"profitable in {out['years_pos']} of {out['years']} years")
    if not (out["excess"] > 0 and out["ex_t"] >= ex_min):
        why.append(f"excess {out['excess']:+.3f}% t {out['ex_t']:.1f}")
    out["deflated"] = deflated_t(out["t"], n_trials).probability if not np.isnan(out["t"]) else float("nan")
    out["pass"], out["why"] = not why, "; ".join(why) or "PASS"
    return out


def main() -> None:
    names, days, trades, control = load()
    trades["excess"] = [g - control[HOLD[r]].get(d, np.nan) for r, d, g in zip(trades["rule"], trades["date"], trades["gross"])]
    print(
        f"{names} stocks, {days[0].date()} .. {days[-1].date()}; delivery fees {FEES * 100:.3f}% + slippage {2 * SLIPPAGE * 100:.2f}% "
        f"a round trip on Rs {POSITION:,.0f}",
        flush=True,
    )
    with TRIALS.open() as f:
        n_trials = sum(int(r["variants"] or 0) for r in csv.DictReader(f)) + len(HOLD)
    results = [judge(k, trades[trades["rule"] == k], n_trials) for k in HOLD]
    print(f"\n{'rule':5}{'trades':>8}{'dates':>7}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}{'2x slip':>9}{'win':>6}"
          f"{'avg win':>9}{'avg loss':>9}{'excess':>9}{'ex t':>6}{'yrs+':>6}{'defl':>6}")  # fmt: skip
    for r in results:
        if not r["trades"]:
            print(f"{r['rule']:5} no trades")
            continue
        print(f"{r['rule']:5}{r['trades']:>8}{r['dates']:>7}{r['gross']:>+8.3f}%{r['net']:>+8.3f}%{r['t']:>6.1f}{r['h1']:>+9.3f}%"
              f"{r['h2']:>+9.3f}%{r['slip2']:>+8.3f}%{r['win']:>5.0f}%{r['avg_win']:>+8.2f}%{r['avg_loss']:>+8.2f}%{r['excess']:>+8.3f}%"
              f"{r['ex_t']:>6.1f}{r['years_pos']:>3}/{r['years']:<2}{r['deflated']:>5.2f}")  # fmt: skip
    print(f"\n(deflated: probability the t beats the best of {n_trials} trials by luck)")
    for r in results:
        print(f"  {r['rule']}: {'PASS' if r['pass'] else 'fails - ' + r['why']}")
    years = sorted(trades["date"].dt.year.unique())
    print("\nNet a trade by calendar year:\nrule " + "".join(f"{y:>9}" for y in years))
    for k in HOLD:
        t = trades[trades["rule"] == k]
        cells = [
            (
                f"{net(t.loc[t['date'].dt.year == y, 'gross']).mean() * 100:>+8.2f}%"
                if (t["date"].dt.year == y).any()
                else f"{'-':>9}"
            )
            for y in years
        ]
        print(f"{k:5}" + "".join(cells) + "   trades: " + " ".join(str(int((t["date"].dt.year == y).sum())) for y in years))
    recent = [judge(k, trades[(trades["rule"] == k) & (trades["date"] >= RECENT_FROM)], n_trials, recent=True) for k in HOLD]
    print(
        f"\nRecent market only, from {RECENT_FROM.date()} (>= {RECENT_TRADES} trades, t >= {RECENT_T}, excess t >= {RECENT_EX_T}):"
    )
    for r in recent:
        if r["trades"]:
            print(f"  {r['rule']}: {r['trades']} trades, net {r['net']:+.3f}%, t {r['t']:.1f}, halves {r['h1']:+.3f}/{r['h2']:+.3f}%, "
                  f"excess {r['excess']:+.3f}% -> {'PASS' if r['pass'] else 'fails - ' + r['why']}")  # fmt: skip
        else:
            print(f"  {r['rule']}: no trades")
    passed = [r for r in results if r["pass"]]
    print("\nCANDIDATES: " + (", ".join(r["rule"] for r in sorted(passed, key=lambda r: -r["t"])) if passed else "none"))
    rec = [r for r in recent if r["pass"] and r["rule"] not in {p["rule"] for p in passed}]
    print("RECENT-ONLY CANDIDATES (paper test only, weaker evidence): " + (", ".join(r["rule"] for r in rec) or "none"))
    if "--record" not in sys.argv:
        print("\n(not recorded: add --record to append these rows to the trial register)")
        return
    span = f"daily bars {days[0].date()} to {days[-1].date()}, {names} NSE stocks"
    with TRIALS.open("a", newline="") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow([date.today().isoformat(), r["rule"], 1, "gap-swing", DESCRIPTIONS[r["rule"]], span, r["trades"],
                        f"{r.get('net', float('nan')):+.3f}", f"{r.get('t', float('nan')):.1f}",
                        "pass" if r["pass"] else ("recent-only pass" if r["rule"] in {x["rule"] for x in rec} else "fail"),
                        "scripts/research_gap_swing.py"])  # fmt: skip


if __name__ == "__main__":
    main()
