"""Seven short-term (2-20 session) strategies for a new delivery account, on 10 years of Angel One daily bars (pre-declared
2026-11-16, before the data was downloaded or any result seen; trial ids ST1-ST7; the owner's request for 5-7 candidates).

    python scripts/research_short_term.py download         # Angel One daily bars of every NSE equity (resumable, ~1.5 h)
    python scripts/research_short_term.py                  # the table and the verdicts
    python scripts/research_short_term.py --record         # the same, and the rows appended to the trial register

Why these seven. Already failed on this market and not repeated here: gap rules held for days (G1-G4), holding after results or
an earnings surprise (H1, H2), and every intraday price pattern (ORB, gap-and-go, N1-N6, EMA/Supertrend/RSI/Bollinger/VWAP). The
seven below are the other well-documented short-horizon effects: buying short pull-backs inside an uptrend (ST1, ST2, ST3), a
volatility squeeze breaking out (ST4), one-week momentum and one-week reversal across stocks (ST5, ST6), and a plain 20-day
breakout that is not a gap (ST7). A delivery round trip costs about 0.93% plus the Rs 23.60 DP charge, so only a move of a few
percent can pay for itself; nothing here is intraday.

Data. research_cache/angel_daily_all/: Angel One ONE_DAY candles from 2016-11-01 for every NSE equity in the instrument list
(today's listings only, so long rules are flattered: the excess over the median liquid stock on the same dates takes that out).
Angel One candles are not adjusted: an overnight fall to under 75% of the previous close that is within 6% of a common split or
bonus ratio is treated as one and earlier prices divided by it; a trade whose window still holds an overnight fall under 70% is
dropped (counted). Prices and volumes only from Angel One; nothing from Yahoo.

Common to every rule. Signal at the close of session D, entry at the open of D+1 (ST4: a buy-stop during D+1), never at a price
already seen. Liquid at D: previous close Rs 100 or more and the 20 sessions before D average Rs 10 crore traded or more. Judged
at Rs 10,000 a trade (the DP charge is 0.24% of it), 5 bps slippage a side, delivery fees as in src.engine.costs (net also shown
at Rs 5,000 and Rs 25,000). An exit signalled at a close fills at the next open. A stop is checked on each session's low from the
entry session on and fills at the stop, or at the open when a session opens below it. One trade per stock per rule at a time.
Long only. Each trade judged alone (no position limit), as in G1-G4.

ST1 RSI(2) pull-back. Close above its 200-day average; Wilder RSI over 2 closes under 10. Exit at the first close above the 5-day
    average, at most 10 sessions. No stop.
ST2 Three-day sell-off in a strong trend. 50-day average above the 200-day and close above the 200-day; three lower closes in a row
    ending at the lowest close of 10 sessions. Exit at the first close above the previous session's high, at most 7 sessions.
ST3 Momentum leader back at its 20-day average. Top 20% of liquid stocks by 126-session return at D; D's low at or under the
    20-day average and its close above it; close above the 50-day average. Hold 10 sessions; stop 8% under the entry.
ST4 NR7 inside-day breakout. D inside the session before (lower high, higher low) and the narrowest range of 7 sessions; close above
    the 50-day average, which is above the 200-day. Buy-stop at D's high during D+1 (filled at the open when it opens above); no
    trade if not reached. Stop at D's low (a stop touched on the entry session counts as hit); hold 10 sessions.
ST5 One-week momentum. On a week's last session, the 10 liquid stocks above their 200-day average with the highest 20-session
    return. Hold 5 sessions. No stop.
ST6 One-week reversal. On a week's last session, among the 200 most traded liquid stocks above their 200-day average, the 10 with
    the lowest 5-session return. Hold 5 sessions. No stop.
ST7 20-day breakout without a gap. Close above the highest close of the 20 sessions before; volume at least 1.5x its 20-session
    average; close above the 200-day average; D opened less than 2% above the previous close. Exit at the first close under the
    lowest low of the 10 sessions before, at most 20 sessions; stop 8% under the entry.

Gate (G1-G4's, word for word), each rule on its own: at least 300 trades; mean net > 0; date-clustered t of net >= 2.6; both halves
of the dates > 0; net > 0 with the slippage doubled; net > 0 in at least 6 calendar years with 20 or more trades; excess over the
median liquid stock over the same entry-to-exit window > 0 with date-clustered t >= 2.0. The deflated t against every trial in the
register is printed beside it. Second route: the same rules from 2024-11-20 only (>= 150 trades, t >= 2.0, excess t >= 1.5; a
weaker candidate, paper only). If several pass, the highest t is the account's candidate, the owner's decision; the others stay
recorded. Nothing is re-run with other thresholds after the first result. Nothing here changes a live account."""

import csv
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.significance import deflated_t  # noqa: E402

CACHE = ROOT / "research_cache" / "angel_daily_all"
TRIALS = ROOT / "docs" / "trading" / "research-trials.csv"
START, CHUNK_DAYS, PACE = datetime(2016, 11, 1), 1900, 1.0
POSITION, MIN_PRICE, MIN_VALUE = 10_000.0, 100.0, 1e8
MIN_TRADES, T_MIN, T_EXCESS, MIN_YEARS, YEAR_TRADES = 300, 2.6, 2.0, 6, 20
RECENT_FROM, RECENT_TRADES, RECENT_T, RECENT_EX_T = pd.Timestamp("2024-11-20"), 150, 2.0, 1.5
SPLIT_FALL, SPLIT_MATCH, BROKEN = 0.75, 0.06, 0.70
FACTORS = (1.25, 4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)
RULES = ("ST1", "ST2", "ST3", "ST4", "ST5", "ST6", "ST7")
DESCRIPTIONS = {
    "ST1": "short-term: RSI(2) under 10 above the 200-day average, out on a close above the 5-day average (max 10)",
    "ST2": "short-term: three lower closes to a 10-day low in a strong trend, out on a close above the previous high (max 7)",
    "ST3": "short-term: 126-day momentum top 20% back at its 20-day average, hold 10, stop 8%",
    "ST4": "short-term: NR7 inside-day buy-stop breakout in an uptrend, stop at the inside day's low, hold 10",
    "ST5": "short-term: weekly top 10 by 20-day return above the 200-day average, hold 5",
    "ST6": "short-term: weekly 10 biggest 5-day losers among the 200 most traded above the 200-day average, hold 5",
    "ST7": "short-term: 20-day closing breakout on 1.5x volume without a gap, 10-day-low exit (max 20), stop 8%",
}


def cost(position: float) -> float:
    return (india_delivery_fees("BUY", position) + india_delivery_fees("SELL", position)) / position


def net(gross, slip: float = SLIPPAGE, position: float = POSITION):
    return gross - cost(position) - 2 * slip


# -- data ---------------------------------------------------------------------------------------------------------------------
def download() -> None:
    import src.config  # noqa: F401  (loads ../.env)
    from src.data.angel import AngelSession, load_tokens

    api, tokens = AngelSession().api(), load_tokens(ROOT / "research_cache")
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sorted(tokens) if not (CACHE / f"{s}.pkl").exists()]
    print(f"{len(tokens)} NSE equities, {len(todo)} to fetch", flush=True)
    for i, symbol in enumerate(todo):
        rows: List[list] = []
        start = START
        while start < datetime.now():
            end = min(start + timedelta(days=CHUNK_DAYS), datetime.now())
            params = {"exchange": "NSE", "symboltoken": tokens[symbol], "interval": "ONE_DAY"}
            params.update(fromdate=f"{start:%Y-%m-%d} 09:15", todate=f"{end:%Y-%m-%d} 15:30")
            for attempt in range(5):
                try:
                    rows += (api.getCandleData(params) or {}).get("data") or []
                    break
                except Exception as e:  # the broker's rate limit answers with text, not JSON
                    print(f"{symbol} {start:%Y-%m-%d}: {type(e).__name__}, retrying", flush=True)
                    time.sleep(PACE * 5 * (attempt + 1))
            time.sleep(PACE)
            start = end + timedelta(days=1)
        frame = pd.DataFrame(rows, columns=["at", "Open", "High", "Low", "Close", "Volume"]).drop_duplicates("at")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("at"))).tz_localize(None).normalize()
        frame.sort_index().to_pickle(CACHE / f"{symbol}.pkl")  # an empty frame too: a listing with no daily bars
        if i % 100 == 0:
            print(f"{i + 1}/{len(todo)} {symbol}: {len(frame)} bars", flush=True)
    print("done", flush=True)


def adjust(frame: pd.DataFrame) -> pd.DataFrame:
    """Undo splits and bonuses: an overnight fall under SPLIT_FALL within SPLIT_MATCH of a common ratio."""
    out = frame.copy()
    ratio = (out["Open"] / out["Close"].shift(1)).to_numpy()
    for i in np.flatnonzero(np.nan_to_num(ratio, nan=1.0) < SPLIT_FALL):
        factor = min(FACTORS, key=lambda f: abs(math.log(f * ratio[i])))
        if abs(math.log(factor * ratio[i])) <= SPLIT_MATCH:
            out.iloc[:i, out.columns.get_indexer(["Open", "High", "Low", "Close"])] /= factor
            out.iloc[:i, out.columns.get_loc("Volume")] *= factor
    return out


def load() -> Dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(CACHE.glob("*.pkl")):
        df = pd.read_pickle(path)
        if len(df) >= 300:
            df = df.astype(float).dropna()
            df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]  # a bar with a zero price is a data error
            out[path.stem] = adjust(df)
    return out


# -- indicators -----------------------------------------------------------------------------------------------------------------
def mean_before(a: np.ndarray, w: int) -> np.ndarray:
    cs, out = np.cumsum(np.insert(a, 0, 0.0)), np.full(len(a), np.nan)
    out[w:] = (cs[w : len(a)] - cs[: len(a) - w]) / w
    return out


def mean_to(a: np.ndarray, w: int) -> np.ndarray:
    cs, out = np.cumsum(np.insert(a, 0, 0.0)), np.full(len(a), np.nan)
    out[w - 1 :] = (cs[w:] - cs[: len(a) - w + 1]) / w
    return out


def rsi2(c: np.ndarray) -> np.ndarray:
    d = pd.Series(c).diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).to_numpy()


class Stock:
    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.o, self.h, self.l, self.c, self.v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        c, n = self.c, len(df.index)
        self.n = n
        self.cp = np.r_[np.nan, c[:-1]]
        self.tv = mean_before(c * self.v, 20)
        self.vavg = mean_before(self.v, 20)
        self.ma5, self.ma20, self.ma50, self.ma200 = (mean_to(c, w) for w in (5, 20, 50, 200))
        self.liquid = np.nan_to_num((self.tv >= MIN_VALUE) & (self.cp >= MIN_PRICE), nan=0).astype(bool)
        s = pd.Series(c)
        self.ret5, self.ret20, self.ret126 = ((s / s.shift(w) - 1).to_numpy() for w in (5, 20, 126))
        gap = self.o / self.cp
        self.broken = np.nan_to_num(gap, nan=1.0) < BROKEN

    def trade(self, e: int, max_hold: int, entry: Optional[float] = None, stop: Optional[float] = None, out=None):
        """(gross, exit index) of a long bought at `entry` (default the open) in session e, or None (no trade / no exit yet)."""
        o, lo, n = self.o, self.l, self.n
        entry = o[e] if entry is None else entry
        if stop is not None and stop >= entry:
            return None
        for k in range(e, min(e + max_hold, n)):
            if stop is not None and lo[k] <= stop:
                return (min(o[k], stop) if k > e else stop) / entry - 1, k
            if out is not None and out[k]:
                return (o[k + 1] / entry - 1, k + 1) if k + 1 < n else None
        x = e + max_hold
        return (o[x] / entry - 1, x) if x < n else None


def signals_per_stock(st: Stock, top126: np.ndarray) -> Dict[str, list]:
    c, h, lo, n = st.c, st.h, st.l, st.n
    above = np.nan_to_num(c > st.ma200, nan=0).astype(bool)
    liq = st.liquid & above
    found: Dict[str, list] = {r: [] for r in ("ST1", "ST2", "ST3", "ST4", "ST7")}

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

    # ST1
    r2 = rsi2(c)
    walk("ST1", liq & (r2 < 10), lambda d: st.trade(d + 1, 10, out=np.nan_to_num(c > st.ma5, nan=0).astype(bool)))
    # ST2
    lower3 = np.r_[np.zeros(3, bool), (c[3:] < c[2:-1]) & (c[2:-1] < c[1:-2]) & (c[1:-2] < c[:-3])]
    low10 = c <= pd.Series(c).rolling(10).min().to_numpy()
    trend = np.nan_to_num(st.ma50 > st.ma200, nan=0).astype(bool)
    above_prev_high = np.r_[False, c[1:] > h[:-1]]
    walk("ST2", liq & trend & lower3 & low10, lambda d: st.trade(d + 1, 7, out=above_prev_high))
    # ST3
    touch = np.nan_to_num((lo <= st.ma20) & (c > st.ma20) & (c > st.ma50), nan=0).astype(bool)
    walk("ST3", st.liquid & top126 & touch, lambda d: st.trade(d + 1, 10, stop=st.o[d + 1] * 0.92))
    # ST4
    rng = h - lo
    inside = np.r_[False, (h[1:] < h[:-1]) & (lo[1:] > lo[:-1])]
    nr7 = rng <= pd.Series(rng).rolling(7).min().to_numpy()
    up = np.nan_to_num((c > st.ma50) & (st.ma50 > st.ma200), nan=0).astype(bool)

    def st4(d):
        e = d + 1
        if h[e] <= h[d]:
            return None
        return st.trade(e, 10, entry=max(st.o[e], h[d]), stop=lo[d])

    walk("ST4", st.liquid & inside & nr7 & up, st4)
    # ST7
    hi20 = pd.Series(c).rolling(20).max().shift(1).to_numpy()
    low10x = np.r_[False, (c < pd.Series(lo).rolling(10).min().shift(1).to_numpy())[1:]]
    no_gap = np.nan_to_num(st.o / st.cp - 1 < 0.02, nan=0).astype(bool)
    vol = np.nan_to_num(st.v >= 1.5 * st.vavg, nan=0).astype(bool)
    brk = np.nan_to_num(c > hi20, nan=0).astype(bool)
    walk("ST7", liq & brk & vol & no_gap, lambda d: st.trade(d + 1, 20, stop=st.o[d + 1] * 0.92, out=low10x))
    return found


def run(stocks: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    built = {s: Stock(df) for s, df in stocks.items()}
    days = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in built.values()])))

    def matrix(attr):
        return pd.DataFrame({s: pd.Series(getattr(b, attr), index=b.index) for s, b in built.items()}).reindex(days)

    liquid = matrix("liquid").astype(float).fillna(0.0).astype(bool)
    above = matrix("c") > matrix("ma200")
    opens = matrix("o")
    ret126 = matrix("ret126").where(liquid)
    top126 = ret126.rank(axis=1, pct=True) >= 0.8
    rows = []
    for s, b in built.items():
        found = signals_per_stock(b, top126[s].reindex(b.index).fillna(False).to_numpy(bool))
        for rule, trades in found.items():
            rows += [(rule, s, b.index[e], b.index[x], g) for e, g, x in trades]
    # ST5 / ST6: across stocks on each week's last session
    week = days.to_period("W-SUN").asi8
    last = days[:-1][week[:-1] != week[1:]]
    ret20, ret5, tv = matrix("ret20"), matrix("ret5"), matrix("tv")
    for d in last:
        ok = liquid.loc[d] & above.loc[d]
        if ok.sum() < 30:
            continue
        mom = ret20.loc[d][ok].dropna().nlargest(10).index
        big = tv.loc[d][ok].dropna().nlargest(200).index
        rev = ret5.loc[d][big].dropna().nsmallest(10).index
        for rule, picks in (("ST5", mom), ("ST6", rev)):
            for s in picks:
                b = built[s]
                i = b.index.searchsorted(d)
                if i + 1 >= b.n or b.index[i] != d:
                    continue
                r = b.trade(i + 1, 5)
                if r is not None and not b.broken[i + 1 : r[1] + 1].any():
                    rows.append((rule, s, b.index[i + 1], b.index[r[1]], r[0]))
    trades = pd.DataFrame(rows, columns=["rule", "sym", "date", "exit", "gross"])
    # the median liquid stock over the same window: what a random entry on that date would have made
    log_open = np.log(opens)
    control = {}
    for e, x in set(zip(trades["date"], trades["exit"])):
        control[(e, x)] = float(np.nanmedian((log_open.loc[x] - log_open.loc[e])[liquid.loc[e]].to_numpy()))
    trades["excess"] = trades["gross"] - [math.expm1(control[(e, x)]) for e, x in zip(trades["date"], trades["exit"])]
    return trades, days, liquid


# -- verdicts --------------------------------------------------------------------------------------------------------------------
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
    hold = (t["exit"] - t["date"]).dt.days.mean()
    out = {
        "rule": rule, "trades": len(t), "dates": len(per), "gross": t["gross"].mean() * 100, "net": t["net"].mean() * 100,
        "t": tstat(per), "h1": per.iloc[:half].mean() * 100, "h2": per.iloc[half:].mean() * 100, "slip2": t["net2"].mean() * 100,
        "win": (t["net"] > 0).mean() * 100, "excess": t["excess"].mean() * 100, "ex_t": tstat(ex),
        "years_pos": int((years["m"] > 0).sum()), "years": len(years), "hold": hold,
        "n5": net(t["gross"], position=5_000).mean() * 100, "n25": net(t["gross"], position=25_000).mean() * 100,
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
    stocks = load()
    trades, days, _ = run(stocks)
    print(
        f"{len(stocks)} NSE stocks (Angel One daily), {days[0].date()} .. {days[-1].date()}; delivery fees "
        f"{cost(POSITION) * 100:.3f}% + slippage {2 * SLIPPAGE * 100:.2f}% a round trip on Rs {POSITION:,.0f}",
        flush=True,
    )
    with TRIALS.open() as f:
        n_trials = sum(int(r["variants"] or 0) for r in csv.DictReader(f)) + len(RULES)
    results = [judge(k, trades[trades["rule"] == k], n_trials) for k in RULES]
    print(f"\n{'rule':5}{'trades':>8}{'days':>6}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}{'2x slip':>9}"
          f"{'win':>6}{'excess':>9}{'ex t':>6}{'yrs+':>7}{'defl':>6}{'@5k':>9}{'@25k':>9}")  # fmt: skip
    for r in results:
        if not r["trades"]:
            print(f"{r['rule']:5} no trades")
            continue
        print(f"{r['rule']:5}{r['trades']:>8}{r['hold']:>6.1f}{r['gross']:>+8.3f}%{r['net']:>+8.3f}%{r['t']:>6.1f}{r['h1']:>+9.3f}%"
              f"{r['h2']:>+9.3f}%{r['slip2']:>+8.3f}%{r['win']:>5.0f}%{r['excess']:>+8.3f}%{r['ex_t']:>6.1f}"
              f"{r['years_pos']:>4}/{r['years']:<2}{r['deflated']:>6.2f}{r['n5']:>+8.3f}%{r['n25']:>+8.3f}%")  # fmt: skip
    print(f"\n(days: average calendar days held; deflated: probability the t beats the best of {n_trials} trials by luck)")
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
            print(f"  {r['rule']}: {r['trades']} trades, net {r['net']:+.3f}%, t {r['t']:.1f}, halves {r['h1']:+.3f}/{r['h2']:+.3f}%, "
                  f"excess {r['excess']:+.3f}% -> {'PASS' if r['pass'] else 'fails - ' + r['why']}")  # fmt: skip
    passed = sorted([r for r in results if r["pass"]], key=lambda r: -r["t"])
    print("\nCANDIDATES: " + (", ".join(r["rule"] for r in passed) if passed else "none"))
    rec = [r for r in recent if r["pass"] and r["rule"] not in {p["rule"] for p in passed}]
    print("RECENT-ONLY CANDIDATES (paper test only, weaker evidence): " + (", ".join(r["rule"] for r in rec) or "none"))
    trades.to_pickle(ROOT / "logs" / "research_short_term_trades.pkl")
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
                        "scripts/research_short_term.py"])  # fmt: skip


if __name__ == "__main__":
    download() if sys.argv[1:2] == ["download"] else main()
