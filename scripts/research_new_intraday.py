"""Six new intraday ideas for the ORB account, on 8 years of Angel One 5-minute bars (pre-declared 2026-11-06 before any data
was downloaded or any result seen; trial ids N1-N6).

    python scripts/research_new_intraday.py download   # Angel One 5-minute candles to research_cache/angel5m/ (resumes)
    python scripts/research_new_intraday.py test       # prints the table and the verdicts, appends to the trial register

Why these. Every price-pattern rule tested so far (ORB and ~25 variants, gap fades, VWAP reversion, popular indicators, an ML
filter) lost to the ~0.21% round trip, and the live ORB account (-0.20% a trade on 173) matched its 8-year test (-0.23%). The
only rule with a gross move well above costs traded right after results: moves with information behind them continue. So
five of the six look for information or a crowd effect without a filing, and the sixth is a market-level timing rule on an ETF.
None of them was tested before in this project.

Data. The 200 stocks with the highest median daily traded value in the live feed's saved bars (angel_bars in intraday.db), plus
NIFTYBEES and the Nifty 50 index, from 2018-01-01. Today's liquid stocks only, which flatters long rules (they are past winners);
the excess test against the median stock of the same list is there to take that out. Angel One's candles are not adjusted, so a
session that opens more than 25% from the previous close is dropped (a split or bonus), and so is a look-back holding one.

Common to every stock rule. A session needs at least 70 of its 75 bars, the 09:15 bar, a previous close of Rs 100 or more and
a 20-session average traded value of Rs 10 crore or more. Rs 5,000 a trade, entry at the open of the bar after the signal bar,
5 bps slippage a side, the live intraday fees (src.engine.costs.intraday_fees), out at the 15:15 open (the 15:10 close if that bar
is missing). A stop is checked on every bar from the entry bar on; a bar that opens beyond the stop fills at its open. One trade per
stock per rule a day, no daily cap. VWAP is the session's cumulative (high + low + close) / 3 weighted by volume.

N1 52-week high. The stock opens below the highest high of its previous 250 sessions (250 needed). Signal: the first bar from
   09:30 to 14:00 that closes above that high and above VWAP. Long; stop at the session's low up to the signal; skipped if the stop
   is more than 3% or less than 0.5% away. Reason: anchoring, investors under-react to news that lifts a stock through its high.
N2 Volume shock, day after. Yesterday traded at least 4 times its average volume of the 20 sessions before, rose 2% or more on the
   previous close and closed in the top quarter of its range. Today opens between -1% and +4% on yesterday's close. Long at the
   09:20 open, stop 2% below the entry. Reason: an unexplained volume shock is information arriving; informed buying continues.
N3 NR7 breakout. Yesterday's range (high - low over close) was the narrowest of the last 7 sessions. Signal: the first bar from
   09:30 that closes above the 09:15-09:30 high, closing by 11:00. Long; stop at the 09:15-09:30 low; skipped if more than 3% or
   less than 0.5% away. Reason: compression before expansion, so the breakout should carry further than an ordinary one.
N4 Afternoon strength. At the 14:00 bar's close the stock is 2% or more above the previous close, within 0.5% of the session's
   high and above VWAP, and the median stock of the list is above its own open. Long at the 14:05 open, stop 1% below. Reason:
   institutions build positions late in the day and the strongest stocks are being bought into the close.
N5 Exhaustion short. Up 12% or more over the previous 3 sessions, opens 0.5% or more above yesterday's close, then the first bar
   from 09:30 that closes below VWAP, closing by 11:00. Short; stop at the session's high up to the signal; skipped if more than
   3% or less than 0.5% away. Reason: late buyers chasing a run fail, and a gap up that cannot hold VWAP is the first sign.
N6 Market intraday momentum (Gao, Han, Li and Zhou 2018). The Nifty 50's move from the previous close to the 09:40 bar's close
   (09:45) gives the side; NIFTYBEES is bought (or sold short) at the 14:45 open and closed at 15:15. Every session with both
   series; no stop. Reason: the first half hour's move predicts the last half hour's, documented in the US and other markets.

Gate, each rule on its own, all of these: at least 300 trades; mean net > 0; date-clustered t of net >= 2.6; both halves of the
dates > 0; net > 0 with the slippage doubled; and (N1-N5) the excess over the median stock of the list in the same entry-to-15:15
window, no stop, > 0 with date-clustered t >= 2.0. The deflated t against every trial in the register is printed beside it. A pass
makes the rule a candidate for the ORB account in paper (orb-v6), the owner's decision; more than one pass, the one with the highest
t is offered first. Second route (added before any data was downloaded, at the owner's point that today's market may behave
differently): the same rules on sessions from 2024-11-20 only (SEBI's F&O curbs), >= 150 trades, net > 0 with t >= 2.0, both halves
> 0, > 0 at twice the slippage, excess > 0 with t >= 1.5. A rule passing only there is a weaker candidate, for a paper test only.
Net by calendar year is printed for every rule. Nothing here changes a live account. A neighbouring variant is never a second chance. The download is kept until
the owner has decided on the result; if no rule passes it is deleted then, and only this script and its log remain."""

import csv
import json
import pickle
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config  # noqa: E402, F401  (loads .env: the Angel One keys)
from src.data.angel import AngelSession, load_tokens  # noqa: E402
from src.engine.costs import SLIPPAGE, intraday_fees  # noqa: E402
from src.research.significance import deflated_t  # noqa: E402

CACHE, OUT = ROOT / "research_cache", ROOT / "research_cache" / "angel5m"
TRIALS = ROOT / "docs" / "trading" / "research-trials.csv"
START, CHUNK_DAYS, PACE, TRIES = date(2018, 1, 1), 90, 1.1, 5
N_STOCKS, NIFTY_TOKEN, ETF = 200, "99926000", "NIFTYBEES"
SLOTS = 75  # 09:15 .. 15:25
EXIT_SLOT, LAST_SLOT = 72, 71  # 15:15 open, else the 15:10 bar's close
POSITION, MIN_PRICE, MIN_VALUE, SPLIT = 5000.0, 100.0, 1e8, 0.25
MIN_TRADES, T_MIN, T_EXCESS = 300, 2.6, 2.0
# The recent market alone (owner's point, added before any data was downloaded): from SEBI's F&O curbs of 2024-11-20, with retail
# flows, weekly expiries and costs as they are now. A weaker bar, because two years hold fewer trades.
RECENT_FROM, RECENT_TRADES, RECENT_T, RECENT_EX_T = pd.Timestamp("2024-11-20"), 150, 2.0, 1.5
FEES = (intraday_fees("BUY", POSITION) + intraday_fees("SELL", POSITION)) / POSITION


def slot(hh: int, mm: int) -> int:
    return ((hh - 9) * 60 + mm - 15) // 5


# --------------------------------------------------------------------------------------------------------------- download
def universe() -> list:
    con = sqlite3.connect(ROOT / "intraday.db")
    rows = con.execute("select symbol, day, sum(close * volume) from angel_bars group by symbol, day").fetchall()
    con.close()
    frame = pd.DataFrame(rows, columns=["symbol", "day", "value"])
    ranked = frame.groupby("symbol")["value"].median().sort_values(ascending=False)
    return [s for s in ranked.index if s != ETF][:N_STOCKS]


def fetch(api, token: str, exchange: str = "NSE") -> list:
    rows: list = []
    start = START
    while start <= date.today():
        end = min(start + timedelta(days=CHUNK_DAYS - 1), date.today())
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": f"{start} 09:15",
            "todate": f"{end} 15:30",
        }
        for attempt in range(TRIES):
            try:
                rows += api.getCandleData(params).get("data") or []
                break
            except Exception:
                if attempt == TRIES - 1:
                    raise
                time.sleep(PACE * 5 * (attempt + 1))
        time.sleep(PACE)
        start = end + timedelta(days=1)
    return rows


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens(CACHE)
    wanted = {s: tokens[s] for s in universe() if s in tokens}
    wanted[ETF] = tokens[ETF]
    wanted["NIFTY"] = NIFTY_TOKEN
    (OUT / "universe.json").write_text(json.dumps(sorted(set(wanted) - {ETF, "NIFTY"})))
    api = AngelSession().api()
    for pass_no in range(3):
        missing = [s for s in wanted if not (OUT / f"{s}.pkl").exists()]
        print(f"pass {pass_no + 1}: {len(wanted) - len(missing)} of {len(wanted)} saved", flush=True)
        for symbol in missing:
            try:
                rows = fetch(api, wanted[symbol])
            except Exception as e:
                print(f"{symbol}: {type(e).__name__}", flush=True)
                continue
            frame = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
            (OUT / f"{symbol}.pkl").write_bytes(pickle.dumps(frame))
            print(f"{symbol}: {len(frame)} bars", flush=True)
        if not missing:
            break
    print("finished", flush=True)


# ------------------------------------------------------------------------------------------------------------------- load
def grid(frame: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """(5, days, 75) float32 of open, high, low, close, volume; NaN where a bar is missing."""
    out = np.full((5, len(days), SLOTS), np.nan, dtype=np.float32)
    if frame.empty:
        return out
    ts = frame["ts"]
    s = ((ts.dt.hour - 9) * 60 + ts.dt.minute - 15) // 5
    d = days.get_indexer(ts.dt.normalize())
    ok = (s >= 0) & (s < SLOTS) & (d >= 0)
    for k, col in enumerate("ohlcv"):
        out[k, d[ok], s[ok]] = frame[col].values[ok]
    return out


def load():
    names = json.loads((OUT / "universe.json").read_text())
    names = [n for n in names if (OUT / f"{n}.pkl").exists()]
    frames = {}
    for n in names + [ETF, "NIFTY"]:
        frame = pickle.loads((OUT / f"{n}.pkl").read_bytes())  # nosec B301
        frame["ts"] = pd.to_datetime(frame["ts"], format="ISO8601").dt.tz_localize(None)
        frames[n] = frame
    days = pd.DatetimeIndex(sorted(set().union(*(f["ts"].dt.normalize().unique() for f in frames.values() if len(f)))))
    bars = np.stack([grid(frames[n], days) for n in names], axis=1)  # (5, stocks, days, 75)
    return names, days, bars, grid(frames[ETF], days), grid(frames["NIFTY"], days)


# ------------------------------------------------------------------------------------------------------------- simulation
def daily(o, h, l, c, v):
    """Per stock-day open, high, low, close, volume, bar count; and the previous valid session's index."""
    n_bars = np.sum(~np.isnan(c), axis=-1)
    with np.errstate(all="ignore"):
        d_open = o[..., 0]
        d_high, d_low = np.nanmax(h, axis=-1), np.nanmin(l, axis=-1)
        d_vol = np.nansum(v, axis=-1)
    last = SLOTS - 1 - np.argmax(~np.isnan(c[..., ::-1]), axis=-1)
    d_close = np.take_along_axis(c, last[..., None], axis=-1)[..., 0]
    d_close[n_bars == 0] = np.nan
    return d_open, d_high, d_low, d_close, d_vol, n_bars


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[:, d - k] for each d (NaN before the start); sessions are the calendar of all stocks, a stock's missing days NaN."""
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def vwap(h, l, c, v):
    tp = (h + l + c) / 3
    with np.errstate(all="ignore"):
        return np.nancumsum(np.nan_to_num(tp) * np.nan_to_num(v), axis=-1) / np.nancumsum(np.nan_to_num(v), axis=-1)


def run_trade(side, o, h, l, c, entry_slot, stop):
    """Gross return of one trade, and the no-stop gross return of the same window (for the excess)."""
    entry = o[entry_slot]
    if np.isnan(entry):
        return None
    exit_px, end = np.nan, (o[EXIT_SLOT] if not np.isnan(o[EXIT_SLOT]) else c[LAST_SLOT])
    if np.isnan(end):
        return None
    if stop is not None:
        for s in range(entry_slot, EXIT_SLOT):
            if np.isnan(l[s]):
                continue
            if side > 0 and l[s] <= stop:
                exit_px = min(o[s], stop) if not np.isnan(o[s]) else stop
                break
            if side < 0 and h[s] >= stop:
                exit_px = max(o[s], stop) if not np.isnan(o[s]) else stop
                break
    if np.isnan(exit_px):
        exit_px = end
    return side * (exit_px / entry - 1), side * (end / entry - 1), entry


def net(gross: float, slip: float = SLIPPAGE) -> float:
    return gross - 2 * slip - FEES


def first(cond: np.ndarray, lo: int, hi: int):
    """The first slot in [lo, hi] where cond holds, else None."""
    hits = np.flatnonzero(cond[lo : hi + 1])
    return lo + int(hits[0]) if len(hits) else None


def scan(names, days, bars, etf, nifty):
    o, h, l, c, v = bars
    d_open, d_high, d_low, d_close, d_vol, n_bars = daily(o, h, l, c, v)
    valid = (n_bars >= 70) & ~np.isnan(o[..., 0])
    # previous valid session of each stock: carry the last valid close forward one session at a time
    S, D = d_close.shape
    prev_idx = np.full((S, D), -1)
    last_seen = np.full(S, -1)
    for d in range(D):
        prev_idx[:, d] = last_seen
        last_seen = np.where(valid[:, d], d, last_seen)

    def prev(a, idx):
        out = np.full(idx.shape, np.nan)
        ok = idx >= 0
        rows = np.nonzero(ok)
        out[ok] = a[rows[0], idx[ok]]
        return out

    pc = prev(d_close, prev_idx)  # previous close
    pidx2 = np.where(prev_idx >= 0, prev(prev_idx.astype(float), prev_idx), -1).astype(int)
    value = np.where(valid, d_close * d_vol, np.nan)
    val20 = pd.DataFrame(value.T).rolling(20, min_periods=15).mean().shift(1).values.T  # calendar sessions, close enough
    vol20 = pd.DataFrame(np.where(valid, d_vol, np.nan).T).rolling(20, min_periods=15).mean().shift(1).values.T
    gap = d_open / pc - 1
    tradable = valid & (pc >= MIN_PRICE) & (val20 >= MIN_VALUE) & (np.abs(gap) <= SPLIT)
    vw = vwap(h, l, c, v)
    with np.errstate(all="ignore"):
        bench = np.nanmedian(o[..., EXIT_SLOT][:, :, None] / o - 1, axis=0)  # (days, 75): median stock, entry slot -> 15:15
        mkt_up_14 = np.nanmedian(c[..., slot(14, 0)] / o[..., 0] - 1, axis=0) > 0
    # look-backs on each stock's own valid sessions
    hi_dense = np.where(valid, d_high, np.nan)
    high250 = np.full((S, D), np.nan)
    nr7 = np.zeros((S, D), dtype=bool)
    ret3 = np.full((S, D), np.nan)
    for i in range(S):
        idx = np.flatnonzero(valid[i])
        if len(idx) < 10:
            continue
        hh = pd.Series(hi_dense[i, idx])
        h250 = hh.rolling(250, min_periods=250).max().shift(1).values
        rng = pd.Series((d_high[i, idx] - d_low[i, idx]) / d_close[i, idx])
        is_nr7 = (rng == rng.rolling(7, min_periods=7).min()).shift(1, fill_value=False).values
        cl = pd.Series(d_close[i, idx])
        step = cl / cl.shift(1) - 1
        big = step.abs().rolling(3, min_periods=3).max() > SPLIT
        r3 = (cl / cl.shift(3) - 1).where(~big).shift(1).values
        high250[i, idx], nr7[i, idx], ret3[i, idx] = h250, is_nr7, r3
    y_vol = prev(d_vol, prev_idx)
    y_vol20 = prev(vol20, prev_idx)
    y_ret = prev(d_close, prev_idx) / prev(d_close, pidx2) - 1
    y_loc = (prev(d_close, prev_idx) - prev(d_low, prev_idx)) / (prev(d_high, prev_idx) - prev(d_low, prev_idx))

    trades = {k: [] for k in ("N1", "N2", "N3", "N4", "N5", "N6")}
    s930, s1100, s1400 = slot(9, 30), slot(10, 55), slot(14, 0)
    for i in range(S):
        for d in np.flatnonzero(tradable[i]):
            oo, hh_, ll, cc, vv = o[i, d], h[i, d], l[i, d], c[i, d], vw[i, d]

            def add(rule, side, sig, stop, oo=oo, hh_=hh_, ll=ll, cc=cc, d=d):
                if sig is None or sig + 1 >= EXIT_SLOT:
                    return
                if stop is not None:
                    entry = oo[sig + 1]
                    dist = abs(entry / stop - 1) if not np.isnan(entry) else np.nan
                    if not 0.005 <= dist <= 0.03:
                        return
                r = run_trade(side, oo, hh_, ll, cc, sig + 1, stop)
                if r:
                    g, nostop, _ = r
                    trades[rule].append((d, g, nostop - side * bench[d, sig + 1]))

            if not np.isnan(high250[i, d]) and oo[0] < high250[i, d]:
                sig = first((cc > high250[i, d]) & (cc > vv), s930, s1400)
                add("N1", 1, sig, None if sig is None else np.nanmin(ll[: sig + 1]))
            if (y_vol[i, d] >= 4 * y_vol20[i, d]) and y_ret[i, d] >= 0.02 and y_loc[i, d] >= 0.75 and -0.01 <= gap[i, d] <= 0.04:
                r = run_trade(1, oo, hh_, ll, cc, 1, oo[1] * 0.98 if not np.isnan(oo[1]) else None)
                if r and not np.isnan(oo[1]):
                    trades["N2"].append((d, r[0], r[1] - bench[d, 1]))
            if nr7[i, d]:
                or_h, or_l = np.nanmax(hh_[:3]), np.nanmin(ll[:3])
                add("N3", 1, first(cc > or_h, s930, s1100), or_l)
            if (
                mkt_up_14[d]
                and cc[s1400] / pc[i, d] - 1 >= 0.02
                and cc[s1400] >= 0.995 * np.nanmax(hh_[: s1400 + 1])
                and cc[s1400] > vv[s1400]
            ):
                e = s1400 + 1
                if not np.isnan(oo[e]):
                    r = run_trade(1, oo, hh_, ll, cc, e, oo[e] * 0.99)
                    if r:
                        trades["N4"].append((d, r[0], r[1] - bench[d, e]))
            if ret3[i, d] >= 0.12 and gap[i, d] >= 0.005:
                sig = first(cc < vv, s930, s1100)
                add("N5", -1, sig, None if sig is None else np.nanmax(hh_[: sig + 1]))
    # N6
    eo, ec = etf[0], etf[3]
    nd = daily(*[x[None] for x in nifty])
    n_close, n_bars = nd[3][0], nd[5][0]
    for d in range(1, D):
        if n_bars[d] < 70 or n_bars[d - 1] < 70:
            continue
        r1 = nifty[3][d, slot(9, 40)] / n_close[d - 1] - 1
        if np.isnan(r1) or r1 == 0:
            continue
        side = 1 if r1 > 0 else -1
        r = run_trade(side, eo[d], etf[1][d], etf[2][d], ec[d], slot(14, 45), None)
        if r:
            trades["N6"].append((d, r[0], np.nan))
    return trades


# ---------------------------------------------------------------------------------------------------------------- verdict
def by_date(rows, col, f=lambda x: x):
    frame = pd.DataFrame(rows, columns=["d", "g", "x"])
    return frame.assign(y=f(frame[col])).groupby("d")["y"].mean().sort_index()


def tstat(per: pd.Series) -> float:
    return per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 2 and per.std(ddof=1) > 0 else float("nan")


def judge(rule, rows, n_trials, min_trades=MIN_TRADES, t_min=T_MIN, ex_t_min=T_EXCESS):
    if not rows:
        return {"rule": rule, "trades": 0, "pass": False, "why": "no trades"}
    g = np.array([r[1] for r in rows])
    per = by_date(rows, "g", net)
    half = len(per) // 2
    out = {
        "rule": rule,
        "trades": len(rows),
        "dates": len(per),
        "gross": g.mean() * 100,
        "net": net(g).mean() * 100,
        "t": tstat(per),
        "h1": per.iloc[:half].mean() * 100,
        "h2": per.iloc[half:].mean() * 100,
        "slip2": net(g, 2 * SLIPPAGE).mean() * 100,
        "win": (net(g) > 0).mean() * 100,
    }
    why = []
    if out["trades"] < min_trades:
        why.append(f"{out['trades']} trades < {min_trades}")
    if not (out["net"] > 0 and out["t"] >= t_min):
        why.append(f"net {out['net']:+.3f}% t {out['t']:.1f}")
    if not (out["h1"] > 0 and out["h2"] > 0):
        why.append(f"halves {out['h1']:+.3f}/{out['h2']:+.3f}")
    if out["slip2"] <= 0:
        why.append(f"2x slip {out['slip2']:+.3f}%")
    if rule != "N6":
        ex = by_date(rows, "x")
        out["excess"], out["ex_t"] = pd.DataFrame(rows)[2].mean() * 100, tstat(ex)
        if not (out["excess"] > 0 and out["ex_t"] >= ex_t_min):
            why.append(f"excess {out['excess']:+.3f}% t {out['ex_t']:.1f}")
    out["deflated"] = deflated_t(out["t"], n_trials).probability if not np.isnan(out["t"]) else float("nan")
    out["pass"], out["why"] = not why, "; ".join(why) or "PASS"
    return out


def test() -> None:
    names, days, bars, etf, nifty = load()
    print(
        f"{len(names)} stocks, {len(days)} sessions {days[0].date()} .. {days[-1].date()}; fees {FEES * 100:.3f}% + slippage "
        f"{2 * SLIPPAGE * 100:.2f}% a round trip",
        flush=True,
    )
    trades = scan(names, days, bars, etf, nifty)
    with TRIALS.open() as f:
        n_trials = sum(int(r["variants"] or 0) for r in csv.DictReader(f)) + 6
    print(
        f"\n{'rule':5}{'trades':>8}{'dates':>7}{'gross':>9}{'net':>9}{'t':>6}{'1st half':>10}{'2nd half':>10}{'2x slip':>9}"
        f"{'win':>6}{'excess':>9}{'ex t':>6}{'defl':>6}"
    )
    results = [judge(k, v, n_trials) for k, v in trades.items()]
    for r in results:
        if not r["trades"]:
            print(f"{r['rule']:5} no trades")
            continue
        print(
            f"{r['rule']:5}{r['trades']:>8}{r['dates']:>7}{r['gross']:>+8.3f}%{r['net']:>+8.3f}%{r['t']:>6.1f}{r['h1']:>+9.3f}%"
            f"{r['h2']:>+9.3f}%{r['slip2']:>+8.3f}%{r['win']:>5.0f}%{r.get('excess', np.nan):>+8.3f}%{r.get('ex_t', np.nan):>6.1f}"
            f"{r['deflated']:>6.2f}"
        )
    print(f"\n(deflated: probability the t beats the best of {n_trials} trials by luck)")
    for r in results:
        print(f"  {r['rule']}: {'PASS' if r['pass'] else 'fails - ' + r['why']}")
    print("\nNet a trade by calendar year (does the market still behave this way?):")
    years = sorted({days[d].year for v in trades.values() for d, _, _ in v})
    print("rule " + "".join(f"{y:>9}" for y in years))
    for k, v in trades.items():
        cells = []
        for y in years:
            g = np.array([x[1] for x in v if days[x[0]].year == y])
            cells.append(f"{net(g).mean() * 100:>+8.2f}%" if len(g) else f"{'-':>9}")
        print(f"{k:5}" + "".join(cells) + "   trades: " + " ".join(str(sum(days[x[0]].year == y for x in v)) for y in years))
    cut = days.searchsorted(RECENT_FROM)
    recent = [judge(k, [x for x in v if x[0] >= cut], n_trials, RECENT_TRADES, RECENT_T, RECENT_EX_T) for k, v in trades.items()]
    print(
        f"\nRecent market only, from {RECENT_FROM.date()} (>= {RECENT_TRADES} trades, t >= {RECENT_T}, excess t >= {RECENT_EX_T}):"
    )
    for r in recent:
        if r["trades"]:
            print(
                f"  {r['rule']}: {r['trades']} trades, net {r['net']:+.3f}%, t {r['t']:.1f}, halves {r['h1']:+.3f}/{r['h2']:+.3f}%, "
                f"excess {r.get('excess', float('nan')):+.3f}% -> {'PASS' if r['pass'] else 'fails - ' + r['why']}"
            )
        else:
            print(f"  {r['rule']}: no trades")
    passed = [r for r in results if r["pass"]]
    print("\nCANDIDATES: " + (", ".join(r["rule"] for r in sorted(passed, key=lambda r: -r["t"])) if passed else "none"))
    rec = [r for r in recent if r["pass"] and r["rule"] not in {p["rule"] for p in passed}]
    print("RECENT-ONLY CANDIDATES (paper test only, weaker evidence): " + (", ".join(r["rule"] for r in rec) or "none"))
    span = f"Angel One 5-minute {days[0].date()} to {days[-1].date()}, {len(names)} stocks + NIFTYBEES"
    with TRIALS.open("a", newline="") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow(
                [
                    date.today().isoformat(),
                    r["rule"],
                    1,
                    "new-intraday",
                    DESCRIPTIONS[r["rule"]],
                    span,
                    r["trades"],
                    f"{r.get('net', float('nan')):+.3f}",
                    f"{r.get('t', float('nan')):.1f}",
                    "pass" if r["pass"] else ("recent-only pass" if r["rule"] in {x["rule"] for x in rec} else "fail"),
                    "scripts/research_new_intraday.py",
                ]
            )


DESCRIPTIONS = {
    "N1": "52-week high breakout above VWAP 09:30-14:00, stop at the day's low, out 15:15",
    "N2": "Day after an unexplained volume shock (4x volume, +2%, top-quarter close): long 09:20, 2% stop, out 15:15",
    "N3": "NR7 day then a 15-minute opening-range breakout by 11:00, stop at the range low, out 15:15",
    "N4": "Afternoon strength: +2%, near the high, above VWAP at 14:00 on an up tape; long 14:05, 1% stop, out 15:15",
    "N5": "Exhaustion short: +12% over 3 sessions, gap up, first close below VWAP by 11:00; stop at the day's high",
    "N6": "Market intraday momentum: Nifty's first half hour sets the side of NIFTYBEES 14:45-15:15",
}


if __name__ == "__main__":
    started = datetime.now()
    {"download": download, "test": test}[sys.argv[1]]()
    print(f"took {datetime.now() - started}")
