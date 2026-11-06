"""E2, E4, E5: screening tests on the data available today (pre-declared 2026-11-06, before any of them was run; owner's
approval of the experiments in the intraday R&D report, and the owner's request to use Yahoo data while Angel One refuses candles).

    python scripts/research_yahoo_experiments.py <bars_dir>   # bars_dir: the saved-bar grids written for the N1-N6 first look

These are SCREENS, not passes. Yahoo's history is short (5-minute bars for about 60 sessions, hourly for about 3 years) and
the saved archive is about 80 sessions, so a rule that clears its screen is only a candidate for the 8-year Angel One test
(scripts/research_new_intraday.py data), where it is judged by the project's full gate. A rule that fails its screen is
dropped. Costs everywhere: the live intraday fees on Rs 5,000 plus 5 bps slippage a side (0.21% a round trip).

E4, ORB changes, each against the live ORB rule on the same stock-days (saved bars, 501 stocks, src.intraday.strategy):
  base  the live rule: src.intraday.strategy.first_breakout (15-minute range 0.3-2.5% wide, close above it and above VWAP on
        1.5x volume, 09:30-14:00), entry at the signal bar's close as the paper broker fills, stop at the range low, target
        2R, out at the 15:15 open. Stop checked before target within a bar; a bar opening past either fills at its open.
  E4a   5-minute opening range (the 09:15 bar), signals from 09:20, the same filters, stop at that bar's low, no target.
  E4b   the live rule, with the stop moved to the entry once a bar's high reaches +1R.
  E4c   the live rule, out at the close of the 12th bar after entry (one hour) unless that close is at least +0.5R.
E5, sector-led ORB: the live rule's signals, taken only when, at the signal bar, the median move since the open of the stock's
  sector peers (Yahoo's sector label; at least 3 other stocks of the list) is above 0 and above the median stock's, and the
  stock's own move since the open is above its sector's median.
Screen for E4 and E5: mean net per date better than the base on the dates both trade, date-clustered paired t >= 2.0, the
  difference positive in both halves of the dates, and mean net > 0.

E2, noise-area trend rule (Zarattini, Barbon and Aziz 2024, adapted to the Nifty), on Yahoo hourly bars from about 2023-11:
  the Nifty 50 (^NSEI) sets the signal, NIFTYBEES carries the trade. For each hourly check k (bar closes at 10:15, 11:15,
  12:15, 13:15 and 14:15), sigma_k is the mean over the previous 14 sessions of |index close at k / index open - 1|. The
  upper band is max(open, previous close) x (1 + sigma_k), the lower min(open, previous close) x (1 - sigma_k). The first
  check that closes above the upper band buys NIFTYBEES at the next bar's open; below the lower band sells it short. Exit:
  at a later check that closes back inside the band (below the upper band for a long, above the lower for a short), at the
  next bar's open; otherwise at the 15:15 open. At most one trade a day. No other parameter.
Screen for E2: at least 150 trades; mean net > 0 with t >= 2.0 (one trade a day, so trades are dates); both halves > 0;
  > 0 at twice the slippage. Also printed, not judged: the same rule with no costs, by year and by side."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import research_new_intraday as R  # noqa: E402

from src.engine.costs import SLIPPAGE  # noqa: E402
from src.intraday.strategy import RANGE_BARS, TARGET_R, first_breakout  # noqa: E402

net = R.net
MINUTES = np.array([555 + 5 * s for s in range(R.SLOTS)])  # 09:15 .. 15:25 as minutes since midnight


def exit_long(o, h, l, c, start, entry, stop, target, be_at=None, time_bar=None, half_r=None):
    """Gross return of a long from `entry`, stepping bars from `start` (the bar after the signal) to the 15:15 open."""
    risk = entry - stop
    for s in range(start, R.EXIT_SLOT):
        if np.isnan(o[s]):
            continue
        if l[s] <= stop:
            return min(o[s], stop) / entry - 1
        if target is not None and h[s] >= target:
            return max(o[s], target) / entry - 1
        if be_at is not None and h[s] >= be_at:
            stop = max(stop, entry)
        if time_bar is not None and s == time_bar and c[s] < entry + half_r * risk:
            return c[s] / entry - 1
    end = o[R.EXIT_SLOT] if not np.isnan(o[R.EXIT_SLOT]) else c[R.LAST_SLOT]
    return end / entry - 1


def orb_trades(bars, sector_ok=None):
    """{variant: [(date index, gross, base gross for the same stock-day or nan)]} over all stock-days."""
    o, h, l, c, v = bars
    S, D = c.shape[:2]
    out = {k: [] for k in ("base", "E4a", "E4b", "E4c", "E5")}
    for i in range(S):
        for d in range(D):
            cc = c[i, d]
            if np.isnan(o[i, d, 0]) or np.sum(~np.isnan(cc)) < 70:
                continue
            oo, hh, ll, vv = o[i, d], h[i, d], l[i, d], v[i, d]
            ok = ~np.isnan(cc)
            # the live rule on the session's bars (all 75 present in the selected days, or skipped)
            if not ok[:RANGE_BARS].all():
                continue
            arrays = [hh, ll, cc, vv]
            if np.isnan(arrays[2]).any():
                continue
            b = first_breakout(*arrays, MINUTES)
            base = np.nan
            if b is not None and b.index + 1 < R.EXIT_SLOT:
                entry, stop = cc[b.index], b.range_low
                target = entry + TARGET_R * (entry - stop)
                base = exit_long(oo, hh, ll, cc, b.index + 1, entry, stop, target)
                out["base"].append((d, base, base))
                out["E4b"].append(
                    (d, exit_long(oo, hh, ll, cc, b.index + 1, entry, stop, target, be_at=entry + (entry - stop)), base)
                )
                out["E4c"].append(
                    (d, exit_long(oo, hh, ll, cc, b.index + 1, entry, stop, target, time_bar=b.index + 12, half_r=0.5), base)
                )
                if sector_ok is not None and sector_ok(i, d, b.index):
                    out["E5"].append((d, base, base))
            # E4a: the 09:15 bar alone as the range
            r_hi, r_lo = hh[0], ll[0]
            width = (r_hi - r_lo) / r_hi
            if 0.003 <= width <= 0.025:
                vw, ratio = R.vwap(hh, ll, cc, vv), _ratio(vv)
                hits = np.flatnonzero((cc > r_hi) & (cc > vw) & (ratio >= 1.5) & (MINUTES >= 560) & (MINUTES < 840))
                hits = hits[hits >= 1]
                if len(hits) and hits[0] + 1 < R.EXIT_SLOT:
                    s0 = hits[0]
                    out["E4a"].append((d, exit_long(oo, hh, ll, cc, s0 + 1, cc[s0], r_lo, None), base))
    return out


def _ratio(volume):
    running = np.cumsum(volume)
    prior = np.concatenate([[np.nan], running[:-1] / np.arange(1, len(volume))])
    with np.errstate(all="ignore"):
        return np.where(prior > 0, volume / prior, 0.0)


def per_date(rows, col=1, slip=SLIPPAGE):
    f = pd.DataFrame(rows, columns=["d", "g", "b"])
    f["n"] = net(f[f.columns[col]], slip)
    return f.groupby("d")["n"].mean()


def screen_vs_base(name, rows, base_rows):
    if not rows:
        print(f"{name}: no trades")
        return False
    mine, base = per_date(rows), per_date(base_rows)
    both = pd.concat([mine, base], axis=1, keys=["m", "b"]).dropna()
    diff = both.m - both.b
    half = len(diff) // 2
    t = R.tstat(diff)
    g = np.array([r[1] for r in rows])
    ok = diff.mean() > 0 and t >= 2.0 and diff.iloc[:half].mean() > 0 and diff.iloc[half:].mean() > 0 and mine.mean() > 0
    print(
        f"{name}: {len(rows)} trades on {len(mine)} dates; gross {g.mean() * 100:+.3f}%, net {net(g).mean() * 100:+.3f}% a trade, "
        f"{mine.mean() * 100:+.3f}% a date; against the live rule on {len(both)} shared dates {both.m.mean() * 100:+.3f}% vs "
        f"{both.b.mean() * 100:+.3f}%, difference {diff.mean() * 100:+.3f}% (t {t:.1f}; halves {diff.iloc[:half].mean() * 100:+.3f}"
        f" / {diff.iloc[half:].mean() * 100:+.3f}) -> {'CANDIDATE for the 8-year test' if ok else 'dropped'}"
    )
    return ok


# ------------------------------------------------------------------------------------------------------------------- E5
def sectors(names, cache: Path):
    have = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [n for n in names if n not in have]
    if missing:
        import yfinance as yf

        def look(n):
            try:
                info = yf.Ticker(n + ".NS").info or {}
                return n, info.get("sector")
            except Exception:
                return n, None

        with ThreadPoolExecutor(8) as pool:
            have.update(dict(pool.map(look, missing)))
        cache.write_text(json.dumps(have))
    return [have.get(n) for n in names]


def sector_filter(names, bars, labels):
    o, c = bars[0], bars[3]
    with np.errstate(all="ignore"):
        move = c / o[..., :1] - 1  # (S, D, 75): each stock's move since its open at every bar
        market = np.nanmedian(move, axis=0)  # (D, 75)
    groups = {}
    for i, s in enumerate(labels):
        if s:
            groups.setdefault(s, []).append(i)

    def ok(i, d, s):
        sec = labels[i]
        if not sec:
            return False
        peers = [j for j in groups[sec] if j != i and not np.isnan(move[j, d, s])]
        if len(peers) < 3:
            return False
        med = np.median(move[peers, d, s])
        return med > 0 and med > market[d, s] and move[i, d, s] > med

    return ok


# ------------------------------------------------------------------------------------------------------------------- E2
def e2() -> None:
    import yfinance as yf

    def get(t):
        f = yf.download(t, period="730d", interval="60m", progress=False, auto_adjust=False)
        f.columns = [c[0] for c in f.columns] if isinstance(f.columns, pd.MultiIndex) else f.columns
        f.index = f.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return f

    idx, etf = get("^NSEI"), get("NIFTYBEES.NS")
    checks = ["10:15", "11:15", "12:15", "13:15", "14:15"]  # bar closes; the bar starting an hour earlier closes then
    days = sorted(set(idx.index.normalize()) & set(etf.index.normalize()))
    rows, prev_close, hist = [], None, []
    for day in days:
        di = idx[idx.index.normalize() == day]
        de = etf[etf.index.normalize() == day]
        starts = [t.strftime("%H:%M") for t in di.index]
        if len(di) < 7 or starts[0] != "09:15" or len(de) < 7:
            prev_close = float(di["Close"].iloc[-1]) if len(di) else prev_close
            continue
        op = float(di["Open"].iloc[0])
        closes = {}
        for t in di.index:
            closes[(t + pd.Timedelta(hours=1)).strftime("%H:%M")] = float(di.loc[t, "Close"])
        moves = {k: abs(closes[k] / op - 1) for k in checks if k in closes}
        if prev_close is not None and len(hist) >= 14 and len(moves) == len(checks):
            sigma = {k: np.mean([h[k] for h in hist[-14:]]) for k in checks}
            up_ref, lo_ref = max(op, prev_close), min(op, prev_close)
            side, entry_at = 0, None
            for k in checks:
                if closes[k] > up_ref * (1 + sigma[k]):
                    side, entry_at = 1, k
                    break
                if closes[k] < lo_ref * (1 - sigma[k]):
                    side, entry_at = -1, k
                    break
            if side:
                e_open = {(t).strftime("%H:%M"): float(de.loc[t, "Open"]) for t in de.index}
                entry = e_open.get(entry_at)  # the ETF bar that starts when the signal bar closes
                exit_px = e_open.get("15:15")
                for k in checks[checks.index(entry_at) + 1 :]:
                    inside = closes[k] < up_ref * (1 + sigma[k]) if side > 0 else closes[k] > lo_ref * (1 - sigma[k])
                    if inside:
                        exit_px = e_open.get(k)
                        break
                if entry and exit_px:
                    rows.append((day, side, side * (exit_px / entry - 1)))
        if len(moves) == len(checks):
            hist.append(moves)
        prev_close = float(di["Close"].iloc[-1])
    f = pd.DataFrame(rows, columns=["day", "side", "g"])
    if f.empty:
        print("E2: no trades")
        return
    f["n"], f["n2"] = net(f.g), net(f.g, 2 * SLIPPAGE)
    half = len(f) // 2
    t = R.tstat(f.n)
    ok = (
        len(f) >= 150
        and f.n.mean() > 0
        and t >= 2.0
        and f.n.iloc[:half].mean() > 0
        and f.n.iloc[half:].mean() > 0
        and f.n2.mean() > 0
    )
    print(
        f"E2: {len(f)} trades over {len(days)} sessions ({days[0].date()} .. {days[-1].date()}); gross {f.g.mean() * 100:+.3f}%, "
        f"net {f.n.mean() * 100:+.3f}% a trade (t {t:.1f}); halves {f.n.iloc[:half].mean() * 100:+.3f} / "
        f"{f.n.iloc[half:].mean() * 100:+.3f}%; 2x slippage {f.n2.mean() * 100:+.3f}%; win {(f.n > 0).mean() * 100:.0f}% "
        f"-> {'CANDIDATE for the 8-year test' if ok else 'dropped'}"
    )
    print(
        "   by side: "
        + ", ".join(f"{'long' if s > 0 else 'short'} {len(x)} at {x.n.mean() * 100:+.3f}%" for s, x in f.groupby("side"))
    )
    print("   by year: " + ", ".join(f"{y}: {len(x)} at {x.n.mean() * 100:+.3f}%" for y, x in f.groupby(f.day.dt.year)))


def main() -> None:
    bars_dir = Path(sys.argv[1])
    R.OUT = bars_dir
    names, days, bars, _, _ = R.load()
    print(f"saved bars: {len(names)} stocks, {len(days)} sessions {days[0].date()} .. {days[-1].date()}", flush=True)
    labels = sectors(names, bars_dir / "sectors.json")
    print(f"sector labels for {sum(1 for x in labels if x)} of {len(names)} stocks", flush=True)
    trades = orb_trades(bars, sector_filter(names, bars, labels))
    b = trades["base"]
    g = np.array([r[1] for r in b])
    print(f"\nbase, the live ORB rule: {len(b)} trades, gross {g.mean() * 100:+.3f}%, net {net(g).mean() * 100:+.3f}% a trade")
    for k in ("E4a", "E4b", "E4c", "E5"):
        screen_vs_base(k, trades[k], b)
    print()
    e2()


if __name__ == "__main__":
    main()
