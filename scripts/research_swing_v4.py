"""Swing exits and robustness (pre-declared 2026-10-26; follows research_swing_v2/v3, techniques from the skills.sh
backtesting skills: trade-order Monte Carlo, entry delay, cross-symbol halves).

    python scripts/research_swing_v4.py

PART 1, exits, on the live configuration (Rs 25,000, 15 slots x 8.8% under the 90% cap, 15% stop, MA200 exit), all NSE
stocks, 10 years, the cached ranked candidates of research_swing_v2:
  base       live
  trail25    stop trails 25% below the highest high since entry (never below the 15% stop)
  trail30    30%
  trail20    20%
  lock20     stop moves to the entry once the high is 20% above it
  lock10     stop moves to the entry once the high is 10% above it
Gate, fixed before running: Sharpe at least 0.05 above base's in BOTH the full period and the last 3 years, max drawdown no
worse in both, return not lower than base minus 10% of base's in both, both halves of the full period positive, and for the
trail variants the neighbouring trail also keeps Sharpe >= base's in both windows (a plateau, not one lucky number).

PART 2, robustness of the live base (no gate; it says how much to trust any difference above):
  * trade-order Monte Carlo: the base's closed trades resampled with replacement 2,000 times into 150 trades (the number
    the account makes in about two years) at 8.8% sizing: the spread of the total return and of the worst drawdown;
    also the chance that the first 21 trades (what the live account has done) lose 9% or more;
  * entry delay: every entry one session later;
  * cross-symbol: the base on two random halves of the stocks (odd and even by name hash)."""

import logging
import pickle
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_swing_v2 import CACHE, LiveRun, make_source, metrics  # noqa: E402
from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim, capital = screen_config(settings), settings.risk, settings.paper_initial_cash
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    bear = pd.Series(False, index=close.index)
    start_all = close.index[MOMENTUM_BARS]
    windows = (("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3)))

    def run(window_start, symbols=None, ranked_by_day=None, **kw):
        w0 = close.index[close.index.searchsorted(window_start)]
        warm = close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))]
        sub = {s: df.loc[warm:] for s, df in bars.items() if symbols is None or s in symbols}
        rk = ranked_by_day or ranked
        if symbols is not None:
            rk = {d: [c for c in cs if c.symbol in symbols] for d, cs in rk.items()}
        src, lo, hi = make_source(rk, cfg, bear)
        limits = replace(lim, min_position_pct=lo, max_position_pct=hi)
        res = simulate(sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                       cash_yield=0.0, cash_yield_from=w0, **kw)
        res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
        return res, metrics(LiveRun(res, capital, limits), capital, w0)

    variants = {
        "base": {}, "trail20": dict(trail_pct=0.20), "trail25": dict(trail_pct=0.25), "trail30": dict(trail_pct=0.30),
        "lock10": dict(lock_at=0.10), "lock20": dict(lock_at=0.20),
    }
    out, base_trades = {}, None
    for wname, w0 in windows:
        for name, kw in variants.items():
            res, m = run(w0, **kw)
            out[(wname, name)] = m
            if wname == "full" and name == "base":
                base_trades = res.trades
        print(wname, "done", flush=True)
    head = f"{'variant':9s}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'avg':>8}{'half1':>8}{'half2':>8}"
    for wname, _ in windows:
        print(f"\n== {wname} ==\n{head}")
        for name in variants:
            s = out[(wname, name)]
            print(f"{name:9s}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}{s['trades']:7d}"
                  f"{s['win_rate']:5.0%}{s['expectancy_pct']:+8.2%}{s['h1']:+8.1%}{s['h2']:+8.1%}")

    def beats(k, extra=False):
        ok = True
        for w, _ in windows:
            b, v = out[(w, "base")], out[(w, k)]
            ok &= v["sharpe"] >= b["sharpe"] + (0.0 if extra else 0.05) and v["max_drawdown"] >= b["max_drawdown"]
            ok &= v["total_return"] >= b["total_return"] - 0.10 * abs(b["total_return"])
        v = out[("full", k)]
        return ok and v["h1"] > 0 and v["h2"] > 0

    print("\n== exit gate vs base ==")
    neighbours = {"trail20": ["trail25"], "trail25": ["trail20", "trail30"], "trail30": ["trail25"]}
    for k in variants:
        if k == "base":
            continue
        plateau = all(beats(n, extra=True) for n in neighbours.get(k, []))
        print(f"{k:9s}{'PASS' if beats(k) and plateau else 'FAIL'} (own gate {beats(k)}, neighbours {plateau})")

    print("\n== PART 2: robustness of the live base ==")
    rng = np.random.default_rng(7)
    rets = np.array([t.return_pct for t in base_trades])
    print(f"base trades {len(rets)}: mean {rets.mean():+.2%}, median {np.median(rets):+.2%}, win {np.mean(rets > 0):.0%}")
    size = 0.088
    totals, dds = [], []
    for _ in range(2000):
        r = rng.choice(rets, 150, replace=True) * size
        curve = np.cumprod(1 + r)
        totals.append(curve[-1] - 1)
        dds.append((curve / np.maximum.accumulate(curve) - 1).min())
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    print(f"150 trades at 8.8%: total return 5/50/95th pct {q(totals, 5):+.1%} / {q(totals, 50):+.1%} / {q(totals, 95):+.1%};"
          f" worst drawdown median {q(dds, 50):.1%}, 5th pct {q(dds, 5):.1%}")
    firsts = [np.prod(1 + rng.choice(rets, 21, replace=True) * size) - 1 for _ in range(5000)]
    print(f"first 21 trades: 5/50/95th pct {q(firsts, 5):+.1%} / {q(firsts, 50):+.1%} / {q(firsts, 95):+.1%}; "
          f"P(loss >= 4% of the account from 21 trades, as the live account's closed+open trades) = {np.mean(np.array(firsts) <= -0.04):.0%}")

    delayed = {}
    days = sorted(ranked)
    for i, d in enumerate(days):
        delayed[d] = ranked[days[i - 1]] if i else []
    for wname, w0 in windows:
        _, m = run(w0, ranked_by_day=delayed)
        b = out[(wname, "base")]
        print(f"entry one session later, {wname}: return {m['total_return']:+.1%} / Sharpe {m['sharpe']:.2f} (base {b['total_return']:+.1%} / {b['sharpe']:.2f})")
    names = sorted(bars)
    for part in (0, 1):
        syms = {s for s in names if zlib.crc32(s.encode()) % 2 == part}
        for wname, w0 in windows:
            _, m = run(w0, symbols=syms)
            print(f"half {part} of the stocks, {wname}: return {m['total_return']:+.1%} / Sharpe {m['sharpe']:.2f} / maxDD {m['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
