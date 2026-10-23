"""Swing improvements, tested on the live configuration over all NSE stocks (pre-declared 2026-10-23).

    python scripts/research_swing_v2.py [--years 10]

Why: the paper account lost ~9% in four weeks. The 21 live positions showed (a) stocks with 3%+ daily volatility lost
12.7% on average against 2.1% for the calmer ones, (b) 13 of 21 buys went in on two days, in a falling market, (c) positions
of Rs 760-1,900 pay a fixed Rs 16 charge per sale. Each variant changes ONE thing from the live configuration and is
judged against it on the same ranked candidates (the scan runs once):

  base        live: 15 positions x 8.8%, 15% stop, MA200 trend exit, 4% volatility cap
  volcap3     screener volatility cap 3% (was 4%)
  volcap25    screener volatility cap 2.5%
  invvol      size inversely to the stock's volatility (8.8% at 2.5% a day, 4%-12% range), same average risk
  pace2       at most 2 new buys a day
  pace1       at most 1 new buy a day
  bear_half   half-size buys while Nifty closes below its 200-day average
  bear_pace1  at most 1 new buy a day while Nifty is below its 200-day average
  pos8        8 positions x 11%
  combo       volcap3 + pace2 + bear_half

Gate, fixed before running (a variant is adopted only if ALL hold against `base`): Sharpe at least 0.05 higher over the
full period AND over the last 3 years; max drawdown no worse in both; return not lower than base minus 10% of base's
return in both; positive return in both halves of the full period. Costs: the paper broker's delivery fees and slippage.
Survivorship bias (today's listings only) inflates every row, base included."""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_index_close, load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS, select_candidates  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import simulate  # noqa: E402
from src.research.live_backtest import DEFAULT_CONFIDENCE, ranked_candidates, summarize_run, LiveRun  # noqa: E402

CACHE = Path("research_cache")
MIN_CONF = 0.60
REF_VOL = 0.025  # the daily volatility at which `invvol` gives the live 8.8%


def conf_for(size: float, lo: float, hi: float) -> float:
    """The confidence that makes the risk engine size a position at `size` when it scales `lo`..`hi` over 0.6..1.0."""
    if hi <= lo:
        return DEFAULT_CONFIDENCE
    return MIN_CONF + (1 - MIN_CONF) * min(1.0, max(0.0, (size - lo) / (hi - lo)))


def make_source(ranked, cfg, bear, vol_cap=None, pace=None, bear_pace=None, bear_scale=1.0, inv_vol=False, base_pct=0.088):
    lo, hi = (0.04, 0.12) if (inv_vol or bear_scale != 1.0) else (base_pct, base_pct)

    def source(day, pf):
        cands = ranked.get(day, [])
        if vol_cap is not None:
            cands = [c for c in cands if c.daily_volatility <= vol_cap]
        picks = select_candidates(cands, cfg.max_candidates, pf.equity, cfg.affordable_pct)
        in_bear = bool(bear.get(day, False))
        limit = pace
        if in_bear and bear_pace is not None:
            limit = bear_pace if limit is None else min(limit, bear_pace)
        out = []
        for c in picks:
            if c.symbol in pf.positions:
                continue
            size = base_pct
            if inv_vol:
                size = min(hi, max(lo, base_pct * REF_VOL / max(c.daily_volatility, 1e-4)))
            if in_bear:
                size *= bear_scale
            out.append((c.symbol, conf_for(size, lo, hi)))
            if limit is not None and len(out) >= limit:
                break
        return out

    return source, lo, hi


def metrics(run, capital, start):
    r = run.result
    s = summarize_run(run)
    curve = r.equity
    half = curve.index[len(curve) // 2]
    h1, h2 = curve.loc[:half], curve.loc[half:]
    s["h1"], s["h2"] = float(h1.iloc[-1] / h1.iloc[0] - 1), float(h2.iloc[-1] / h2.iloc[0] - 1)
    yearly = curve.groupby(curve.index.year).apply(lambda x: float(x.iloc[-1] / x.iloc[0] - 1))
    s["yearly"] = yearly
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg = screen_config(settings)
    capital = settings.paper_initial_cash
    bars = load_ohlc_nse(args.years)
    nifty = load_index_close("^NSEI", args.years)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    print(f"{len(bars)} stocks, {close.index[0].date()} .. {close.index[-1].date()}; capital {capital:,.0f}, "
          f"{settings.risk.max_open_positions} positions x {settings.risk.min_position_pct:.1%}", flush=True)
    start_all = close.index[MOMENTUM_BARS]
    dates = [d for d in close.index if d >= start_all]
    cache = CACHE / f"ranked_nse_{args.years}y.pkl"
    if cache.exists():
        ranked = pickle.loads(cache.read_bytes())  # nosec B301
    else:
        print("scanning every day with the live filter ...", flush=True)
        ranked = ranked_candidates(close, volume, dates, cfg, None)
        cache.write_bytes(pickle.dumps(ranked))
    ma200 = nifty.rolling(200).mean()
    bear = (nifty < ma200).reindex(close.index, method="ffill").fillna(False)

    lim = settings.risk
    from dataclasses import replace

    variants = {
        "base": dict(),
        "volcap3": dict(vol_cap=0.03),
        "volcap25": dict(vol_cap=0.025),
        "invvol": dict(inv_vol=True),
        "pace2": dict(pace=2),
        "pace1": dict(pace=1),
        "bear_half": dict(bear_scale=0.5),
        "bear_pace1": dict(bear_pace=1),
        "pos8": dict(_limits=replace(lim, max_open_positions=8, min_position_pct=0.11, max_position_pct=0.11), base_pct=0.11),
        "combo": dict(vol_cap=0.03, pace=2, bear_scale=0.5),
    }
    windows = [("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3))]
    results = {}
    for wname, w0 in windows:
        w0 = close.index[close.index.searchsorted(w0)]
        warm = close.index.searchsorted(w0 - pd.Timedelta(days=330))
        sub = {s: df.loc[close.index[warm]:] for s, df in bars.items()}
        for name, kw in variants.items():
            kw = dict(kw)
            limits = kw.pop("_limits", lim)
            src, lo, hi = make_source(ranked, cfg, bear, **kw)
            limits = replace(limits, min_position_pct=lo, max_position_pct=hi)
            res = simulate(sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                           cash_yield=0.0, cash_yield_from=w0)
            from src.research.backtest import Result

            res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
            results[(wname, name)] = metrics(LiveRun(res, capital, limits), capital, w0)
        print(f"{wname} done", flush=True)

    head = f"{'variant':12s}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'avg':>8}{'fees%':>7}{'half1':>8}{'half2':>8}"
    for wname, _ in windows:
        print(f"\n== {wname} ==\n{head}")
        for name in variants:
            s = results[(wname, name)]
            print(f"{name:12s}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}{s['trades']:7d}"
                  f"{s['win_rate']:5.0%}{s['expectancy_pct']:+8.2%}{s['fees_pct_of_capital']:7.1%}{s['h1']:+8.1%}{s['h2']:+8.1%}")
    print("\n== gate (vs base) ==")
    for name in variants:
        if name == "base":
            continue
        ok = True
        why = []
        for wname, _ in windows:
            b, v = results[(wname, "base")], results[(wname, name)]
            if v["sharpe"] < b["sharpe"] + 0.05:
                ok = False
                why.append(f"{wname}: Sharpe {v['sharpe']:.2f} vs {b['sharpe']:.2f}")
            if v["max_drawdown"] < b["max_drawdown"]:
                ok = False
                why.append(f"{wname}: drawdown {v['max_drawdown']:.1%} vs {b['max_drawdown']:.1%}")
            if v["total_return"] < b["total_return"] - 0.10 * abs(b["total_return"]):
                ok = False
                why.append(f"{wname}: return {v['total_return']:+.0%} vs {b['total_return']:+.0%}")
        v = results[("full", name)]
        if v["h1"] <= 0 or v["h2"] <= 0:
            ok = False
            why.append("a half of the full period is negative")
        print(f"{name:12s}{'PASS' if ok else 'FAIL'}  " + "; ".join(why))
    print("\nyearly returns, base vs combo (full run):")
    print(pd.DataFrame({"base": results[("full", "base")]["yearly"], "combo": results[("full", "combo")]["yearly"]}).round(3).to_string())
    _ = np


if __name__ == "__main__":
    main()
