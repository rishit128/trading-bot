"""Multi-year, multi-regime research on candidate signals for the Indian market, with Indian trading costs.

PRE-DECLARED before any result was seen (so the goalposts cannot move). A candidate passes only if ALL hold:
  1. net CAGR over the full period            > 10%
  2. net Sharpe over the full period          > 0.5
  3. held-out final N years: Sharpe > 0.5 AND CAGR > 0   (parameters are literature defaults, never tuned on the data)
  4. full-period Sharpe                       > Nifty 50 buy-and-hold Sharpe
  5. profitable in at least 60% of calendar years
  6. max drawdown                             better than -30%
     (the original plan said -20%; declared relaxed up front because a long-only Indian equity strategy cannot be
      expected to stay under -20% through the 2020 crash)
  7. still profitable (CAGR > 0) when costs are doubled

Honest limits: today's Nifty 500 members only (survivorship bias flatters momentum/trend results, so a pass means "not
rejected", not "proven"); six candidates were tried, so one lucky pass is plausible; daily bars, trades at the close
one day after the signal; cash earns 0%; costs are approximate."""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import engine, signals  # noqa: E402
from src.data.price_history import load_index_close, load_universe  # noqa: E402

CRITERIA = ("CAGR>10%", "Sharpe>0.5", "held-out ok", "beats Nifty Sharpe", ">=60% yrs +", "maxDD>-30%", "2x costs +")


def evaluate(name, weights, close, args, nifty_sharpe):
    base = engine.portfolio_returns(close, weights, args.cost)
    dbl = engine.portfolio_returns(close, weights, args.cost * 2)
    full = engine.summarize(base)
    cut = base.index[-1] - pd.DateOffset(years=args.holdout_years)
    held = engine.summarize(engine.slice_period(base, start=cut))
    checks = (
        full["cagr"] > 0.10, full["sharpe"] > 0.5, held["sharpe"] > 0.5 and held["cagr"] > 0,
        full["sharpe"] > nifty_sharpe, full["pct_years_positive"] >= 0.6, full["max_drawdown"] > -0.30,
        engine.cagr(dbl["net"]) > 0,
    )
    return {"name": name, "full": full, "held": held, "cagr_2x": engine.cagr(dbl["net"]), "checks": checks,
            "yearly": engine.yearly_returns(base["net"]), "net": base["net"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, choices=(50, 100, 500), default=500)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.0012, help="cost per side as a fraction of traded value")
    ap.add_argument("--min-traded-value", type=float, default=1e8, help="Rs, 60-day average daily value traded")
    ap.add_argument("--holdout-years", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="re-download data instead of using research_cache/")
    ap.add_argument("--membership-json", type=str, default=None,
                    help="JSON {date: [symbols]} of daily index membership; makes the universe survivorship-free")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    membership = None
    if args.membership_json:
        import json

        membership = json.loads(Path(args.membership_json).read_text())
        print(f"point-in-time universe applied from {args.membership_json}: "
              f"{len(membership)} dated membership snapshots")
    close, volume, nifty = load_universe(args.index, args.years, refresh=args.refresh, membership=membership)
    print(f"Nifty {args.index}: {close.shape[1]} stocks, {close.index[0].date()} .. {close.index[-1].date()} "
          f"({len(close) / engine.TRADING_DAYS:.1f} years), costs {args.cost:.2%}/side, hold-out last {args.holdout_years}y\n")

    nifty_bh = engine.benchmark_returns(nifty)
    nifty_stats = engine.summarize(nifty_bh)
    mtv = args.min_traded_value
    candidates = {
        "momentum 12-1 (top 20, monthly)": signals.xs_momentum(close, volume, mtv),
        "low volatility (top 20, monthly)": signals.low_volatility(close, volume, mtv),
        "donchian 55/20 breakout (max 20)": signals.donchian_breakout(close, volume, mtv),
        "RSI(2) dip in uptrend (max 20)": signals.rsi2_mean_reversion(close, volume, mtv),
        "current trend rule, no stops (max 20)": signals.trend_rule(close, volume, mtv),
    }
    # Round 2 (declared before running): does an index-relative filter or a market-regime filter improve the signal?
    nifty_aligned = nifty.reindex(close.index).ffill()
    candidates.update({
        "R2 momentum 6-1 (top 20, monthly)": signals.xs_momentum_6m(close, volume, mtv),
        "R2 trend rule + beats Nifty 6m": signals.trend_rule_beating_index(close, volume, mtv, nifty_aligned),
        "R2 trend rule + Nifty>200d regime": signals.regime_filter(signals.trend_rule(close, volume, mtv), nifty_aligned),
        "R2 momentum 12-1 + Nifty>200d regime": signals.regime_filter(signals.xs_momentum(close, volume, mtv), nifty_aligned),
    })
    results = [evaluate(n, w, close, args, nifty_stats["sharpe"]) for n, w in candidates.items()]
    timing = signals.index_trend_timing(nifty).reindex(close.index).fillna(0.0)
    results.append(evaluate("Nifty 50 above 200-day (else cash)", timing, nifty.to_frame("INDEX").reindex(close.index), args,
                            nifty_stats["sharpe"]))

    n500 = load_index_close("^CRSLDX", args.years, refresh=args.refresh).reindex(close.index).ffill()
    benches = {
        "BENCH Nifty 50 buy&hold": nifty_bh,
        "BENCH Nifty 500 index (real)": engine.benchmark_returns(n500),
        "BENCH equal-weight liquid stocks": engine.portfolio_returns(close, signals.equal_weight_universe(close, volume, mtv), args.cost),
    }
    cut = close.index[-1] - pd.DateOffset(years=args.holdout_years)
    print(f"{'':40s} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>7} {'turn/y':>6} {'expo':>5} {'yrs+':>5} | held-out {args.holdout_years}y CAGR  Sharpe | 2x-cost CAGR")
    for name, res in benches.items():
        f, h = engine.summarize(res), engine.summarize(engine.slice_period(res, start=cut))
        print(f"{name:40s} {f['cagr']:7.1%} {f['sharpe']:6.2f} {f['max_drawdown']:7.1%} {f['turnover_per_year']:6.1f} "
              f"{f['avg_exposure']:5.2f} {f['pct_years_positive']:5.0%} | {h['cagr']:18.1%} {h['sharpe']:7.2f} |")
    print()
    for r in results:
        f, h = r["full"], r["held"]
        print(f"{r['name']:40s} {f['cagr']:7.1%} {f['sharpe']:6.2f} {f['max_drawdown']:7.1%} {f['turnover_per_year']:6.1f} "
              f"{f['avg_exposure']:5.2f} {f['pct_years_positive']:5.0%} | {h['cagr']:18.1%} {h['sharpe']:7.2f} | {r['cagr_2x']:11.1%}")

    print("\nCalendar-year net returns (regimes):")
    table = pd.DataFrame({r["name"][:34]: r["yearly"] for r in results})
    table["Nifty 50"] = engine.yearly_returns(nifty_bh["net"])
    print((table * 100).round(1).to_string())

    ew = benches["BENCH equal-weight liquid stocks"]
    real = engine.benchmark_returns(n500)
    bias = engine.cagr(ew["net"]) - engine.cagr(real["net"])
    print("\nPOST-HOC (added after seeing results; NOT part of the pre-declared rules):")
    print(f"  Survivorship check: an equal-weight basket of today's Nifty 500 members earned {engine.cagr(ew['net']):.1%}/yr vs "
          f"{engine.cagr(real['net']):.1%}/yr for the real Nifty 500 index -> roughly {bias:.1%}/yr of every stock-picking "
          f"number above is survivorship bias, not skill.")
    cut = close.index[-1] - pd.DateOffset(years=args.holdout_years)
    ew_held = engine.slice_period(ew, start=cut)
    print(f"  {'vs the equal-weight basket (same bias):':40s} {'excess CAGR':>11} {'info ratio':>10} | held-out excess CAGR | Sharpe minus basket (full / held-out)")
    for r in results[:-1]:
        excess = r["net"] - ew["net"]
        ir = float(excess.mean() / excess.std() * (engine.TRADING_DAYS ** 0.5)) if excess.std() > 1e-12 else 0.0
        held_excess = engine.cagr(engine.slice_period(r["net"].to_frame("net"), start=cut)["net"]) - engine.cagr(ew_held["net"])
        print(f"  {r['name']:40s} {r['full']['cagr'] - engine.cagr(ew['net']):11.1%} {ir:10.2f} | {held_excess:20.1%} | "
              f"{r['full']['sharpe'] - engine.sharpe(ew['net']):+.2f} / {r['held']['sharpe'] - engine.sharpe(ew_held['net']):+.2f}")

    print("\nPre-declared checklist (all 7 must pass):")
    for r in results:
        failed = [c for c, ok in zip(CRITERIA, r["checks"]) if not ok]
        print(f"  {'PASS' if not failed else 'FAIL'}  {r['name']}" + (f"   failed: {', '.join(failed)}" if failed else ""))
    pd.DataFrame({r["name"]: r["net"] for r in results}).to_csv("research_results.csv")


if __name__ == "__main__":
    main()
