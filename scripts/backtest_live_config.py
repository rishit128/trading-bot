"""Backtest the LIVE configuration (account size, position limits, sizing, exits, costs and scanner) on Indian history.

    python scripts/backtest_live_config.py                 # Nifty 500, 10 years, settings from .env
    python scripts/backtest_live_config.py --years 5 --index 100

It runs three setups on the SAME scanner rankings so the effect of each difference is visible:
  1. live            the paper account as configured now (e.g. Rs 20,000, 5 positions, 10-25% sizing)
  2. live limits, big account   the same limits on Rs 10 lakh: isolates the effect of a small account (whole shares,
                     fixed per-sale charge, unaffordable stocks)
  3. research setup  Rs 10 lakh, 10 positions of a flat 5% (what the earlier research tested)
and reports each over the full period and the last 3 years, next to the Nifty 50 and an equal-weight basket of the same
stocks. The AI is replaced by a constant approval confidence; today's index members only, so survivorship bias inflates
every stock-picking row (compare rows to the basket, not to zero)."""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.kit import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.universe import MOMENTUM_BARS, delivery_average  # noqa: E402
from src.research.data import load_index_close, load_ohlc_universe  # noqa: E402
from src.research.live_backtest import (DEFAULT_CONFIDENCE, equal_weight_basket, ranked_candidates,  # noqa: E402
                                        research_setup, run_live_config, summarize_run)

BIG = 1_000_000.0
HEAD = f"{'':46s}{'return':>9}{'CAGR':>8}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'avg/trade':>10}{'PF':>6}{'fees':>7}"


def row(label, s):
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "  n/a"
    return (f"{label:46s}{s['total_return']:+9.1%}{s['cagr']:+8.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}"
            f"{s['trades']:7d}{s['win_rate']:5.0%}{s['expectancy_pct']:+10.2%}{pf:>6}{s['fees_pct_of_capital']:7.1%}")


def curve_row(label, curve):
    r = curve.pct_change().dropna()
    years = len(curve) / 252
    total = float(curve.iloc[-1] / curve.iloc[0] - 1)
    sharpe = float(r.mean() / r.std() * 252 ** 0.5) if r.std() > 0 else 0.0
    dd = float((curve / curve.cummax() - 1).min())
    return f"{label:46s}{total:+9.1%}{(1 + total) ** (1 / years) - 1:+8.1%}{sharpe:7.2f}{dd:8.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, choices=(50, 100, 500), default=500)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="stand-in for the AI's approval confidence")
    ap.add_argument("--recent-years", type=int, default=3, help="also report this many most recent years (delivery data covers ~3)")
    ap.add_argument("--refresh", action="store_true", help="re-download the price cache")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = load_settings()
    live_cfg = screen_config(settings)
    big_cfg = dataclasses.replace(live_cfg, affordable_pct=settings.risk.min_position_pct)
    res_limits = research_setup(settings.risk)
    res_cfg = dataclasses.replace(live_cfg, affordable_pct=res_limits.min_position_pct)
    print(f"live settings: capital Rs {settings.paper_initial_cash:,.0f}, {settings.risk.max_open_positions} positions, "
          f"{settings.risk.min_position_pct:.0%}-{settings.risk.max_position_pct:.0%} sizing, stop {settings.risk.stop_loss_pct:.0%}, "
          f"trend exit, candidates {live_cfg.max_candidates}, AI stand-in confidence {args.confidence}")

    bars = load_ohlc_universe(args.index, args.years, refresh=args.refresh)
    nifty = load_index_close("^NSEI", args.years, refresh=args.refresh)
    print(f"Nifty {args.index}: {len(bars)} stocks with data, {nifty.index[0].date()} .. {nifty.index[-1].date()}")
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()

    periods = [("full period", close.index[MOMENTUM_BARS]),
               (f"last {args.recent_years}y", close.index[-1] - pd.DateOffset(years=args.recent_years))]
    deliv = None
    if settings.delivery_filter:
        from src.research.delivery import load_delivery
        deliv = delivery_average(load_delivery(years=3))

    for name, start in periods:
        start = close.index[close.index.searchsorted(start)]
        dates = [d for d in close.index if d >= start]
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        window = {s: df.loc[close.index[warm]:] for s, df in bars.items()}
        print(f"\n=== {name}: {start.date()} .. {close.index[-1].date()} ({len(dates)} trading days) ===")
        print("scanning every day with the live filter ...", flush=True)
        variants = [("no delivery filter", None)] + ([("with delivery filter (live default)", deliv)] if deliv is not None and name != "full period" else [])
        for tag, avg in variants:
            ranked = ranked_candidates(close, volume, dates, live_cfg, avg)
            print(f"\n{tag}\n{HEAD}")
            for label, limits, cfg, capital in (
                    (f"live: Rs {settings.paper_initial_cash:,.0f}, {settings.risk.max_open_positions} pos", settings.risk, live_cfg, settings.paper_initial_cash),
                    ("live limits on Rs 10 lakh", settings.risk, big_cfg, BIG),
                    ("research setup: Rs 10 lakh, 10 x 5%", res_limits, res_cfg, BIG)):
                run = run_live_config(window, limits, cfg, capital, ranked=ranked, confidence=args.confidence, start=start)
                s = summarize_run(run)
                print(row(label, s) + f"   exits {s['exits']}, open {s['open_at_end']}")
            print(curve_row("equal-weight basket, same stocks", equal_weight_basket(bars, start, close.index[-1], 100.0)))
            print(curve_row("Nifty 50 buy & hold", nifty.loc[start:]))


if __name__ == "__main__":
    main()
