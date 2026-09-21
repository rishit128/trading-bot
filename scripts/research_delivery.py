"""Does NSE delivery percentage improve the signals? (Round 3; declared before any result was seen.)

Delivery % history is only ~3 years, so everything here is compared with an equal-weight basket over the SAME window and
is weak evidence at best. A candidate is only interesting if it beats its own unfiltered version, net of costs, in the
full window AND in the last year, and keeps a positive excess over the basket. Delivery filter: a stock's 20-day
average delivery % must be above that day's cross-sectional median (a literature-style default, not tuned)."""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import engine, signals  # noqa: E402
from src.research.data import load_universe  # noqa: E402
from src.research.delivery import load_delivery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--cost", type=float, default=0.0012)
    ap.add_argument("--min-traded-value", type=float, default=1e8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    close, volume, nifty = load_universe(500, 10)
    deliv = load_delivery(args.years)
    print(f"delivery data: {len(deliv)} trading days, {deliv.index[0].date()} .. {deliv.index[-1].date()}")
    avg = deliv.reindex(close.index).rolling(20, min_periods=10).mean().reindex(columns=close.columns)
    high_delivery = avg.gt(avg.median(axis=1), axis=0)
    mtv = args.min_traded_value
    liquid = signals.liquid_mask(close, volume, mtv)

    mom = close.shift(21) / close.shift(252) - 1
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    enter = (close > ma50) & (ma50 > ma200) & (signals.rsi_series(close, 14) < 70) & liquid
    candidates = {
        "BASKET equal-weight liquid": signals.equal_weight_universe(close, volume, mtv),
        "momentum 12-1 (top 20)": signals.xs_momentum(close, volume, mtv),
        "D1 momentum 12-1 + high delivery": signals._rebalanced(close, mom, liquid & mom.notna() & high_delivery, 20, False),
        "trend rule": signals.trend_rule(close, volume, mtv),
        "D2 trend rule + high delivery": signals._capped_equal_weight(
            signals.stateful(enter & high_delivery, close < ma200), 20),
    }
    start, last_year = deliv.index[0], close.index[-1] - pd.DateOffset(years=1)
    res = {n: engine.slice_period(engine.portfolio_returns(close, w, args.cost), start=start) for n, w in candidates.items()}
    basket = res["BASKET equal-weight liquid"]["net"]
    print(f"\nwindow {start.date()} .. {close.index[-1].date()}  ({args.cost:.2%}/side)")
    print(f"{'':36s} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>7} {'excess':>7} | last-1y excess")
    for n, r in res.items():
        f = engine.summarize(r)
        ex = engine.cagr(r["net"]) - engine.cagr(basket)
        ex1 = engine.cagr(engine.slice_period(r, start=last_year)["net"]) - engine.cagr(engine.slice_period(res["BASKET equal-weight liquid"], start=last_year)["net"])
        print(f"{n:36s} {f['cagr']:7.1%} {f['sharpe']:6.2f} {f['max_drawdown']:7.1%} {ex:7.1%} | {ex1:7.1%}")


if __name__ == "__main__":
    main()
