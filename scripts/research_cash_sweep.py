"""Does sweeping idle cash into a liquid fund make the live configuration better? (2026-09-26)

Background: 10 positions x 5% means the bot is at most ~50% invested (measured: 45% on average over 2017-2026, with all
10 slots full 84% of the time), and in the simulator the rest earned nothing. A liquid mutual fund (direct plan, held
outside demat) has no DP charge or STT on redemption and allows instant redemption up to Rs 50,000, so for a Rs 20,000
account it is close to cash that earns a rate. The trades themselves are unchanged; only idle cash earns.

PRE-DECLARED before any result was seen:
  rate 5.0%  a central assumption for Indian liquid funds over 2017-2026 (they ranged roughly 3-7% a year)
  rate 3.5%  a deliberately low stress rate; ADOPTION IS JUDGED ON THIS ONE, so a pass does not rest on optimism
A rate PASSES only if, on EVERY window, against the live configuration without the sweep:
  1. higher total return
  2. Sharpe at least as high
  3. maximum drawdown no worse
  4. still positive with slippage doubled
Also reported, not gated: the same run against the equal-weight basket on Sharpe and drawdown.

Not modelled: tax on liquid-fund gains, the rate varying over time, and a one-day redemption delay on a day the bot
buys (the simulator treats the swept cash as available at once, which instant redemption covers up to Rs 50,000).

    python scripts/research_cash_sweep.py
    python scripts/research_cash_sweep.py --universe nse   # all NSE stocks instead of today's Nifty 500
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse, load_ohlc_universe  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE  # noqa: E402
from src.research.backtest import curve_metrics  # noqa: E402
from src.research.live_backtest import (DEFAULT_CONFIDENCE, delivery_average_or_none, equal_weight_basket,
                                        ranked_candidates,  # noqa: E402
                                        run_live_config, summarize_run)

RATES = (0.05, 0.035)
JUDGED_RATE = 0.035


def run(window, ranked, settings, cfg, start, confidence, rate, slippage=SLIPPAGE) -> dict:
    return summarize_run(run_live_config(window, settings.risk, cfg, settings.paper_initial_cash, ranked=ranked,
                                         confidence=confidence, start=start, cash_yield=rate, slippage=slippage))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=("nifty500", "nse"), default="nifty500")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--recent-years", type=int, default=3)
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = load_settings()
    cfg = screen_config(settings)
    bars = load_ohlc_nse(args.years) if args.universe == "nse" else load_ohlc_universe(500, args.years)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    print(f"live settings: capital Rs {settings.paper_initial_cash:,.0f}, {settings.risk.max_open_positions} x "
          f"{settings.risk.max_position_pct:.0%}, stop {settings.risk.stop_loss_pct:.0%}; universe {args.universe} "
          f"({len(bars)} stocks)\npass bar (judged at {JUDGED_RATE:.1%}): beats the no-sweep run on return, Sharpe and "
          f"drawdown on every window, and stays positive at 2x slippage\n")

    periods = [("full period", close.index[MOMENTUM_BARS], None),
               (f"last {args.recent_years}y", close.index[-1] - pd.DateOffset(years=args.recent_years), None)]
    avg = delivery_average_or_none(3) if settings.delivery_filter else None
    if avg is not None:
        periods.append((f"last {args.recent_years}y + delivery filter",
                        close.index[-1] - pd.DateOffset(years=args.recent_years), avg))

    verdict = {rate: True for rate in RATES}
    for name, start_ts, deliv in periods:
        start = close.index[close.index.searchsorted(start_ts)]
        dates = [d for d in close.index if d >= start]
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        window = {s: df.loc[close.index[warm]:] for s, df in bars.items()}
        ranked = ranked_candidates(close, volume, dates, cfg, deliv)
        basket = curve_metrics(equal_weight_basket(bars, start, close.index[-1], 100.0))
        base = run(window, ranked, settings, cfg, start, args.confidence, 0.0)
        print(f"=== {name}: {start.date()} .. {close.index[-1].date()} ===")
        print(f"  {'equal-weight basket':28s} return {basket['total_return']:+8.1%}  Sharpe {basket['sharpe']:4.2f}  "
              f"maxDD {basket['max_drawdown']:6.1%}")
        print(f"  {'live, no sweep':28s} return {base['total_return']:+8.1%}  Sharpe {base['sharpe']:4.2f}  "
              f"maxDD {base['max_drawdown']:6.1%}")
        for rate in RATES:
            s = run(window, ranked, settings, cfg, start, args.confidence, rate)
            doubled = run(window, ranked, settings, cfg, start, args.confidence, rate, SLIPPAGE * 2)["total_return"]
            gates = {"higher return": s["total_return"] > base["total_return"],
                     "Sharpe >= no-sweep": s["sharpe"] >= base["sharpe"],
                     "drawdown no worse": s["max_drawdown"] >= base["max_drawdown"],
                     "positive at 2x slippage": doubled > 0}
            verdict[rate] &= all(gates.values())
            vs_basket = (f"vs basket: Sharpe {'above' if s['sharpe'] >= basket['sharpe'] else 'below'}, drawdown "
                         f"{'smaller' if s['max_drawdown'] >= basket['max_drawdown'] else 'larger'}")
            print(f"  {f'live + sweep at {rate:.1%}':28s} return {s['total_return']:+8.1%}  Sharpe {s['sharpe']:4.2f}  "
                  f"maxDD {s['max_drawdown']:6.1%}  | {' '.join(('PASS ' if ok else 'FAIL ') + g for g, ok in gates.items())}"
                  f" | {vs_basket}")
        print()

    for rate in RATES:
        print(f"{rate:.1%}: {'PASSES' if verdict[rate] else 'FAILS'} on every window"
              f"{'  <- judged rate' if rate == JUDGED_RATE else ''}")
    print("ADOPT: the sweep clears the bar at the stress rate." if verdict[JUDGED_RATE] else
          "DO NOT ADOPT: the sweep fails the bar at the stress rate.")


if __name__ == "__main__":
    main()
