"""Is the Rs 1,000 minimum ticket that passed research_swing_v6 real, or one lucky path? (pre-declared 2026-11-03, before running.)

    python scripts/research_swing_v6b.py

A small change to which trades are taken can move a 10-year compounded result a long way through a few large winners, so a pass
is checked three ways on the same data:
  sweep    Rs 700, 800, 900, 1,100, 1,200, 1,300 at Rs 25,000
  capital  base and Rs 1,000 at Rs 20,000, 35,000 and 50,000 (the larger the account, the fewer buys the ticket touches)
  skipped  what the buys it refuses actually did, net of their fees
Gate: Sharpe above base in both windows at every sweep value (no cliff around 1,000), not lower than base at any capital in both
windows, and the skipped trades' mean net return below the taken trades'. A fail leaves the live rules alone."""

import logging
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_swing_v2 import CACHE, LiveRun, make_source, metrics  # noqa: E402
from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402

SWEEP = (700.0, 800.0, 900.0, 1100.0, 1200.0, 1300.0)
CAPITALS = (20_000.0, 35_000.0, 50_000.0)
LIVE_CAPITAL = 25_000.0


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim = screen_config(settings), settings.risk
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    calm = pd.Series(False, index=close.index)
    start_all = close.index[MOMENTUM_BARS]
    windows = (("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3)))
    runs = [(LIVE_CAPITAL, 0.0), (LIVE_CAPITAL, 1000.0)]
    runs += [(LIVE_CAPITAL, ticket) for ticket in SWEEP]
    runs += [(capital, ticket) for capital in CAPITALS for ticket in (0.0, 1000.0)]
    out, trades = {}, {}
    for wname, w0 in windows:
        w0 = close.index[close.index.searchsorted(w0)]
        warm = close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))]
        sub = {s: df.loc[warm:] for s, df in bars.items()}
        for capital, ticket in runs:
            src, lo, hi = make_source(ranked, cfg, calm)
            limits = replace(lim, min_position_pct=lo, max_position_pct=hi)
            res = simulate(
                sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                cash_yield=0.0, cash_yield_from=w0, min_ticket=ticket,
            )  # fmt: skip
            res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
            out[(wname, capital, ticket)] = metrics(LiveRun(res, capital, limits), capital, w0)
            trades[(wname, capital, ticket)] = res.trades
            print(wname, capital, ticket, "done", flush=True)

    print(f"\n{'window':8s}{'capital':>8}{'ticket':>7}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}")
    for (w, capital, ticket), s in out.items():
        print(
            f"{w:8s}{capital:8,.0f}{ticket:7,.0f}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}"
            f"{s['trades']:7d}{s['win_rate']:5.0%}"
        )

    print("\nsmall buys in the base run (what the ticket refuses), Rs 25,000, mean net a trade:")
    for w, _ in windows:
        base = trades[(w, LIVE_CAPITAL, 0.0)]
        small = [t for t in base if t.qty * t.entry_price < 1000.0]
        rest = [t for t in base if t.qty * t.entry_price >= 1000.0]
        mean = lambda ts: sum(t.return_pct for t in ts) / len(ts) if ts else float("nan")  # noqa: E731
        print(f"  {w:8s} under Rs 1,000: n={len(small)} mean {mean(small):+.1%} | the rest: n={len(rest)} mean {mean(rest):+.1%}")

    sweep_ok = all(
        out[(w, LIVE_CAPITAL, t)]["sharpe"] > out[(w, LIVE_CAPITAL, 0.0)]["sharpe"] for w, _ in windows for t in (*SWEEP, 1000.0)
    )
    capital_ok = all(out[(w, c, 1000.0)]["sharpe"] >= out[(w, c, 0.0)]["sharpe"] for w, _ in windows for c in CAPITALS)
    small_ok = True
    for w, _ in windows:
        base = trades[(w, LIVE_CAPITAL, 0.0)]
        small = [t.return_pct for t in base if t.qty * t.entry_price < 1000.0]
        rest = [t.return_pct for t in base if t.qty * t.entry_price >= 1000.0]
        small_ok &= bool(small) and sum(small) / len(small) < sum(rest) / len(rest)
    print(
        f"\ngate: sweep {sweep_ok}, capital {capital_ok}, skipped trades worse {small_ok} -> {'PASS' if sweep_ok and capital_ok and small_ok else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
