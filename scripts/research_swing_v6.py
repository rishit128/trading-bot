"""Swing: a minimum position size and smaller positions when volatility is high (pre-declared 2026-11-03, from the account
analysis: fees took 1.59% a trade on Rs 650-2,400 positions, and the 16-23 Oct fall hit positions just raised to 8.8%).

    python scripts/research_swing_v6.py

Variants change one thing on the live configuration (15 slots x 8.8% under the 90% cap, 15% stop, MA200 exit), all NSE stocks,
10 years, the cached candidates of research_swing_v2, Rs 25,000:
  base              live
  ticket1000/1500/2000  no buy worth less than this
  vix18/22/26       half size when India VIX closed above this on the signal day (known before the next open's fill)
Gate, fixed before running: Sharpe at least 0.05 above base in BOTH the full period and the last 3 years, max drawdown no worse in
both, return not lower than base minus 10% of it in both, both halves of the full period positive, and the neighbouring values
also keep Sharpe >= base's in both windows (a plateau). A pass makes the rule a candidate for the next swing version; a fail
leaves the live rules alone."""

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
from src.data.price_history import load_index_close, load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import Result, simulate  # noqa: E402

TICKETS = {"base": 0.0, "ticket1000": 1000.0, "ticket1500": 1500.0, "ticket2000": 2000.0}
VIX = {"vix18": 18.0, "vix22": 22.0, "vix26": 26.0}
NEIGHBOURS = {
    "ticket1000": ["ticket1500"],
    "ticket1500": ["ticket1000", "ticket2000"],
    "ticket2000": ["ticket1500"],
    "vix18": ["vix22"],
    "vix22": ["vix18", "vix26"],
    "vix26": ["vix22"],
}


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim, capital = screen_config(settings), settings.risk, settings.paper_initial_cash
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    vix = load_index_close("^INDIAVIX", 10).reindex(close.index).ffill()
    calm = pd.Series(False, index=close.index)
    start_all = close.index[MOMENTUM_BARS]
    windows = (("full", start_all), ("last3y", close.index[-1] - pd.DateOffset(years=3)))
    out = {}
    for wname, w0 in windows:
        w0 = close.index[close.index.searchsorted(w0)]
        warm = close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))]
        sub = {s: df.loc[warm:] for s, df in bars.items()}
        for name in [*TICKETS, *VIX]:
            high = vix > VIX[name] if name in VIX else calm
            src, lo, hi = make_source(ranked, cfg, high, bear_scale=0.5 if name in VIX else 1.0)
            limits = replace(lim, min_position_pct=lo, max_position_pct=hi)
            res = simulate(
                sub, {}, limits, start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src,
                cash_yield=0.0, cash_yield_from=w0, min_ticket=TICKETS.get(name, 0.0),
            )  # fmt: skip
            res = Result(res.equity.loc[w0:], [t for t in res.trades if t.entry_date >= w0], res.open_at_end)
            out[(wname, name)] = metrics(LiveRun(res, capital, limits), capital, w0)
            print(wname, name, "done", flush=True)
    for wname, _ in windows:
        print(
            f"\n== {wname} ==\n{'variant':12s}{'return':>9}{'Sharpe':>7}{'maxDD':>8}{'trades':>7}{'win':>5}{'half1':>8}{'half2':>8}"
        )
        for name in [*TICKETS, *VIX]:
            s = out[(wname, name)]
            print(
                f"{name:12s}{s['total_return']:+9.1%}{s['sharpe']:7.2f}{s['max_drawdown']:8.1%}{s['trades']:7d}"
                f"{s['win_rate']:5.0%}{s['h1']:+8.1%}{s['h2']:+8.1%}"
            )

    def beats(k, margin):
        ok = True
        for w, _ in windows:
            b, v = out[(w, "base")], out[(w, k)]
            ok &= v["sharpe"] >= b["sharpe"] + margin and v["max_drawdown"] >= b["max_drawdown"]
            ok &= v["total_return"] >= b["total_return"] - 0.10 * abs(b["total_return"])
        v = out[("full", k)]
        return ok and v["h1"] > 0 and v["h2"] > 0

    def plateau(k):
        return all(out[(w, n)]["sharpe"] >= out[(w, "base")]["sharpe"] for w, _ in windows for n in NEIGHBOURS[k])

    print("\n== gate against base ==")
    for k in NEIGHBOURS:
        print(
            f"{k:12s}{'PASS' if beats(k, 0.05) and plateau(k) else 'FAIL'} (own gate {beats(k, 0.05)}, neighbours {plateau(k)})"
        )


if __name__ == "__main__":
    main()
