"""A core index sleeve beside the swing strategy (pre-declared 2026-10-26).

    python scripts/research_core_sleeve.py

Why: the residual-edge test (STRATEGY.md, 2026-10-26) found the swing strategy's alpha over an equal-weight basket of all NSE
stocks not significant (t = 0.97); its return is mostly broad-market exposure. A cheaper way to hold that exposure is index
ETFs, with the stock-picking sleeve kept for what it may add. The strategy's daily equity curve (live configuration, all NSE
stocks, 10 years, as in research_swing_v2) is blended with ETF returns, rebalanced to the target mix on the first trading day
of each month (rebalancing costs ignored: about 0.2% on the traded part, a few turns a year).

Cores: broad = 50% NIFTYBEES + 50% JUNIORBEES (Nifty 50 and Next 50; since 2009), mid = 50% NIFTYBEES + 50% MID150BEES
(Nifty Midcap 150; since 2019, so only the last 3 years and the 2019-on window). Core weights 0, 25, 40, 50, 60, 100%.

Gate, fixed before running (a core weight is adopted only if all hold against 0%): Sharpe at least 0.05 higher AND max
drawdown no worse AND return not lower than 0%'s minus 25% of it, in BOTH windows (full 10 years with the broad core, and the
last 3 years with the broad core), and the neighbouring weights also have Sharpe >= 0%'s in both windows (a plateau)."""

import logging
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.research_swing_v2 import CACHE, make_source  # noqa: E402
from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import simulate  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.40, 0.50, 0.60, 1.0)


def etf_returns(symbol: str) -> pd.Series:
    import yfinance as yf

    h = yf.Ticker(symbol + ".NS").history(period="max", auto_adjust=True)["Close"]
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    r = h.pct_change().dropna()
    return r[r.abs() < 0.30]  # Yahoo carries a bad split adjustment in NIFTYBEES in Dec 2019 (-90% then +897%)


def blend(strategy: pd.Series, core: pd.Series, weight: float) -> pd.Series:
    """Daily returns of `weight` in the core and the rest in the strategy, rebalanced on each month's first trading day."""
    idx = strategy.index.intersection(core.index)
    s, c = strategy.loc[idx], core.loc[idx]
    out, wc = [], weight
    month = None
    for d in idx:
        if month != (d.year, d.month):
            wc, month = weight, (d.year, d.month)
        r = wc * c.loc[d] + (1 - wc) * s.loc[d]
        out.append(r)
        total = wc * (1 + c.loc[d]) + (1 - wc) * (1 + s.loc[d])
        wc = wc * (1 + c.loc[d]) / total
    return pd.Series(out, index=idx)


def stats(r: pd.Series) -> dict:
    curve = (1 + r).cumprod()
    return {
        "ret": float(curve.iloc[-1] - 1),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
        "dd": float((curve / curve.cummax() - 1).min()),
    }


def main():
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg, lim, capital = screen_config(settings), settings.risk, settings.paper_initial_cash
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    ranked = pickle.loads((CACHE / "ranked_nse_10y.pkl").read_bytes())  # nosec B301
    w0 = close.index[MOMENTUM_BARS]
    warm = close.index[close.index.searchsorted(w0 - pd.Timedelta(days=330))]
    src, lo, hi = make_source(ranked, cfg, pd.Series(False, index=close.index))
    res = simulate({s: df.loc[warm:] for s, df in bars.items()}, {}, replace(lim, min_position_pct=lo, max_position_pct=hi),
                   start_equity=capital, fees=india_delivery_fees, slippage=SLIPPAGE, buy_source=src)
    strat = res.equity.loc[w0:].pct_change().dropna()
    nifty, junior, mid = etf_returns("NIFTYBEES"), etf_returns("JUNIORBEES"), etf_returns("MID150BEES")
    cores = {"broad (Nifty50+Next50)": ((nifty + junior) / 2).dropna(),
             "mid (Nifty50+Midcap150)": ((nifty + mid) / 2).dropna()}
    last3 = strat.index[-1] - pd.DateOffset(years=3)
    table = {}
    for cname, core in cores.items():
        for wname, start in (("full", strat.index[0]), ("last3y", last3)):
            sub = strat.loc[start:]
            if cname.startswith("mid") and wname == "full":
                sub = strat.loc[core.index[0]:]  # the ETF exists from 2019
                wname = "since2019"
            for w in WEIGHTS:
                table[(cname, wname, w)] = stats(blend(sub, core, w))
    for cname in cores:
        for wname in ("full", "since2019", "last3y"):
            if (cname, wname, 0.0) not in table:
                continue
            print(f"\n== core {cname}, {wname} ==\n{'core wt':>8}{'return':>9}{'Sharpe':>7}{'maxDD':>8}")
            for w in WEIGHTS:
                s = table[(cname, wname, w)]
                print(f"{w:8.0%}{s['ret']:+9.1%}{s['sharpe']:7.2f}{s['dd']:8.1%}")
    print("\n== gate (broad core; full and last3y) ==")
    def ok(w, margin):
        for wn in ("full", "last3y"):
            b, v = table[("broad (Nifty50+Next50)", wn, 0.0)], table[("broad (Nifty50+Next50)", wn, w)]
            if not (v["sharpe"] >= b["sharpe"] + margin and v["dd"] >= b["dd"] and v["ret"] >= b["ret"] - 0.25 * abs(b["ret"])):
                return False
        return True
    for i, w in enumerate(WEIGHTS):
        if w in (0.0, 1.0):
            continue
        near = [WEIGHTS[j] for j in (i - 1, i + 1) if 0 < j < len(WEIGHTS) - 1]
        plateau = all(ok(n, 0.0) for n in near)
        print(f"{w:5.0%}  {'PASS' if ok(w, 0.05) and plateau else 'FAIL'} (own {ok(w, 0.05)}, neighbours {near}: {plateau})")


if __name__ == "__main__":
    main()
