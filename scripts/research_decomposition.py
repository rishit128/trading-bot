"""Experiment 1 (2026-09-26): which live rule destroys the momentum edge?

The 10-year research found one edge: holding the top stocks by 12-1 month momentum beat the equal-weight basket by
~11%/yr. The live setup ranks by that same momentum but adds filters, a different holding rule, a stop, a target, halts
and small-account sizing, and it trails the basket badly. This walks from pure momentum to the live setup ONE rule at a
time, through the same event simulator, fees and slippage, on every NSE stock (what the live scanner scans).

PRE-DECLARED before any result was seen:
  Chain (each step adds one rule to the previous; Rs 10 lakh, 10 positions x 10%, so the signal is not hidden by
  small-account fees or cash; halts off until their own step):
    S0 pure momentum: hold the top 10 by 12-1 momentum among liquid stocks (price >= Rs 100, >= Rs 10 crore/day,
       a year of history), rebalanced at each month end
    S1 + uptrend filter (price > MA50 > MA200)        S2 + RSI(14) < 70        S3 + daily volatility <= 4%
    S4 live holding rule instead of monthly rebalancing: buy daily from the top 15 into free slots, sell only on the
       MA200 trend exit                              S5 + 15% stop            S6 + 100% target
    S7 + daily-loss and drawdown halts (the live setup's rules, at full size)
    S8 live account: Rs 20,000, 10 x 5%, 80% exposure cap, affordability rule (the running configuration)
  Leave-one-out from S7: drop the uptrend filter / the RSI cap / the volatility cap / the stop, and S7 with
  monthly top-10 rebalancing instead of the trend exit.
  Judged against an equal-weight basket of the SAME liquid universe (price and liquidity floors, known the day before),
  over the full period and the last 3 years.
  A live rule is HARMFUL only if dropping it raises BOTH excess CAGR over that basket AND Sharpe in BOTH windows.

CONFIRMATION (--confirm), declared after the chain above and before running it: at the live account (Rs 20,000,
10 x 5%, live halts) compare dropping the RSI cap, dropping the uptrend filter, and dropping both, against the current
rules. A change is CONFIRMED only if, on BOTH universes (all NSE and today's Nifty 500), it has higher CAGR AND higher
Sharpe in both windows AND in both halves of the full period, still has higher CAGR at doubled slippage, and its max
drawdown is no more than 5 points worse in any window.

Caveats: today's listed stocks only (delisted ones are missing, which flatters every row and the basket alike); RSI
here is Wilder's recursive form over the full history (the live scanner seeds it on a 400-day window: tiny differences).

    python scripts/research_decomposition.py            # the chain and leave-one-out
    python scripts/research_decomposition.py --confirm  # the confirmation at the live account, both universes
"""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RiskLimits  # noqa: E402
from src.data.price_history import load_index_close, load_ohlc_nse, load_ohlc_universe  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.backtest import ExitPolicy, curve_metrics, simulate  # noqa: E402

MIN_PRICE, MIN_TRADED_VALUE, TOP_N, CANDIDATES, CONFIDENCE = 100.0, 1e8, 10, 15, 0.75
BIG, LIVE_CAPITAL = 1_000_000.0, 20_000.0

FULL_SIZE = RiskLimits(min_position_pct=0.10, max_position_pct=0.10, max_portfolio_exposure_pct=1.0,
                       max_open_positions=10, stop_loss_pct=0.99, take_profit_pct=10.0,
                       max_daily_loss_pct=1.0, max_drawdown_pct=1.0, drawdown_pause_days=0.0)
LIVE_EXIT_RULES = dict(stop_loss_pct=0.15, take_profit_pct=1.00)
LIVE_HALTS = dict(max_daily_loss_pct=0.02, max_drawdown_pct=0.20, drawdown_pause_days=30.0)
LIVE_ACCOUNT = RiskLimits(**{**dataclasses.asdict(FULL_SIZE), **LIVE_EXIT_RULES, **LIVE_HALTS,
                             "min_position_pct": 0.05, "max_position_pct": 0.05, "max_portfolio_exposure_pct": 0.80})


class Tables:
    """Every factor the scanner uses, for every stock and day, computed once (values known at that day's close)."""

    def __init__(self, bars: Dict[str, pd.DataFrame]):
        self.close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
        volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).reindex_like(self.close)
        self.mom = self.close.shift(21) / self.close.shift(252) - 1
        ma50, ma200 = self.close.rolling(50).mean(), self.close.rolling(200).mean()
        delta = self.close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        vol = self.close.pct_change(fill_method=None).rolling(63).std()
        traded = (self.close * volume).rolling(20).mean()
        history = self.close.notna().cumsum() >= max(MOMENTUM_BARS, 200)
        self.base = (self.close >= MIN_PRICE) & (traded >= MIN_TRADED_VALUE) & history & self.mom.notna()
        self.rules = {"trend": (self.close > ma50) & (ma50 > ma200), "rsi": rsi < 70, "vol": vol <= 0.04}
        self.symbols = np.array(self.close.columns)
        self._ranked: Dict[tuple, Dict[pd.Timestamp, List[str]]] = {}

    def ranked(self, filters: tuple) -> Dict[pd.Timestamp, List[str]]:
        """day -> eligible symbols, best 12-1 momentum first (cached per filter set)."""
        if filters not in self._ranked:
            ok = self.base.copy()
            for f in filters:
                ok &= self.rules[f]
            score = self.mom.where(ok).to_numpy()
            out = {}
            for i, day in enumerate(self.close.index):
                row = score[i]
                idx = np.flatnonzero(~np.isnan(row))
                out[day] = list(self.symbols[idx[np.argsort(-row[idx])]])
            self._ranked[filters] = out
        return self._ranked[filters]

    def basket(self, start: pd.Timestamp) -> pd.Series:
        """Equal-weight daily holding of the liquid universe (eligibility known at the previous close)."""
        r = self.close.pct_change(fill_method=None)
        held = self.base.shift(1, fill_value=False)
        daily = r.where(held).mean(axis=1).fillna(0.0).loc[start:]
        return (1 + daily).cumprod()


def month_ends(index: pd.DatetimeIndex) -> set:
    s = pd.Series(index, index=index)
    return set(s.groupby([index.year, index.month]).max())


def run(bars, tables: Tables, start, limits: RiskLimits, filters: tuple, rebalance: bool, trend_exit: bool,
        capital: float, affordable: Optional[float] = None, slippage: float = SLIPPAGE) -> dict:
    ranked = tables.ranked(filters)
    ends = month_ends(tables.close.index)

    def buy(day, pf):
        if day < start or (rebalance and day not in ends):
            return []
        picks = ranked.get(day, [])[:TOP_N if rebalance else CANDIDATES]
        if affordable is not None:
            cap = pf.equity * affordable
            picks = [s for s in picks if tables.close.at[day, s] <= cap]
        return [(s, CONFIDENCE) for s in picks]

    def exit_(day, sym):
        return rebalance and day in ends and day >= start and sym not in ranked.get(day, [])[:TOP_N]

    result = simulate(bars, {}, limits, start_equity=capital, exits=ExitPolicy(trend_exit=trend_exit),
                      fees=india_delivery_fees, slippage=slippage, buy_source=buy,
                      exit_source=exit_ if rebalance else None)
    eq = result.equity.loc[start:]
    trades = [t for t in result.trades if t.entry_date >= start]
    m = curve_metrics(eq, capital)
    years = len(eq) / 252
    mid = len(eq) // 2
    return {"return": m["total_return"], "cagr": (1 + m["total_return"]) ** (1 / years) - 1, "sharpe": m["sharpe"],
            "dd": m["max_drawdown"], "trades": len(trades),
            "halves": (float(eq.iloc[mid] / capital - 1), float(eq.iloc[-1] / eq.iloc[mid] - 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--recent-years", type=int, default=3)
    ap.add_argument("--confirm", action="store_true", help="run the confirmation at the live account instead")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.confirm:
        confirm(args.years, args.recent_years)
        return
    bars = load_ohlc_nse(args.years)
    tables = Tables(bars)
    nifty = load_index_close("^NSEI", args.years)
    close = tables.close
    windows = [("full period", close.index[MOMENTUM_BARS]),
               (f"last {args.recent_years}y", close.index[close.index.searchsorted(close.index[-1] - pd.DateOffset(years=args.recent_years))])]

    live_rules = RiskLimits(**{**dataclasses.asdict(FULL_SIZE), **LIVE_EXIT_RULES})
    live_halted = RiskLimits(**{**dataclasses.asdict(live_rules), **LIVE_HALTS})
    no_stop = RiskLimits(**{**dataclasses.asdict(live_halted), "stop_loss_pct": 0.99})
    all_filters = ("trend", "rsi", "vol")
    # name -> (limits, filters, monthly rebalance, trend exit, capital, affordability)
    steps: Dict[str, tuple] = {
        "S0 pure momentum, monthly top 10": (FULL_SIZE, (), True, False, BIG, None),
        "S1 + uptrend filter": (FULL_SIZE, ("trend",), True, False, BIG, None),
        "S2 + RSI < 70": (FULL_SIZE, ("trend", "rsi"), True, False, BIG, None),
        "S3 + volatility <= 4%": (FULL_SIZE, all_filters, True, False, BIG, None),
        "S4 live holding (daily buys, MA200 exit)": (FULL_SIZE, all_filters, False, True, BIG, None),
        "S5 + 15% stop": (dataclasses.replace(FULL_SIZE, stop_loss_pct=0.15), all_filters, False, True, BIG, None),
        "S6 + 100% target": (live_rules, all_filters, False, True, BIG, None),
        "S7 + loss/drawdown halts": (live_halted, all_filters, False, True, BIG, None),
        "S8 live account (Rs 20k, 10 x 5%)": (LIVE_ACCOUNT, all_filters, False, True, LIVE_CAPITAL, 0.05),
    }
    loo: Dict[str, tuple] = {
        "L1 S7 without uptrend filter": (live_halted, ("rsi", "vol"), False, True, BIG, None),
        "L2 S7 without RSI cap": (live_halted, ("trend", "vol"), False, True, BIG, None),
        "L3 S7 without volatility cap": (live_halted, ("trend", "rsi"), False, True, BIG, None),
        "L4 S7 without 15% stop": (no_stop, all_filters, False, True, BIG, None),
        "L5 S7 with monthly top-10 rebalancing": (live_halted, all_filters, True, False, BIG, None),
    }
    results: Dict[str, Dict[str, dict]] = {}
    for wname, start in windows:
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        window = {s: df.loc[close.index[warm]:] for s, df in bars.items()}
        basket = tables.basket(start)
        bm = curve_metrics(basket, 1.0)
        b_cagr = (1 + bm["total_return"]) ** (252 / len(basket)) - 1
        n = nifty.loc[start:]
        print(f"\n=== {wname}: {start.date()} .. {close.index[-1].date()} ===")
        print(f"  {'liquid equal-weight basket':44s} return {bm['total_return']:+8.1%}  CAGR {b_cagr:+6.1%}  "
              f"Sharpe {bm['sharpe']:4.2f}  maxDD {bm['max_drawdown']:6.1%}")
        print(f"  {'Nifty 50':44s} return {n.iloc[-1] / n.iloc[0] - 1:+8.1%}")
        for name, (lim, filt, reb, tx, cap, aff) in {**steps, **loo}.items():
            r = run(window, tables, start, lim, filt, reb, tx, cap, aff)
            r["excess"] = r["cagr"] - b_cagr
            results.setdefault(name, {})[wname] = r
            print(f"  {name:44s} return {r['return']:+8.1%}  CAGR {r['cagr']:+6.1%}  excess {r['excess']:+6.1%}  "
                  f"Sharpe {r['sharpe']:4.2f}  maxDD {r['dd']:6.1%}  trades {r['trades']:4d}  "
                  f"halves {r['halves'][0]:+.0%}/{r['halves'][1]:+.0%}", flush=True)

    print("\n=== verdict (pre-declared): a live rule is harmful if dropping it raises excess CAGR AND Sharpe in both windows ===")
    ref = results["S7 + loss/drawdown halts"]
    for name in loo:
        better = all(results[name][w]["excess"] > ref[w]["excess"] and results[name][w]["sharpe"] > ref[w]["sharpe"]
                     for w, _ in windows)
        deltas = "; ".join(f"{w}: excess {results[name][w]['excess'] - ref[w]['excess']:+.1%}, "
                           f"Sharpe {results[name][w]['sharpe'] - ref[w]['sharpe']:+.2f}" for w, _ in windows)
        print(f"  {'HARMFUL' if better else 'not shown harmful':18s} {name}  ({deltas})")


def confirm(years: int, recent_years: int) -> None:
    """The pre-declared confirmation (module docstring): candidate filter removals at the live account, both universes."""
    current = ("trend", "rsi", "vol")
    candidates = {"drop RSI cap": ("trend", "vol"), "drop uptrend filter": ("rsi", "vol"), "drop both": ("vol",)}
    verdict = {name: True for name in candidates}
    for label, bars in (("all NSE", load_ohlc_nse(years)), ("Nifty 500 (today's members)", load_ohlc_universe(500, years))):
        tables = Tables(bars)
        close = tables.close
        windows = [("full period", close.index[MOMENTUM_BARS]),
                   (f"last {recent_years}y", close.index[close.index.searchsorted(close.index[-1] - pd.DateOffset(years=recent_years))])]
        print(f"\n##### {label}: {len(bars)} stocks")
        for wname, start in windows:
            warm = close.index.searchsorted(start - pd.Timedelta(days=330))
            window = {s: df.loc[close.index[warm]:] for s, df in bars.items()}
            live = run(window, tables, start, LIVE_ACCOUNT, current, False, True, LIVE_CAPITAL, 0.05)
            live2 = run(window, tables, start, LIVE_ACCOUNT, current, False, True, LIVE_CAPITAL, 0.05, SLIPPAGE * 2)
            print(f"=== {wname} from {start.date()}")
            print(f"  {'current live rules':22s} CAGR {live['cagr']:+6.1%}  Sharpe {live['sharpe']:4.2f}  maxDD {live['dd']:6.1%}  "
                  f"halves {live['halves'][0]:+.0%}/{live['halves'][1]:+.0%}  trades {live['trades']}")
            for name, filters in candidates.items():
                r = run(window, tables, start, LIVE_ACCOUNT, filters, False, True, LIVE_CAPITAL, 0.05)
                r2 = run(window, tables, start, LIVE_ACCOUNT, filters, False, True, LIVE_CAPITAL, 0.05, SLIPPAGE * 2)
                checks = {"CAGR": r["cagr"] > live["cagr"], "Sharpe": r["sharpe"] > live["sharpe"],
                          "2x slippage": r2["cagr"] > live2["cagr"], "drawdown": r["dd"] >= live["dd"] - 0.05}
                if wname == "full period":
                    checks["both halves"] = r["halves"][0] > live["halves"][0] and r["halves"][1] > live["halves"][1]
                verdict[name] &= all(checks.values())
                print(f"  {name:22s} CAGR {r['cagr']:+6.1%}  Sharpe {r['sharpe']:4.2f}  maxDD {r['dd']:6.1%}  "
                      f"halves {r['halves'][0]:+.0%}/{r['halves'][1]:+.0%}  trades {r['trades']}  | "
                      + " ".join(("PASS " if ok else "FAIL ") + k for k, ok in checks.items()), flush=True)
    print("\n=== confirmation verdict ===")
    for name, ok in verdict.items():
        print(f"  {'CONFIRMED' if ok else 'not confirmed':14s} {name}")


if __name__ == "__main__":
    main()
