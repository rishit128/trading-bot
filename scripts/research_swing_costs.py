"""Swing position count at the account's real delivery costs (declared 2026-11-11, before any result).

Why: until 2026-11-11 every swing backtest charged a zero-brokerage plan (about 0.24% a round trip plus Rs 15.93 a sale). The
account's real Angel One plan is 0.3% brokerage a side: about 0.93% a round trip plus a Rs 23.60 DP charge on each sale, which is
2.0% of a Rs 2,200 position. The 2026-10-23 position-count sweep (no count passed) was run at the old costs, so it is run again.
A changed cost model is new information, not a second look at the same test.

Rules (fixed): the live configuration from .env (scanner, 12-1 momentum ranking, risk engine, 15% stop, MA200 trend exit, paper
fees and slippage, AI stand-in confidence 0.75) on every NSE stock in the 10-year cache, one change per row: n positions of
0.90/n of equity each, n in 4, 5, 6, 8, 10, 12; the base is the live setting.

Gate (the 2026-10-23 one, unchanged): a count passes only if its Sharpe is at least the base's in BOTH the full period and the
last 3 years, its drawdown is no worse than the base's in both, and both neighbouring counts in the list also have a Sharpe at
least the base's in both windows. If several pass, the one nearest the live count is taken. Today's listings only: survivorship
bias lifts every row alike, so only differences between rows mean anything.

Second question, added 2026-11-11 before its result (`blend`): the core index sleeve at the real costs. The 2026-10-26 blend
test failed its gate with the strategy costed at zero brokerage. Same method and the same gate, word for word: the live
configuration's daily equity curve blended with a broad core (half NIFTYBEES, half JUNIORBEES; Angel One daily bars), re-balanced
on the first session of each month, re-balancing costs ignored; core weights 0, 25, 40, 50, 60, 100%. A weight is adopted only if,
against 0%, its Sharpe is at least 0.05 higher AND its drawdown is no worse AND its return is not lower than 0%'s minus a quarter
of it, in BOTH the full period and the last 3 years, and its neighbouring weights have a Sharpe at least 0%'s in both windows. If
several pass, the smallest is taken (the least change).

    python scripts/research_swing_costs.py            # the position counts
    python scripts/research_swing_costs.py blend      # the core sleeve"""

import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import screen_config  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.price_history import load_index_close, load_ohlc_nse  # noqa: E402
from src.data.universe import MOMENTUM_BARS  # noqa: E402
from src.engine.costs import india_delivery_fees  # noqa: E402
from src.research.live_backtest import DEFAULT_CONFIDENCE, ranked_candidates, run_live_config, summarize_run  # noqa: E402

COUNTS = (4, 5, 6, 8, 10, 12)
INVESTED = 0.90
WEIGHTS = (0.0, 0.25, 0.40, 0.50, 0.60, 1.0)
DAILY = Path(__file__).resolve().parent.parent / "research_cache" / "angel_daily"  # scripts/research_etf_rotation.py download


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    base_cfg = screen_config(settings)
    bars = load_ohlc_nse(10)
    nifty = load_index_close("^NSEI", 10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    position = settings.paper_initial_cash * settings.risk.max_position_pct
    cost = (india_delivery_fees("BUY", position) + india_delivery_fees("SELL", position)) / position
    print(
        f"{len(bars)} NSE stocks to {close.index[-1].date()}; capital Rs {settings.paper_initial_cash:,.0f}; base "
        f"{settings.risk.max_open_positions} positions of {settings.risk.max_position_pct:.1%}; fees {cost:.2%} a round trip on "
        f"Rs {position:,.0f}"
    )
    setups = [("base (live)", settings.risk, base_cfg)]
    for n in COUNTS:
        limits = dataclasses.replace(
            settings.risk, max_open_positions=n, min_position_pct=INVESTED / n, max_position_pct=INVESTED / n
        )
        setups.append((f"{n} x {INVESTED / n:.1%}", limits, dataclasses.replace(base_cfg, affordable_pct=INVESTED / n)))

    stats: dict = {}
    for window, start in (("full", close.index[MOMENTUM_BARS]), ("last 3y", close.index[-1] - pd.DateOffset(years=3))):
        start = close.index[close.index.searchsorted(start)]
        dates = [d for d in close.index if d >= start]
        warm = close.index.searchsorted(start - pd.Timedelta(days=330))
        frames = {s: df.loc[close.index[warm] :] for s, df in bars.items()}
        index = nifty.loc[start:]
        print(
            f"\n=== {window}: {start.date()} .. {close.index[-1].date()}; Nifty 50 {index.iloc[-1] / index.iloc[0] - 1:+.0%} ==="
        )
        print(
            f"{'setup':16s}{'return':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'win':>6}{'fees/capital':>14}",
            flush=True,
        )
        for label, limits, cfg in setups:
            # the scan depends on the smallest affordable position, so each setup is ranked with its own
            ranked = ranked_candidates(close, volume, dates, cfg, None)
            s = summarize_run(
                run_live_config(
                    frames, limits, cfg, settings.paper_initial_cash, ranked=ranked, confidence=DEFAULT_CONFIDENCE, start=start
                )
            )
            stats[(label, window)] = s
            print(
                f"{label:16s}{s['total_return']:+9.0%}{s['cagr']:+8.1%}{s['sharpe']:8.2f}{s['max_drawdown']:8.1%}"
                f"{s['trades']:8d}{s['win_rate']:6.0%}{s['fees_pct_of_capital']:14.0%}",
                flush=True,
            )

    def sharpe_ok(label: str) -> bool:
        return all(stats[(label, w)]["sharpe"] >= stats[("base (live)", w)]["sharpe"] for w in ("full", "last 3y"))

    def drawdown_ok(label: str) -> bool:
        return all(stats[(label, w)]["max_drawdown"] >= stats[("base (live)", w)]["max_drawdown"] for w in ("full", "last 3y"))

    labels = [label for label, *_ in setups[1:]]
    passed = [
        label
        for i, label in enumerate(labels)
        if sharpe_ok(label) and drawdown_ok(label) and all(sharpe_ok(labels[j]) for j in (i - 1, i + 1) if 0 <= j < len(labels))
    ]
    print("\nPASSED: " + (", ".join(passed) if passed else "none (the live count stays)"))


def blend(strategy: pd.Series, core: pd.Series, weight: float) -> pd.Series:
    """Daily returns of `weight` in the core and the rest in the strategy, re-balanced on each month's first session."""
    days = strategy.index.intersection(core.index)
    out, share, month = [], weight, None
    for day, s, c in zip(days, strategy.loc[days], core.loc[days]):
        if month != (day.year, day.month):
            share, month = weight, (day.year, day.month)
        out.append(share * c + (1 - share) * s)
        share = share * (1 + c) / (share * (1 + c) + (1 - share) * (1 + s))
    return pd.Series(out, index=days)


def curve_stats(returns: pd.Series) -> dict:
    curve = (1 + returns).cumprod()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    return {"ret": float(curve.iloc[-1] - 1), "sharpe": sharpe, "dd": float((curve / curve.cummax() - 1).min())}


def core_sleeve() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    cfg = screen_config(settings)
    bars = load_ohlc_nse(10)
    close = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).sort_index()
    volume = pd.DataFrame({s: df["Volume"] for s, df in bars.items()}).sort_index()
    start = close.index[MOMENTUM_BARS]
    warm = close.index.searchsorted(start - pd.Timedelta(days=330))
    frames = {s: df.loc[close.index[warm] :] for s, df in bars.items()}
    ranked = ranked_candidates(close, volume, [d for d in close.index if d >= start], cfg, None)
    run = run_live_config(frames, settings.risk, cfg, settings.paper_initial_cash, ranked=ranked, start=start)
    strategy = run.result.equity.loc[start:].pct_change().dropna()
    strategy.index = pd.DatetimeIndex(strategy.index).tz_localize(None).normalize()
    etf = {s: pd.read_pickle(DAILY / f"{s}.pkl")["Close"].pct_change().dropna() for s in ("NIFTYBEES", "JUNIORBEES")}
    core = ((etf["NIFTYBEES"] + etf["JUNIORBEES"]) / 2).dropna()
    windows = {"full": strategy.index[0], "last 3y": strategy.index[-1] - pd.DateOffset(years=3)}
    table = {(w, x): curve_stats(blend(strategy.loc[first:], core, x)) for w, first in windows.items() for x in WEIGHTS}
    for w in windows:
        print(f"\n== broad core, {w} ==\n{'core':>6}{'return':>9}{'Sharpe':>8}{'maxDD':>8}")
        for x in WEIGHTS:
            t = table[(w, x)]
            print(f"{x:6.0%}{t['ret']:+9.0%}{t['sharpe']:8.2f}{t['dd']:8.1%}")

    def ok(x: float, margin: float) -> bool:
        return all(
            table[(w, x)]["sharpe"] >= table[(w, 0.0)]["sharpe"] + margin
            and table[(w, x)]["dd"] >= table[(w, 0.0)]["dd"]
            and table[(w, x)]["ret"] >= table[(w, 0.0)]["ret"] - 0.25 * abs(table[(w, 0.0)]["ret"])
            for w in windows
        )

    inner = WEIGHTS[1:-1]
    passed = [
        x for i, x in enumerate(inner) if ok(x, 0.05) and all(ok(inner[j], 0.0) for j in (i - 1, i + 1) if 0 <= j < len(inner))
    ]
    print("\nPASSED: " + (", ".join(f"{x:.0%}" for x in passed) if passed else "none (the sleeve stays at its size)"))


if __name__ == "__main__":
    core_sleeve() if sys.argv[1:] == ["blend"] else main()
