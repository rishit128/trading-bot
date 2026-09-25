"""No-LLM baseline for India: the trend rule (price > MA50 > MA200, RSI < 70 -> BUY; price < MA200 -> SELL) run through the
same simulator/risk engine as live, WITH Indian delivery costs, vs equal-weight buy&hold and the Nifty 50 index.

  --years 10 --lookback 2200   test across all regimes since 2016 instead of just the last year
  --exits                      compare three exit styles chosen in advance (not tuned): the bot's tight 2%/5% bracket,
                               a wider 8%/20% bracket, and "trend exit only" (no stop/target, sell on an MA200 break),
                               then break the bot's own losses down by exit reason, holding period, year and symbol

Caveats: today's index constituents only (survivorship bias flatters results), daily bars (stops assumed to fill at
their level, no circuit limits), max 10 positions of 5% each so the strategy is at most half invested. It does NOT test
the LLM agent: only whether the rule the scanner and agent are built around has an edge net of costs."""
import argparse
import dataclasses
import io
import sys
from pathlib import Path

import httpx
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.backtest import (TIME_LIMITED, ExitPolicy, analyze_trades, buy_and_hold_curve, curve_metrics, rule_signals, simulate,  # noqa: E402
                          trade_metrics)
from src.config import load_settings  # noqa: E402
from src.data.india import NIFTY_LISTS, _HEADERS  # noqa: E402
from src.engine.costs import india_delivery_fees  # noqa: E402


def download(tickers, years):
    df = yf.download(tickers, period=f"{years}y", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        if t in df.columns.get_level_values(0):
            sub = df[t].dropna(subset=["Open", "High", "Low", "Close"])
            if len(sub):
                out[t] = sub
    return out


def line(label, r, capital):
    m, t = curve_metrics(r.equity, capital), trade_metrics(r.trades)
    extra = f"{t['trades']:6d}  {t['win_rate']:.0%}  {t['avg_trade_return']:+.2%}  {t['exits']}" if t["trades"] else "     0"
    print(f"{label:40s} {m['total_return']:+8.2%} {m['max_drawdown']:8.2%} {m['sharpe']:7.2f}  {extra}")


def print_analysis(trades):
    a = analyze_trades(trades)
    if not a:
        return
    fmt = {"win_rate": "{:.0%}".format, "avg_return": "{:+.2%}".format, "pnl": "{:,.0f}".format}
    for title, key in (("by exit reason", "by_reason"), ("by holding period", "by_holding"), ("by year", "by_year")):
        print(f"\n  P&L {title} (net of fees):")
        print("  " + a[key].to_string(formatters=fmt).replace("\n", "\n  "))
    print(f"\n  worst symbols by P&L: {', '.join(f'{s} {v:,.0f}' for s, v in a['worst_symbols'].items())}")
    if a["median_days_stopped_out"] is not None:
        print(f"  median days before a stop-out: {a['median_days_stopped_out']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, choices=(50, 100, 500), default=50)
    ap.add_argument("--years", type=int, default=3, help="years of history to download")
    ap.add_argument("--lookback", type=int, default=250, help="trading days to backtest")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--exits", action="store_true", help="compare exit styles and analyse the bot's losses")
    args = ap.parse_args()

    symbols = pd.read_csv(io.StringIO(httpx.get(NIFTY_LISTS[args.index], headers=_HEADERS, timeout=30).text))["Symbol"].tolist()
    nifty = download(["^NSEI"], args.years)["^NSEI"]
    calendar = nifty.index[-args.lookback:]
    raw = download([s + ".NS" for s in symbols], args.years)
    need = args.lookback + 210
    bars = {}
    for t, df in raw.items():
        if len(df) >= need and df.index.isin(calendar).sum() >= 0.98 * len(calendar):
            bars[t.removesuffix(".NS")] = df.reindex(nifty.index).ffill().dropna()
    print(f"Nifty {args.index}: {len(symbols)} listed, {len(bars)} usable (>= {need} bars, few gaps)", flush=True)

    dates = list(calendar[::args.step])
    settings = load_settings()
    capital = settings.paper_initial_cash  # same account size as the paper broker: flat per-sale charges scale with it
    signals = rule_signals(bars, dates)
    window = {s: df.loc[dates[0]:] for s, df in bars.items()}

    print(f"period {dates[0].date()} .. {calendar[-1].date()}, decisions every {args.step} trading days\n")
    if args.exits:
        wide = dataclasses.replace(settings.risk, stop_loss_pct=0.08, take_profit_pct=0.20)
        trend_only = dataclasses.replace(settings.risk, stop_loss_pct=0.60, take_profit_pct=100.0)
        protective = dataclasses.replace(settings.risk, stop_loss_pct=0.08, take_profit_pct=100.0)
        runs = (("bot default: 2% stop / 5% target / 10d", settings.risk, 10),
                ("wider: 8% stop / 20% target / 60d", wide, 60),
                ("trend exit only (no stop/target)", trend_only, 10 ** 6),
                # Added AFTER seeing the three above, to mirror what the live bot can actually do (no time exit, a
                # broker-side protective stop): wide 8% stop, no practical target, exit on the MA200 trend break.
                ("live-style: 8% protective stop + trend exit", protective, 10 ** 6),
                ("live-style variant: 8%/20% and no time limit", wide, 10 ** 6))
        mid = dates[len(dates) // 2]
        halves = (("full", None, None), ("1st half", None, mid), ("2nd half", mid, None))
        print(f"{'exit style (with Indian costs)':40s} " + " ".join(f"{name + ' ret':>14} {'trades':>6}" for name, _, _ in halves)
              + f"   (halves split at {mid.date()})")
        results = {}
        for label, limits, hold in runs:
            row = []
            for name, start, end in halves:
                w = {s: df.loc[start or dates[0]:end] for s, df in bars.items()}
                sg = {s: {d: v for d, v in sig.items() if (start is None or d >= start) and (end is None or d <= end)}
                      for s, sig in signals.items()}
                r = simulate(w, sg, limits, start_equity=capital, exits=ExitPolicy(max_hold_days=None if hold >= 10 ** 6 else hold),
                             fees=india_delivery_fees)
                if name == "full":
                    results[label] = r
                m = curve_metrics(r.equity, capital)
                row.append(f"{m['total_return']:+14.1%} {len(r.trades):6d}")
            print(f"{label:40s} " + " ".join(row))
        print()
        print(f"{'':40s} {'return':>8} {'maxDD':>8} {'sharpe':>7}  trades  win   avg/trade  exits  (full period)")
        for label, r in results.items():
            line(label, r, capital)
        line("bot default, zero costs", simulate(window, signals, settings.risk, start_equity=capital, exits=TIME_LIMITED), capital)
        print()
        print("Failure analysis of the bot's current exits:")
        print_analysis(results[runs[0][0]].trades)
        print()
    else:
        print(f"{'':40s} {'return':>8} {'maxDD':>8} {'sharpe':>7}  trades  win   avg/trade  exits")
        for label, fees in (("trend rule, with Indian costs", india_delivery_fees), ("trend rule, zero costs", None)):
            line(label, simulate(window, signals, settings.risk, start_equity=capital, fees=fees, exits=TIME_LIMITED), capital)
    bh = curve_metrics(buy_and_hold_curve(window, capital), capital)
    idx = curve_metrics(buy_and_hold_curve({"NIFTY": nifty.loc[dates[0]:]}, capital), capital)
    print(f"{'buy & hold, same stocks (100% in)':40s} {bh['total_return']:+8.2%} {bh['max_drawdown']:8.2%} {bh['sharpe']:7.2f}")
    print(f"{'Nifty 50 index buy & hold':40s} {idx['total_return']:+8.2%} {idx['max_drawdown']:8.2%} {idx['sharpe']:7.2f}")


if __name__ == "__main__":
    main()
