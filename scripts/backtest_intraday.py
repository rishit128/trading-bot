"""Backtest the opening-range-breakout on Yahoo's last ~60 days of 5-minute bars (all Yahoo provides).

Honest limits: ~3 months is one market regime and a small sample; today's Nifty 100 members only; entry fills at the
signal bar's close plus slippage (the bot buys at the live price a moment later); if a bar touches both stop and target
the stop is assumed first; costs and slippage are charged, cash earns 0%. Parameters are fixed in src/intraday/strategy.py."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.india import IST, SUFFIX, fetch_index_symbols  # noqa: E402
from src.intraday import strategy as st  # noqa: E402


def download_5m(symbols, days):
    import yfinance as yf

    out = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i + 50]
        wide = yf.download([s + SUFFIX for s in batch], period=f"{days}d", interval="5m", group_by="ticker",
                           auto_adjust=True, progress=False, threads=True, timeout=60)
        top = set(wide.columns.get_level_values(0))
        for s in batch:
            if s + SUFFIX in top:
                df = wide[s + SUFFIX].dropna(subset=["Close"])
                if len(df):
                    df.index = df.index.tz_convert(IST)
                    out[s] = df
    return out


def simulate(day_bars, setup, slippage):
    """Enter at the signal close (+slippage), then walk the later bars: stop, target, or 15:15 exit. Returns (entry, exit, why)."""
    entry = setup.signal_price * (1 + slippage)
    later = day_bars[day_bars.index > setup.bar_time]
    for ts, bar in later.iterrows():
        if ts.time() >= st.SQUARE_OFF:
            return entry, float(bar["Open"]) * (1 - slippage), "EOD"
        if bar["Open"] <= setup.stop:
            return entry, float(bar["Open"]) * (1 - slippage), "STOP"
        if bar["Low"] <= setup.stop:
            return entry, setup.stop * (1 - slippage), "STOP"
        if bar["High"] >= setup.target:
            return entry, setup.target * (1 - slippage), "TARGET"
    last = later["Close"].iloc[-1] if len(later) else setup.signal_price
    return entry, float(last) * (1 - slippage), "EOD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=59)
    ap.add_argument("--index", type=int, default=100)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--max-positions", type=int, default=5)
    args = ap.parse_args()
    symbols = fetch_index_symbols(args.index)
    data = download_5m(symbols, args.days)
    print(f"{len(data)} of {len(symbols)} Nifty {args.index} stocks with 5-minute data")

    setups = []  # (date, setup, entry, exit, why)
    for sym, df in data.items():
        for day, g in df.groupby(df.index.date):
            if len(g) < 60:  # a partial session
                continue
            s = st.find_setup(sym, g)
            if s:
                entry, exit_, why = simulate(g, s, args.slippage)
                setups.append((day, s, entry, exit_, why))
    if not setups:
        print("no signals"); return

    equity, rows = args.capital, []
    for day in sorted({d for d, *_ in setups}):
        todays = sorted((x for x in setups if x[0] == day), key=lambda x: x[1].bar_time)[:args.max_positions]
        day_pnl = 0.0
        for _, s, entry, exit_, why in todays:
            qty = st.position_size(equity, entry, s.stop)
            if qty <= 0:
                continue
            fees = st.intraday_fees("BUY", qty * entry) + st.intraday_fees("SELL", qty * exit_)
            pnl = qty * (exit_ - entry) - fees
            day_pnl += pnl
            rows.append({"day": day, "symbol": s.symbol, "why": why, "pnl": pnl, "ret": exit_ / entry - 1, "fees": fees})
        equity += day_pnl
    t = pd.DataFrame(rows)
    daily = t.groupby("day")["pnl"].sum()
    wins = t[t.pnl > 0]
    print(f"\ntrades {len(t)} over {len(daily)} days | win rate {len(wins) / len(t):.0%} | avg win {wins.pnl.mean():,.0f} "
          f"avg loss {t[t.pnl <= 0].pnl.mean():,.0f} | fees paid {t.fees.sum():,.0f}")
    print(f"exits: {t.why.value_counts().to_dict()}")
    print(f"start {args.capital:,.0f} -> end {equity:,.0f}  ({equity / args.capital - 1:+.2%} over {len(daily)} trading days)")
    dd = ((args.capital + daily.cumsum()) / (args.capital + daily.cumsum()).cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    print(f"daily Sharpe (annualised) {sharpe:.2f} | max drawdown {dd:.1%} | profitable days {(daily > 0).mean():.0%}")
    half = len(daily) // 2
    print(f"first half P&L {daily.iloc[:half].sum():+,.0f} | second half P&L {daily.iloc[half:].sum():+,.0f}")
    print("by exit:", t.groupby("why").pnl.agg(["count", "sum"]).round(0).to_dict("index"))


if __name__ == "__main__":
    main()
