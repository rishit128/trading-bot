"""What the recorded entry context says about how the swing trades did (forward test; see STRATEGY.md, 2026-10-23/26).

    python scripts/context_report.py [--min 100]

Each swing BUY stores Angel One's spread, order-book balance, 52-week position, F&O put-call and build-up, the market regime,
crude and results timing (orders.context_json). This joins them to the outcome of the trade (closed: its net return;
open: marked at the latest Yahoo close) and splits the trades by each field: count, average return, win share. It prints
"too few" under the pre-declared 100 entries instead of a verdict, and never changes a rule: adopting one is a separate,
pre-declared test."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402


def load(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    orders = pd.read_sql(
        "select o.symbol, o.created_at, o.fill_price, o.context_json from orders o "
        "where o.side='BUY' and o.status='filled' and o.context_json is not null", con)
    trades = pd.read_sql("select symbol, opened_at, entry_price, exit_price, fees, qty from paper_trades", con)
    held = pd.read_sql("select symbol, avg_price, opened_at from paper_positions", con)
    rows = []
    for _, o in orders.iterrows():
        note = json.loads(o.context_json)
        closed = trades[(trades.symbol == o.symbol) & (abs(pd.to_datetime(trades.opened_at) - pd.to_datetime(o.created_at)) < pd.Timedelta(hours=1))]
        if len(closed):
            t = closed.iloc[0]
            ret, state = t.exit_price / t.entry_price - 1 - t.fees / (t.entry_price * t.qty), "closed"
        else:
            pos = held[held.symbol == o.symbol]
            if pos.empty:
                continue
            import yfinance as yf

            last = float(yf.Ticker(o.symbol + ".NS").history(period="5d")["Close"].iloc[-1])
            ret, state = last / float(pos.iloc[0].avg_price) - 1, "open"
        fno = note.get("fno") if isinstance(note.get("fno"), dict) else {}
        rows.append({
            "symbol": o.symbol, "ret": ret, "state": state,
            "market": str(note.get("market", "")).split("/")[0] or None,
            "crude": None if note.get("crude_1m") is None else ("rising" if note["crude_1m"] > 0 else "falling"),
            "results": "due or just out" if note.get("results_in_days") is not None or note.get("results_filed_days_ago") is not None else "none known",
            "fno": fno.get("build_up") or ("F&O stock, no signal" if fno else "not F&O"),
            "book": None if note.get("book_imbalance") is None else ("buyers" if note["book_imbalance"] > 0 else "sellers"),
            "near_52w_high": None if note.get("range_52w") is None else ("top third" if note["range_52w"] > 2 / 3 else "lower"),
            "spread": None if note.get("spread_pct") is None else ("wide (>0.15%)" if note["spread_pct"] > 0.15 else "tight"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=100)
    args = ap.parse_args()
    df = load(load_settings().database_url.replace("sqlite:///", ""))
    print(f"swing entries with a stored context: {len(df)} ({(df.state == 'closed').sum() if len(df) else 0} closed)")
    if df.empty:
        return
    verdict = "pre-declared minimum reached" if len(df) >= args.min else f"too few: {args.min} needed before any rule is considered"
    print(f"-> {verdict}\n")
    for col in ("market", "crude", "results", "fno", "book", "near_52w_high", "spread"):
        g = df.dropna(subset=[col]).groupby(col).ret.agg(n="size", avg="mean", win=lambda s: (s > 0).mean())
        if not g.empty:
            print(f"by {col}:\n{g.round(3).to_string()}\n")


if __name__ == "__main__":
    main()
