"""P0 B1: run the phase ablation over one historical window for one symbol.

    DET / +LLM / +LEARNING / +CONTEXT / +REFLECTION / FULL

Usage:
  python scripts/ablate_phases.py RELIANCE --start 2023-01-01 --end 2024-06-30
  python scripts/ablate_phases.py RELIANCE --offline          # mechanical-only, requires no API key

Phase 2 (learning) reads the closed paper trades of `--database-url` (default sqlite:///trading.db),
point-in-time, so an ablation on live history has no look-ahead. Without a configured model the LLM
phases are skipped and the report says so; the mechanical-only answer is still a valid baseline the
review can use."""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.engine.paper_broker import india_delivery_fees  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.research.ablation import run_ablation  # noqa: E402


def download(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol + ".NS", start=start, end=end, auto_adjust=True, progress=False, interval="1d")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--start", type=date.fromisoformat, default=None)
    ap.add_argument("--end", type=date.fromisoformat, default=None)
    ap.add_argument("--database-url", default="sqlite:///trading.db")
    ap.add_argument("--offline", action="store_true", help="no API key: only the mechanical DET baseline is produced")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    end = args.end or date.today()
    start = args.start or (end - pd.Timedelta(days=700)).date()
    bars = download(args.symbol, start.isoformat(), (end + pd.Timedelta(days=1)).isoformat())
    if len(bars) < 200:
        raise SystemExit(f"need >=200 bars for warm-up, got {len(bars)}")
    dates = list(bars.index[200:])
    print(f"{args.symbol}: {len(bars)} bars {bars.index[0].date()} .. {bars.index[-1].date()}, "
          f"{len(dates)} decision dates")

    llm, sessions = None, None
    if not args.offline:
        try:
            llm = LLMClient.from_settings(load_settings(), cache_ttl_seconds=0)
            sessions = make_session_factory(args.database_url)
        except Exception as e:
            print(f"no model configured ({e}); running mechanical-only (use --offline to silence this)")
    result = run_ablation(args.symbol, {args.symbol: bars}, dates, llm=llm, sessions=sessions,
                          fees=india_delivery_fees)
    print(result.table())
    if any(r.skipped for r in result.reports.values()):
        print("note: LLM phases were skipped (no model configured); only DET is a real result here.")


if __name__ == "__main__":
    main()