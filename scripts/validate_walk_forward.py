"""P0 C1: walk-forward validation of one phase over one symbol.

    python scripts/validate_walk_forward.py RELIANCE --n-windows 4
    python scripts/validate_walk_forward.py RELIANCE --phase +LLM     # requires a configured model
    python scripts/validate_walk_forward.py RELIANCE --offline        # mechanical DET, no API key

The decision timeline is carved into non-overlapping hold-out blocks; every block is evaluated with
point-in-time learning and the OOS result is the continuous curve of all hold-out blocks stitched
together. Without a model, `--phase` falls back to DET (mechanical, offline-valid)."""
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
from src.research.ablation import DET  # noqa: E402
from src.research.walk_forward import run_walk_forward  # noqa: E402


def download(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol + ".NS", start=start, end=end, auto_adjust=True, progress=False, interval="1d")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--start", type=date.fromisoformat, default=None)
    ap.add_argument("--end", type=date.fromisoformat, default=None)
    ap.add_argument("--phase", default=DET, choices=[DET, "+LLM", "+LEARNING", "+CONTEXT", "+REFLECTION", "FULL"])
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=0)
    ap.add_argument("--database-url", default="sqlite:///trading.db")
    ap.add_argument("--offline", action="store_true", help="mechanical DET only, no API key")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    end = args.end or date.today()
    start = args.start or (end - pd.Timedelta(days=900)).date()
    bars = download(args.symbol, start.isoformat(), (end + pd.Timedelta(days=1)).isoformat())
    if len(bars) < 200:
        raise SystemExit(f"need >=200 bars for warm-up, got {len(bars)}")
    dates = list(bars.index[200:])
    print(f"{args.symbol}: {len(bars)} bars {bars.index[0].date()} .. {bars.index[-1].date()}, "
          f"{len(dates)} decision dates")

    llm, sessions = None, None
    if not args.offline and args.phase != DET:
        try:
            llm = LLMClient.from_settings(load_settings(), cache_ttl_seconds=0)
            sessions = make_session_factory(args.database_url)
        except Exception as e:
            print(f"no model configured ({e}); falling back to {DET}")
            args.phase = DET
    result = run_walk_forward(args.symbol, {args.symbol: bars}, dates, phase=args.phase,
                              n_windows=args.n_windows, min_train=args.min_train, llm=llm, sessions=sessions,
                              fees=india_delivery_fees)
    print(result.summary())


if __name__ == "__main__":
    main()