"""Backtest the intraday engine on the archived 5-minute bars, by running the LIVE engine and paper broker over them.

Nothing is re-implemented: every 5 minutes of every stored session `IntradayEngine.step()` runs against a `PaperBroker`, so
sizing, skips, stops, targets, the daily-loss halt, the square-off, fees and slippage are exactly what trades live (see
src/research/intraday_replay.py for the few things a replay must assume). One continuous account, so results compound.

Fill the archive first (scripts/intraday_archive.py refresh); it grows by itself while the engine runs.

    python scripts/backtest_intraday.py [--start 2026-08-01] [--end 2026-09-25] [--capital 20000] [--strategy gapgo]
    python scripts/backtest_intraday.py --bars-file research_cache/intraday_5m_2y.pkl   # a pickle of {symbol: 5-minute bars}"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import open_bar_archive  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.research.intraday_replay import STRATEGIES, ReplayConfig, format_summary, replay, summarise  # noqa: E402


def main():
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", help="first session (ISO date); default: the oldest archived")
    ap.add_argument("--end", help="last session (ISO date); default: the newest archived")
    ap.add_argument("--capital", type=float, default=settings.paper_initial_cash,
                    help="starting equity (default: PAPER_INITIAL_CASH)")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="orb", help="the rule to replay (default: the breakout)")
    ap.add_argument("--bars-file", help="replay a pickle of {symbol: 5-minute bars} instead of the archive")
    args = ap.parse_args()
    bars = pickle.loads(Path(args.bars_file).read_bytes()) if args.bars_file else open_bar_archive().load(args.start, args.end)
    if not bars:
        sys.exit("the archive has no bars for that range: run scripts/intraday_archive.py refresh")
    config = ReplayConfig.from_settings(settings, args.capital)
    print(f"replaying {args.strategy} on {len(bars)} stocks with {config}")
    result = replay(bars, config, start=args.start, end=args.end, strategy=args.strategy,
                    progress=lambda day: print(f"  {day}", end="\r", flush=True))
    print("\n" + format_summary(summarise(result)))


if __name__ == "__main__":
    main()
