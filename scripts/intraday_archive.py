"""The archive of 5-minute bars (table intraday_bars in the intraday database): what is stored, and how to fill it.

Yahoo keeps only ~60 sessions of 5-minute data, so the intraday engine saves each finished session itself. Use this to see
what has accumulated, to backfill the window Yahoo still has, or to bring in an older pickle cache.

    python scripts/intraday_archive.py status
    python scripts/intraday_archive.py refresh             # download Yahoo's last 59 sessions, store the missing ones
    python scripts/intraday_archive.py import-cache FILE   # a pickle of {symbol: DataFrame} (or {"label", "data"})"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import open_bar_archive  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.intraday_bars import BarArchive, fetch_5m_bars  # noqa: E402
from src.intraday.universe import select_universe  # noqa: E402


def status(archive: BarArchive) -> None:
    days = archive.days()
    if not days:
        print("the archive is empty")
        return
    bars = archive.load()
    rows = sum(len(df) for df in bars.values())
    print(f"{len(days)} sessions, {days[0]} .. {days[-1]}; {len(bars)} stocks; {rows:,} bars")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("status", "refresh", "import-cache"))
    ap.add_argument("file", nargs="?", help="the pickle to import (import-cache)")
    args = ap.parse_args()
    archive = open_bar_archive()
    if args.command == "refresh":
        symbols, label = select_universe(load_settings().intraday_universe, 100)
        print(f"downloading 59 sessions for {len(symbols)} stocks [{label}] ...")
        print(f"added {archive.save(fetch_5m_bars(symbols, '59d')):,} bars")
    elif args.command == "import-cache":
        if not args.file:
            ap.error("import-cache needs the pickle file")
        loaded = pickle.loads(Path(args.file).read_bytes())
        print(f"added {archive.save(loaded['data'] if set(loaded) == {'label', 'data'} else loaded):,} bars")
    status(archive)


if __name__ == "__main__":
    main()
