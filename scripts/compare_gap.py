"""Gap-and-go against its gap-only control: per trade, date by date, the same stock on the same day, costs, drawdown and
how much of each account was used, from the two paper accounts (see src/research/gap_compare.py for the rule).

    python scripts/compare_gap.py
    python scripts/compare_gap.py --gapgo-database-url sqlite:///gapgo.db --gaponly-database-url sqlite:///gaponly.db
On the server: sudo docker compose -f docker-compose.aws.yml exec intraday-bot python scripts/compare_gap.py"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import gapgo_db_url, gaponly_db_url, make_paper_broker  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.database import IntradaySignalRecord  # noqa: E402
from src.engine.costs import intraday_fees  # noqa: E402
from src.research.gap_compare import capped_signals, compare, format_comparison  # noqa: E402


def account(settings, url):
    """(closed trades, starting cash, recorded signals) of one paper account."""
    broker = make_paper_broker(settings, url, fees=intraday_fees)
    with broker.sessions() as s:
        signals = pd.DataFrame([{"outcome": r.outcome, "reason": r.reason} for r in s.scalars(select(IntradaySignalRecord))])
    return broker.trade_history(), broker.summary()["initial_cash"], signals


def main():
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gapgo-database-url", default=gapgo_db_url())
    ap.add_argument("--gaponly-database-url", default=gaponly_db_url())
    args = ap.parse_args()
    go_trades, go_cash, go_signals = account(settings, args.gapgo_database_url)
    only_trades, only_cash, only_signals = account(settings, args.gaponly_database_url)
    result = compare(
        go_trades,
        only_trades,
        go_cash,
        only_cash,
        capped={"gap-and-go": capped_signals(go_signals), "gap-only": capped_signals(only_signals)},
    )
    print(format_comparison(result, settings.currency))


if __name__ == "__main__":
    main()
