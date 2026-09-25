"""Attach real outcomes to the decisions the bot has made (the bot also does this once a day). Safe to re-run: only
matured, unlabelled decisions are touched.

    python scripts/label_outcomes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.data.india import SUFFIX  # noqa: E402
from src.data.market_data import fetch_daily_bars  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.learning.outcomes import HORIZONS, backfill_bar_dates, label_outcomes  # noqa: E402


def main():
    settings = load_settings()
    sessions = make_session_factory(settings.database_url)
    print(f"bar dates backfilled on old decisions: {backfill_bar_dates(sessions)}")
    r = label_outcomes(sessions, lambda symbol: fetch_daily_bars(symbol, suffix=SUFFIX), HORIZONS, settings.risk.stop_loss_pct)
    print(f"new outcomes {r.labelled}, already done {r.already_done}, not yet matured {r.not_mature}, no price data {r.no_data}")
    if r.symbols_without_bars:
        print("no price data for:", ", ".join(r.symbols_without_bars))


if __name__ == "__main__":
    main()
