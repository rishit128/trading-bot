"""Launcher for the intraday engine (its own paper account, dry run unless live)."""
import os
import sys

from src.app.kit import intraday_db_url, make_paper_broker
from src.config import load_settings
from src.data.india import fetch_index_symbols
from src.intraday.engine import IntradayEngine
from src.intraday.strategy import intraday_fees
from src.monitoring.telegram import Notifier


def run_intraday(live: bool) -> None:
    """Run the intraday opening-range-breakout engine against its own paper account until interrupted."""
    try:
        settings = load_settings()
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    broker = make_paper_broker(settings, intraday_db_url(), fees=intraday_fees)
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    notify = Notifier(token, chat_id).send if token and chat_id else None
    universe = fetch_index_symbols(100)
    engine = IntradayEngine(broker, universe, broker.clock, notify=notify, live=live)
    mode = "PAPER ORDERS (separate intraday account)" if live else "DRY RUN (signals only)"
    print(f"intraday: opening-range breakout on {len(universe)} Nifty 100 stocks, {mode}; square-off 15:15 IST")
    print("note: the 58-day backtest of this rule LOST money (see STRATEGY.md); this is paper trading to gather live evidence")
    if notify:
        notify(f"[INTRADAY] started: {mode}")
    engine.run_loop()
