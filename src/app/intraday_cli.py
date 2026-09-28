"""Launcher for the intraday engine (its own paper account, dry run unless live)."""
import os
import sys

from src.app.wiring import intraday_db_url, make_paper_broker
from src.config import load_settings
from src.data.intraday_bars import BarArchive
from src.intraday.engine import IntradayEngine
from src.intraday.strategy import intraday_fees
from src.intraday.universe import select_universe
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
    universe, label = select_universe(settings.intraday_universe)
    # Its own sizing (INTRADAY_* settings), not the swing account's: see Settings.intraday_risk_pct.
    engine = IntradayEngine(broker, universe, broker.clock, notify=notify, live=live,
                            max_positions=settings.intraday_max_positions, risk_pct=settings.intraday_risk_pct,
                            max_position_pct=settings.intraday_max_position_pct, universe_source=label,
                            archive=BarArchive(broker.sessions) if settings.intraday_archive else None)
    mode = "PAPER ORDERS (separate intraday account)" if live else "DRY RUN (signals only)"
    print(f"intraday: opening-range breakout on {len(universe)} stocks [{label}], {mode}; square-off 15:15 IST")
    print(f"budget {settings.currency}{settings.paper_initial_cash:,.0f}, max {settings.intraday_max_positions} positions, "
          f"risk {settings.intraday_risk_pct:.2%} of equity per trade, "
          f"position cap {settings.intraday_max_position_pct:.0%} of equity")
    print("note: the backtest of this rule LOST money and an audit found no edge (see STRATEGY.md); "
          "this is paper trading to gather live evidence")
    if notify:
        notify(f"[INTRADAY] started: {mode}")
    engine.run_loop()
