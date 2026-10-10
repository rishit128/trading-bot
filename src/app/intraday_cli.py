"""Launcher for the intraday engines: the opening-range breakout and gap-and-go, each in its own paper account (dry run
unless live), stepped together every five minutes."""
import os
import sys

from src.app.wiring import gapgo_db_url, intraday_db_url, make_paper_broker
from src.config import load_settings
from src.data.intraday_bars import BarArchive, fetch_5m_bars
from src.intraday.engine import IntradayEngine, run_together
from src.intraday.strategy import GAP_ENTRY_UNTIL, find_gap_setup, intraday_fees
from src.intraday.universe import select_universe
from src.monitoring.telegram import Notifier
from src.versions import GAPGO_VERSION


def run_intraday(live: bool) -> None:
    """Run both intraday paper strategies, each against its own account, until interrupted."""
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
    # Gap-and-go: same sizing and universe, its own account; the 5-day download supplies the previous session's close.
    gap_broker = make_paper_broker(settings, gapgo_db_url(), fees=intraday_fees)
    gapgo = IntradayEngine(gap_broker, universe, gap_broker.clock, fetch_bars=lambda symbols: fetch_5m_bars(symbols, "5d"),
                           notify=notify, live=live, max_positions=settings.intraday_max_positions,
                           risk_pct=settings.intraday_risk_pct, max_position_pct=settings.intraday_max_position_pct,
                           universe_source=label, find_setup=find_gap_setup, entry_until=GAP_ENTRY_UNTIL, label="GAP-GO",
                           version=GAPGO_VERSION)
    mode = "PAPER ORDERS (separate intraday accounts)" if live else "DRY RUN (signals only)"
    print(f"intraday: opening-range breakout on {len(universe)} stocks [{label}], {mode}; square-off 15:15 IST")
    print(f"budget {settings.currency}{settings.paper_initial_cash:,.0f}, max {settings.intraday_max_positions} positions, "
          f"risk {settings.intraday_risk_pct:.2%} of equity per trade, "
          f"position cap {settings.intraday_max_position_pct:.0%} of equity")
    print("note: the backtest of this rule LOST money and an audit found no edge (see STRATEGY.md); "
          "this is paper trading to gather live evidence")
    print("gap-and-go (own account): gap >= 2%, then a close above the opening range by 10:30; held to the square-off. "
          "Positive after costs in both years of the 2-year test but not proven: paper only, to gather forward evidence")
    if notify:
        notify(f"[INTRADAY] started: {mode}")
        notify(f"[GAP-GO] started: {mode}")
    run_together([engine, gapgo])
