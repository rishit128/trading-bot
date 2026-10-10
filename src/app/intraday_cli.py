"""Launcher for the intraday engines, each in its own paper account (dry run unless live), stepped together every five
minutes: the opening-range breakout, gap-and-go, and gap-only (gap-and-go's control: the same gaps, no breakout)."""

import os
import sys
from typing import Any

from src.app.wiring import gapgo_db_url, gaponly_db_url, intraday_db_url, make_paper_broker
from src.config import load_settings
from src.data.intraday_bars import BarArchive, fetch_5m_bars
from src.engine.costs import intraday_fees
from src.intraday.engine import IntradayEngine, run_together
from src.intraday.strategy import GAP_ENTRY_UNTIL, GAP_ONLY_ENTRY_UNTIL, find_gap_only_setup, find_gap_setup
from src.intraday.universe import select_universe
from src.monitoring.telegram import Notifier
from src.versions import GAPGO_VERSION, GAPONLY_VERSION

# The gap experiments: (Telegram label, database, rule, last bar that may signal, version). Same sizing and universe as
# the breakout engine; the 5-day download supplies the previous session's close.
GAP_EXPERIMENTS = (
    ("GAP-GO", gapgo_db_url, find_gap_setup, GAP_ENTRY_UNTIL, GAPGO_VERSION),
    ("GAP-ONLY", gaponly_db_url, find_gap_only_setup, GAP_ONLY_ENTRY_UNTIL, GAPONLY_VERSION),
)


def run_intraday(live: bool) -> None:
    """Run the three intraday paper strategies, each against its own account, until interrupted."""
    try:
        settings = load_settings()
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    broker = make_paper_broker(settings, intraday_db_url(), fees=intraday_fees)
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    notify = Notifier(token, chat_id).send if token and chat_id else None
    universe, label = select_universe(settings.intraday_universe)

    def engine(account, **rule: Any) -> IntradayEngine:
        """One engine on its own paper account, with the shared universe and its own sizing (INTRADAY_* settings, not
        the swing account's: see Settings.intraday_risk_pct)."""
        return IntradayEngine(
            account,
            universe,
            account.clock,
            notify=notify,
            live=live,
            universe_source=label,
            max_positions=settings.intraday_max_positions,
            risk_pct=settings.intraday_risk_pct,
            max_position_pct=settings.intraday_max_position_pct,
            **rule,
        )

    engines = [engine(broker, archive=BarArchive(broker.sessions) if settings.intraday_archive else None)]
    for name, db_url, rule, until, version in GAP_EXPERIMENTS:
        engines.append(
            engine(
                make_paper_broker(settings, db_url(), fees=intraday_fees),
                fetch_bars=lambda symbols: fetch_5m_bars(symbols, "5d"),
                find_setup=rule,
                entry_until=until,
                label=name,
                version=version,
                record_capped=True,
            )
        )
    mode = "PAPER ORDERS (separate intraday accounts)" if live else "DRY RUN (signals only)"
    print(f"intraday: opening-range breakout on {len(universe)} stocks [{label}], {mode}; square-off 15:15 IST")
    print(
        f"budget {settings.currency}{settings.paper_initial_cash:,.0f} per account, max {settings.intraday_max_positions} "
        f"positions, risk {settings.intraday_risk_pct:.2%} of equity per trade, "
        f"position cap {settings.intraday_max_position_pct:.0%} of equity"
    )
    print(
        "note: the backtest of this rule LOST money and an audit found no edge (see STRATEGY.md); "
        "this is paper trading to gather live evidence"
    )
    print(
        "gap-and-go (own account): gap >= 2%, then a close above the opening range by 10:30; held to the square-off. "
        "Positive after costs in both years of the 2-year test but not proven: paper only, to gather forward evidence"
    )
    print(
        "gap-only (own account, gap-and-go's control): the same gap, bought on the 09:30 close without a breakout; "
        "compare the two with scripts/compare_gap.py"
    )
    if notify:
        for running in engines:
            notify(f"[{running.label}] started: {mode}")
    run_together(engines)
