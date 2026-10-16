"""Launcher for the intraday engines, each in its own paper account (dry run unless live), stepped together every five
minutes: the opening-range breakout, gap-and-go, and gap-only (gap-and-go's control: the same gaps, no breakout)."""

import os
from typing import Any

from src.app.wiring import (
    gapgo_db_url,
    gaponly_db_url,
    intraday_db_url,
    load_settings_or_exit,
    make_paper_broker,
    open_house_calls,
)
from src.data.intraday_bars import BarArchive, fetch_5m_bars, shared_download
from src.engine.costs import intraday_fees
from src.intraday.engine import IntradayEngine, run_together
from src.intraday.strategy import (
    GAP_ENTRY_UNTIL,
    GAP_ONLY_ENTRY_UNTIL,
    GAP_ONLY_LATEST_START,
    find_gap_only_setup,
    find_gap_setup,
)
from src.intraday.universe import select_universe
from src.monitoring.heartbeat import Heartbeat
from src.monitoring.telegram import Notifier
from src.versions import GAPGO_VERSION, GAPONLY_VERSION

# The gap experiments, in the order they are stepped: (Telegram label, database, rule, last bar that may signal, version,
# latest start of the 09:30 step). Gap-only goes first because it trades on that one step and must not wait behind the
# others' work (the first engine also pays for the shared download); the breakout engine, which has all day, goes last.
# Same sizing and universe as the breakout engine; the 5-day download supplies the previous session's close.
GAP_EXPERIMENTS = (
    ("GAP-ONLY", gaponly_db_url, find_gap_only_setup, GAP_ONLY_ENTRY_UNTIL, GAPONLY_VERSION, GAP_ONLY_LATEST_START),
    ("GAP-GO", gapgo_db_url, find_gap_setup, GAP_ENTRY_UNTIL, GAPGO_VERSION, None),
)


def run_intraday(live: bool) -> None:
    """Run the three intraday paper strategies, each against its own account, until interrupted."""
    settings = load_settings_or_exit()
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
            max_daily_loss_pct=settings.intraday_max_daily_loss_pct,
            **rule,
        )

    # One download per cycle for all three engines (5 days: the gap rules need the previous close; the breakout rule uses
    # only today's rows). record_capped: a signal the 5-position limit left out is still recorded (as SKIPPED), so the
    # forward record covers every signal of the day, not only the first five.
    bars = shared_download(lambda symbols: fetch_5m_bars(symbols, "5d"))
    # The breakout engine also reads the broker's research calls (today's Intraday buys join its scan; an open SELL call
    # blocks a buy). The gap engines do not: they are a matched pair, and an input on one side would spoil the comparison.
    house_calls = open_house_calls()
    engines = [
        engine(
            make_paper_broker(settings, db_url(), fees=intraday_fees),
            fetch_bars=bars,
            find_setup=rule,
            entry_until=until,
            label=name,
            version=version,
            step_deadline=deadline,
            record_capped=True,
        )
        for name, db_url, rule, until, version, deadline in GAP_EXPERIMENTS
    ]
    engines.append(
        engine(
            broker,
            fetch_bars=bars,
            archive=BarArchive(broker.sessions) if settings.intraday_archive else None,
            record_capped=True,
            house_calls_fn=house_calls.current,
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
    run_together(engines, heartbeat=Heartbeat.from_setting(settings.heartbeat_file))
