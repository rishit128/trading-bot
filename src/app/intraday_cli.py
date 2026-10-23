"""Launcher for the intraday engines, each in its own paper account (dry run unless live), stepped together every five
minutes: the opening-range breakout, gap-and-go, gap-only (gap-and-go's control: the same gaps, no breakout) and the
news-catalyst engine (exchange filings read by the AI, with its filing watcher in a background thread)."""

import dataclasses
import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Set

from sqlalchemy import select

from src.app.wiring import (
    gapgo_db_url,
    gaponly_db_url,
    intraday_db_url,
    load_settings_or_exit,
    make_paper_broker,
    news_db_url,
    open_house_calls,
)
from src.data.angel import CACHE_DIR, AngelStream, make_stream
from src.data.angel_bars import AngelBars
from src.data.angel_quotes import AngelQuotes, EntryContext
from src.data.intraday_bars import BarArchive, fetch_5m_bars, shared_download
from src.data.market_context import MarketContextProvider
from src.data.market_regime import RegimeProvider, with_regime
from src.database import PaperPositionRecord, make_session_factory
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
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    notify = Notifier(token, chat_id).send if token and chat_id else None
    universe, label = select_universe(settings.intraday_universe)
    # Angel One streams the universe into angel_bars; its bars, prices and quote checks serve every account here (and the swing
    # bot's quote checks and prices), Yahoo being the automatic backup. See src.data.angel_bars.
    sessions = make_session_factory(intraday_db_url())
    angel = start_angel(settings, sessions, universe, notify)
    stream = angel.stream if angel else None
    broker = make_paper_broker(settings, intraday_db_url(), fees=intraday_fees, stream=stream)

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
            context_fn=angel.context if angel else None,
            **rule,
        )

    # One download per cycle for all three engines (5 days: the gap rules need the previous close; the breakout rule uses
    # only today's rows). record_capped: a signal the 5-position limit left out is still recorded (as SKIPPED), so the
    # forward record covers every signal of the day, not only the first five.
    def yahoo_bars(symbols):
        return fetch_5m_bars(symbols, "5d")

    bars = shared_download(AngelBars(sessions, stream, yahoo_bars, settings.angel_bars) if stream else yahoo_bars)
    # The breakout engine also reads the broker's research calls and notes them on its signals; it trades on them (today's
    # Intraday buys join its scan, an open SELL call blocks a buy) only if INTRADAY_CALLS_ACT is on, so by default it stays
    # a clean control. The gap engines do not read them: they are a matched pair, and an input on one side would spoil the
    # comparison.
    house_calls = open_house_calls()
    engines = [
        engine(
            make_paper_broker(settings, db_url(), fees=intraday_fees, stream=stream),
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
            act_on_calls=settings.intraday_calls_act,
        )
    )
    news = build_news_engine(settings, universe, bars, notify, live, angel) if settings.news_engine else None
    if news is not None:
        engines.append(news)  # last: its AI work runs in the watcher's thread, and it must not delay the 09:30 gap step
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
    if news is not None:
        print(
            "news-catalyst (own account): exchange filings (BSE feed, NSE as backup) read by the AI; long only on material "
            "news by fixed rules (STRATEGY.md, News-Catalyst v1). Untested: paper only"
        )
    if notify:
        for running in engines:
            notify(f"[{running.label}] started: {mode}")
    run_together(engines, heartbeat=Heartbeat.from_setting(settings.heartbeat_file))


class Angel(NamedTuple):
    stream: AngelStream
    context: Callable[[str, float], EntryContext]  # the quote checks and entry record, with the market regime added
    tokens: Dict[str, str]


def start_angel(settings, sessions, universe, notify) -> Optional[Angel]:
    """Start streaming Angel One's prices for the universe; None (with the reason printed) when it is off or cannot be used."""
    if not settings.angel_stream:
        return None
    made = make_stream(sessions, universe, CACHE_DIR, notify)
    if made is None:
        return None
    stream, tokens = made
    stream.start()
    quotes = AngelQuotes(lambda: stream.api, tokens, universe)
    return Angel(stream, with_regime(quotes.entry_context, RegimeProvider(MarketContextProvider(), sessions)), tokens)


def swing_holdings(sessions) -> Set[str]:
    """The symbols the swing account holds now (its own database, read directly: no prices are needed)."""
    with sessions() as s:
        return set(s.scalars(select(PaperPositionRecord.symbol)))


def build_news_engine(settings, universe, bars, notify, live: bool, angel: Optional[Angel] = None) -> Optional[Any]:
    """The news-catalyst engine on its own paper account, with its filing watcher started; None (and a printed reason) when it
    cannot run: it needs an AI model and the BSE -> NSE symbol map. `angel` is set when Angel One is in use: it
    drops stocks Angel One lists only as BE (trade-to-trade), gives the live price when a filing is seen and the circuit,
    spread and F&O checks at each entry."""
    from src.data.filings import FeedReader, load_bse_nse_map
    from src.data.india import IndiaClock
    from src.llm import LLMClient
    from src.news.engine import NewsEngine
    from src.news.universe import NewsUniverse
    from src.news.watcher import FilingWatcher

    if not settings.news_models or not os.getenv("OPENROUTER_API_KEY"):
        print("news-catalyst engine NOT started: it needs OPENROUTER_API_KEY and a model to read the filings")
        return None
    try:
        mapping = load_bse_nse_map(CACHE_DIR)
    except Exception as e:
        print(f"news-catalyst engine NOT started: no BSE -> NSE symbol map ({type(e).__name__}: {e})")
        return None
    swing_sessions = make_session_factory(settings.database_url)
    broker = make_paper_broker(settings, news_db_url(), fees=intraday_fees, stream=angel.stream if angel else None)
    news_universe = NewsUniverse([s for s in universe if angel is None or s in angel.tokens])
    calendar = IndiaClock()
    FilingWatcher(
        broker.sessions,
        FeedReader(mapping),
        LLMClient.from_settings(dataclasses.replace(settings, models=settings.news_models)),
        lambda: set(news_universe.symbols()),
        is_session_day=calendar.is_open,
        notify=notify,
        price_fn=angel.stream.last_price if angel else lambda symbol: None,
        held_fn=lambda: swing_holdings(swing_sessions),
    ).start()
    symbols = news_universe.symbols()
    return NewsEngine(
        broker,
        symbols,
        broker.clock,
        fetch_bars=bars,
        notify=notify,
        live=live,
        universe_source=f"news universe: top {len(symbols)} by traded value, price >= Rs 100, >= Rs 50 cr a day",
        risk_pct=settings.intraday_risk_pct,
        max_position_pct=settings.intraday_max_position_pct,
        universe_fn=news_universe.symbols,
        context_fn=angel.context if angel else None,
    )
