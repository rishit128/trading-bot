"""Assembling the bot: the pipeline with its agents, broker and Telegram, the startup checks, the daily jobs, and the
commands `main.py` dispatches to. Nothing here decides anything about trading: it only wires the parts together."""

import dataclasses
import logging
import os
import sys
from typing import Callable, Dict, Tuple

from src.agents.technical import TechnicalAgent
from src.app.portfolio_agent import (
    PortfolioCommand,
    make_explainer,
    make_research_sink,
    run_standalone,
    strip_html,
    to_html,
)
from src.app.reports import print_positions
from src.app.wiring import (
    build_market_wiring,
    gapgo_db_url,
    gaponly_db_url,
    intraday_db_url,
    load_settings_or_exit,
    make_paper_broker,
    news_db_url,
    open_house_calls,
)
from src.config import Settings
from src.control import Control
from src.data.fundamentals import FundamentalsProvider
from src.data.india import SUFFIX
from src.data.market_context import MarketContextProvider
from src.data.market_data import fetch_daily_bars
from src.data.market_pulse import MarketPulse
from src.database import make_session_factory
from src.engine.costs import intraday_fees
from src.learning.jobs import DailyOutcomeLabelling, WeeklyLearningReport
from src.llm import LLMClient
from src.monitoring.heartbeat import Heartbeat
from src.monitoring.telegram import CommandListener, Notifier, handle_command
from src.ops.holdout import HoldoutReserve
from src.ops.preflight import build_checks, critical_failures, format_results, run_checks
from src.pipeline import TradingPipeline
from src.runner import run_cycle, run_loop

log = logging.getLogger(__name__)


def build_pipeline(live: bool) -> TradingPipeline:
    """Validate configuration and keys, then wire the LLM, agents, broker, Telegram and the LangGraph pipeline."""
    settings = load_settings_or_exit()
    if live:
        settings = dataclasses.replace(settings, dry_run=False)
    if not settings.models:
        sys.exit("OPENROUTER_MODEL is not set in .env")
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Missing in .env: OPENROUTER_API_KEY")

    llm = LLMClient.from_settings(settings)
    sessions = make_session_factory(settings.database_url)
    control = Control(sessions)
    kit = build_market_wiring(settings, sessions)
    house_calls = open_house_calls()
    intraday_broker = gapgo_broker = gaponly_broker = news_broker = None
    try:
        intraday_broker = make_paper_broker(settings, intraday_db_url(), fees=intraday_fees)
        gapgo_broker = make_paper_broker(settings, gapgo_db_url(), fees=intraday_fees)
        gaponly_broker = make_paper_broker(settings, gaponly_db_url(), fees=intraday_fees)
        news_broker = make_paper_broker(settings, news_db_url(), fees=intraday_fees)
    except Exception as e:  # the intraday views are optional; never stop the swing bot for them
        print(f"intraday status commands unavailable: {type(e).__name__}")
    notify = None
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        notify = Notifier(token, chat_id).send
        # /portfolio: the read-only real-account agent. It shares this Telegram bot, so this listener hands it the OTP.
        # /research_start keeps the same login open through the day and feeds the house-call snapshot.
        portfolio = PortfolioCommand(notify, make_explainer(llm), research_sink=make_research_sink(house_calls, to_html(notify)))
        extra: Dict[str, Tuple[str, Callable[[], str]]] = {
            "/portfolio": (PortfolioCommand.HELP, portfolio),
            "/research_start": (PortfolioCommand.RESEARCH_START_HELP, portfolio.research_start),
            "/research_stop": (PortfolioCommand.RESEARCH_STOP_HELP, portfolio.research_stop),
        }
        CommandListener(
            token,
            chat_id,
            lambda text: handle_command(
                text, control, kit.broker, settings, intraday_broker, extra, gapgo_broker, gaponly_broker, news_broker
            ),
            intercept=portfolio.inbox.offer,
        ).start()
    else:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); alerts and /pause disabled")
    market_provider = MarketContextProvider()

    def market_context(snapshot):
        """Market situation for the stock's prompt (index regime, VIX, relative strength); a failure means none."""
        try:
            return market_provider.context(snapshot.symbol, snapshot)
        except Exception as e:
            log.warning("market context failed for %s: %s: %s", snapshot.symbol, type(e).__name__, e)
            return None

    agents = [TechnicalAgent(llm, use_cot=settings.llm_cot, use_reflect=settings.llm_reflect)]
    pipeline = TradingPipeline(
        settings,
        agents,
        kit.broker,
        sessions,
        kit.snapshot_fn,
        control=control,
        notify=notify,
        universe_fn=kit.screener.symbols_for if kit.screener else None,
        quote_fn=kit.quote_fn,
        market_context_fn=market_context,
        fundamentals_fn=FundamentalsProvider(),
        broker_label=kit.broker_label,
        house_calls_fn=house_calls.current,
        screen_verdict_fn=kit.screener.verdict if kit.screener else None,
        pulse=MarketPulse(kit.screener.movers) if kit.screener and settings.market_pulse else None,
    )
    return pipeline


def preflight(pipeline) -> list:
    """Run the connectivity checks and print them. Returns the critical failures (empty = safe to start)."""
    settings = pipeline.settings if pipeline is not None else load_settings_or_exit()
    broker = (
        pipeline.broker
        if pipeline is not None
        else build_market_wiring(settings, make_session_factory(settings.database_url)).broker
    )
    results = run_checks(build_checks(settings, broker, os.environ))
    print("Startup checks:\n" + format_results(results))
    return critical_failures(results)


def build_outcome_labeller(pipeline) -> DailyOutcomeLabelling:
    """The once-a-day job that attaches real outcomes to the decisions the bot has made, so it can learn from all of them."""
    return DailyOutcomeLabelling(
        pipeline.sessions,
        pipeline.control,
        lambda symbol: fetch_daily_bars(symbol, suffix=SUFFIX),
        stop_pct=pipeline.settings.risk.stop_loss_pct,
        notify=pipeline.notify,
    )


def build_holdout_callback(pipeline):
    """The per-cycle hook for --holdout: replays the approved decisions into the independent reserve and prints a mark."""
    import yfinance as yf

    from src.engine.costs import SLIPPAGE

    def reserve_bars(sym: str):
        """Daily OHLCV for one symbol; a failure means the symbol is priced out of the reserve, never guessed."""
        try:
            df = yf.download(sym + ".NS", period="2y", auto_adjust=True, progress=False, interval="1d")
            if df is not None and not df.empty and hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna() if df is not None and not df.empty else None
        except Exception as e:
            log.warning("holdout: no daily bars for %s (%s); excluded", sym, e)
            return None

    reserve = HoldoutReserve(pipeline.sessions, reserve_bars, slippage=SLIPPAGE, cash_yield=pipeline.settings.cash_yield_pct)

    def mark_reserve():
        m = reserve.mark()
        print(
            f"holdout reserve: equity {m['equity']:,.2f} (return {m['return_pct']:.2%}), "
            f"{m['closed_trades']} closed, {m['open_positions']} open, {m['decisions']} decisions replayed"
        )

    print(
        "holdout reserve enabled: every cycle replays the approved decisions into an independent account "
        "(never tuned, never read back by the paper account)"
    )
    return mark_reserve


def run_portfolio_agent(show: bool, otp_from: str, use_ai: bool) -> int:
    """--portfolio: one read-only run of the real-account agent while the trading bot is NOT running (when it is, send
    /portfolio to it instead: one Telegram bot can have only one reader)."""
    import getpass

    settings = load_settings_or_exit()
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    telegram = Notifier(token, chat_id).send if token and chat_id else None

    def notify(text: str) -> None:
        print(strip_html(text))  # the terminal has no HTML rendering; Telegram gets the formatted version
        if telegram:
            telegram(text)

    use_ai = use_ai and bool(settings.models) and bool(os.getenv("OPENROUTER_API_KEY"))
    explain = make_explainer(LLMClient.from_settings(settings)) if use_ai else None
    result = run_standalone(
        token,
        chat_id,
        notify,
        explain,
        headless=not show,
        otp_from=otp_from,
        ask_terminal=lambda: getpass.getpass("OTP sent to your mobile (4 digits): ").strip(),
    )
    if result.get("run_dir"):
        print(f"saved under {result['run_dir']}")
    return 1 if result.get("error") else 0


def show_intraday_reports(settings: Settings) -> None:
    """--intraday-report: positions, trade history and status of each intraday paper account."""
    for title, url in (
        ("INTRADAY: opening-range breakout", intraday_db_url()),
        ("GAP-AND-GO (paper experiment)", gapgo_db_url()),
        ("GAP-ONLY (paper control for gap-and-go)", gaponly_db_url()),
        ("NEWS-CATALYST (paper experiment)", news_db_url()),
    ):
        print("=" * 60)
        print(title)
        print("=" * 60)
        print_positions(settings, url)


def run_trading_bot(live: bool, holdout: bool, loop_minutes: int | None) -> None:
    """The swing bot: build it, run the startup checks, then one cycle or a loop (`--loop`)."""
    pipeline = build_pipeline(live)
    st = pipeline.settings
    failed = preflight(pipeline)
    if failed:
        sys.exit("Startup checks failed: " + "; ".join(f"{r.name} ({r.detail})" for r in failed))
    mode = f"PAPER ORDERS ({pipeline.broker_label})" if not st.dry_run else "DRY RUN"
    scope = (
        f"whole {st.market.upper()} market (top candidates + holdings)" if pipeline.universe_fn else f"watchlist {st.watchlist}"
    )
    print(f"market={st.market.upper()} mode={mode} universe={scope}")
    print("note: news sentiment is disabled (no reliable free Indian news source)")

    reserve_callback = build_holdout_callback(pipeline) if holdout else None
    labeller = build_outcome_labeller(pipeline)
    weekly_report = WeeklyLearningReport(
        pipeline.sessions,
        pipeline.control,
        pipeline.notify,
        make_reserve=lambda: HoldoutReserve(
            pipeline.sessions, lambda symbol: fetch_daily_bars(symbol, suffix=SUFFIX), cash_yield=st.cash_yield_pct
        ),
        primary_model=st.models[0] if st.models else None,
    )

    def after_cycle() -> None:
        """Chores after each cycle: mark the holdout reserve (if enabled), label matured outcomes (once a day) and send
        the learning report (once a week)."""
        if reserve_callback is not None:
            reserve_callback()
        labeller()
        weekly_report()

    if loop_minutes:
        pipeline.notify(f"Trading bot started: {st.market.upper()} {mode}, every {loop_minutes} min while the market is open.")
        run_loop(pipeline, loop_minutes, after_cycle=after_cycle, heartbeat=Heartbeat.from_setting(st.heartbeat_file))
    else:
        run_cycle(pipeline)
        after_cycle()
