"""Command-line entry point: builds the market kit, agents and pipeline, then runs one cycle, a loop, or a utility command."""
import argparse
import dataclasses
import logging
import os
import sys

from src.agents.agents import SentimentAgent, TechnicalAgent
from src.agents.history import history_stats
from src.app.intraday_cli import run_intraday
from src.app.kit import build_kit, intraday_db_url, make_paper_broker
from src.app.reports import print_candidates, print_graphs, print_positions, print_report
from src.config import load_settings
from src.control import Control
from src.data.market_context import MarketContextProvider
from src.database import make_session_factory
from src.intraday.strategy import intraday_fees
from src.llm import LLMClient
from src.logging_setup import configure_logging
from src.monitoring.telegram import CommandListener, Notifier, handle_command
from src.pipeline import TradingPipeline
from src.preflight import build_checks, critical_failures, format_results, run_checks
from src.runner import run_cycle, run_loop

log = logging.getLogger(__name__)


def build_pipeline(live: bool) -> TradingPipeline:
    """Validate configuration and keys, then wire the LLM, agents, broker, Telegram and the LangGraph pipeline."""
    try:
        settings = load_settings()
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    if live:
        settings = dataclasses.replace(settings, dry_run=False)
    if not settings.models:
        sys.exit("OPENROUTER_MODEL is not set in .env")
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Missing in .env: OPENROUTER_API_KEY")

    llm = LLMClient(settings.models, cache_ttl_seconds=settings.llm_cache_hours * 3600)
    sessions = make_session_factory(settings.database_url)
    control = Control(sessions)
    kit = build_kit(settings, sessions)
    intraday_broker = None
    try:
        intraday_broker = make_paper_broker(settings, intraday_db_url(), fees=intraday_fees)
    except Exception as e:  # the intraday view is optional; never stop the swing bot for it
        print(f"intraday status commands unavailable: {type(e).__name__}")
    notify = None
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        notify = Notifier(token, chat_id).send
        CommandListener(token, chat_id, lambda text: handle_command(text, control, kit.broker, settings, intraday_broker)).start()
    else:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); alerts and /pause disabled")
    def make_history_fn(sessions):
        """Phase 2 hook: closed paper-trade stats for setups similar to the snapshot; failures mean no adjustment."""

        def history(snapshot):
            try:
                return history_stats(sessions, snapshot)
            except Exception as e:
                log.warning("history lookup failed for %s: %s: %s", snapshot.symbol, type(e).__name__, e)
                return None

        return history

    market_provider = MarketContextProvider()

    def market_context(snapshot):
        """Phase 3 hook: index regime / VIX for the stock; a failure means no context adjustment."""
        try:
            return market_provider.context(snapshot.symbol, snapshot)
        except Exception as e:
            log.warning("market context failed for %s: %s: %s", snapshot.symbol, type(e).__name__, e)
            return None

    agents = [TechnicalAgent(
        llm, use_cot=settings.llm_cot, use_learning=settings.llm_learning, use_context=settings.llm_context,
        use_reflect=settings.llm_reflect, history_fn=make_history_fn(sessions),
        max_adjustment=settings.llm_max_adjust,
    )] + ([SentimentAgent(llm)] if kit.use_news else [])
    pipeline = TradingPipeline(
        settings, agents, kit.broker, sessions, kit.snapshot_fn, kit.headlines_fn, control=control, notify=notify,
        universe_fn=kit.screener.symbols_for if kit.screener else None, quote_fn=kit.quote_fn,
        market_context_fn=market_context,
    )
    pipeline.broker_label = kit.broker_label
    return pipeline


def preflight(pipeline) -> list:
    """Run the connectivity checks and print them. Returns the critical failures (empty = safe to start)."""
    try:
        settings = pipeline.settings if pipeline is not None else load_settings()
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    broker = pipeline.broker if pipeline is not None else build_kit(settings, make_session_factory(settings.database_url)).broker
    results = run_checks(build_checks(settings, broker, os.environ))
    print("Startup checks:\n" + format_results(results))
    return critical_failures(results)


def build_holdout_callback(pipeline):
    """The per-cycle hook for --holdout: replays the approved decisions into the independent reserve and prints a mark."""
    from src.engine.convention import SLIPPAGE
    from src.holdout import HoldoutReserve

    import yfinance as yf

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

    reserve = HoldoutReserve(pipeline.sessions, reserve_bars, slippage=SLIPPAGE)

    def mark_reserve():
        m = reserve.mark()
        print(f"holdout reserve: equity {m['equity']:,.2f} (return {m['return_pct']:.2%}), "
              f"{m['closed_trades']} closed, {m['open_positions']} open, {m['decisions']} decisions replayed")

    print("holdout reserve enabled: every cycle replays the approved decisions into an independent account "
          "(never tuned, never read back by the paper account)")
    return mark_reserve


def main() -> None:
    """Parse arguments and run the requested command."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="AI trading bot (Indian NSE stocks, paper trading).")
    parser.add_argument("--loop", type=int, metavar="MINUTES", help="repeat every N minutes while the market is open (N >= 1)")
    parser.add_argument("--screen", action="store_true", help="only scan the whole market and print the candidates")
    parser.add_argument("--graph", action="store_true", help="print the LangGraph decision workflow as a Mermaid diagram")
    parser.add_argument("--check", action="store_true", help="test keys and connectivity (OpenRouter, broker, data, Telegram) and exit")
    parser.add_argument("--report", action="store_true", help="print the paper-trading account summary (India simulator)")
    parser.add_argument("--positions", action="store_true", help="every open position (P&L, buy date), closed-trade history, overall status")
    parser.add_argument("--intraday", action="store_true", help="run the intraday breakout engine (own paper account; add --live to place paper orders)")
    parser.add_argument("--intraday-report", action="store_true", help="positions, trade history and status of the intraday paper account")
    parser.add_argument("--live", action="store_true", help="place paper orders (default: dry run, decisions only)")
    parser.add_argument("--holdout", action="store_true",
                        help="after each cycle, mark the independent holdout reserve account (see docs/holdout.md)")
    args = parser.parse_args()
    if args.loop is not None and args.loop < 1:
        parser.error("--loop must be at least 1 minute")

    try:
        configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FORMAT", "text"))
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    if args.intraday:
        run_intraday(args.live)
        return
    if args.intraday_report:
        try:
            settings = load_settings()
        except ValueError as e:
            sys.exit(f"Invalid configuration: {e}")
        print_positions(settings, intraday_db_url())
        return
    if args.screen or args.report or args.graph or args.positions:
        try:
            settings = load_settings()
        except ValueError as e:
            sys.exit(f"Invalid configuration: {e}")
        (print_graphs if args.graph else print_candidates if args.screen else print_positions if args.positions else print_report)(settings)
        return

    if args.check:
        sys.exit(0 if not preflight(None) else 1)

    pipeline = build_pipeline(args.live)
    st = pipeline.settings
    failed = preflight(pipeline)
    if failed:
        sys.exit("Startup checks failed: " + "; ".join(f"{r.name} ({r.detail})" for r in failed))
    mode = f"PAPER ORDERS ({pipeline.broker_label})" if not st.dry_run else "DRY RUN"
    scope = f"whole {st.market.upper()} market (top candidates + holdings)" if pipeline.universe_fn else f"watchlist {st.watchlist}"
    print(f"market={st.market.upper()} mode={mode} universe={scope}")
    print("note: news sentiment is disabled (no reliable free Indian news source)")

    reserve_callback = build_holdout_callback(pipeline) if args.holdout else None

    if args.loop:
        pipeline.notify(f"Trading bot started: {st.market.upper()} {mode}, every {args.loop} min while the market is open.")
        run_loop(pipeline, args.loop, after_cycle=reserve_callback)
    else:
        run_cycle(pipeline)
        if reserve_callback is not None:
            reserve_callback()


if __name__ == "__main__":
    main()
