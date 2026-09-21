"""Command-line entry point: builds the market kit, agents and pipeline, then runs one cycle, a loop, or a utility command."""
import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

from src.agents.agents import SentimentAgent, TechnicalAgent
from src.config import Settings, load_settings
from src.control import Control
from src.data.market_data import fetch_headlines, fetch_snapshot
from src.data.universe import ScreenConfig, UniverseScreener, alpaca_bar_fetcher, list_tradable_stocks
from src.database import make_session_factory
from src.llm import LLMClient
from src.logging_setup import configure_logging
from src.monitoring.telegram import CommandListener, Notifier, handle_command
from src.pipeline import TradingPipeline
from src.preflight import build_checks, critical_failures, format_results, run_checks
from src.runner import run_cycle, run_loop
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph


@dataclass
class MarketKit:
    """Everything that differs per market: broker, data functions, universe scanner, and whether news sentiment is available."""
    broker: object
    snapshot_fn: Callable
    headlines_fn: Callable[[str], Sequence[str]]
    screener: Optional[UniverseScreener]
    broker_label: str
    use_news: bool = True  # registers the sentiment agent; off where no reliable news source exists
    quote_fn: Optional[Callable[[str], float]] = None  # live price at execution time


def screen_config(settings: Settings) -> ScreenConfig:
    """Scanner thresholds taken from settings."""
    return ScreenConfig(settings.max_candidates, settings.min_price, settings.min_traded_value, settings.max_daily_volatility)


def build_us_screener(settings: Settings) -> UniverseScreener:
    """Whole-US-market scanner backed by Alpaca assets and bars."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    trading, data = TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)

    def symbols():
        """List tradable US common stocks from Alpaca."""
        return list_tradable_stocks(
            trading.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
        )

    return UniverseScreener(symbols, alpaca_bar_fetcher(data, settings.data_feed), screen_config(settings))


def build_india_screener(settings: Settings) -> UniverseScreener:
    """Whole-NSE scanner backed by NSE's stock list and Yahoo bars."""
    from src.data.india import fetch_nse_symbols, yf_bar_fetcher

    return UniverseScreener(fetch_nse_symbols, yf_bar_fetcher(), screen_config(settings), chunk=200)


def build_kit(settings: Settings, sessions) -> MarketKit:
    """Assemble the broker and data functions for the configured market."""
    if settings.market == "us":
        from src.engine.broker import AlpacaBroker

        screener = build_us_screener(settings) if settings.universe == "market" else None
        return MarketKit(AlpacaBroker(), fetch_snapshot, fetch_headlines, screener, "Alpaca paper")

    from src.data.india import SUFFIX, IndiaClock, IntradayFeed
    from src.engine.paper_broker import PaperBroker

    clock, feed = IndiaClock(), IntradayFeed()
    broker = PaperBroker(sessions, feed, clock, initial_cash=settings.paper_initial_cash)
    screener = build_india_screener(settings) if settings.universe == "market" else None
    # Analyse the last COMPLETED session only (stable prompts all day -> cacheable, and matches the backtest), then
    # size/bracket the order off the live price. No reliable free Indian news source, so no sentiment agent.
    return MarketKit(
        broker, lambda s: fetch_snapshot(s, suffix=SUFFIX, as_of=clock.last_completed_session), lambda s: [], screener,
        "built-in paper simulator", use_news=False, quote_fn=feed.last_price,
    )


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
    required = ["OPENROUTER_API_KEY"] + (["ALPACA_API_KEY", "ALPACA_SECRET_KEY"] if settings.market == "us" else [])
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")

    llm = LLMClient(settings.models, cache_ttl_seconds=settings.llm_cache_hours * 3600)
    sessions = make_session_factory(settings.database_url)
    control = Control(sessions)
    kit = build_kit(settings, sessions)
    notify = None
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        notify = Notifier(token, chat_id).send
        CommandListener(token, chat_id, lambda text: handle_command(text, control, kit.broker, settings)).start()
    else:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); alerts and /pause disabled")
    agents = [TechnicalAgent(llm)] + ([SentimentAgent(llm)] if kit.use_news else [])
    pipeline = TradingPipeline(
        settings, agents, kit.broker, sessions, kit.snapshot_fn, kit.headlines_fn, control=control, notify=notify,
        universe_fn=kit.screener.symbols_for if kit.screener else None, quote_fn=kit.quote_fn,
    )
    pipeline.broker_label = kit.broker_label
    return pipeline


def print_graphs(settings: Settings) -> None:
    """Print the three LangGraph workflows as Mermaid diagrams."""
    names = ["technical"] + (["sentiment"] if settings.market == "us" else [])
    stub = SimpleNamespace(agents={n: None for n in names})
    analysis, decision = build_analysis_graph(stub), build_decision_graph(stub)
    for title, graph in (("CYCLE: scan -> analyse all stocks in parallel -> decide each in order", build_cycle_graph(stub, analysis, decision)),
                         (f"ANALYSIS (per stock): agents {names} run in parallel", analysis),
                         ("DECISION (per stock): combine -> risk -> execute", decision)):
        print(f"%% {title}")
        print(graph.get_graph().draw_mermaid())


def print_candidates(settings: Settings) -> None:
    """Print today's scanner candidates."""
    screener = build_india_screener(settings) if settings.market == "india" else build_us_screener(settings)
    cur = settings.currency
    for c in screener.candidates():
        print(f"{c.symbol:12s} price={cur}{c.price:10,.2f} 3m={c.momentum_63d:+.1%} vol/day={c.daily_volatility:.1%} "
              f"rsi={c.rsi:4.1f} traded/day={cur}{c.avg_traded_value / 1e6:9,.1f}M")


def print_report(settings: Settings) -> None:
    """Print the paper-trading account summary (India simulator)."""
    if settings.market != "india":
        sys.exit("--report covers the built-in paper simulator (MARKET=india); for US see your Alpaca paper dashboard.")
    from src.data.india import IndiaClock, IntradayFeed
    from src.engine.paper_broker import PaperBroker

    s = PaperBroker(make_session_factory(settings.database_url), IntradayFeed(), IndiaClock(),
                    initial_cash=settings.paper_initial_cash).summary()
    cur = settings.currency
    win = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "n/a"
    print(f"Paper account ({settings.market.upper()}): started {cur}{s['initial_cash']:,.0f} -> equity {cur}{s['equity']:,.0f} "
          f"({s['return_pct']:+.2%})\ncash {cur}{s['cash']:,.0f} | open positions {s['open_positions']} | closed trades "
          f"{s['closed_trades']} (win rate {win}) | realized net P&L {cur}{s['realized_net_pnl']:,.0f} | fees paid "
          f"{cur}{s['fees_paid']:,.0f}\nexits: {s['exits']}")


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


def main() -> None:
    """Parse arguments and run the requested command."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="AI trading bot (India NSE by default; set MARKET=us for US stocks).")
    parser.add_argument("--loop", type=int, metavar="MINUTES", help="repeat every N minutes while the market is open (N >= 1)")
    parser.add_argument("--screen", action="store_true", help="only scan the whole market and print the candidates")
    parser.add_argument("--graph", action="store_true", help="print the LangGraph decision workflow as a Mermaid diagram")
    parser.add_argument("--check", action="store_true", help="test keys and connectivity (OpenRouter, broker, data, Telegram) and exit")
    parser.add_argument("--report", action="store_true", help="print the paper-trading account summary (India simulator)")
    parser.add_argument("--live", action="store_true", help="place paper orders (default: dry run, decisions only)")
    args = parser.parse_args()
    if args.loop is not None and args.loop < 1:
        parser.error("--loop must be at least 1 minute")

    try:
        configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FORMAT", "text"))
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")
    if args.screen or args.report or args.graph:
        try:
            settings = load_settings()
        except ValueError as e:
            sys.exit(f"Invalid configuration: {e}")
        (print_graphs if args.graph else print_candidates if args.screen else print_report)(settings)
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
    if st.market == "india":
        print("note: news sentiment is disabled for India (no reliable free news source)")

    if args.loop:
        pipeline.notify(f"Trading bot started: {st.market.upper()} {mode}, every {args.loop} min while the market is open.")
        run_loop(pipeline, args.loop)
    else:
        run_cycle(pipeline)


if __name__ == "__main__":
    main()
