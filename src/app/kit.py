"""Wiring shared by the command-line entry point and the reports: the market kit and the paper-broker factory."""
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from src.config import Settings
from src.data.market_data import fetch_snapshot
from src.data.universe import ScreenConfig, UniverseScreener


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
    return ScreenConfig(settings.max_candidates, settings.min_price, settings.min_traded_value, settings.max_daily_volatility,
                        affordable_pct=settings.risk.min_position_pct)


def build_india_screener(settings: Settings) -> UniverseScreener:
    """Whole-NSE scanner backed by NSE's stock list and Yahoo bars."""
    from src.data.india import fetch_nse_symbols, yf_bar_fetcher

    return UniverseScreener(fetch_nse_symbols, yf_bar_fetcher(), screen_config(settings), chunk=200,
                            use_delivery_filter=settings.delivery_filter)


def intraday_db_url() -> str:
    """The intraday paper account lives in its own database so swing and intraday results never mix."""
    return os.getenv("INTRADAY_DATABASE_URL", "sqlite:///intraday.db")


def make_paper_broker(settings: Settings, database_url: Optional[str] = None, fees: Optional[Callable] = None):
    """A PaperBroker on the given database (default: the swing account) with live NSE prices and the IST market clock."""
    from src.data.india import IndiaClock, IntradayFeed
    from src.database import make_session_factory
    from src.engine.paper_broker import PaperBroker, india_delivery_fees

    return PaperBroker(make_session_factory(database_url or settings.database_url), IntradayFeed(), IndiaClock(),
                       initial_cash=settings.paper_initial_cash, fees=fees or india_delivery_fees)


def build_kit(settings: Settings, sessions) -> MarketKit:
    """Assemble the broker and data functions for the configured market."""
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
