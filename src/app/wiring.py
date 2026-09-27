"""Wiring shared by the command-line entry point and the reports: the market wiring and the paper-broker factory."""
import os
from dataclasses import dataclass
from typing import Callable, Optional

from src.config import Settings
from src.engine.ports import Broker
from src.data.market_data import cached_per_session, fetch_snapshot
from src.data.universe import ScreenConfig, UniverseScreener


@dataclass
class MarketWiring:
    """Everything that differs per market: broker, data functions and universe scanner."""
    broker: Broker
    snapshot_fn: Callable
    screener: Optional[UniverseScreener]
    broker_label: str
    quote_fn: Optional[Callable[[str], float]] = None  # live price at execution time


def screen_config(settings: Settings) -> ScreenConfig:
    """Scanner thresholds taken from settings."""
    return ScreenConfig(settings.max_candidates, settings.min_price, settings.min_traded_value, settings.max_daily_volatility,
                        affordable_pct=settings.risk.min_position_pct)


def build_india_screener(settings: Settings) -> UniverseScreener:
    """Whole-NSE scanner: the stock list from Yahoo (or a hand-saved NSE list), bars from Yahoo. The list is pre-cut to
    stocks Yahoo shows trading at least half the liquidity floor, so the daily bar download stays small."""
    from src.data.india import load_nse_symbols, yf_bar_fetcher

    return UniverseScreener(lambda: load_nse_symbols(min_traded_value=settings.min_traded_value), yf_bar_fetcher(),
                            screen_config(settings), chunk=200, use_delivery_filter=settings.delivery_filter)


def intraday_db_url() -> str:
    """The intraday paper account lives in its own database so swing and intraday results never mix."""
    return os.getenv("INTRADAY_DATABASE_URL", "sqlite:///intraday.db")


def make_paper_broker(settings: Settings, database_url: Optional[str] = None, fees: Optional[Callable] = None):
    """A PaperBroker on the given database (default: the swing account) with live NSE prices and the IST market clock.

    Only the swing account sweeps idle cash (CASH_YIELD_PCT): the intraday account is flat every night, so a liquid-fund
    sweep does not describe it. The report commands build the swing broker through here too, so they credit interest
    the same way the running bot does."""
    from src.data.india import IndiaClock, IntradayFeed
    from src.database import make_session_factory
    from src.engine.paper_broker import PaperBroker
    from src.engine.costs import india_delivery_fees

    url = database_url or settings.database_url
    return PaperBroker(make_session_factory(url), IntradayFeed(), IndiaClock(),
                       initial_cash=settings.paper_initial_cash, fees=fees or india_delivery_fees,
                       cash_yield=settings.cash_yield_pct if url == settings.database_url else 0.0)


def build_market_wiring(settings: Settings, sessions) -> MarketWiring:
    """Assemble the broker and data functions for the configured market."""
    from src.data.india import SUFFIX, IndiaClock, IntradayFeed
    from src.engine.paper_broker import PaperBroker

    clock, feed = IndiaClock(), IntradayFeed()
    broker = PaperBroker(sessions, feed, clock, initial_cash=settings.paper_initial_cash,
                         cash_yield=settings.cash_yield_pct)
    screener = build_india_screener(settings) if settings.universe == "market" else None
    # Analyse the last COMPLETED session only (stable prompts all day -> cacheable, and matches the backtest), then
    # size/bracket the order off the live price.
    return MarketWiring(
        broker, cached_per_session(lambda s: fetch_snapshot(s, suffix=SUFFIX, as_of=clock.last_completed_session),
                                   clock.last_completed_session),
        screener, "built-in paper simulator", quote_fn=feed.last_price,
    )
