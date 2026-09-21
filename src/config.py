"""Settings and risk limits, loaded from environment variables with validation and per-market defaults."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "dots-studio/dots-3-note-preview:free",
    "nex-agi/nex-n2.5-pro:free",
)


MARKET_DEFAULTS = {
    "india": dict(currency="₹", min_price=100.0, min_traded_value=100_000_000.0,  # Rs 10 crore/day
                  watchlist=("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")),
    "us": dict(currency="$", min_price=10.0, min_traded_value=20_000_000.0,
               watchlist=("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA")),
}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk limits; validated so a typo (for example 5 instead of 0.05) refuses to start."""
    max_position_pct: float = 0.05
    max_portfolio_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.20
    min_confidence: float = 0.60
    # 2%/5% (the original plan) was stopped out within ~1 day by normal noise and lost money in every window tested;
    # a wide protective stop with a rule-based trend exit did best in both halves of a 9-year test (see STRATEGY.md).
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 1.00  # effectively no target: winners run until the trend exit
    max_open_positions: int = 10

    def __post_init__(self):
        if self.max_open_positions < 1:
            raise ValueError(f"max_open_positions must be >= 1, got {self.max_open_positions}")
        # Percent limits are fractions; a typo like MAX_POSITION_PCT=5 must fail loudly, not allow 500% positions.
        for name in ("max_position_pct", "max_portfolio_exposure_pct", "max_daily_loss_pct",
                     "max_drawdown_pct", "stop_loss_pct"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a fraction in (0, 1], got {getattr(self, name)}")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")
        if self.max_position_pct > self.max_portfolio_exposure_pct:
            raise ValueError("max_position_pct cannot exceed max_portfolio_exposure_pct")


@dataclass(frozen=True)
class Settings:
    """All runtime settings for one run."""
    market: str = "india"  # "india": NSE via Yahoo + built-in paper broker; "us": Alpaca paper trading
    currency: str = "₹"
    paper_initial_cash: float = 1_000_000.0  # India paper account starting balance
    watchlist: tuple = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
    dry_run: bool = True
    database_url: str = "sqlite:///trading.db"
    models: tuple = ()
    rebuy_cooldown_hours: float = 24.0
    universe: str = "market"  # "market": scan every listed US stock; "watchlist": only WATCHLIST
    max_candidates: int = 15
    min_price: float = 100.0
    min_traded_value: float = 100_000_000.0
    max_daily_volatility: float = 0.04
    auto_protect: bool = True  # place a protective stop on any position found without one (live orders only)
    trend_exit: bool = True  # sell a held stock when its last completed close is below its 200-day average
    analysis_workers: int = 3  # stocks analysed concurrently (LLM calls are latency-bound; keep modest for free tiers)
    llm_cache_hours: float = 12.0
    data_feed: str = "sip"
    risk: RiskLimits = field(default_factory=RiskLimits)

    def __post_init__(self):
        if self.market not in MARKET_DEFAULTS:
            raise ValueError(f"market must be one of {sorted(MARKET_DEFAULTS)}, got {self.market!r}")
        if self.paper_initial_cash <= 0:
            raise ValueError("paper_initial_cash must be > 0")
        if self.universe not in ("market", "watchlist"):
            raise ValueError(f"universe must be 'market' or 'watchlist', got {self.universe!r}")
        if self.data_feed not in ("sip", "iex"):
            raise ValueError(f"data_feed must be 'sip' or 'iex', got {self.data_feed!r}")
        if self.max_candidates < 1 or self.min_price < 0 or self.min_traded_value < 0:
            raise ValueError("max_candidates must be >= 1 and min_price/min_traded_value must be >= 0")
        if self.analysis_workers < 1 or self.llm_cache_hours < 0:
            raise ValueError("analysis_workers must be >= 1 and llm_cache_hours must be >= 0")
        if not 0 < self.max_daily_volatility < 1:
            raise ValueError("max_daily_volatility must be a fraction in (0, 1), e.g. 0.04")


def load_settings() -> Settings:
    """Read and validate settings from the environment; raises ValueError on bad values."""
    configured = [os.getenv("OPENROUTER_MODEL"), os.getenv("OPENROUTER_FALLBACK_MODEL")]
    models = tuple(dict.fromkeys(m for m in configured + list(DEFAULT_MODELS) if m))
    market = os.getenv("MARKET", "india").strip().lower()
    defaults = MARKET_DEFAULTS.get(market, MARKET_DEFAULTS["india"])
    watchlist = tuple(
        s.strip().upper() for s in os.getenv("WATCHLIST", ",".join(defaults["watchlist"])).split(",") if s.strip()
    )
    risk = RiskLimits(
        max_position_pct=_float("MAX_POSITION_PCT", 0.05),
        max_portfolio_exposure_pct=_float("MAX_PORTFOLIO_EXPOSURE_PCT", 0.80),
        max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.02),
        max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 0.20),
        min_confidence=_float("MIN_CONFIDENCE", 0.60),
        stop_loss_pct=_float("STOP_LOSS_PCT", 0.08),
        take_profit_pct=_float("TAKE_PROFIT_PCT", 1.00),
        max_open_positions=int(_float("MAX_OPEN_POSITIONS", 10)),
    )
    return Settings(
        market=market,
        currency=defaults["currency"],
        paper_initial_cash=_float("PAPER_INITIAL_CASH", 1_000_000.0),
        watchlist=watchlist,
        dry_run=os.getenv("DRY_RUN", "true").strip().lower() != "false",
        database_url=os.getenv("DATABASE_URL", "sqlite:///trading.db"),
        models=models,
        rebuy_cooldown_hours=_float("REBUY_COOLDOWN_HOURS", 24.0),
        universe=os.getenv("UNIVERSE", "market").strip().lower(),
        max_candidates=int(_float("MAX_CANDIDATES", 15)),
        min_price=_float("MIN_PRICE", defaults["min_price"]),
        min_traded_value=_float("MIN_TRADED_VALUE", defaults["min_traded_value"]),
        max_daily_volatility=_float("MAX_DAILY_VOLATILITY", 0.04),
        trend_exit=os.getenv("TREND_EXIT", "true").strip().lower() != "false",
        auto_protect=os.getenv("AUTO_PROTECT", "true").strip().lower() != "false",
        analysis_workers=int(_float("ANALYSIS_WORKERS", 3)),
        llm_cache_hours=_float("LLM_CACHE_HOURS", 12.0),
        data_feed=os.getenv("DATA_FEED", "sip").strip().lower(),
        risk=risk,
    )
