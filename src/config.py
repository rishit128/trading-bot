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
}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk limits; validated so a typo (for example 5 instead of 0.05) refuses to start."""
    # Position size scales linearly with signal confidence: min_position_pct at min_confidence, max_position_pct at
    # confidence 1.0. A flat size treated a 0.61 and a 0.95 signal alike; this puts more capital behind stronger signals.
    # Defaults are equal (flat 5% regardless of confidence) so existing configs are unaffected; set MIN_POSITION_PCT
    # below MAX_POSITION_PCT to enable confidence-scaled sizing.
    min_position_pct: float = 0.05
    max_position_pct: float = 0.05
    max_portfolio_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.20
    # Calendar days new buys stay halted after the drawdown limit is hit, after which the peak is rebased and trading
    # resumes (see engine.risk_engine.drawdown_pause). 0 = the halt never ends by itself (only /rebase clears it).
    drawdown_pause_days: float = 30.0
    min_confidence: float = 0.60
    # 2%/5% (the original plan) was stopped out within ~1 day by normal noise and lost money in every window tested.
    # On the live scanner's momentum stocks even 8% was too tight: across four sizing setups and both halves of a 9-year
    # test, 15% beat 8% on return and profit factor, and no stop beat both (STRATEGY.md, 2026-09-25). 15% is kept as a
    # safety net against gaps and data outages rather than as the exit; the MA200 trend exit does the real work.
    stop_loss_pct: float = 0.15
    take_profit_pct: float = 1.00  # effectively no target: winners run until the trend exit
    max_open_positions: int = 10

    def __post_init__(self):
        if self.max_open_positions < 1:
            raise ValueError(f"max_open_positions must be >= 1, got {self.max_open_positions}")
        # Percent limits are fractions; a typo like MAX_POSITION_PCT=5 must fail loudly, not allow 500% positions.
        for name in ("max_position_pct", "min_position_pct", "max_portfolio_exposure_pct", "max_daily_loss_pct",
                     "max_drawdown_pct", "stop_loss_pct"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a fraction in (0, 1], got {getattr(self, name)}")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.drawdown_pause_days < 0:
            raise ValueError(f"drawdown_pause_days must be >= 0, got {self.drawdown_pause_days}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")
        if self.min_position_pct > self.max_position_pct:
            raise ValueError("min_position_pct cannot exceed max_position_pct")
        if self.max_position_pct > self.max_portfolio_exposure_pct:
            raise ValueError("max_position_pct cannot exceed max_portfolio_exposure_pct")


@dataclass(frozen=True)
class Settings:
    """All runtime settings for one run."""
    market: str = "india"  # NSE via Yahoo + built-in paper broker
    currency: str = "₹"
    paper_initial_cash: float = 1_000_000.0  # India paper account starting balance
    watchlist: tuple = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
    dry_run: bool = True
    database_url: str = "sqlite:///trading.db"
    models: tuple = ()
    rebuy_cooldown_hours: float = 24.0
    universe: str = "market"  # "market": scan every listed NSE stock; "watchlist": only WATCHLIST
    max_candidates: int = 15
    min_price: float = 100.0
    min_traded_value: float = 100_000_000.0
    max_daily_volatility: float = 0.04
    auto_protect: bool = True  # place a protective stop on any position found without one (live orders only)
    trend_exit: bool = True  # sell a held stock when its last completed close is below its 200-day average
    # Only above the day's cross-sectional median 20-day NSE delivery %. Combined with 12-1 momentum ranking, a
    # single 3-year test added ~14%/yr over the same-window basket (STRATEGY.md, 2026-09-22) -- promising, unproven.
    delivery_filter: bool = True
    analysis_workers: int = 3  # stocks analysed concurrently (LLM calls are latency-bound; keep modest for free tiers)
    llm_cache_hours: float = 12.0
    llm_cot: bool = True  # chain-of-thought prompting: 5-step reasoning chain stored with each decision (Phase 1)
    llm_learning: bool = True  # Phase 2: adjust from closed paper-trade record of similar setups
    llm_context: bool = True  # Phase 3: adjust when the market regime is notable (risk-off / euphoric)
    llm_reflect: bool = True  # Phase 4: self-critique of the assembled call before it is final
    llm_max_adjust: float = 0.15  # largest confidence change one refinement phase may make (review doc 2.4)
    risk: RiskLimits = field(default_factory=RiskLimits)

    def __post_init__(self):
        if self.market not in MARKET_DEFAULTS:
            raise ValueError(f"market must be one of {sorted(MARKET_DEFAULTS)}, got {self.market!r}")
        if self.paper_initial_cash <= 0:
            raise ValueError("paper_initial_cash must be > 0")
        if self.universe not in ("market", "watchlist"):
            raise ValueError(f"universe must be 'market' or 'watchlist', got {self.universe!r}")
        if self.max_candidates < 1 or self.min_price < 0 or self.min_traded_value < 0:
            raise ValueError("max_candidates must be >= 1 and min_price/min_traded_value must be >= 0")
        if self.analysis_workers < 1 or self.llm_cache_hours < 0:
            raise ValueError("analysis_workers must be >= 1 and llm_cache_hours must be >= 0")
        if not 0 <= self.llm_max_adjust <= 1:
            raise ValueError("llm_max_adjust must be a fraction in [0, 1]")
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
        min_position_pct=_float("MIN_POSITION_PCT", _float("MAX_POSITION_PCT", 0.05)),
        max_position_pct=_float("MAX_POSITION_PCT", 0.05),
        max_portfolio_exposure_pct=_float("MAX_PORTFOLIO_EXPOSURE_PCT", 0.80),
        max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.02),
        max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 0.20),
        drawdown_pause_days=_float("DRAWDOWN_PAUSE_DAYS", 30.0),
        min_confidence=_float("MIN_CONFIDENCE", 0.60),
        stop_loss_pct=_float("STOP_LOSS_PCT", 0.15),
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
        delivery_filter=os.getenv("DELIVERY_FILTER", "true").strip().lower() != "false",
        auto_protect=os.getenv("AUTO_PROTECT", "true").strip().lower() != "false",
        analysis_workers=int(_float("ANALYSIS_WORKERS", 3)),
        llm_cache_hours=_float("LLM_CACHE_HOURS", 12.0),
        llm_cot=os.getenv("LLM_COT", "true").strip().lower() != "false",
        llm_learning=os.getenv("LLM_LEARNING", "true").strip().lower() != "false",
        llm_context=os.getenv("LLM_CONTEXT", "true").strip().lower() != "false",
        llm_reflect=os.getenv("LLM_REFLECT", "true").strip().lower() != "false",
        llm_max_adjust=_float("LLM_MAX_ADJUST", 0.15),
        risk=risk,
    )
