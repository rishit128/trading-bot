"""Settings and risk limits, loaded from environment variables with validation and per-market defaults."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# Free OpenRouter models that answered on 2026-10-21 (nvidia/nemotron-3-super-120b-a12b:free and nex-agi/nex-n2.5-pro:free,
# the earlier defaults, are gone). Tried after OPENROUTER_MODEL and OPENROUTER_FALLBACK_MODEL.
DEFAULT_MODELS = (
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
)
# The models that read exchange filings for the news engine, best first. Chosen on 2026-10-21 by labelling 11 real filings
# (results, orders, routine) against answers checked by hand: nemotron-3-ultra 11/11 in three runs out of three; dots-3-note
# 9-10/11 (it takes the standalone profit over the consolidated one); nemotron-3.5-lightning 9/11 with every number right.
# Unusable that day: minimax-m3.1-flash (errors on every call), gemma-4 (Google's free quota always exhausted), inkling (403).
NEWS_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3.5-lightning:free",
)


@dataclass(frozen=True)
class MarketDefaults:
    """What differs per market: the currency symbol, the scanner's price and liquidity floors, and the default watchlist."""

    currency: str
    min_price: float
    min_traded_value: float
    watchlist: tuple


MARKET_DEFAULTS = {
    "india": MarketDefaults(
        currency="₹",
        min_price=100.0,
        min_traded_value=100_000_000.0,  # Rs 10 crore/day
        watchlist=("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"),
    ),
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
    # Refuse a buy whose round-trip fees would exceed this fraction of the position (fixed per-sale charges make tiny
    # positions uneconomic: one Rs 363 share pays ~4.4%). 0 disables. Needs the broker's fee schedule (see RiskEngine).
    max_fee_drag_pct: float = 0.04
    max_open_positions: int = 10

    def __post_init__(self):
        if self.max_open_positions < 1:
            raise ValueError(f"max_open_positions must be >= 1, got {self.max_open_positions}")
        # Percent limits are fractions; a typo like MAX_POSITION_PCT=5 must fail loudly, not allow 500% positions.
        for name in (
            "max_position_pct",
            "min_position_pct",
            "max_portfolio_exposure_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "stop_loss_pct",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a fraction in (0, 1], got {getattr(self, name)}")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if not 0 <= self.max_fee_drag_pct < 1:
            raise ValueError(f"max_fee_drag_pct must be in [0, 1), got {self.max_fee_drag_pct}")
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
    # Annual rate the paper account's idle cash earns, standing in for a liquid mutual fund sweep. 0 = off (cash earns
    # nothing). The bot is only ~45% invested (10 positions x 5%), so this matters; see STRATEGY.md (2026-09-26).
    cash_yield_pct: float = 0.0
    universe: str = "market"  # "market": scan every listed NSE stock; "watchlist": only WATCHLIST
    max_candidates: int = 15
    min_price: float = 100.0
    min_traded_value: float = 100_000_000.0
    max_daily_volatility: float = 0.04
    trend_exit: bool = True  # sell a held stock when its last completed close is below its 200-day average
    research_rule_gate: bool = False  # a BUY on a stock only a broker research call put on the list needs the rule's BUY too
    market_pulse: bool = True  # also analyse stocks of surging sectors (src.data.market_pulse); False = the screener only
    # Only above the day's cross-sectional median 20-day NSE delivery %. Combined with 12-1 momentum ranking, a
    # single 3-year test added ~14%/yr over the same-window basket (STRATEGY.md, 2026-09-22) -- promising, unproven.
    # Off by default since 2026-09-26: its data (NSE's daily bhavcopy) may not be downloaded automatically under NSE's
    # terms, so it only works with files saved by hand each trading day (see README, 'Files you download').
    delivery_filter: bool = False
    analysis_workers: int = 3  # stocks analysed concurrently (LLM calls are latency-bound; keep modest for free tiers)
    # Free OpenRouter models are reasoning models: hidden thinking eats max_tokens and leaves the answer empty or cut off.
    # Switching it off gave valid answers 3-10x faster on all three configured models (src/llm.py).
    llm_reasoning_off: bool = True
    llm_min_interval: float = 3.0  # seconds between request starts (the free tier allows ~20 requests a minute)
    llm_concurrency: int = 2  # requests in flight at once
    # Sampling temperature. 0 asks for the model's single most likely answer: the live record showed the same stock on the
    # same bar answered HOLD one time and BUY the next, which turns the 0.60 confidence gate into a coin toss.
    llm_temperature: float = 0.0
    llm_cot: bool = True  # the full analysis prompt (chart, company, market) with its reasoning stored per decision
    llm_reflect: bool = True  # reflection: a veto-only critic of each BUY/SELL before it is final
    # The intraday engine (its own paper account) is sized by RISK, not by the swing settings above: a position is the
    # smaller of "the stop-out costs intraday_risk_pct of equity" and "at most intraday_max_position_pct of equity", cash
    # only (no leverage). Borrowing the swing 5% left a Rs 20,000 account unable to buy any stock above Rs 985, silently
    # skipping 63% of the breakouts the strategy had been tested on (STRATEGY.md, intraday audit 2026-09-28).
    intraday_risk_pct: float = 0.005
    intraday_max_position_pct: float = 0.20
    intraday_max_daily_loss_pct: float = 0.02  # an intraday account takes no new entries once down this much on the day
    intraday_max_positions: int = 5
    intraday_calls_act: bool = (
        False  # the breakout engine trades on the broker's calls (False: it only notes them on its signals)
    )
    angel_stream: bool = True  # stream Angel One's live prices into angel_bars (src.data.angel); needs the ANGEL_* keys
    angel_bars: str = (
        "auto"  # who serves the intraday engines' bars: auto (Angel One after 5 sessions that agreed with Yahoo) | on | off
    )
    news_engine: bool = True  # the news-catalyst paper account (src.news) runs beside the other intraday engines
    news_models: tuple = NEWS_MODELS  # the models that label filings for it, best first (NEWS_MODELS, comma list)
    intraday_universe: tuple = ()  # explicit symbols; empty = the Nifty 100 file if saved, else Yahoo's top 100 by traded value
    intraday_archive: bool = True  # save every finished session's 5-minute bars (Yahoo keeps only ~60 sessions)
    heartbeat_file: str = ""  # where the loop records each cycle's outcome for the container health check; empty = off
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
        if self.analysis_workers < 1:
            raise ValueError("analysis_workers must be >= 1")
        if self.llm_min_interval < 0 or self.llm_concurrency < 0:
            raise ValueError("llm_min_interval and llm_concurrency must be >= 0 (0 concurrency = unlimited)")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError(f"llm_temperature must be in [0, 2], got {self.llm_temperature}")
        if not 0 <= self.cash_yield_pct <= 0.15:
            raise ValueError(f"cash_yield_pct must be an annual fraction in [0, 0.15], e.g. 0.05, got {self.cash_yield_pct}")
        if not 0 < self.max_daily_volatility < 1:
            raise ValueError("max_daily_volatility must be a fraction in (0, 1), e.g. 0.04")
        for name in ("intraday_risk_pct", "intraday_max_position_pct", "intraday_max_daily_loss_pct"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a fraction in (0, 1], got {getattr(self, name)}")
        if self.intraday_max_positions < 1:
            raise ValueError(f"intraday_max_positions must be >= 1, got {self.intraday_max_positions}")
        if self.intraday_max_positions * self.intraday_max_position_pct > 1.0:
            raise ValueError("intraday_max_positions x intraday_max_position_pct cannot exceed 100% of equity (no leverage)")


def load_settings() -> Settings:
    """Read and validate settings from the environment; raises ValueError on bad values."""
    configured = [os.getenv("OPENROUTER_MODEL"), os.getenv("OPENROUTER_FALLBACK_MODEL")]
    models = tuple(dict.fromkeys(m for m in configured + list(DEFAULT_MODELS) if m))
    market = os.getenv("MARKET", "india").strip().lower()
    defaults = MARKET_DEFAULTS.get(market, MARKET_DEFAULTS["india"])
    watchlist = tuple(s.strip().upper() for s in os.getenv("WATCHLIST", ",".join(defaults.watchlist)).split(",") if s.strip())
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
        max_fee_drag_pct=_float("MAX_FEE_DRAG_PCT", 0.04),
        max_open_positions=int(_float("MAX_OPEN_POSITIONS", 10)),
    )
    return Settings(
        market=market,
        currency=defaults.currency,
        paper_initial_cash=_float("PAPER_INITIAL_CASH", 1_000_000.0),
        watchlist=watchlist,
        dry_run=os.getenv("DRY_RUN", "true").strip().lower() != "false",
        database_url=os.getenv("DATABASE_URL", "sqlite:///trading.db"),
        models=models,
        rebuy_cooldown_hours=_float("REBUY_COOLDOWN_HOURS", 24.0),
        cash_yield_pct=_float("CASH_YIELD_PCT", 0.0),
        universe=os.getenv("UNIVERSE", "market").strip().lower(),
        max_candidates=int(_float("MAX_CANDIDATES", 15)),
        min_price=_float("MIN_PRICE", defaults.min_price),
        min_traded_value=_float("MIN_TRADED_VALUE", defaults.min_traded_value),
        max_daily_volatility=_float("MAX_DAILY_VOLATILITY", 0.04),
        trend_exit=os.getenv("TREND_EXIT", "true").strip().lower() != "false",
        research_rule_gate=os.getenv("RESEARCH_RULE_GATE", "false").strip().lower() == "true",
        market_pulse=os.getenv("MARKET_PULSE", "true").strip().lower() != "false",
        delivery_filter=os.getenv("DELIVERY_FILTER", "false").strip().lower() == "true",
        analysis_workers=int(_float("ANALYSIS_WORKERS", 3)),
        llm_reasoning_off=os.getenv("LLM_REASONING_OFF", "true").strip().lower() != "false",
        llm_min_interval=_float("LLM_MIN_INTERVAL", 3.0),
        llm_concurrency=int(_float("LLM_CONCURRENCY", 2)),
        llm_temperature=_float("LLM_TEMPERATURE", 0.0),
        llm_cot=os.getenv("LLM_COT", "true").strip().lower() != "false",
        llm_reflect=os.getenv("LLM_REFLECT", "true").strip().lower() != "false",
        intraday_risk_pct=_float("INTRADAY_RISK_PCT", 0.005),
        intraday_max_position_pct=_float("INTRADAY_MAX_POSITION_PCT", 0.20),
        intraday_max_daily_loss_pct=_float("INTRADAY_MAX_DAILY_LOSS_PCT", 0.02),
        intraday_max_positions=int(_float("INTRADAY_MAX_POSITIONS", 5)),
        intraday_calls_act=os.getenv("INTRADAY_CALLS_ACT", "false").strip().lower() == "true",
        angel_stream=os.getenv("ANGEL_STREAM", "true").strip().lower() != "false",
        angel_bars=os.getenv("ANGEL_BARS", "auto").strip().lower(),
        news_engine=os.getenv("NEWS_ENGINE", "true").strip().lower() != "false",
        news_models=tuple(m.strip() for m in (os.getenv("NEWS_MODELS") or "").split(",") if m.strip()) or NEWS_MODELS,
        intraday_universe=tuple(x.strip().upper() for x in os.getenv("INTRADAY_UNIVERSE", "").split(",") if x.strip()),
        intraday_archive=os.getenv("INTRADAY_ARCHIVE", "true").strip().lower() != "false",
        heartbeat_file=os.getenv("HEARTBEAT_FILE", "").strip(),
        risk=risk,
    )
