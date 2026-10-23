"""The market's state in a few fixed words, for people and for the record: trend (Nifty against its 50- and 200-day averages),
risk (India VIX against its own year) and today's tone (how many of the streamed stocks are above yesterday's close).

Rules are fixed here and not tuned. The regime is recorded with every entry (swing and intraday) and shown by /market; it does
not gate any trade. The bot's own pre-declared test of a "no new buys below the 200-day average" gate FAILED (STRATEGY.md,
2026-10-09: it halved the all-NSE return), so a regime rule is only adopted if the recorded entries show a difference in a
later test fixed in advance."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

import pandas as pd
from sqlalchemy import select

from src.data.angel_quotes import EntryContext
from src.data.india import IST
from src.database import AngelBarRecord

log = logging.getLogger(__name__)

BULL_MOMENTUM_DAYS = 20
STRESS_VIX, STRESS_PERCENTILE, ELEVATED_PERCENTILE = 22.0, 0.90, 0.60
STRONG_BREADTH, WEAK_BREADTH = 0.60, 0.40
CACHE_SECONDS = 300.0


@dataclass(frozen=True)
class Regime:
    trend: str  # BULL | BEAR | SIDEWAYS
    risk: str  # CALM | ELEVATED | STRESS
    tone: Optional[str]  # STRONG | WEAK | MIXED, from today's breadth; None without live data
    facts: Dict[str, object]

    def short(self) -> str:
        return f"{self.trend}/{self.risk}" + (f"/{self.tone}" if self.tone else "")

    def note(self) -> Dict[str, object]:
        return {"regime": self.short(), **self.facts}

    def text(self) -> str:
        f = self.facts
        lines = [
            f"Market: {self.short()}",
            f"Nifty {f['nifty']:,.0f}: {f['vs_ma50']:+.1%} vs its 50-day, {f['vs_ma200']:+.1%} vs its 200-day, "
            f"{f['return_20d']:+.1%} over {BULL_MOMENTUM_DAYS} sessions -> {self.trend}",
        ]
        if f.get("vix") is not None:
            lines.append(f"India VIX {f['vix']:.1f} ({f['vix_percentile']:.0%} of its year's readings are lower) -> {self.risk}")
        if self.tone:
            lines.append(
                f"Today: {f['advancers']} stocks up, {f['decliners']} down ({f['breadth']:.0%} above yesterday's close) -> {self.tone}"
            )
        return "\n".join(lines)


def trend_of(close: float, ma50: float, ma200: float, return_20d: float) -> str:
    if ma200 < close and ma200 < ma50 and return_20d > 0:  # (the index's trend, not the stock rule in src.engine.rules)
        return "BULL"
    if ma50 < ma200 and ma200 > close:
        return "BEAR"
    return "SIDEWAYS"


def risk_of(vix: Optional[float], percentile: Optional[float]) -> str:
    if vix is None or percentile is None:
        return "CALM"
    if vix >= STRESS_VIX or percentile >= STRESS_PERCENTILE:
        return "STRESS"
    return "ELEVATED" if percentile >= ELEVATED_PERCENTILE else "CALM"


def tone_of(breadth: float) -> str:
    return "STRONG" if breadth >= STRONG_BREADTH else "WEAK" if breadth <= WEAK_BREADTH else "MIXED"


def live_breadth(sessions, now: datetime) -> Optional[Dict[str, float]]:
    """From the saved Angel One bars: stocks whose latest price today is above (advancers) or below (decliners) their previous
    session's last close. None until about 20 stocks have a bar today."""
    today = now.astimezone(IST).date()
    since = (today - timedelta(days=7)).isoformat()
    with sessions() as s:
        rows = s.execute(
            select(AngelBarRecord.symbol, AngelBarRecord.day, AngelBarRecord.ts, AngelBarRecord.close).where(
                AngelBarRecord.day >= since
            )
        ).all()
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=["symbol", "day", "ts", "close"]).sort_values("ts")
    last = frame.groupby(["symbol", "day"])["close"].last().unstack("day")
    if today.isoformat() not in last.columns:
        return None
    earlier = [c for c in last.columns if c < today.isoformat()]
    if not earlier:
        return None
    previous = last[earlier].ffill(axis=1).iloc[:, -1]
    both = pd.DataFrame({"now": last[today.isoformat()], "before": previous}).dropna()
    if len(both) < 20:
        return None
    up, down = int((both["now"] > both["before"]).sum()), int((both["now"] < both["before"]).sum())
    return {"advancers": up, "decliners": down, "breadth": up / len(both)}


class RegimeProvider:
    """The current Regime (recomputed at most every CACHE_SECONDS); None when the index series cannot be loaded."""

    def __init__(
        self, market, sessions=None, clock: Callable[[], float] = time.monotonic, now_fn=lambda: datetime.now(timezone.utc)
    ):
        self.market, self.sessions, self.now_fn = market, sessions, now_fn
        self.clock = clock
        self._at = -CACHE_SECONDS
        self._regime: Optional[Regime] = None

    def current(self) -> Optional[Regime]:
        if self.clock() - self._at < CACHE_SECONDS:
            return self._regime
        try:
            self._regime = self._compute()
        except Exception as e:
            log.warning("could not compute the market regime (%s: %s)", type(e).__name__, e)
            self._regime = None
        self._at = self.clock()
        return self._regime

    def _compute(self) -> Optional[Regime]:
        nifty = self.market.index_series()
        if nifty is None or len(nifty) < 200:
            return None
        close, ma50, ma200 = float(nifty.iloc[-1]), float(nifty.tail(50).mean()), float(nifty.tail(200).mean())
        ret = close / float(nifty.iloc[-1 - BULL_MOMENTUM_DAYS]) - 1
        facts: Dict[str, object] = {"nifty": close, "vs_ma50": close / ma50 - 1, "vs_ma200": close / ma200 - 1, "return_20d": ret}
        vix_series = self.market.vix_series()
        vix = percentile = None
        if vix_series is not None and len(vix_series) >= 30:
            vix, percentile = float(vix_series.iloc[-1]), float(vix_series.tail(252).rank(pct=True).iloc[-1])
        facts.update(vix=vix, vix_percentile=percentile)
        tone = None
        breadth = live_breadth(self.sessions, self.now_fn()) if self.sessions is not None else None
        if breadth:
            facts.update(breadth)
            tone = tone_of(breadth["breadth"])
        return Regime(trend_of(close, ma50, ma200, ret), risk_of(vix, percentile), tone, facts)


def with_regime(
    context_fn: Callable[[str, float], EntryContext], provider: RegimeProvider
) -> Callable[[str, float], EntryContext]:
    """`context_fn` with the market's regime added to what is recorded with each entry (record only: it changes no decision)."""

    def wrapped(symbol: str, price: float) -> EntryContext:
        context, regime = context_fn(symbol, price), provider.current()
        return EntryContext(context.skip, {**context.note, "market": regime.short()} if regime else context.note)

    return wrapped
