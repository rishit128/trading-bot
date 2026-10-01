"""Replay stored 5-minute bars through the SAME engine and paper broker that trade live.

A backtest with its own simulator is a different system from the one that trades: the old intraday backtest had separate
sizing (it ignored that a Rs 20,000 account cannot buy a Rs 3,000 share), separate exit logic and a different universe, and
its conclusions did not describe the live results. Here nothing is re-implemented: `IntradayEngine.step()` runs every five
minutes of every historical session against a `PaperBroker`, so entries, sizing, skips, stops, targets, the daily loss halt,
the square-off, fees and slippage are the live code paths.

What the replay must assume (Yahoo's live behaviour cannot be reproduced from finished bars):
  * at each cycle (15 seconds past a bar boundary) the bar in progress shows only its OPEN, as if ~20 seconds of it had
    traded; finished bars are complete;
  * an order fills at that open plus the broker's 0.05% slippage, like the live fill at the latest price;
  * stop and target orders fill at their level (or the open on a gap): no circuit limits, no partial fills;
  * volume is the final volume of each bar, whereas live sees it settle (Yahoo revises recent bars)."""
import tempfile
from dataclasses import dataclass, field
from datetime import date as Date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.config import RiskLimits, Settings
from src.data.india import IST, IndiaClock
from src.data.intraday_bars import COLUMNS
from src.database import IntradaySignalRecord, make_session_factory
from src.engine.costs import SLIPPAGE
from src.engine.paper_broker import PaperBroker
from src.engine.ports import BAR
from src.intraday.engine import IntradayEngine
from src.intraday.strategy import intraday_fees

FIRST_CYCLE, LAST_CYCLE = dtime(9, 20, 15), dtime(15, 25, 15)  # the engine's cycles, 15 s past each 5-minute boundary


class ReplayFeed:
    """A price feed over stored bars, showing what Yahoo would have shown at `self.now` (see the module docstring)."""

    def __init__(self, bars: Dict[str, pd.DataFrame]):
        self.now = datetime(1970, 1, 1, tzinfo=timezone.utc)
        frames = {s: df.sort_index() for s, df in bars.items() if len(df)}
        self._index = {s: pd.DatetimeIndex(df.index) for s, df in frames.items()}
        self._values = {s: df[COLUMNS].to_numpy(float) for s, df in frames.items()}   # columns: Open High Low Close Volume
        self._starts = {s: idx.tz_convert("UTC").astype("int64").to_numpy() for s, idx in self._index.items()}
        self._bar_ns = int(BAR.total_seconds() * 1e9)

    def _upto(self, symbol: str, since_ns: Optional[int] = None) -> pd.DataFrame:
        """Finished bars up to now, plus the bar in progress with only its open printed."""
        starts = self._starts[symbol]
        now_ns = pd.Timestamp(self.now).value
        finished = int(np.searchsorted(starts + self._bar_ns, now_ns, side="right"))   # bars that ended at or before now
        end = finished + 1 if finished < len(starts) and starts[finished] <= now_ns else finished
        low = 0 if since_ns is None else int(np.searchsorted(starts, since_ns, side="left"))
        values = self._values[symbol][low:end].copy()
        if end > finished and len(values):
            values[-1, 1:4] = values[-1, 0]      # High = Low = Close = Open: only the first print of the forming bar is known
        return pd.DataFrame(values, index=self._index[symbol][low:end], columns=COLUMNS)

    def last_price(self, symbol: str) -> float:
        view = self._upto(symbol) if symbol in self._values else None
        if view is None or view.empty:
            raise ValueError(f"{symbol}: no price data available")
        return float(view["Close"].iloc[-1])

    def bars_since(self, symbol: str, since: datetime) -> pd.DataFrame:
        if symbol not in self._values:
            return pd.DataFrame()
        view = self._upto(symbol)
        return view[pd.DatetimeIndex(view.index).tz_convert("UTC") > pd.Timestamp(since).tz_convert("UTC")]

    def today(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """What the engine's per-cycle download returns: today's session so far, per symbol that has bars."""
        session_start = pd.Timestamp(self.now).tz_convert(IST).normalize().tz_convert("UTC").value
        out = {}
        for s in symbols:
            if s in self._values:
                view = self._upto(s, since_ns=session_start)
                if len(view):
                    out[s] = view
        return out


@dataclass(frozen=True)
class ReplayConfig:
    """The engine's settings for a replay. The defaults ARE the settings' defaults (read from `Settings`, not copied), so a
    replay cannot drift from what the engine trades with unless a value is overridden on purpose."""
    capital: float = Settings.paper_initial_cash
    risk_pct: float = Settings.intraday_risk_pct
    max_position_pct: float = Settings.intraday_max_position_pct
    max_positions: int = Settings.intraday_max_positions
    max_daily_loss_pct: float = RiskLimits.max_daily_loss_pct
    slippage: Optional[float] = None  # None = the broker's default (0.05% a side)

    @classmethod
    def from_settings(cls, settings: Settings, capital: Optional[float] = None) -> "ReplayConfig":
        """The configuration the live intraday engine runs with (INTRADAY_* settings), optionally on a different account size."""
        return cls(capital=settings.paper_initial_cash if capital is None else capital, risk_pct=settings.intraday_risk_pct,
                   max_position_pct=settings.intraday_max_position_pct, max_positions=settings.intraday_max_positions,
                   max_daily_loss_pct=settings.risk.max_daily_loss_pct)


@dataclass
class ReplayResult:
    config: ReplayConfig
    trades: pd.DataFrame        # closed trades, oldest first
    signals: pd.DataFrame       # every breakout the engine saw and what it did (ENTERED / SKIPPED ...)
    daily_equity: pd.Series     # equity at the end of each session
    days: List[Date] = field(default_factory=list)


def session_days(bars: Dict[str, pd.DataFrame], start: Optional[str] = None, end: Optional[str] = None) -> List[Date]:
    """Every session date present in the bars, within [start, end] (ISO dates), oldest first."""
    days = sorted({d for df in bars.values() for d in pd.DatetimeIndex(df.index).tz_convert(IST).date})
    return [d for d in days if (start is None or d.isoformat() >= start) and (end is None or d.isoformat() <= end)]


def _run_session(engine: IntradayEngine, feed: ReplayFeed, day: Date) -> None:
    """Every engine cycle of one session: 09:20:15 to 15:25:15, a step every five minutes."""
    t = datetime.combine(day, FIRST_CYCLE, tzinfo=IST)
    while t.timetz().replace(tzinfo=None) <= LAST_CYCLE:
        feed.now = t.astimezone(timezone.utc)
        engine.step()
        t += BAR


def replay(bars: Dict[str, pd.DataFrame], config: ReplayConfig = ReplayConfig(), universe: Optional[List[str]] = None,
           start: Optional[str] = None, end: Optional[str] = None,
           progress: Optional[Callable[[Date], None]] = None) -> ReplayResult:
    """Run the live engine over every stored session in [start, end] and return what it did. One continuous account, so
    equity compounds and drawdowns carry across days exactly as they would live."""
    days = session_days(bars, start, end)
    feed, clock = ReplayFeed(bars), IndiaClock()
    symbols = sorted(bars) if universe is None else universe
    # ignore_cleanup_errors: on Windows SQLite still holds replay.db open when the folder is removed
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sessions = make_session_factory(f"sqlite:///{Path(tmp) / 'replay.db'}")
        broker = PaperBroker(sessions, feed, clock, initial_cash=config.capital, fees=intraday_fees,
                             now_fn=lambda: feed.now,
                             slippage=SLIPPAGE if config.slippage is None else config.slippage)
        engine = IntradayEngine(broker, symbols, clock, fetch_bars=feed.today, live=True, max_positions=config.max_positions,
                                risk_pct=config.risk_pct, max_position_pct=config.max_position_pct,
                                max_daily_loss_pct=config.max_daily_loss_pct, universe_source="", now_fn=lambda: feed.now)
        equity: Dict[Date, float] = {}
        for day in days:
            _run_session(engine, feed, day)
            equity[day] = broker.portfolio().equity
            if progress:
                progress(day)
        trades = pd.DataFrame(broker.trade_history()[::-1])
        with sessions() as s:
            signals = pd.DataFrame([{c.name: getattr(r, c.name) for c in IntradaySignalRecord.__table__.columns}
                                    for r in s.scalars(select(IntradaySignalRecord).order_by(IntradaySignalRecord.id))])
    return ReplayResult(config, trades, signals, pd.Series(equity, dtype=float), days)


@dataclass(frozen=True)
class ReplaySummary:
    """Headline numbers of a replay."""
    sessions: int
    trades: int
    start: float
    end: float
    total_return: float
    max_drawdown: float        # deepest fall of the end-of-session equity curve from its running peak (0 or negative)
    sharpe: float              # annualised, from the daily P&L
    profitable_days: float     # share of sessions with a positive P&L
    win_rate: float            # share of trades with a positive net P&L
    fees: float
    exits: Dict[str, int]      # STOP / TARGET / SIGNAL (the 15:15 square-off), by count
    signals: Dict[str, int]    # ENTERED / SKIPPED / ..., by count
    first_half_pnl: float
    second_half_pnl: float


def summarise(result: ReplayResult) -> ReplaySummary:
    """Headline numbers of a replay: return, drawdown, win rate, fees, exits, skips."""
    equity, start = result.daily_equity, result.config.capital
    end = float(equity.iloc[-1]) if len(equity) else start
    daily_pnl = equity.diff().fillna(equity.iloc[0] - start) if len(equity) else equity
    curve = pd.concat([pd.Series([start]), equity.reset_index(drop=True)])
    trades, signals = result.trades, result.signals
    half = len(daily_pnl) // 2
    return ReplaySummary(
        sessions=len(result.days), trades=len(trades), start=start, end=end, total_return=end / start - 1,
        max_drawdown=float((curve / curve.cummax() - 1).min()),
        sharpe=float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0.0,
        profitable_days=float((daily_pnl > 0).mean()) if len(daily_pnl) else 0.0,
        win_rate=float((trades["net_pnl"] > 0).mean()) if len(trades) else 0.0,
        fees=float(trades["fees"].sum()) if len(trades) else 0.0,
        exits={str(k): int(v) for k, v in trades["reason"].value_counts().items()} if len(trades) else {},
        signals={str(k): int(v) for k, v in signals["outcome"].value_counts().items()} if len(signals) else {},
        first_half_pnl=float(daily_pnl.iloc[:half].sum()), second_half_pnl=float(daily_pnl.iloc[half:].sum()))


def format_summary(r: ReplaySummary) -> str:
    """The summary as text for a terminal."""
    return "\n".join([
        f"{r.sessions} sessions, {r.trades} trades | win rate {r.win_rate:.0%} | fees paid Rs {r.fees:,.0f}",
        f"exits {r.exits} | signals {r.signals}",
        f"start Rs {r.start:,.0f} -> end Rs {r.end:,.0f} ({r.total_return:+.2%})",
        f"daily Sharpe (annualised) {r.sharpe:.2f} | max drawdown {r.max_drawdown:.1%} | profitable days {r.profitable_days:.0%}",
        f"first half P&L {r.first_half_pnl:+,.0f} | second half P&L {r.second_half_pnl:+,.0f}"])
