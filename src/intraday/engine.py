"""Intraday engine: every 5 minutes look for opening-range breakouts on liquid stocks, buy in a SEPARATE paper account,
and square everything off at 15:15 IST. Stops and targets are settled by the paper broker from 5-minute bars."""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

from src.data.india import IST, SUFFIX
from src.intraday import strategy as st

log = logging.getLogger(__name__)
BAR = timedelta(minutes=5)


def fetch_today_bars(symbols: List[str], download: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
    """Today's 5-minute bars (IST index) per symbol in one bulk call. Symbols with no data are omitted."""
    if download is None:
        import yfinance as yf

        def download(tickers, **kw):
            """Bulk yfinance download with a timeout."""
            return yf.download(tickers, group_by="ticker", auto_adjust=True, progress=False, threads=True, timeout=30, **kw)

    wide = download([s + SUFFIX for s in symbols], period="1d", interval="5m")
    out: Dict[str, pd.DataFrame] = {}
    if wide is None or wide.empty:
        return out
    top = set(wide.columns.get_level_values(0))
    for s in symbols:
        if s + SUFFIX in top:
            df = wide[s + SUFFIX].dropna(subset=["Close"])
            if len(df):
                df = df.copy()
                df.index = df.index.tz_convert(IST)
                out[s] = df
    return out


@dataclass
class IntradayEngine:
    """One engine per intraday paper account. `broker` is a PaperBroker on its own database with intraday fees."""
    broker: object
    universe: List[str]
    clock: object
    fetch_bars: Callable[[List[str]], Dict[str, pd.DataFrame]] = fetch_today_bars
    notify: Optional[Callable[[str], object]] = None
    live: bool = False  # False: log signals only, place nothing
    max_positions: int = 5
    min_position_pct: float = 0.02  # position size at a barely-qualifying breakout (Setup.strength == 0.0)
    max_position_pct: float = 0.05  # position size at a much stronger breakout (Setup.strength == 1.0)
    max_daily_loss_pct: float = 0.02
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    _day: Optional[str] = field(default=None, init=False)
    _done_today: Set[str] = field(default_factory=set, init=False)
    _known: Dict[str, int] = field(default_factory=dict, init=False)

    def _say(self, text: str) -> None:
        log.info(text)
        if self.notify:
            self.notify("[INTRADAY] " + text)

    def _new_day(self, today: str) -> None:
        """Forget yesterday; anything already traded or held today (e.g. before a restart) counts as done."""
        self._day, self._done_today, self._known = today, set(), {}
        for t in self.broker.trade_history():
            if t["closed_at"].astimezone(IST).date().isoformat() == today:
                self._done_today.add(t["symbol"])

    def step(self) -> str:
        """One 5-minute cycle. Returns a short status string (useful in tests and logs)."""
        now = self.now_fn()
        if not self.clock.is_open(now):
            return "closed"
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._day != today:
            self._new_day(today)
        pf = self.broker.portfolio()  # settles any stop/target hit since the last cycle
        self._report_exits(pf)
        if local.time() >= st.SQUARE_OFF:
            return self._square_off(pf)
        if local.time() < st.ENTRY_FROM:
            return "waiting for the opening range"
        start = pf.start_of_day_equity or pf.equity
        if pf.equity < start * (1 - self.max_daily_loss_pct):
            return "daily loss limit reached: no new entries"
        if len(pf.positions) >= self.max_positions:
            return "max positions open"
        candidates = [s for s in self.universe if s not in pf.positions and s not in self._done_today]
        bars = self.fetch_bars(candidates)
        entered = 0
        for sym, df in bars.items():
            df = df[df.index + BAR <= local]  # completed bars only
            if len(pf.positions) + entered >= self.max_positions:
                break
            setup = st.find_setup(sym, df) if len(df) else None
            if setup is None or setup.bar_time < df.index[-1] - BAR:  # a stale breakout from earlier in the day: skip
                continue
            entered += self._enter(setup, pf.equity)
        return f"scanned {len(bars)} stocks, entered {entered}"

    def _enter(self, setup: st.Setup, equity: float) -> int:
        qty = st.position_size(equity, setup.signal_price, setup.stop, setup.strength,
                               self.min_position_pct, self.max_position_pct)
        self._done_today.add(setup.symbol)  # one attempt per stock per day, filled or not
        if qty <= 0:
            return 0
        text = (f"BUY {setup.symbol} x{qty} breakout above {setup.range_high:,.2f}; stop {setup.stop:,.2f}, "
                f"target {setup.target:,.2f}, volume strength {setup.strength:.0%}")
        if not self.live:
            self._say("DRY RUN signal: " + text)
            return 0
        try:
            self.broker.buy_with_bracket(setup.symbol, qty, setup.signal_price,
                                         1 - setup.stop / setup.signal_price, setup.target / setup.signal_price - 1)
        except Exception as e:
            log.warning("intraday buy %s failed: %s", setup.symbol, e)
            return 0
        self._known[setup.symbol] = qty
        self._say(text)
        return 1

    def _report_exits(self, pf) -> None:
        """Tell the user about positions that closed since the last cycle (stop or target)."""
        gone = [s for s in self._known if s not in pf.positions]
        if not gone:
            return
        recent = {t["symbol"]: t for t in self.broker.trade_history(limit=20)}
        for s in gone:
            t = recent.get(s)
            if t:
                self._say(f"{t['reason']} {s}: {t['net_pnl']:+,.0f} net")
            self._known.pop(s, None)

    def _square_off(self, pf) -> str:
        closed = 0
        for sym, qty in list(pf.position_qty.items()):
            try:
                self.broker.sell(sym, qty)
                closed += 1
            except Exception as e:
                log.warning("square-off %s failed: %s", sym, e)
        if closed:
            for t in self.broker.trade_history(limit=closed):
                self._say(f"SQUARE-OFF {t['symbol']}: {t['net_pnl']:+,.0f} net")
            self._known.clear()
        return f"square-off: closed {closed}"

    def run_loop(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Run a step at every 5-minute boundary (plus a few seconds for Yahoo to publish the bar), forever."""
        while True:
            try:
                log.info("intraday cycle: %s", self.step())
            except Exception:
                log.exception("intraday cycle failed")
            now = datetime.now(timezone.utc)
            nxt = (now + BAR).replace(second=15, microsecond=0)
            nxt -= timedelta(minutes=nxt.minute % 5)
            sleep(max(30.0, (nxt - now).total_seconds()))
