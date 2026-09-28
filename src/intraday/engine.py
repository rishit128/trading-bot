"""Intraday engine: every 5 minutes look for opening-range breakouts on liquid stocks, buy in a SEPARATE paper account,
and square everything off at 15:15 IST. Stops and targets are settled by the paper broker from 5-minute bars.

Every breakout it sees is recorded (table intraday_signals) with what became of it: entered, skipped and why (a stock too
dear for the account, no cash), a dry-run signal, or a failed order. The universe it scanned is recorded once a day
(intraday_universe). A position left over from an earlier day (the process was down at the square-off) is closed at the
next session's first cycle: this account is meant to be flat every night. With an archive, each finished session's
5-minute bars are saved after the close (see src.data.intraday_bars), so history for testing rules keeps growing."""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.data.india import IST
from src.data.intraday_bars import BarArchive, fetch_today_bars, session_rows
from src.database import IntradaySignalRecord, IntradayUniverseRecord
from src.engine.enums import SignalOutcome
from src.engine.paper_broker import PaperBroker
from src.engine.ports import BAR, Fill, MarketClock
from src.intraday import strategy as st
from src.versions import INTRADAY_VERSION

log = logging.getLogger(__name__)
ARCHIVE_FROM = dtime(15, 35)  # the session's last bar (15:25-15:30) is published by then


@dataclass
class IntradayEngine:
    """One engine per intraday paper account. `broker` is a PaperBroker on its own database with intraday fees."""
    broker: PaperBroker
    universe: List[str]
    clock: MarketClock
    fetch_bars: Callable[[List[str]], Dict[str, pd.DataFrame]] = fetch_today_bars
    notify: Optional[Callable[[str], object]] = None
    live: bool = False  # False: log signals only, place nothing
    max_positions: int = 5
    risk_pct: float = 0.005  # equity lost if a stop is hit, per trade
    max_position_pct: float = 0.20  # largest position, as a fraction of equity (cash only, no leverage)
    max_daily_loss_pct: float = 0.02
    universe_source: str = ""  # where `universe` came from (recorded daily); empty = not recorded
    archive: Optional[BarArchive] = None  # save every finished session's bars here; None = do not archive
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    _day: Optional[str] = field(default=None, init=False)
    _done_today: Set[str] = field(default_factory=set, init=False)
    _known: Dict[str, int] = field(default_factory=dict, init=False)
    _archived: Optional[str] = field(default=None, init=False)  # the session date whose bars were last archived

    def _say(self, text: str) -> None:
        log.info(text)
        if self.notify:
            self.notify("[INTRADAY] " + text)

    def _new_day(self, today: str) -> None:
        """Forget yesterday; anything already traded today (e.g. before a restart) counts as done. Records the universe."""
        self._day, self._done_today, self._known = today, set(), {}
        for t in self.broker.trade_history():
            if t["closed_at"].astimezone(IST).date().isoformat() == today:
                self._done_today.add(t["symbol"])
        if self.universe_source:
            self._store(IntradayUniverseRecord(day=today, source=self.universe_source, size=len(self.universe),
                                               symbols_json=json.dumps(self.universe), system_version=INTRADAY_VERSION))

    def _store(self, row) -> None:
        """Save an audit row. A failure here is logged and never stops trading."""
        try:
            with self.broker.sessions() as s:
                s.add(row)
                s.commit()
        except Exception:
            log.exception("could not record %s", type(row).__name__)

    def _record(self, setup: st.Setup, outcome: SignalOutcome, qty: int = 0, reason: Optional[str] = None) -> None:
        self._store(IntradaySignalRecord(
            day=self._day or "", symbol=setup.symbol, bar_time=setup.bar_time.isoformat(), signal_price=setup.signal_price,
            stop=setup.stop, target=setup.target, range_high=setup.range_high, range_low=setup.range_low,
            volume_ratio=setup.volume_ratio, strength=setup.strength, outcome=outcome, qty=qty, reason=reason,
            system_version=INTRADAY_VERSION))

    def step(self) -> str:
        """One 5-minute cycle. Returns a short status string (useful in tests and logs)."""
        now = self.now_fn()
        if not self.clock.is_open(now):
            self._archive_session(now)
            return "closed"
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._day != today:
            self._new_day(today)
        pf = self.broker.portfolio()  # settles any stop/target hit since the last cycle
        if pf.positions and self._close_stale(today):
            pf = self.broker.portfolio()
        self._report_exits(pf)
        if local.time() >= st.SQUARE_OFF:
            return self._square_off(pf)
        if local.time() < st.ENTRY_FROM:
            return "waiting for the opening range"
        day_start_equity = pf.start_of_day_equity or pf.equity
        if pf.equity < day_start_equity * (1 - self.max_daily_loss_pct):
            return "daily loss limit reached: no new entries"
        if len(pf.positions) >= self.max_positions:
            return "max positions open"
        return self._scan(pf, local)

    def _scan(self, pf, local: datetime) -> str:
        """Look for a fresh breakout in every stock not already held or traded today, and try to buy each one found."""
        candidates = [s for s in self.universe if s not in pf.positions and s not in self._done_today]
        bars = self.fetch_bars(candidates)
        entered = skipped = 0
        cash = pf.cash
        for sym, df in bars.items():
            if len(pf.positions) + entered >= self.max_positions:
                break
            # today's finished bars
            df = session_rows(df[(pd.DatetimeIndex(df.index) + BAR) <= pd.Timestamp(local)], local.date())
            setup = st.find_setup(sym, df) if len(df) else None
            if setup is None or setup.bar_time < df.index[-1] - BAR:  # a stale breakout from earlier in the day: skip
                continue
            outcome, fill = self._enter(setup, pf.equity, cash)
            if outcome == SignalOutcome.SKIPPED:
                skipped += 1
            elif fill is not None:
                entered += 1
                cash -= (fill.filled_qty or 0) * (fill.avg_price or setup.signal_price)
        return f"scanned {len(bars)} stocks, entered {entered}" + (f", skipped {skipped}" if skipped else "")

    def _enter(self, setup: st.Setup, equity: float, cash: float) -> Tuple[SignalOutcome, Optional[Fill]]:
        """Try to buy one breakout. Returns (outcome, fill); the fill is None unless a paper order was placed. Every outcome
        (ENTERED, SKIPPED, DRY_RUN, FAILED) is recorded with its reason."""
        qty, why = st.position_size(equity, setup.signal_price, setup.stop, self.risk_pct, self.max_position_pct, cash)
        self._done_today.add(setup.symbol)  # one attempt per stock per day, filled or not
        if qty <= 0:
            log.info("intraday SKIP %s at %.2f: %s", setup.symbol, setup.signal_price, why)
            self._record(setup, SignalOutcome.SKIPPED, 0, why)
            return SignalOutcome.SKIPPED, None
        text = (f"BUY {setup.symbol} x{qty} breakout above {setup.range_high:,.2f}; stop {setup.stop:,.2f}, "
                f"target {setup.target:,.2f}, volume {setup.volume_ratio:.1f}x average")
        if not self.live:
            self._say("DRY RUN signal: " + text)
            self._record(setup, SignalOutcome.DRY_RUN, qty)
            return SignalOutcome.DRY_RUN, None
        try:
            fill = self.broker.buy_with_bracket(setup.symbol, qty, setup.signal_price,
                                                1 - setup.stop / setup.signal_price, setup.target / setup.signal_price - 1)
        except Exception as e:
            log.warning("intraday buy %s failed: %s", setup.symbol, e)
            self._record(setup, SignalOutcome.FAILED, qty, str(e))
            return SignalOutcome.FAILED, None
        self._known[setup.symbol] = qty
        self._say(text)
        self._record(setup, SignalOutcome.ENTERED, qty)
        return SignalOutcome.ENTERED, fill

    def _archive_session(self, now: datetime) -> None:
        """Once a session has finished, save its 5-minute bars for the whole universe. Runs at most once a day (a holiday, when
        Yahoo still serves the previous session, saves nothing and is not retried); a failed download is retried next cycle."""
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self.archive is None or self._archived == today or local.weekday() >= 5 or local.time() < ARCHIVE_FROM:
            return
        try:
            fetched = self.fetch_bars(self.universe)
        except Exception:
            log.exception("could not download the session's bars for the archive; will retry")
            return
        rows = {s: session_rows(df, local.date()) for s, df in fetched.items()}
        today_bars = {s: df for s, df in rows.items() if len(df)}  # a holiday: only the previous session came back
        added = self.archive.save(today_bars)
        self._archived = today
        log.info("archived %d bars for %d stocks (%s)", added, len(today_bars), today)

    def _close_stale(self, today: str) -> bool:
        """Sell any position bought on an earlier day (the square-off was missed): this account must be flat overnight."""
        closed = 0
        for h in self.broker.holdings():
            opened = h["opened_at"].astimezone(IST).date().isoformat()
            if opened >= today:
                continue
            try:
                self.broker.sell(h["symbol"], h["qty"])
            except Exception as e:
                log.warning("could not close the stale %s position: %s", h["symbol"], e)
                continue
            closed += 1
            self._known.pop(h["symbol"], None)
            self._say(f"STALE {h['symbol']}: held over from {opened}, closed at the first cycle of {today}")
        return closed > 0

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
