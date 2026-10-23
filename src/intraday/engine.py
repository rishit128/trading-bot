"""Intraday engine: every 5 minutes look for opening-range breakouts on liquid stocks, buy in a SEPARATE paper account,
and square everything off at 15:15 IST. Stops and targets are settled by the paper broker from 5-minute bars.

Every breakout it sees is recorded (table intraday_signals) with what became of it: entered, skipped and why (a stock too
dear for the account, no cash), a dry-run signal, or a failed order. The universe it scanned is recorded once a day
(intraday_universe). A position left over from an earlier day (the process was down at the square-off) is closed at the
next session's first cycle: this account is meant to be flat every night. With an archive, each finished session's
5-minute bars are saved after the close (see src.data.intraday_bars), so history for testing rules keeps growing.

The rule is pluggable (`find_setup`): the opening-range breakout by default, or gap-and-go (`strategy.find_gap_setup`),
each in its own paper account and database, run side by side by `run_together`.

Broker research calls (`house_calls_fn`, the breakout engine only): each recorded signal notes the open call on its stock,
if any (`house_call`). With `act_on_calls` the stocks of today's open Intraday BUY calls are also added to the scan, where
they must pass the same breakout rule, and a stock the broker has an open SELL call on is not bought (recorded as SKIPPED);
without it the calls are recorded only, which keeps the breakout engine a clean control. Without a fresh snapshot of calls
(no portal session that day) the engine runs exactly as before."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from sqlalchemy import func, select

from src.data.angel_quotes import EntryContext
from src.data.house_calls import HouseCall, intraday_buys, open_sells
from src.data.india import IST
from src.data.intraday_bars import BarArchive, fetch_today_bars, session_rows
from src.database import IntradaySignalRecord, IntradayUniverseRecord
from src.engine.enums import SignalOutcome
from src.engine.paper_broker import PaperBroker
from src.engine.ports import BAR, Fill, MarketClock
from src.intraday import strategy as st
from src.monitoring.heartbeat import Heartbeat
from src.versions import INTRADAY_VERSION

log = logging.getLogger(__name__)
ARCHIVE_FROM = dtime(15, 35)  # the session's last bar (15:25-15:30) is published by then
MIN_COVERAGE = 0.8  # a scan that returns bars for fewer of the stocks asked for than this raises an alert
STALE_FILL = timedelta(minutes=10)  # a fill priced from a bar that started longer ago than this raises an alert


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
    # (symbol, today's finished bars, previous session's close) -> a setup or None: the rule this account trades
    find_setup: Callable[[str, pd.DataFrame, Optional[float]], Optional[st.Setup]] = st.find_setup
    entry_until: dtime = st.ENTRY_UNTIL  # no bar starting at or after this can signal: the scan stops one bar later
    label: str = "INTRADAY"  # prefix of its Telegram messages
    version: str = INTRADAY_VERSION  # recorded with every signal and universe row
    record_capped: bool = False  # also record (as SKIPPED) the signals the position limit left out: the gap experiments
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    house_calls_fn: Optional[Callable[[], List[HouseCall]]] = None  # broker research calls; None = not used
    act_on_calls: bool = True  # False: the calls are only noted on each signal (the live setting is INTRADAY_CALLS_ACT)
    # Angel One's circuit limits, spread, order book and F&O data for a symbol about to be bought; a buy within 1% of the upper
    # circuit is refused and the rest is stored on the signal. None, or an error from it, leaves the entry on Yahoo's data alone.
    context_fn: Optional[Callable[[str, float], EntryContext]] = None
    step_deadline: Optional[dtime] = None  # the first step of the day at or after 09:30 must start by this, else an alert
    _day: Optional[str] = field(default=None, init=False)
    _contexts: Dict[str, EntryContext] = field(default_factory=dict, init=False)  # today's, by symbol: one request per entry
    _calls: List[HouseCall] = field(default_factory=list, init=False)  # this cycle's snapshot of house calls
    _done_today: Set[str] = field(default_factory=set, init=False)
    _known: Dict[str, int] = field(default_factory=dict, init=False)
    _archived: Optional[str] = field(default=None, init=False)  # the session date whose bars were last archived
    _verified: Optional[str] = field(default=None, init=False)  # the session date whose signals were last verified
    _step_at: Optional[datetime] = field(default=None, init=False)  # when the current cycle started
    _seen_volume: Dict[str, float] = field(default_factory=dict, init=False)  # signal bar volume as this cycle saw it
    _flags: Set[str] = field(default_factory=set, init=False)  # faults currently alerted (cleared when they clear)
    _with_data: int = field(default=0, init=False)  # the most stocks any of today's scans returned bars for
    _deadline_day: Optional[str] = field(default=None, init=False)  # the day the step deadline was last checked
    _summarised: Optional[str] = field(default=None, init=False)  # the session date whose summary line was sent

    def _say(self, text: str) -> None:
        log.info(text)
        if self.notify:
            self.notify(f"[{self.label}] " + text)

    def _flag(self, key: str, bad: bool, text: str) -> None:
        """Alert once when a fault appears, stay quiet while it lasts, and forget it when it clears (so its return is
        alerted again)."""
        if not bad:
            self._flags.discard(key)
        elif key not in self._flags:
            self._flags.add(key)
            self._say("⚠️ ALERT: " + text)

    def _new_day(self, today: str) -> None:
        """Forget yesterday; anything already traded today (e.g. before a restart) counts as done. Records the universe.
        Positions already open (a restart) are remembered before anything is settled, so their exits are still reported."""
        _, held = self.broker.valuation()
        self._day, self._done_today, self._known = today, set(), {h["symbol"]: h["qty"] for h in held}
        self._contexts = {}
        self._with_data = 0
        for t in self.broker.trade_history():
            if t["closed_at"].astimezone(IST).date().isoformat() == today:
                self._done_today.add(t["symbol"])
        with self.broker.sessions() as s:  # every signal already recorded today (entered, skipped, dry run): one attempt each
            self._done_today.update(s.scalars(select(IntradaySignalRecord.symbol).where(IntradaySignalRecord.day == today)))
        if self.universe_source:
            self._store(
                IntradayUniverseRecord(
                    day=today,
                    source=self.universe_source,
                    size=len(self.universe),
                    symbols_json=json.dumps(self.universe),
                    system_version=self.version,
                )
            )

    def _store(self, row) -> None:
        """Save an audit row. A failure here is logged and never stops trading."""
        try:
            with self.broker.sessions() as s:
                s.add(row)
                s.commit()
        except Exception:
            log.exception("could not record %s", type(row).__name__)

    def _entry_context(self, symbol: str, price: float) -> EntryContext:
        """Angel One's checks for buying `symbol` now (see `context_fn`); asked once per symbol a day."""
        if self.context_fn is None:
            return EntryContext()
        if symbol not in self._contexts:
            try:
                self._contexts[symbol] = self.context_fn(symbol, price)
            except Exception as e:
                log.warning("Angel One quote failed for %s (%s: %s); entering without it", symbol, type(e).__name__, e)
                self._contexts[symbol] = EntryContext()
        return self._contexts[symbol]

    def _record(self, setup: st.Setup, outcome: SignalOutcome, qty: int = 0, reason: Optional[str] = None) -> None:
        self._store(
            IntradaySignalRecord(
                day=self._day or "",
                symbol=setup.symbol,
                bar_time=setup.bar_time.isoformat(),
                signal_price=setup.signal_price,
                stop=setup.stop,
                target=setup.target,
                range_high=setup.range_high,
                range_low=setup.range_low,
                volume_ratio=setup.volume_ratio,
                strength=setup.strength,
                outcome=outcome,
                qty=qty,
                reason=reason,
                system_version=self.version,
                house_call=self._house_tag(setup.symbol),
                step_at=self._step_at,
                seen_volume=self._seen_volume.get(setup.symbol),
                context_json=json.dumps(self._contexts[setup.symbol].note) if setup.symbol in self._contexts else None,
            )
        )

    def _house_tag(self, symbol: str) -> Optional[str]:
        """The newest open house call on this stock, as "Intraday buy" etc., or None."""
        mine = [c for c in self._calls if c.symbol == symbol and c.is_open]
        if not mine:
            return None
        call = max(mine, key=lambda c: c.published)
        return f"{call.kind} {call.side}"[:32]

    def _read_house_calls(self) -> List[HouseCall]:
        if self.house_calls_fn is None:
            return []
        try:
            return list(self.house_calls_fn())
        except Exception:
            log.exception("could not read the house research calls; scanning without them")
            return []

    def step(self) -> str:
        """One 5-minute cycle. Returns a short status string (useful in tests and logs)."""
        now = self.now_fn()
        self._step_at = now
        if not self.clock.is_open(now):
            self._archive_session(now)
            self._verify_signals(now)
            self._summarise(now)
            return "closed"
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._day != today:
            self._new_day(today)
        self._check_start(local)
        pf = self.broker.portfolio()  # settles any stop/target hit since the last cycle
        self._flag(
            "stops",
            bool(self.broker.unchecked_exits),
            f"could not check the stop/target of {', '.join(sorted(self.broker.unchecked_exits))}: the price feed failed, "
            "so those positions are unprotected until it returns",
        )
        self._flag(
            "at-cost",
            bool(self.broker.valued_at_cost),
            f"no price for {', '.join(sorted(self.broker.valued_at_cost))}: valued at cost, so equity and the daily-loss "
            "limit are not reliable",
        )
        if pf.positions and self._close_stale(today):
            pf = self.broker.portfolio()
        self._report_exits(pf)
        if local.time() >= st.SQUARE_OFF:
            return self._square_off(pf)
        if local.time() < st.ENTRY_FROM:
            return "waiting for the opening range"
        if (
            local >= datetime.combine(local.date(), self.entry_until, IST) + BAR
        ):  # the last allowed bar has finished and been seen
            return "entry window closed: managing open positions only"
        day_start_equity = pf.start_of_day_equity or pf.equity
        if pf.equity < day_start_equity * (1 - self.max_daily_loss_pct):
            return "daily loss limit reached: no new entries"
        if len(pf.positions) >= self.max_positions and not self.record_capped:
            return "max positions open"
        return self._scan(pf, local)

    def _scan(self, pf, local: datetime) -> str:
        """Look for a fresh breakout in every stock not already held or traded today, and try to buy each one found."""
        self._calls = self._read_house_calls()
        added = [s for s in intraday_buys(self._calls, local.date()) if s not in self.universe] if self.act_on_calls else []
        sells = open_sells(self._calls) if self.act_on_calls else {}
        candidates = [s for s in self.universe + added if s not in pf.positions and s not in self._done_today]
        try:
            bars = self.fetch_bars(candidates)
        except Exception as e:
            self._flag("coverage", True, f"the 5-minute download failed ({type(e).__name__}: {e}): no scan this cycle")
            raise
        self._with_data = max(self._with_data, len(bars))
        self._flag(
            "coverage",
            bool(candidates) and len(bars) < MIN_COVERAGE * len(candidates),
            f"the scan got bars for only {len(bars)} of {len(candidates)} stocks (under {MIN_COVERAGE:.0%}): "
            "breakouts may be missed",
        )
        entered = skipped = 0
        cash = pf.cash
        for sym, df in bars.items():
            full = len(pf.positions) + entered >= self.max_positions
            if full and not self.record_capped:
                break
            finished = df[(pd.DatetimeIndex(df.index) + BAR) <= pd.Timestamp(local)]
            earlier = finished[pd.DatetimeIndex(finished.index).tz_convert(IST).date < local.date()]
            prev_close = float(earlier["Close"].iloc[-1]) if len(earlier) else None  # only when the feed returns it
            df = session_rows(finished, local.date())
            setup = self.find_setup(sym, df, prev_close) if len(df) else None
            if setup is None or setup.bar_time < df.index[-1] - BAR:  # a stale breakout from earlier in the day: skip
                continue
            if setup.bar_time in df.index:
                self._seen_volume[setup.symbol] = float(df["Volume"][setup.bar_time])
            if setup.symbol in sells:  # the house says sell this stock: a long breakout against that call is not taken
                self._done_today.add(setup.symbol)
                self._record(setup, SignalOutcome.SKIPPED, 0, f"the broker has an open {sells[setup.symbol].kind} SELL call")
                skipped += 1
                continue
            if full:  # only with record_capped: the signal is real, the account had no free position
                self._done_today.add(setup.symbol)
                self._record(setup, SignalOutcome.SKIPPED, 0, f"position limit reached ({self.max_positions} open)")
                skipped += 1
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
        exit_plan = (
            f"target {setup.target:,.2f}"
            if setup.target < setup.signal_price * (1 + st.NO_TARGET)
            else "held to the 15:15 square-off"
        )
        text = (
            f"BUY {setup.symbol} x{qty} breakout above {setup.range_high:,.2f}; stop {setup.stop:,.2f}, "
            f"{exit_plan}, volume {setup.volume_ratio:.1f}x average"
        )
        if not self.live:
            self._say("DRY RUN signal: " + text)
            self._record(setup, SignalOutcome.DRY_RUN, qty)
            return SignalOutcome.DRY_RUN, None
        context = self._entry_context(setup.symbol, setup.signal_price)
        if context.skip:
            self._record(setup, SignalOutcome.SKIPPED, 0, context.skip)
            return SignalOutcome.SKIPPED, None
        try:
            fill = self.broker.buy_with_bracket(
                setup.symbol, qty, setup.signal_price, 1 - setup.stop / setup.signal_price, setup.target / setup.signal_price - 1
            )
        except Exception as e:
            log.warning("intraday buy %s failed: %s", setup.symbol, e)
            self._record(setup, SignalOutcome.FAILED, qty, str(e))
            return SignalOutcome.FAILED, None
        self._known[setup.symbol] = qty
        self._say(text)
        self._check_fill_age(setup.symbol, fill)
        self._record(setup, SignalOutcome.ENTERED, qty)
        return SignalOutcome.ENTERED, fill

    def _check_start(self, local: datetime) -> None:
        """Alert when the day's first step at or after 09:30 starts after `step_deadline` (gap-only trades on that one step,
        so a late start means a missed day)."""
        today = local.date().isoformat()
        if self.step_deadline is None or self._deadline_day == today or local.time() < st.ENTRY_FROM:
            return
        self._deadline_day = today
        if local.time() > self.step_deadline:
            self._say(
                f"⚠️ ALERT: the {st.ENTRY_FROM:%H:%M} step started late, at {local:%H:%M:%S} (it must start by "
                f"{self.step_deadline:%H:%M}); today's entries may have been missed"
            )

    def _check_fill_age(self, symbol: str, fill: Fill) -> None:
        """Alert when a fill was priced from a bar that started more than STALE_FILL ago: the feed was lagging."""
        if fill.price_bar is None:
            return
        started = pd.Timestamp(fill.price_bar)
        started = started.tz_localize(timezone.utc) if started.tzinfo is None else started.tz_convert(timezone.utc)
        age = self.now_fn() - started.to_pydatetime()
        if age > STALE_FILL:
            self._say(
                f"⚠️ ALERT: {symbol} was filled at {fill.avg_price:,.2f} from a bar that started {age.total_seconds() / 60:.0f} "
                "minutes ago: the price feed is lagging, so the fill may not match the market"
            )

    def _summarise(self, now: datetime) -> None:
        """One line each evening, once a session has finished: universe, stocks with data, signals by outcome and reason,
        entries, exits and net P&L. Only on days this engine traded (it ran a session today)."""
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._summarised == today or self._day != today or local.time() < ARCHIVE_FROM:
            return
        with self.broker.sessions() as s:
            rows = s.execute(
                select(IntradaySignalRecord.outcome, IntradaySignalRecord.reason, func.count())
                .where(IntradaySignalRecord.day == today)
                .group_by(IntradaySignalRecord.outcome, IntradaySignalRecord.reason)
            ).all()
        by_outcome: Dict[str, int] = {}
        reasons: Dict[str, int] = {}
        for outcome, reason, n in rows:
            by_outcome[str(outcome)] = by_outcome.get(str(outcome), 0) + n
            if reason:
                reasons[reason[:40]] = reasons.get(reason[:40], 0) + n
        trades = [t for t in self.broker.trade_history() if t["closed_at"].astimezone(IST).date().isoformat() == today]
        exits: Dict[str, int] = {}
        for t in trades:
            exits[t["reason"]] = exits.get(t["reason"], 0) + 1
        signals = ", ".join(f"{k} {v}" for k, v in sorted(by_outcome.items())) or "none"
        why = "; ".join(f"{r} x{n}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        self._say(
            f"{today}: universe {len(self.universe)}, bars for {self._with_data}; signals: {signals}"
            + (f" ({why})" if why else "")
            + f"; entries {by_outcome.get(str(SignalOutcome.ENTERED), 0)}; exits {len(trades)}"
            + (f" ({', '.join(f'{k} {v}' for k, v in sorted(exits.items()))})" if exits else "")
            + f"; net P&L {sum(t['net_pnl'] for t in trades):+,.0f}"
        )
        self._summarised = today

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

    def _verify_signals(self, now: datetime) -> None:
        """Once a session has finished, compare each of its recorded signals with the final bars: the close and volume of
        the signal bar, and whether the rule run on the final bars up to that bar still signals there. Yahoo revises the
        bars it served during the day; this measures by how much, and whether it changed what was traded. Once a day;
        a failed download is retried next cycle; a stock the download does not return is left unverified."""
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._verified == today or local.weekday() >= 5 or local.time() < ARCHIVE_FROM:
            return
        try:
            with self.broker.sessions() as s:
                rows = list(
                    s.scalars(
                        select(IntradaySignalRecord).where(
                            IntradaySignalRecord.day == today, IntradaySignalRecord.final_close.is_(None)
                        )
                    )
                )
                fetched = self.fetch_bars(sorted({r.symbol for r in rows})) if rows else {}
                for r in rows:
                    self._check_signal(r, fetched.get(r.symbol), local)
                s.commit()
        except Exception:
            log.exception("could not verify the session's signals against the final bars; will retry")
            return
        self._verified = today
        if rows:
            log.info("verified %d signals against the final bars (%s)", len(rows), today)

    def _check_signal(self, row: IntradaySignalRecord, bars: Optional[pd.DataFrame], local: datetime) -> None:
        if bars is None or bars.empty:
            return
        day = session_rows(bars, local.date())
        earlier = bars[pd.DatetimeIndex(bars.index).tz_convert(IST).date < local.date()]
        prev_close = float(earlier["Close"].iloc[-1]) if len(earlier) else None
        at = pd.Timestamp(row.bar_time)
        if at not in day.index:
            return
        row.final_close, row.final_volume = float(day["Close"][at]), float(day["Volume"][at])
        setup = self.find_setup(row.symbol, day[day.index <= at], prev_close)
        row.final_signal = setup is not None and setup.bar_time == at

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
        # trade_history is newest first; reversed, the last trade per symbol in the comprehension is the newest
        recent = {t["symbol"]: t for t in reversed(self.broker.trade_history(limit=20))}
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
                self._flags.discard(f"square-off {sym}")
            except Exception as e:
                log.warning("square-off %s failed: %s", sym, e)
                self._flag(
                    f"square-off {sym}", True, f"square-off of {sym} x{qty} FAILED ({e}): it stays open, retried next cycle"
                )
        if closed:
            for t in self.broker.trade_history(limit=closed):
                self._say(f"SQUARE-OFF {t['symbol']}: {t['net_pnl']:+,.0f} net")
            self._known.clear()
        return f"square-off: closed {closed}"

    def run_loop(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Run this engine alone (see `run_together`)."""
        run_together([self], sleep)


def run_together(
    engines: Sequence[IntradayEngine],
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Heartbeat | None = None,
) -> None:
    """Step every engine at each 5-minute boundary (plus a few seconds for Yahoo to publish the bar), forever. One engine's
    failure never stops the others. `heartbeat` records after each pass whether every engine's step succeeded."""
    while True:
        failed = []
        for engine in engines:
            started = datetime.now(IST)
            try:
                status = engine.step()
                log.info("%s cycle (started %s): %s", engine.label.lower(), f"{started:%H:%M:%S}", status)
            except Exception:
                failed.append(engine.label)
                log.exception("%s cycle failed", engine.label.lower())
        if heartbeat:
            heartbeat.beat(not failed, f"failed: {', '.join(failed)}" if failed else "all engines stepped")
        now = datetime.now(timezone.utc)
        nxt = (now + BAR).replace(second=15, microsecond=0)
        nxt -= timedelta(minutes=nxt.minute % 5)
        sleep(max(30.0, (nxt - now).total_seconds()))
