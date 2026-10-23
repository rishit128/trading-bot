"""The news-catalyst engine (News-Catalyst v1): a fourth intraday paper account that trades only stocks with a material
exchange filing, by the fixed rules of src.news.strategy. The filings and their AI labels come from the watcher
(src.news.watcher) through the news_filings table; this engine decides and trades, on the same 5-minute steps as the
other intraday engines, and inherits their square-off, stale-position close, exit reports and daily summary.

Risk (the plan's): 0.5% of equity per trade, at most 20% of equity and at least Rs 3,000 per position, at most 3 trades a
day and one per stock, no new entries after a 1.5% loss on the day. Every filing of the day gets a rule verdict (fired,
or why not) and, after the close, the move 5, 30 and 120 minutes after it and at the close."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func, select

from src.data.india import IST
from src.data.intraday_bars import session_rows
from src.database import IntradaySignalRecord, NewsFilingRecord
from src.engine.enums import SignalOutcome
from src.engine.ports import BAR, Fill
from src.intraday import strategy as st
from src.intraday.engine import ARCHIVE_FROM, IntradayEngine
from src.news import strategy as ns
from src.news.labels import as_utc
from src.versions import NEWS_VERSION

log = logging.getLogger(__name__)

MAX_TRADES_PER_DAY = 3
MIN_POSITION_VALUE = 3000.0
MOVE_MINUTES = (5, 30, 120)
OPEN = dtime(9, 15)


def _prev_close_and_today(bars: pd.DataFrame, today) -> Tuple[Optional[float], pd.DataFrame]:
    earlier = bars[pd.DatetimeIndex(bars.index).tz_convert(IST).date < today]
    return (float(earlier["Close"].iloc[-1]) if len(earlier) else None), session_rows(bars, today)


def price_at(day: pd.DataFrame, when: datetime) -> Optional[float]:
    """The close of the last bar that finished by `when`; None when no bar had."""
    done = day[(pd.DatetimeIndex(day.index) + BAR) <= pd.Timestamp(when)]
    return float(done["Close"].iloc[-1]) if len(done) else None


def filing_moves(day: pd.DataFrame, prev_close: Optional[float], published: datetime, during: bool) -> Dict[str, Optional[float]]:
    """ref_price and the move from it 5, 30 and 120 minutes after the filing and at the close. A filing outside market hours
    is measured from the previous close, its minutes counted from the open."""
    if not len(day):
        return {}
    session_open = datetime.combine(pd.DatetimeIndex(day.index)[0].date(), OPEN, IST)
    ref: Optional[float]
    if during:
        ref, start = price_at(day, published) or float(day["Open"].iloc[0]), published
    else:
        ref, start = prev_close, session_open
    if not ref:
        return {}
    last_end = pd.DatetimeIndex(day.index)[-1] + BAR
    out: Dict[str, Optional[float]] = {"ref_price": ref}
    for minutes in MOVE_MINUTES:
        when = start + timedelta(minutes=minutes)
        price = price_at(day, when) if pd.Timestamp(when) <= last_end else None
        out[f"move_{minutes}m"] = None if price is None else price / ref - 1
    out["move_close"] = float(day["Close"].iloc[-1]) / ref - 1
    return out


@dataclass
class NewsEngine(IntradayEngine):
    """See the module docstring. `universe_fn` gives the day's news universe (refreshed at each session's first step)."""

    entry_until: dtime = ns.B_UNTIL  # the last bar that can signal is the one in progress at 14:30
    label: str = "NEWS"
    version: str = NEWS_VERSION
    max_daily_loss_pct: float = 0.015
    universe_fn: Optional[Callable[[], List[str]]] = None
    max_trades_per_day: int = MAX_TRADES_PER_DAY
    min_position_value: float = MIN_POSITION_VALUE
    _breadth: Optional[float] = field(default=None, init=False)
    _breadth_day: Optional[str] = field(default=None, init=False)
    _measured: Optional[str] = field(default=None, init=False)

    # ------------------------------------------------------------------ the day
    def _new_day(self, today: str) -> None:
        if self.universe_fn is not None:
            try:
                self.universe = list(self.universe_fn())
            except Exception:
                log.exception("could not refresh the news universe; keeping %d stocks", len(self.universe))
        super()._new_day(today)

    def step(self) -> str:
        status = super().step()
        now = self.now_fn()
        if self.clock.is_open(now):
            local = now.astimezone(IST)
            if st.ENTRY_FROM <= local.time() < st.SQUARE_OFF:
                self._breakeven()
        else:
            self._measure(now)
        return status

    # ------------------------------------------------------------------ entries
    def _filings(self, today: str) -> List[NewsFilingRecord]:
        with self.broker.sessions() as s:
            return list(
                s.scalars(
                    select(NewsFilingRecord)
                    .where(NewsFilingRecord.session == today)
                    .where((NewsFilingRecord.rule_status.is_(None)) | (NewsFilingRecord.rule_status == ns.WAITING))
                    .order_by(NewsFilingRecord.published)
                )
            )

    def _verdict(self, row_id: int, status: str, reason: str, kind: Optional[str] = None, ref: Optional[float] = None) -> None:
        with self.broker.sessions() as s:
            row = s.get(NewsFilingRecord, row_id)
            row.rule_status, row.rule_reason = status, reason[:500]
            if kind:
                row.entry_kind = kind
            if ref is not None and row.ref_price is None:
                row.ref_price = ref
            s.commit()

    def _trades_today(self, today: str) -> int:
        with self.broker.sessions() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(IntradaySignalRecord)
                    .where(IntradaySignalRecord.day == today, IntradaySignalRecord.outcome == SignalOutcome.ENTERED)
                )
                or 0
            )

    def _market_filter(self, local: datetime) -> Optional[float]:
        """Rule C: the share of the universe above its previous close at 09:30, measured once a day; None until known."""
        today = local.date().isoformat()
        if self._breadth_day == today:
            return self._breadth
        bars = self.fetch_bars(self.universe)
        sessions, prev = {}, {}
        for symbol, df in bars.items():
            p, day = _prev_close_and_today(df, local.date())
            if p:
                sessions[symbol], prev[symbol] = day, p
        share = ns.breadth(sessions, prev)
        if share is not None:
            self._breadth, self._breadth_day = share, today
            log.info("news market filter: %.0f%% of %d stocks above the previous close at 09:30", share * 100, len(sessions))
        return share

    def _scan(self, pf, local: datetime) -> str:
        today = local.date().isoformat()
        tradable, fired, entered = [], 0, 0
        for row in self._filings(today):
            why = self._not_tradable(row)
            if why:
                self._verdict(row.id, ns.NO, why)
            elif row.label_status == "LABELLED":  # (a filing the AI has not read yet waits for a later step)
                tradable.append(row)
        if not tradable:
            return "no material filings to act on"
        try:
            bars = self.fetch_bars(sorted({r.symbol for r in tradable}))
        except Exception as e:
            self._flag("coverage", True, f"the 5-minute download failed ({type(e).__name__}: {e})")
            raise
        cash = pf.cash
        for row in tradable:
            kind = self._kind(row)
            df = bars.get(row.symbol)
            if df is None or df.empty:
                continue  # no data this cycle: still WAITING
            finished = df[(pd.DatetimeIndex(df.index) + BAR) <= pd.Timestamp(local)]
            prev_close, day = _prev_close_and_today(finished, local.date())
            if kind == "A":
                check = ns.rule_a(row.symbol, day, prev_close)
            else:
                usual = None
                signal_bar = day.index[(pd.DatetimeIndex(day.index) + BAR) > pd.Timestamp(as_utc(row.published))]
                if len(signal_bar):
                    usual = ns.normal_volume(finished, signal_bar[0], local.date())
                check = ns.rule_b(row.symbol, day, as_utc(row.published), prev_close, usual)
            if check.status != ns.FIRED:
                self._verdict(row.id, check.status, check.reason, kind, check.ref_price)
                continue
            fired += 1
            fill = self._try_entry(row, check, kind, pf, cash, local, day)
            if fill is not None:
                entered += 1
                cash -= (fill.filled_qty or 0) * (fill.avg_price or 0.0)
        return f"{len(tradable)} filings checked, fired {fired}, entered {entered}"

    def _not_tradable(self, row: NewsFilingRecord) -> Optional[str]:
        """Why a filing can never be traded (a final NO), or None while it may still be."""
        if row.symbol not in self.universe:
            return "not in the news universe (a swing holding: read for the warning only)"
        if row.label_status == "PENDING":
            return None  # the AI has not read it yet: decided on a later step
        if row.label_status == "ROUTINE":
            return row.material_reason or "routine"
        if row.label_status == "FAILED":
            return row.material_reason or "no AI label"
        if not row.material:
            return f"not material: {row.material_reason}"
        if row.direction != "up":
            return f"{row.direction} news: logged, not traded (long only)"
        local = as_utc(row.published).astimezone(IST)
        if self._kind(row) == "B" and local.time() > ns.B_UNTIL:
            return f"filed at {local:%H:%M}, after {ns.B_UNTIL:%H:%M}"
        return None

    def _kind(self, row: NewsFilingRecord) -> str:
        """A: filed before the session's open (overnight); B: filed during it."""
        local = as_utc(row.published).astimezone(IST)
        return "B" if local.date().isoformat() == row.session and local.time() >= ns.B_FROM else "A"

    def _try_entry(
        self, row, check: ns.RuleCheck, kind: str, pf, cash: float, local: datetime, day: pd.DataFrame
    ) -> Optional[Fill]:
        """Apply the checks a fired rule must still pass, then buy. Returns the fill of an entered trade, else None."""
        setup = check.setup
        assert setup is not None  # a FIRED check always carries its setup
        today = local.date().isoformat()
        if setup.bar_time < day.index[-1] - BAR:
            self._verdict(row.id, ns.NO, f"rule fired on the {setup.bar_time:%H:%M} bar but was seen too late to trade", kind)
            return None
        share = self._market_filter(local)
        if share is None:
            self._verdict(row.id, ns.WAITING, "market filter not known yet", kind, check.ref_price)
            return None
        if share < ns.BREADTH_MIN:
            self._verdict(
                row.id, ns.NO, f"market filter: {share:.0%} of stocks above the previous close at 09:30 (needs 40%)", kind
            )
            return None
        if row.symbol in pf.positions or row.symbol in self._done_today:
            self._verdict(row.id, ns.NO, "already traded or tried today (one trade per stock per day)", kind)
            return None
        if self._trades_today(today) >= self.max_trades_per_day:
            self._verdict(row.id, ns.NO, f"fired, but {self.max_trades_per_day} trades already today", kind)
            return None
        context = self._entry_context(row.symbol, setup.signal_price)
        self._store_context(row, context)
        if context.skip:
            self._verdict(row.id, ns.NO, context.skip, kind, check.ref_price)
            return None
        outcome, fill = self._enter(setup, pf.equity, cash)
        self._verdict(row.id, ns.FIRED, f"{check.reason}; {outcome}", kind, check.ref_price)
        return fill if outcome == SignalOutcome.ENTERED else None

    def _store_context(self, row: NewsFilingRecord, context) -> None:
        """Keep Angel One's record of this entry (spread, circuit limits, F&O data) on the filing."""
        if context.note:
            with self.broker.sessions() as s:
                s.get(NewsFilingRecord, row.id).context_json = json.dumps(context.note)
                s.commit()

    def _enter(self, setup: st.Setup, equity: float, cash: float) -> Tuple[SignalOutcome, Optional[Fill]]:
        qty, _ = st.position_size(equity, setup.signal_price, setup.stop, self.risk_pct, self.max_position_pct, cash)
        if 0 < qty and qty * setup.signal_price < self.min_position_value:
            self._done_today.add(setup.symbol)
            why = f"position Rs {qty * setup.signal_price:,.0f} is under the Rs {self.min_position_value:,.0f} minimum"
            self._record(setup, SignalOutcome.SKIPPED, 0, why)
            return SignalOutcome.SKIPPED, None
        return super()._enter(setup, equity, cash)

    # ------------------------------------------------------------------ exits
    def _breakeven(self) -> None:
        """Move each position's stop to its entry once a finished bar has reached the entry plus the original risk."""
        held = self.broker.holdings()
        if not held:
            return
        today = self.now_fn().astimezone(IST).date()
        with self.broker.sessions() as s:
            signals = {
                r.symbol: r
                for r in s.scalars(
                    select(IntradaySignalRecord).where(
                        IntradaySignalRecord.day == today.isoformat(), IntradaySignalRecord.outcome == SignalOutcome.ENTERED
                    )
                )
            }
        try:
            bars = self.fetch_bars([h["symbol"] for h in held])
        except Exception:
            log.exception("could not download bars for the break-even check")
            return
        now = pd.Timestamp(self.now_fn())
        for h in held:
            sig, df = signals.get(h["symbol"]), bars.get(h["symbol"])
            if sig is None or df is None:
                continue
            day = session_rows(df[(pd.DatetimeIndex(df.index) + BAR) <= now], today)
            after = day[pd.DatetimeIndex(day.index) > pd.Timestamp(sig.bar_time)]
            if not len(after) or not ns.breakeven_due(sig.signal_price, sig.stop, float(after["High"].max())):
                continue
            entry = float(h["avg_price"])
            if self.broker.move_stop(h["symbol"], entry):
                self._say(f"{h['symbol']}: up 1R, stop moved to the entry {entry:,.2f}")

    def _check_signal(self, row, bars, local) -> None:
        """The final bars' close and volume at the signal bar (the inherited check re-runs the breakout rule, which is not
        this engine's rule, so `final_signal` is left empty)."""
        if bars is None or bars.empty:
            return
        day = session_rows(bars, local.date())
        at = pd.Timestamp(row.bar_time)
        if at in day.index:
            row.final_close, row.final_volume = float(day["Close"][at]), float(day["Volume"][at])

    # ------------------------------------------------------------------ after the close
    def _measure(self, now: datetime) -> None:
        """Once a session has finished: each of its filings' moves (traded or not), then one summary line."""
        local = now.astimezone(IST)
        today = local.date().isoformat()
        if self._measured == today or local.weekday() >= 5 or local.time() < ARCHIVE_FROM:
            return
        with self.broker.sessions() as s:
            rows = list(s.scalars(select(NewsFilingRecord).where(NewsFilingRecord.session == today)))
        if not rows:
            self._measured = today
            return
        try:
            bars = self.fetch_bars(sorted({r.symbol for r in rows}))
        except Exception:
            log.exception("could not download bars to measure the filings' moves; will retry")
            return
        with self.broker.sessions() as s:
            rows = [s.get(NewsFilingRecord, r.id) for r in rows]
            for row in rows:
                if row.rule_status in (None, ns.WAITING):
                    row.rule_status = ns.NO
                    row.rule_reason = (row.rule_reason or "never decided") + " (session ended)"
                df = bars.get(row.symbol)
                if df is None or df.empty:
                    continue
                prev_close, day = _prev_close_and_today(df, local.date())
                during = self._kind(row) == "B"
                for k, v in filing_moves(day, prev_close, as_utc(row.published), during).items():
                    setattr(row, k, v)
            s.commit()
        self._measured = today
        labelled = [r for r in rows if r.label_status == "LABELLED"]
        material = [r for r in labelled if r.material]
        self._say(
            f"{today} filings: {len(rows)} on universe stocks, {len(labelled)} read by the AI, {len(material)} material "
            f"({sum(r.direction == 'up' for r in material)} up, {sum(r.direction == 'down' for r in material)} down); "
            f"rule fired {sum(r.rule_status == ns.FIRED for r in rows)}"
        )
