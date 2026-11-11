"""The broker-calls paper account: it follows the broker's Intraday BUY calls with its own Rs 30,000 and nothing else, to measure
what following them would earn (the owner's decision, 2026-11-05; not proven, see STRATEGY.md).

The calls come from the research watch (`/research_start`: one OTP on Telegram a day, then the portal is read every 10 minutes), so
an entry can be that much later than the call, as for anyone not at the screen. No read, no entries; an alert says so.

Entry: every open Intraday BUY call published today, at the next 5-minute step. Refused, with the reason recorded, only when the
price is already past the call's stop or target, when 5 positions are open, or when position_size cannot buy a share (one share
above the 20% limit, or no cash). No breakout, volume or market check. One attempt per stock per day.
Exit, whichever comes first: the call's stop, the call's target, the broker closing the call (a "book profit" message, or its own
target or stop-loss note: sold at the market price on the step that sees it), or the 15:15 square-off.

SELL calls are not traded (the paper account can only buy). They stay in house_call_events, and scripts/score_house_calls.py scores
them with the rest. Sizing and the daily-loss limit are the other intraday accounts'."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dtime
from typing import List, Optional

import pandas as pd

from src.data.house_calls import INTRADAY_KINDS, HouseCall
from src.data.india import IST
from src.data.intraday_bars import session_rows
from src.database import IntradaySignalRecord
from src.engine.enums import SignalOutcome
from src.engine.ports import BAR
from src.intraday import strategy as st
from src.intraday.engine import IntradayEngine
from src.versions import CALLS_VERSION

log = logging.getLogger(__name__)

CALLS_CASH = 30_000.0  # this account's starting balance; the other paper accounts use PAPER_INITIAL_CASH
# calls-v2 (2026-11-06, the owner's decision): the broker publishes 2-3 Intraday calls a day, so 3 positions sharing Rs 28,000 of the
# Rs 30,000 (each up to 28/90 of equity, about Rs 9,333) instead of the shared 5 x 20%, which left most of the balance idle. The rest
# stays as cash for fees and slippage. The 0.5% risk rule still applies.
CALLS_MAX_POSITIONS, CALLS_DEPLOYED = 3, 28_000 / CALLS_CASH
CALLS_MAX_POSITION_PCT = CALLS_DEPLOYED / CALLS_MAX_POSITIONS
ENTRY_UNTIL = dtime(15, 0)  # the broker publishes all day; nothing is bought in the last few steps before the square-off
ALERT_FROM = dtime(9, 35)  # no fresh read by now: tell the owner, as the account cannot trade without one


def _todays_buys(calls: List[HouseCall], today: date) -> List[HouseCall]:
    """Today's Intraday BUY calls, open or closed, oldest first."""
    mine = [c for c in calls if c.kind_key in INTRADAY_KINDS and c.side == "buy" and c.published.astimezone(IST).date() == today]
    return sorted(mine, key=lambda c: c.published)


@dataclass
class CallsEngine(IntradayEngine):
    """See the module docstring. `universe` is empty: the day's calls are the only stocks it looks at."""

    entry_until: dtime = ENTRY_UNTIL
    max_positions: int = CALLS_MAX_POSITIONS
    max_position_pct: float = CALLS_MAX_POSITION_PCT
    label: str = "BROKER-CALLS"
    version: str = CALLS_VERSION
    record_capped: bool = True

    def step(self) -> str:
        status = super().step()
        now = self.now_fn()
        if self.clock.is_open(now) and now.astimezone(IST).time() < st.SQUARE_OFF:
            self._follow_broker_exits(now.astimezone(IST).date())
        return status

    def _scan(self, pf, local: datetime) -> str:
        self._calls = self._read_house_calls()
        self._flag(
            "no-calls",
            not self._calls and local.time() >= ALERT_FROM,
            "no fresh read of the broker's calls, so this account cannot enter. Send /research_start and type the OTP.",
        )
        today = local.date()
        wanted: dict = {}
        for call in _todays_buys(self._calls, today):
            if call.is_open and call.symbol not in pf.positions and call.symbol not in self._done_today:
                wanted.setdefault(call.symbol, call)
        if not wanted:
            return "no new broker calls"
        try:
            bars = self.fetch_bars(sorted(wanted))
        except Exception as e:
            self._flag("coverage", True, f"the 5-minute download failed ({type(e).__name__}: {e}): broker calls not checked")
            raise
        self._flag("coverage", False, "")
        self._with_data = max(self._with_data, len(bars))
        entered, cash = 0, pf.cash
        for symbol, call in wanted.items():
            df = bars.get(symbol)
            if df is None or df.empty:
                continue  # no price yet: tried again on the next step
            finished = df[(pd.DatetimeIndex(df.index) + BAR) <= pd.Timestamp(local)]
            day = session_rows(finished, today)
            if not len(day):
                continue
            price = float(day["Close"].iloc[-1])
            self._seen_volume[symbol] = float(day["Volume"].iloc[-1])
            setup = st.Setup(symbol, price, call.stop, call.target, call.entry_high, call.entry_low, day.index[-1])
            why = self._refusal(call, price, len(pf.positions) + entered)
            if why:
                self._done_today.add(symbol)
                self._record(setup, SignalOutcome.SKIPPED, 0, why)
                continue
            outcome, fill = self._enter(setup, pf.equity, cash)
            if outcome == SignalOutcome.ENTERED and fill is not None:
                entered += 1
                cash -= (fill.filled_qty or 0) * (fill.avg_price or price)
        return f"{len(wanted)} broker calls checked, entered {entered}"

    def _refusal(self, call: HouseCall, price: float, held: int) -> Optional[str]:
        if not call.stop < price < call.target:
            return f"price {price:,.2f} is already outside the call's stop {call.stop:,.2f} and target {call.target:,.2f}"
        if held >= self.max_positions:
            return f"position limit reached ({self.max_positions} open)"
        return None

    def _entry_text(self, setup: st.Setup, qty: int) -> str:
        return (
            f"BUY {setup.symbol} x{qty} on the broker's Intraday call (range {setup.range_low:,.2f}-{setup.range_high:,.2f}); "
            f"stop {setup.stop:,.2f}, target {setup.target:,.2f}"
        )

    def _follow_broker_exits(self, today: date) -> None:
        """Sell a held stock once the broker has closed all of today's BUY calls on it. A stale read shows no calls, so nothing is sold
        on it: the stop, the target and the square-off still protect the position."""
        held = self.broker.holdings()
        if not held:
            return
        calls = _todays_buys(self._read_house_calls(), today)
        for h in held:
            symbol = h["symbol"]
            mine = [c for c in calls if c.symbol == symbol]
            if not mine or any(c.is_open for c in mine):
                continue
            try:
                self.broker.sell(symbol, h["qty"])
            except Exception as e:
                log.warning("broker exit %s failed: %s", symbol, e)
                self._flag(
                    f"exit {symbol}", True, f"could not sell {symbol} after the broker closed its call ({e}): retried next step"
                )
                continue
            self._flags.discard(f"exit {symbol}")
            self._known.pop(symbol, None)
            trade = self.broker.trade_history(limit=1)[0]
            self._say(f"BROKER EXIT {symbol} ({mine[-1].exit_comment or 'call closed'}): {trade['net_pnl']:+,.0f} net")

    def _check_signal(self, row: IntradaySignalRecord, bars: Optional[pd.DataFrame], local: datetime) -> None:
        """The final bars' close and volume at the signal bar. There is no rule to re-run, so `final_signal` stays empty."""
        if bars is None or bars.empty:
            return
        day = session_rows(bars, local.date())
        at = pd.Timestamp(row.bar_time)
        if at in day.index:
            row.final_close, row.final_volume = float(day["Close"][at]), float(day["Volume"][at])
