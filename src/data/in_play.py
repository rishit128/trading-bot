"""The whole NSE market, read every few minutes from Angel One's quotes (orb-v8, the owner's decision of 2026-11-16).

The breakout account's fixed list of 500 stocks misses most of each day's movers: on 16 Nov 2026 at 10:30, 45 liquid stocks
outside it were up 2% or more, against 25 inside. So every IN_PLAY_EVERY seconds, from the first breakout bar to the last
entry, `InPlay` asks Angel One for a quote of every NSE equity (about 3,000, IN_PLAY_BATCH a request) and:

- adds to the day's scan the liquid movers outside the list (`movers`): up at least IN_PLAY_GAIN on the day, at least
  IN_PLAY_MIN_VALUE traded so far, priced at least IN_PLAY_MIN_PRICE and not pinned near the upper circuit (where nothing can
  be bought), the most traded first, at most IN_PLAY_MAX a day. The stream subscribes them and fetches their earlier bars,
  so the breakout rule sees them exactly as it sees the list's stocks;
- keeps every stock's gain on the day (`gains`), so the market and sector checks read the whole market rather than the list.

Quotes only: nothing here places an order. A failed scan keeps the previous one; a scan older than IN_PLAY_MAX_AGE is not
used (the checks then read the list's own bars)."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from datetime import time as dtime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.data.india import IST

log = logging.getLogger(__name__)

IN_PLAY_GAIN = 0.02  # the breakout rule's own leader line (src.intraday.strategy.ORB_MIN_GAIN)
IN_PLAY_MIN_VALUE = 5e7  # Rs 5 crore traded so far today: a Rs 5,000 position is a rounding error in it
IN_PLAY_MIN_PRICE = 50.0
IN_PLAY_CIRCUIT_ROOM = 0.01  # a price within 1% of the upper circuit: sellers have gone, a buy may not fill
IN_PLAY_MAX = 100  # stocks added a day (the stream's limit is 1,000 tokens; the list holds 500)
IN_PLAY_EVERY = 300.0  # seconds between two scans of the whole market
IN_PLAY_MAX_AGE = 900.0  # seconds a scan is used for
IN_PLAY_BATCH, IN_PLAY_PACE = 50, 0.4  # stocks a quote request, seconds between requests (the broker allows 10 a second)
IN_PLAY_FROM, IN_PLAY_UNTIL = dtime(9, 20), dtime(14, 0)  # the breakout rule's entry window
MIN_QUOTED = 500  # a scan that priced fewer stocks is not a reading of the market


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    previous_close: float
    value: float  # rupees traded today
    upper_circuit: Optional[float]


def parse_quotes(fetched: Sequence[Mapping[str, Any]], symbol_of: Mapping[str, str]) -> List[Quote]:
    """Angel One's FULL quotes -> Quote, for the stocks that traded today and have a previous close."""
    out = []
    for q in fetched:
        symbol = symbol_of.get(str(q.get("symbolToken", "")))
        try:
            price, close, volume = float(q.get("ltp") or 0), float(q.get("close") or 0), float(q.get("tradeVolume") or 0)
            upper = float(q["upperCircuit"]) if q.get("upperCircuit") else None
        except (TypeError, ValueError):
            continue
        if symbol and price > 0 and close > 0 and volume > 0:
            out.append(Quote(symbol, price, close, volume * price, upper))
    return out


def movers(quotes: Sequence[Quote], exclude: Sequence[str] = ()) -> List[str]:
    """The liquid stocks up IN_PLAY_GAIN on the day and not near their upper circuit, the most traded first."""
    skip = set(exclude)
    picked = [
        q
        for q in quotes
        if q.symbol not in skip
        and q.price / q.previous_close - 1 >= IN_PLAY_GAIN
        and q.value >= IN_PLAY_MIN_VALUE
        and q.price >= IN_PLAY_MIN_PRICE
        and not (q.upper_circuit and q.price >= q.upper_circuit * (1 - IN_PLAY_CIRCUIT_ROOM))
    ]
    return [q.symbol for q in sorted(picked, key=lambda q: -q.value)]


@dataclass
class InPlay:
    """See the module docstring. `api_fn` gives the logged-in session (the stream's); `tokens` maps symbol -> token for every
    NSE equity; `add` subscribes newly picked stocks (AngelStream.add)."""

    api_fn: Callable[[], Any]
    tokens: Mapping[str, str]
    base: Sequence[str]  # the fixed list: never "added"
    add: Callable[[List[str]], None]
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    clock: Callable[[], float] = time.monotonic
    pace: float = IN_PLAY_PACE
    _added: List[str] = field(default_factory=list, init=False)
    _gains: Dict[str, float] = field(default_factory=dict, init=False)
    _at: float = field(default=float("-inf"), init=False)
    _day: Optional[date] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def symbols(self) -> List[str]:
        """The stocks added to today's scan."""
        with self._lock:
            return list(self._added) if self._day == self.now_fn().astimezone(IST).date() else []

    def gains(self) -> Optional[Dict[str, float]]:
        """Every quoted stock's gain on the day from the latest scan, or None when it is stale or from another day."""
        with self._lock:
            fresh = self.clock() - self._at <= IN_PLAY_MAX_AGE and self._day == self.now_fn().astimezone(IST).date()
            return dict(self._gains) if fresh and self._gains else None

    def breadth(self) -> Optional[float]:
        """The share of all quoted NSE stocks above their previous close, or None (see `gains`)."""
        gains = self.gains()
        return sum(g > 0 for g in gains.values()) / len(gains) if gains else None

    def quote_all(self) -> List[Quote]:
        api, symbol_of = self.api_fn(), {t: s for s, t in self.tokens.items()}
        if api is None:
            return []
        ids = list(symbol_of)
        fetched: List[Mapping[str, Any]] = []
        for i in range(0, len(ids), IN_PLAY_BATCH):
            try:
                reply = api.getMarketData("FULL", {"NSE": ids[i : i + IN_PLAY_BATCH]}) or {}
                fetched += (reply.get("data") or {}).get("fetched") or []
            except Exception as e:  # one refused batch leaves 50 stocks out of this scan, nothing more
                log.debug("in-play quotes %d failed: %s: %s", i, type(e).__name__, e)
            if self.pace:
                time.sleep(self.pace)
        return parse_quotes(fetched, symbol_of)

    def scan(self) -> int:
        """One scan of the whole market; returns how many stocks it added."""
        today = self.now_fn().astimezone(IST).date()
        quotes = self.quote_all()
        if len(quotes) < MIN_QUOTED:
            log.warning("in-play scan priced only %d stocks; the previous scan is kept", len(quotes))
            return 0
        with self._lock:
            if self._day != today:
                self._day, self._added = today, []
            self._gains = {q.symbol: q.price / q.previous_close - 1 for q in quotes}
            self._at = self.clock()
            room = IN_PLAY_MAX - len(self._added)
            new = movers(quotes, exclude=list(self.base) + self._added)[: max(room, 0)]
            self._added += new
        if new:
            log.info("in-play: %d stocks added to the scan (%s)", len(new), ", ".join(new[:10]))
            self.add(new)
        return len(new)

    def run(self) -> None:
        while True:
            local = self.now_fn().astimezone(IST)
            if local.weekday() < 5 and IN_PLAY_FROM <= local.time() < IN_PLAY_UNTIL:
                started = self.clock()
                try:
                    self.scan()
                except Exception:
                    log.exception("in-play scan failed")
                time.sleep(max(5.0, IN_PLAY_EVERY - (self.clock() - started)))
            else:
                time.sleep(30)

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True, name="in-play").start()
