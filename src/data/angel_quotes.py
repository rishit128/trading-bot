"""Angel One's full quotes and F&O data for an entry in any paper account: the price limits (no buy within 1% of the upper
circuit), and, logged only, the bid-ask spread, how lopsided the order book is, where the price sits in its 52-week range, and
the stock's futures positioning (put-call ratio, open-interest build-up). Nothing but the circuit check acts on them: they are
recorded with the entry so a later, pre-declared test can ask whether any of them separates winners from losers.

Quotes come 50 stocks to a request and are reused for QUOTE_TTL seconds, so the engines stepping one after another at 09:30
share one refresh of the whole streamed universe. Requests are spaced, as the broker refuses bursts; an error is the caller's
to catch (the entry then goes ahead on Yahoo's data alone)."""

import re
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

CIRCUIT_MARGIN = 0.01  # no buy at or above the upper circuit less this share
MIN_GAP = 1.2  # seconds between requests
BATCH = 50  # stocks per quote request
QUOTE_TTL = 60.0  # seconds a quote is reused
FNO_MAX_AGE = 600.0  # seconds a put-call ratio / build-up listing is reused
BUILD_UPS = ("Long Built Up", "Short Built Up", "Short Covering", "Long Unwinding")
FUTURE = re.compile(r"^(.+?)\d{2}[A-Z]{3}\d{2}FUT$")  # 'INDHOTEL27OCT26FUT' -> INDHOTEL


@dataclass(frozen=True)
class EntryContext:
    skip: Optional[str] = None  # why the entry must not go ahead
    note: Dict[str, object] = field(default_factory=dict)  # what to log with it


def _full_quotes(api, ids: List[str]) -> dict:
    return api.getMarketData("FULL", {"NSE": ids})


def _build_up(api, kind: str) -> dict:
    return api.oIBuildup({"expirytype": "NEAR", "datatype": kind})


def _symbol(future: str) -> Optional[str]:
    m = FUTURE.match(future)
    return m.group(1) if m else None


class AngelQuotes:
    """`api_fn` gives the logged-in session (the stream's, or an AngelSession's); `universe` is refreshed together when any of
    its stocks is asked for."""

    def __init__(
        self,
        api_fn: Callable[[], Any],
        tokens: Dict[str, str],
        universe: Optional[List[str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_fn, self.tokens, self.clock = api_fn, tokens, clock
        self.universe = {s for s in universe or [] if s in tokens}
        self.lock = threading.Lock()
        self.last_call = -MIN_GAP
        self.quotes: Dict[str, tuple] = {}  # symbol -> (time, quote)
        self.fno: Dict[str, Dict[str, object]] = {}
        self.fno_at = -FNO_MAX_AGE

    def _call(self, request: Callable[[Any], dict]) -> Any:
        with self.lock:
            time.sleep(max(0.0, self.last_call + MIN_GAP - self.clock()))
            try:
                reply = request(self.api_fn())
            finally:
                self.last_call = self.clock()
        if not reply.get("status"):
            raise RuntimeError(f"Angel One refused: {reply.get('message')}")
        return reply["data"]

    def quote(self, symbol: str) -> dict:
        hit = self.quotes.get(symbol)
        if hit is None or self.clock() - hit[0] >= QUOTE_TTL:
            wanted = sorted(self.universe) if symbol in self.universe else [symbol]
            for start in range(0, len(wanted), BATCH):
                chunk = {self.tokens[s]: s for s in wanted[start : start + BATCH]}
                data = self._call(partial(_full_quotes, ids=list(chunk)))
                for q in data["fetched"]:
                    if q["symbolToken"] in chunk:
                        self.quotes[chunk[q["symbolToken"]]] = (self.clock(), q)
            hit = self.quotes.get(symbol)
            if hit is None:
                raise RuntimeError(f"Angel One returned no quote for {symbol}")
        return hit[1]

    def _futures(self) -> Dict[str, Dict[str, object]]:
        """symbol -> {pcr, build_up}, for every F&O stock; refreshed at most every FNO_MAX_AGE seconds."""
        if self.clock() - self.fno_at < FNO_MAX_AGE:
            return self.fno
        fno: Dict[str, Dict[str, object]] = {}
        for row in self._call(lambda api: api.putCallRatio()):
            if symbol := _symbol(row["tradingSymbol"]):
                fno[symbol] = {"pcr": row["pcr"]}
        for kind in BUILD_UPS:
            for row in self._call(partial(_build_up, kind=kind)):
                if (symbol := _symbol(row["tradingSymbol"])) in fno:
                    fno[symbol]["build_up"] = kind
                    fno[symbol]["oi_change_pct"] = round(float(row["percentChange"]), 2)
        self.fno, self.fno_at = fno, self.clock()
        return fno

    def entry_context(self, symbol: str, price: float) -> EntryContext:
        """The checks and the record for buying `symbol` at about `price`."""
        q = self.quote(symbol)
        bid, ask = q["depth"]["buy"][0]["price"], q["depth"]["sell"][0]["price"]
        buyers, sellers = q.get("totBuyQuan") or 0, q.get("totSellQuan") or 0
        low, high = q.get("52WeekLow") or 0, q.get("52WeekHigh") or 0
        note: Dict[str, object] = {
            "upper_circuit": q["upperCircuit"],
            "lower_circuit": q["lowerCircuit"],
            "bid": bid,
            "ask": ask,
            "spread_pct": round((ask - bid) / ((ask + bid) / 2) * 100, 4) if bid and ask else None,
            "book_imbalance": round((buyers - sellers) / (buyers + sellers), 3) if buyers + sellers else None,  # +1 all buyers
            "range_52w": round((price - low) / (high - low), 3) if high > low else None,  # 0 at the 52-week low, 1 at the high
        }
        try:
            note["fno"] = self._futures().get(symbol)  # context only: a failure here never stops the trade
        except Exception as e:
            note["fno"] = f"unavailable ({type(e).__name__})"
        upper = q["upperCircuit"]
        skip = (
            f"price {price:,.2f} within 1% of the upper circuit {upper:,.2f}"
            if upper and price >= upper * (1 - CIRCUIT_MARGIN)
            else None
        )
        return EntryContext(skip, note)
