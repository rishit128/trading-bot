"""Angel One's full quote and F&O data for a news-engine entry: the price limits (skip a buy within 1% of the upper circuit),
the bid-ask spread (logged, to see what the 0.05% slippage assumption really is), and the stock's futures positioning (put-call
ratio and open-interest build-up, logged only: nothing trades on it). Requests are spaced, as the broker refuses bursts; an
error is the caller's to catch (the entry then goes ahead on Yahoo's data alone)."""

import re
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional

from src.data.angel import AngelStream

CIRCUIT_MARGIN = 0.01  # no buy at or above the upper circuit less this share
MIN_GAP = 1.2  # seconds between requests
FNO_MAX_AGE = 600.0  # seconds a put-call ratio / build-up listing is reused
BUILD_UPS = ("Long Built Up", "Short Built Up", "Short Covering", "Long Unwinding")
FUTURE = re.compile(r"^(.+?)\d{2}[A-Z]{3}\d{2}FUT$")  # 'INDHOTEL27OCT26FUT' -> INDHOTEL


@dataclass(frozen=True)
class NewsContext:
    skip: Optional[str] = None  # why the entry must not go ahead
    note: Dict[str, object] = field(default_factory=dict)  # what to log with it


def _symbol(future: str) -> Optional[str]:
    m = FUTURE.match(future)
    return m.group(1) if m else None


class AngelQuotes:
    def __init__(self, stream: AngelStream, tokens: Dict[str, str], clock: Callable[[], float] = time.monotonic):
        self.stream, self.tokens, self.clock = stream, tokens, clock
        self.lock = threading.Lock()
        self.last_call = -MIN_GAP
        self.fno: Dict[str, Dict[str, object]] = {}
        self.fno_at = -FNO_MAX_AGE

    def _call(self, request: Callable[[], dict]) -> Any:
        with self.lock:
            time.sleep(max(0.0, self.last_call + MIN_GAP - self.clock()))
            try:
                reply = request()
            finally:
                self.last_call = self.clock()
        if not reply.get("status"):
            raise RuntimeError(f"Angel One refused: {reply.get('message')}")
        return reply["data"]

    def quote(self, symbol: str) -> dict:
        data = self._call(lambda: self.stream.api.getMarketData("FULL", {"NSE": [self.tokens[symbol]]}))
        return data["fetched"][0]

    def _futures(self) -> Dict[str, Dict[str, object]]:
        """symbol -> {pcr, build_up}, for every F&O stock; refreshed at most every FNO_MAX_AGE seconds."""
        if self.clock() - self.fno_at < FNO_MAX_AGE:
            return self.fno
        fno: Dict[str, Dict[str, object]] = {}
        for row in self._call(self.stream.api.putCallRatio):
            if symbol := _symbol(row["tradingSymbol"]):
                fno[symbol] = {"pcr": row["pcr"]}
        for kind in BUILD_UPS:
            rows = self._call(partial(self.stream.api.oIBuildup, {"expirytype": "NEAR", "datatype": kind}))
            for row in rows:
                if (symbol := _symbol(row["tradingSymbol"])) in fno:
                    fno[symbol]["build_up"] = kind
                    fno[symbol]["oi_change_pct"] = round(float(row["percentChange"]), 2)
        self.fno, self.fno_at = fno, self.clock()
        return fno

    def news_context(self, symbol: str, price: float) -> NewsContext:
        """The checks and the record for buying `symbol` at about `price`."""
        q = self.quote(symbol)
        bid, ask = q["depth"]["buy"][0]["price"], q["depth"]["sell"][0]["price"]
        note: Dict[str, object] = {
            "upper_circuit": q["upperCircuit"],
            "lower_circuit": q["lowerCircuit"],
            "bid": bid,
            "ask": ask,
            "spread_pct": round((ask - bid) / ((ask + bid) / 2) * 100, 4) if bid and ask else None,
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
        return NewsContext(skip, note)
