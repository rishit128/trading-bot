"""Angel One's live market data (the broker's own feed; Yahoo stays the backup): log in, stream the intraday stocks' ticks,
build 5-minute bars and save them in `angel_bars`. src.data.angel_bars serves them to the engines, src.data.angel_quotes gives
the circuit limits, spread and F&O context. Only data calls are made, never orders. Credentials come from ANGEL_* in .env and
are never logged."""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.data.india import IST

log = logging.getLogger(__name__)

BAR = timedelta(minutes=5)
CACHE_DIR = Path("research_cache")  # the BSE -> NSE symbol map and the Angel One token list (git-ignored)
TOKEN_FILE = "angel_tokens.json"
TOKEN_MAX_AGE = 86400  # the instrument list is refreshed daily
SWEEP = 2.0  # seconds between checks for bars whose five minutes have passed
STALE_AFTER = 90.0  # seconds without a tick before the stream counts as down
SESSION_START, SESSION_END = dtime(9, 15), dtime(15, 30)
LOGIN_FROM, LOGIN_UNTIL = (8, 45), (15, 40)  # IST: log in before the open, drop the stream after the close
QUOTE_MODE = 2  # last price and the day's cumulative volume
CANDLE_PACE = 1.1  # seconds between historical-candle requests (the broker refuses faster ones: 'exceeding access rate')
CANDLE_TRIES = 3
HISTORY_DAYS = 10  # calendar days of earlier sessions fetched for a stock with none saved (about 7 sessions)
REQUIRED_KEYS = ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET")


def credentials_set() -> bool:
    return all(os.getenv(k) for k in REQUIRED_KEYS)


def eq_tokens(master: List[dict]) -> Dict[str, str]:
    """NSE symbol -> Angel One token, for plain equity (-EQ) instruments only: the BE (trade-to-trade) series is left out."""
    return {r["name"]: str(r["token"]) for r in master if r.get("exch_seg") == "NSE" and str(r.get("symbol", "")).endswith("-EQ")}


def load_tokens(cache_dir: Path) -> Dict[str, str]:
    """The symbol -> token map from a daily cache; a failed refresh keeps the old copy, and with none at all it raises."""
    from src.data.filings import scrip_master

    path = cache_dir / TOKEN_FILE
    if path.exists() and time.time() - path.stat().st_mtime < TOKEN_MAX_AGE:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        tokens = eq_tokens(scrip_master())
    except Exception:
        if path.exists():
            log.exception("could not refresh the Angel One instrument list; using the old copy")
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokens), encoding="utf-8")
    return tokens


def _silence_sdk_log() -> None:
    """The SDK logs every failed request with its headers, which carry the login token and key, to the console and to
    logs/<date>/app.log; our own code logs the failures without them."""
    import logzero

    logzero.logger.setLevel(logging.CRITICAL)
    for handler in list(logzero.logger.handlers):
        logzero.logger.removeHandler(handler)


def login():
    """A logged-in SmartConnect session (raises when the broker refuses)."""
    import pyotp
    from SmartApi import SmartConnect

    api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
    _silence_sdk_log()
    totp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
    reply = api.generateSession(os.environ["ANGEL_CLIENT_CODE"], os.environ["ANGEL_PIN"], totp)
    if not reply.get("status"):
        raise RuntimeError(f"Angel One login refused: {reply.get('message')}")
    return api, reply["data"]["jwtToken"], reply["data"]["feedToken"]


class AngelSession:
    """A logged-in session for quote calls from a process that does not stream (the swing bot): it logs in at first use and
    again on each new day, as the broker's tokens last one day."""

    def __init__(self):
        self._api: Any = None
        self._day = ""
        self._lock = threading.Lock()

    def api(self) -> Any:
        today = datetime.now(IST).date().isoformat()
        with self._lock:
            if self._api is None or self._day != today:
                self._api, _, _ = login()
                self._day = today
            return self._api


def bar_start(when: datetime) -> datetime:
    """The UTC start of the 5-minute bar containing `when`."""
    local = when.astimezone(IST)
    return local.replace(minute=local.minute - local.minute % 5, second=0, microsecond=0).astimezone(timezone.utc)


class BarBuilder:
    """Turns ticks into 5-minute bars. A bar is complete once its five minutes have passed; its volume is the day's cumulative
    volume at its last tick less that at the previous bar's last tick. Ticks outside 09:15-15:30 (pre-open, closing session)
    are not part of any bar."""

    def __init__(self):
        self.building: Dict[str, dict] = {}  # symbol -> the bar being built
        self.previous_volume: Dict[str, float] = {}  # symbol -> cumulative volume at the end of its last finished bar
        self.finished: List[dict] = []

    def tick(self, symbol: str, when: datetime, price: float, cumulative_volume: float) -> None:
        if not SESSION_START <= when.astimezone(IST).time() < SESSION_END:
            return
        start = bar_start(when)
        bar = self.building.get(symbol)
        if bar is not None and start < bar["ts"]:
            return  # a late tick for a bar already passed
        if bar is not None and start > bar["ts"]:
            self._finish(symbol)
            bar = None
        if bar is None:
            bar = self.building[symbol] = {"symbol": symbol, "ts": start, "open": price, "high": price, "low": price}
        bar["high"], bar["low"], bar["close"], bar["cum"] = (
            max(bar["high"], price),
            min(bar["low"], price),
            price,
            cumulative_volume,
        )

    def _finish(self, symbol: str) -> None:
        bar = self.building.pop(symbol)
        before = self.previous_volume.get(symbol)
        self.previous_volume[symbol] = bar["cum"]
        if before is None and bar["ts"].astimezone(IST).time() != SESSION_START:
            return  # the stream joined mid-session: this bar's start and its volume are unknown (the backfill supplies it)
        bar["volume"] = bar["cum"] - (before or 0.0)
        self.finished.append(bar)

    def take(self, now: datetime) -> List[dict]:
        """The finished bars, including those whose five minutes have ended by `now` with no later tick (a quiet stock)."""
        for symbol in [s for s, b in self.building.items() if b["ts"] + BAR <= now]:
            self._finish(symbol)
        out, self.finished = self.finished, []
        return out


def bar_row(symbol: str, ts: datetime, o: float, h: float, low: float, c: float, volume: float, seen_at: datetime) -> dict:
    return {
        "symbol": symbol,
        "ts": ts,
        "day": ts.astimezone(IST).date().isoformat(),
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": volume,
        "seen_at": seen_at,
    }


class AngelStream:
    """Streams the given stocks and saves each finished 5-minute bar. `run` keeps the stream up through the session, logging in
    afresh each day; a failed login or dropped stream is retried, and the first failure of a day is sent to `notify`. After a
    (re)connect the bars the stream has not seen are fetched from Angel One's candles; until then `ready()` is False
    and the engines read Yahoo."""

    def __init__(self, sessions, symbols: List[str], tokens: Dict[str, str], notify: Optional[Callable[[str], object]] = None):
        self.sessions, self.notify = sessions, notify
        self.tokens = {s: tokens[s] for s in symbols if s in tokens}
        self.symbol_of = {token: s for s, token in self.tokens.items()}
        self.builder = BarBuilder()
        self.lock = threading.Lock()
        self.api: Any = None  # the logged-in session, shared with src.data.angel_quotes
        self.prices: Dict[str, float] = {}
        self.last_tick = 0.0
        self.backfilled = False
        self.alerted_on = ""
        self.logged_in_on = ""

    def ready(self) -> bool:
        """True while ticks are arriving and the day's earlier bars are in."""
        return self.backfilled and time.monotonic() - self.last_tick < STALE_AFTER

    def last_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol)

    def forming(self, symbol: str) -> Optional[dict]:
        """The bar being built for `symbol` (its highs and lows so far are real prints), or None."""
        with self.lock:
            bar = self.builder.building.get(symbol)
            return dict(bar) if bar else None

    def on_tick(self, _ws, message) -> None:
        if not isinstance(message, dict):
            return
        symbol, stamp = self.symbol_of.get(str(message.get("token", "")).strip('"')), message.get("exchange_timestamp")
        if symbol is None or not stamp:
            return
        price = message["last_traded_price"] / 100
        with self.lock:
            self.last_tick, self.prices[symbol] = time.monotonic(), price
            self.builder.tick(
                symbol,
                datetime.fromtimestamp(stamp / 1000, timezone.utc),
                price,
                float(message.get("volume_trade_for_the_day", 0)),
            )

    def save(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        with self.lock:
            bars = self.builder.take(now)
        return self._insert(
            [bar_row(b["symbol"], b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"], now) for b in bars]
        )

    def _insert(self, rows: List[dict]) -> int:
        from sqlalchemy.dialects.sqlite import insert

        from src.database import AngelBarRecord

        if rows:
            with self.sessions() as s:
                s.execute(insert(AngelBarRecord).prefix_with("OR IGNORE"), rows)
                s.commit()
        return len(rows)

    def has_history(self) -> set:
        """The symbols with bars on an earlier session already saved."""
        from sqlalchemy import select

        from src.database import AngelBarRecord

        today = datetime.now(IST).date().isoformat()
        with self.sessions() as s:
            return set(s.scalars(select(AngelBarRecord.symbol).where(AngelBarRecord.day < today).distinct()))

    def backfill(self) -> None:
        """The bars the stream has not seen, from the broker's candles: today's earlier bars, and for a stock with no earlier
        session saved the previous HISTORY_DAYS calendar days too (one request each), so the engines have the previous close
        and time-of-day volumes from the first day. Mid-session it starts at the next bar boundary, so that the bar the stream
        joined in the middle of is complete."""
        now = datetime.now(timezone.utc)
        if now.astimezone(IST).time() >= dtime(9, 20):
            time.sleep(max(0.0, (bar_start(now) + BAR - now).total_seconds() + 3))
        today, known, failed = datetime.now(IST).date(), self.has_history(), 0
        for symbol, token in self.tokens.items():
            start = today if symbol in known else today - timedelta(days=HISTORY_DAYS)
            candles: list = []
            for attempt in range(CANDLE_TRIES):
                end = datetime.now(IST)
                try:
                    reply = self.api.getCandleData(
                        {
                            "exchange": "NSE",
                            "symboltoken": token,
                            "interval": "FIVE_MINUTE",
                            "fromdate": f"{start} 09:15",
                            "todate": f"{end:%Y-%m-%d %H:%M}",
                        }
                    )
                    candles = reply.get("data") or []
                    break
                except Exception as e:
                    log.debug(
                        "candles for %s failed (try %d of %d): %s: %s", symbol, attempt + 1, CANDLE_TRIES, type(e).__name__, e
                    )
                    time.sleep(CANDLE_PACE * 3 * (attempt + 1))
            else:
                log.warning("candles for %s: gave up after %d tries", symbol, CANDLE_TRIES)
                failed += 1
            seen = datetime.now(timezone.utc)
            rows = []
            for stamp, o, h, low, c, v in candles:
                ts = pd.Timestamp(stamp).tz_convert("UTC").to_pydatetime()
                if ts + BAR <= seen:  # a bar still in progress would be saved incomplete and win over the stream's
                    rows.append(bar_row(symbol, ts, o, h, low, c, float(v), seen))
            self._insert(rows)
            time.sleep(CANDLE_PACE)
        log.info("Angel One backfill done for %d stocks (%d requests failed)", len(self.tokens), failed)
        self.backfilled = True

    def _alert_once(self, text: str) -> None:
        today = datetime.now(IST).date().isoformat()
        if self.notify and self.alerted_on != today:
            self.alerted_on = today
            self.notify(f"[ANGEL] ⚠️ {text}; Yahoo is used instead")

    def _tell_login(self) -> None:
        """Telegram: the day's first successful login (a dropped stream that reconnects is not announced again)."""
        today = datetime.now(IST).date().isoformat()
        if self.notify and self.logged_in_on != today:
            self.logged_in_on = today
            self.notify(f"[ANGEL] ✅ logged in at {datetime.now(IST):%H:%M}; streaming {len(self.symbol_of)} stocks")

    def _stream(self) -> None:
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2

        self.api, jwt, feed = login()
        self._tell_login()
        ws = SmartWebSocketV2(jwt, os.environ["ANGEL_API_KEY"], os.environ["ANGEL_CLIENT_CODE"], feed)
        ws.on_open = lambda _w: ws.subscribe("bars", QUOTE_MODE, [{"exchangeType": 1, "tokens": list(self.symbol_of)}])
        ws.on_data = self.on_tick
        ws.on_error = lambda _w, *e: log.warning("Angel One stream error: %s", e)
        stopped = threading.Event()
        ws.on_close = lambda _w: stopped.set()
        self.builder, self.backfilled = BarBuilder(), False
        threading.Thread(target=ws.connect, daemon=True, name="angel-ws").start()
        threading.Thread(target=self.backfill, daemon=True, name="angel-backfill").start()
        while not stopped.is_set() and self._in_hours():
            time.sleep(SWEEP)
            self.save()
        ws.close_connection()
        self.save(datetime.now(timezone.utc) + BAR)  # the last bar of the session

    @staticmethod
    def _in_hours() -> bool:
        local = datetime.now(IST)
        return local.weekday() < 5 and LOGIN_FROM <= (local.hour, local.minute) < LOGIN_UNTIL

    def run(self) -> None:
        wait = 30.0
        while True:
            if not self._in_hours():
                time.sleep(60)
                continue
            try:
                self._stream()
                wait = 30.0
                time.sleep(5)  # a stream that dropped is reconnected, not hammered
            except Exception as e:
                log.warning("Angel One stream failed (%s: %s); retrying in %.0fs", type(e).__name__, str(e)[:200], wait)
                self._alert_once(f"Angel One stream or login failed ({type(e).__name__})")
                time.sleep(wait)
                wait = min(wait * 2, 600)

    def start(self) -> None:
        threading.Thread(target=self.run, name="angel-stream", daemon=True).start()


def make_stream(sessions, universe: List[str], cache_dir: Path, notify) -> Optional[Tuple[AngelStream, Dict[str, str]]]:
    """The stream and the full token map, or None (with the reason logged) when Angel One cannot be used."""
    if not credentials_set():
        log.warning("angel: ANGEL_* credentials missing, Yahoo only")
        return None
    try:
        tokens = load_tokens(cache_dir)
    except Exception as e:
        log.warning("angel: no instrument list (%s: %s), Yahoo only", type(e).__name__, e)
        return None
    return AngelStream(sessions, universe, tokens, notify), tokens
