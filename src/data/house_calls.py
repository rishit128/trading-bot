"""The broker's research calls ("house calls") from the Integrated portal: parsing the portal's own data response, and the
one-snapshot store the trading code reads them from.

Where they come from: the read-only portfolio agent stays logged in to the portal during market hours (`/research_start`
on Telegram, one OTP a day) and reads the Research list every few minutes. Each call has the stock, the call type
(Intraday, BTST, Swing Trade, Momentum, Stellar Pick), buy or sell, the recommended entry range, stop, target, when it was
published, how long it is valid, and whether it is still open (and if not, how it ended: "Target Hit", "Stoploss Hit").

How they are used (nothing here trades):
  * the swing bot adds the stocks of open Swing Trade / Momentum / Stellar Pick buy calls to its candidates, and shows the
    AI the call on any stock it analyses (`prompt_line`); its own filter, AI and risk engine still decide;
  * the opening-range breakout engine adds the stocks of today's open Intraday buy calls to its scan, and does not buy a
    stock the house has an open SELL call on.

Trading reads only the latest read (`HouseCallStore.replace` swaps it whole); a snapshot older than `MAX_AGE` is treated
as no snapshot at all, so a day without the portal session trades exactly as before. Each read is also added to an
append-only history (`house_call_events`: one row per state of a call, with when it was first and last seen), so the
calls themselves can be scored later. Trading never reads the history."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import delete, func, select

from src.data.india import IST
from src.database import HouseCallEvent, HouseCallRecord

log = logging.getLogger(__name__)

MAX_AGE = timedelta(minutes=30)  # three missed 10-minute reads: the session is gone, stop relying on the calls
SWING_KINDS = frozenset({"swing trade", "momentum", "stellar pick"})  # held for days to weeks: swing bot candidates
INTRADAY_KINDS = frozenset({"intraday"})  # closed the same day: the breakout engine's scan
NO_EXIT = "1900-01-01"  # the portal's ExitDtm for a call that has not been closed
REQUIRED = ("ScripCode", "StopLoss", "TargetPrice", "CMPFrom", "PublishedDtm")


@dataclass(frozen=True)
class HouseCall:
    """One research call, as the portal published it. Prices in rupees; `published` is IST-aware."""

    symbol: str
    kind: str
    side: str
    entry_low: float
    entry_high: float
    stop: float
    target: float
    published: datetime
    valid_to: Optional[date]
    is_open: bool
    exit_comment: str = ""
    exit_at: Optional[datetime] = None  # when the call was closed (IST); None while it is open

    @property
    def kind_key(self) -> str:
        return self.kind.strip().lower()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HouseCall":
        """The inverse of `to_json` (after json.loads): how a stored call is read back for replay."""
        return cls(
            symbol=d["symbol"],
            kind=d["kind"],
            side=d["side"],
            entry_low=float(d["entry_low"]),
            entry_high=float(d["entry_high"]),
            stop=float(d["stop"]),
            target=float(d["target"]),
            published=datetime.fromisoformat(d["published"]),
            valid_to=date.fromisoformat(d["valid_to"]) if d.get("valid_to") else None,
            is_open=bool(d["is_open"]),
            exit_comment=d.get("exit_comment") or "",
            exit_at=datetime.fromisoformat(d["exit_at"]) if d.get("exit_at") else None,
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        d["valid_to"] = self.valid_to.isoformat() if self.valid_to else None
        d["exit_at"] = self.exit_at.isoformat() if self.exit_at else None
        return json.dumps(d, sort_keys=True)


def _price(value: Any) -> Optional[float]:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _published(text: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(text).strip(), "%d %b %Y %I:%M %p").replace(tzinfo=IST)
    except ValueError:
        return None


def _valid_to(text: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(text).strip()[:10])
    except ValueError:
        return None


def _exit_at(text: Any) -> Optional[datetime]:
    """The portal's ExitDtm ("2026-10-15T10:52:00", IST wall time) or None for its "not closed" 1900-01-01 value."""
    text = str(text or "").strip()
    if not text or text.startswith(NO_EXIT):
        return None
    try:
        at = datetime.fromisoformat(text[:19])
    except ValueError:
        return None
    return at.replace(tzinfo=IST) if at.tzinfo is None else at.astimezone(IST)


def _rows(body: Any) -> Optional[list]:
    """The list of call dicts in a response body (the portal sends it as a JSON string), or None if it is not one."""
    for _ in range(2):  # the portal sends a JSON string; some responses are a JSON string inside a JSON string
        if not isinstance(body, str):
            break
        try:
            body = json.loads(body)
        except ValueError:
            return None
    if not isinstance(body, list) or not body or not all(isinstance(r, dict) for r in body):
        return None
    return body


def looks_like_calls(body: Any) -> bool:
    """Whether a portal data response is a research-call list (recognised by its fields, not by the request's name,
    which the portal changes between pages)."""
    rows = _rows(body)
    return rows is not None and all(k in rows[0] for k in REQUIRED)


def why_skipped(row: Dict[str, Any]) -> str:
    """Why `parse_call` gives nothing for this row, in plain words (for the log: no prices or other values)."""
    if str(row.get("RecomType", "Equity")).strip().lower() != "equity":
        return f"type {row.get('RecomType')}"
    bad = [
        name
        for name, ok in (
            ("ScripCode", bool(str(row.get("ScripCode") or "").strip())),
            ("CallType", str(row.get("CallType") or row.get("AppCallType") or "").strip().lower() in ("buy", "sell")),
            ("CMPFrom", _price(row.get("CMPFrom")) is not None),
            ("StopLoss", _price(row.get("StopLoss")) is not None),
            ("TargetPrice", _price(row.get("TargetPrice")) is not None),
            ("PublishedDtm", _published(row.get("PublishedDtm")) is not None),
        )
        if not ok
    ]
    return "unusable " + "/".join(bad) if bad else "duplicate"


def parse_call(row: Dict[str, Any]) -> Optional[HouseCall]:
    """One portal row as a HouseCall; None for anything that is not an equity call with usable prices. The entry range is
    given low-high, high-low, or as one price with "0.00" in CMPTo: all three become (low, high)."""
    if str(row.get("RecomType", "Equity")).strip().lower() != "equity":
        return None  # F&O and commodity calls: options and futures are not traded here
    symbol = str(row.get("ScripCode") or "").strip().upper()
    side = str(row.get("CallType") or row.get("AppCallType") or "").strip().lower()
    a, b = _price(row.get("CMPFrom")), _price(row.get("CMPTo"))
    stop, target, published = _price(row.get("StopLoss")), _price(row.get("TargetPrice")), _published(row.get("PublishedDtm"))
    if not symbol or side not in ("buy", "sell") or a is None or stop is None or target is None or published is None:
        return None
    low, high = (a, a) if b is None else (min(a, b), max(a, b))
    exit_at = _exit_at(row.get("ExitDtm"))
    closed = str(row.get("CallMode") or "").strip().lower() not in ("open", "") or exit_at is not None
    return HouseCall(
        symbol=symbol,
        kind=str(row.get("RecomSubType") or "").strip(),
        side=side,
        entry_low=low,
        entry_high=high,
        stop=stop,
        target=target,
        published=published,
        valid_to=_valid_to(row.get("ValidTo")),
        is_open=not closed,
        exit_comment=str(row.get("ExitComment") or "").strip()[:64],
        exit_at=exit_at,
    )


def parse_calls(bodies: Iterable[Any]) -> List[HouseCall]:
    """Every equity call in the given response bodies, each call once (the same call can come in two responses)."""
    seen, calls = set(), []
    for body in bodies:
        for row in _rows(body) or ():
            call = parse_call(row)
            if call is None:
                log.info("research row skipped: %s %s (%s)", row.get("ScripCode"), row.get("RecomSubType"), why_skipped(row))
                continue
            key = (call.symbol, call.kind_key, call.side, call.published)
            if key not in seen:
                seen.add(key)
                calls.append(call)
    return calls


def prompt_line(call: Optional[HouseCall]) -> str:
    """The call as the swing AI is shown it: the portal's own numbers, nothing added."""
    if call is None:
        return "none"
    entry = f"{call.entry_low:,.2f}" if call.entry_low == call.entry_high else f"{call.entry_low:,.2f}-{call.entry_high:,.2f}"
    status = "open" if call.is_open else f"closed ({call.exit_comment or 'no reason given'})"
    valid = f", valid to {call.valid_to.isoformat()}" if call.valid_to else ""
    return (
        f"the broker's research desk published a {call.kind} {call.side.upper()} call on "
        f"{call.published.strftime('%d %b %Y %H:%M')} IST: entry {entry}, stop {call.stop:,.2f}, target "
        f"{call.target:,.2f}{valid}; status {status}"
    )


def _columns(c: HouseCall) -> Dict[str, Any]:
    """A call as stored (in `house_calls` and `house_call_events`): times in UTC, the validity date as text."""
    return dict(
        symbol=c.symbol,
        kind=c.kind,
        side=c.side,
        entry_low=c.entry_low,
        entry_high=c.entry_high,
        stop=c.stop,
        target=c.target,
        published=c.published.astimezone(timezone.utc),
        valid_to=c.valid_to.isoformat() if c.valid_to else None,
        is_open=c.is_open,
        exit_comment=c.exit_comment,
        exit_at=c.exit_at.astimezone(timezone.utc) if c.exit_at else None,
    )


STATE = ("entry_low", "entry_high", "stop", "target", "valid_to", "is_open", "exit_comment", "exit_at")


def _naive_utc(at: Optional[datetime]) -> Optional[datetime]:
    """A UTC time without its zone: SQLite hands stored times back naive, so stored and fresh values compare this way."""
    return at.astimezone(timezone.utc).replace(tzinfo=None) if at is not None and at.tzinfo else at


def _state_of(cols: Dict[str, Any]) -> tuple:
    return tuple(_naive_utc(cols[k]) if k == "exit_at" else cols[k] for k in STATE)


def _state(e: HouseCallEvent) -> tuple:
    return _state_of({k: getattr(e, k) for k in STATE})


def _aware(at: Optional[datetime]) -> Optional[datetime]:
    """A stored time back in IST (SQLite hands timezone-aware columns back naive, in UTC)."""
    if at is None:
        return None
    return (at if at.tzinfo else at.replace(tzinfo=timezone.utc)).astimezone(IST)


class HouseCallStore:
    """The latest snapshot of house calls, in a database both containers can open (the portal reader writes it, the
    swing and intraday processes read it)."""

    def __init__(self, sessions, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.sessions = sessions
        self.now_fn = now_fn

    def replace(self, calls: List[HouseCall], fetched_at: Optional[datetime] = None) -> None:
        """Swap the stored snapshot for this one, in one transaction (a reader never sees half of each), then add the read
        to the call history (`record_history`). Times are stored in UTC: SQLite keeps no timezone, and `_aware` reads
        them back as UTC."""
        at = (fetched_at or self.now_fn()).astimezone(timezone.utc)
        with self.sessions() as s:
            s.execute(delete(HouseCallRecord))
            s.add_all(HouseCallRecord(fetched_at=at, **_columns(c)) for c in calls)
            s.commit()
        try:
            self.record_history(calls, at)
        except Exception:  # the history is for later study; the snapshot above is what trading reads
            log.exception("could not add this read to the house-call history; the snapshot is stored")

    def record_history(self, calls: List[HouseCall], seen_at: datetime) -> int:
        """Add one read to `house_call_events`: a call whose state is unchanged since its latest row only moves that row's
        `last_seen`; a new call, or one whose state changed, appends a row. Returns the rows appended."""
        at = seen_at.astimezone(timezone.utc)
        added = 0
        with self.sessions() as s:
            latest: Dict[tuple, HouseCallEvent] = {}
            symbols = sorted({c.symbol for c in calls})
            for e in s.scalars(select(HouseCallEvent).where(HouseCallEvent.symbol.in_(symbols)).order_by(HouseCallEvent.id)):
                latest[(e.symbol, e.kind, e.side, _naive_utc(e.published))] = e  # ordered by id: the last one wins
            for c in calls:
                cols = _columns(c)
                key = (c.symbol, c.kind, c.side, _naive_utc(cols["published"]))
                last = latest.get(key)
                if last is not None and _state(last) == _state_of(cols):
                    last.last_seen = at
                    continue
                event = HouseCallEvent(first_seen=at, last_seen=at, **cols)
                s.add(event)
                latest[key] = event  # the same call twice in one read is one row
                added += 1
            s.commit()
        return added

    def fetched_at(self) -> Optional[datetime]:
        with self.sessions() as s:
            at = s.scalar(select(func.max(HouseCallRecord.fetched_at)))
        if at is not None and at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)  # SQLite hands timezone-aware columns back naive
        return at

    def current(self) -> List[HouseCall]:
        """The stored calls if the snapshot is fresh (younger than MAX_AGE), else none. Never raises: a database problem
        means no calls, and trading carries on as without them."""
        try:
            at = self.fetched_at()
            if at is None or self.now_fn() - at > MAX_AGE:
                return []
            with self.sessions() as s:
                rows = list(s.scalars(select(HouseCallRecord)))
        except Exception:
            log.exception("could not read the house calls; trading without them")
            return []
        return [
            HouseCall(
                symbol=r.symbol,
                kind=r.kind,
                side=r.side,
                entry_low=r.entry_low,
                entry_high=r.entry_high,
                stop=r.stop,
                target=r.target,
                published=_aware(r.published) or r.published,
                valid_to=_valid_to(r.valid_to) if r.valid_to else None,
                is_open=r.is_open,
                exit_comment=r.exit_comment,
                exit_at=_aware(r.exit_at),
            )
            for r in rows
        ]


def swing_buys(calls: Iterable[HouseCall]) -> List[str]:
    """Stocks with an open Swing Trade / Momentum / Stellar Pick BUY call: candidates for the swing bot."""
    return list(dict.fromkeys(c.symbol for c in calls if c.is_open and c.side == "buy" and c.kind_key in SWING_KINDS))


def intraday_buys(calls: Iterable[HouseCall], today: date) -> List[str]:
    """Stocks with an open Intraday BUY call published today: added to the breakout engine's scan."""
    return list(
        dict.fromkeys(
            c.symbol
            for c in calls
            if c.is_open and c.side == "buy" and c.kind_key in INTRADAY_KINDS and c.published.astimezone(IST).date() == today
        )
    )


def open_sells(calls: Iterable[HouseCall]) -> Dict[str, HouseCall]:
    """Stocks the house currently says to SELL (any call type), each with its newest such call."""
    out: Dict[str, HouseCall] = {}
    for c in sorted(calls, key=lambda c: c.published):
        if c.is_open and c.side == "sell":
            out[c.symbol] = c
    return out


def call_for(
    calls: Iterable[HouseCall], symbol: str, now: datetime, closed_within: timedelta = timedelta(days=3)
) -> Optional[HouseCall]:
    """The call the AI should see for a stock: its newest open call, else its newest call closed in the last few days
    (a fresh "Stoploss Hit" is worth knowing about), else None."""
    mine = sorted((c for c in calls if c.symbol == symbol), key=lambda c: c.published, reverse=True)
    for c in mine:
        if c.is_open:
            return c
    for c in mine:
        if c.exit_at is not None and now - c.exit_at <= closed_within:
            return c
    return None


def recently_stopped(
    calls: Iterable[HouseCall], symbol: str, now: datetime, within: timedelta = timedelta(days=1)
) -> Optional[HouseCall]:
    """The house's call on this stock that hit its stop-loss in the last day, if any (a warning for a stock we hold)."""
    for c in calls:
        if c.symbol == symbol and not c.is_open and c.exit_at is not None and now - c.exit_at <= within:
            if "stop" in c.exit_comment.lower():
                return c
    return None
