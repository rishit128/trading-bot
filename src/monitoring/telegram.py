"""Telegram alerts and remote commands (/status, /positions, /history, /pnl, /pause, /resume, /rebase)."""

import html
import logging
import threading
import time
from functools import partial, reduce
from typing import Callable, Mapping, Optional, Tuple

import httpx

from src.data.india import IST
from src.engine.paper_report import group_by_day, group_by_week, merge_periods, totals
from src.engine.ports import ListsHoldings, ListsTrades

log = logging.getLogger(__name__)
API = "https://api.telegram.org"
HELP = (
    "/status - equity, cash, mode\n/positions - open positions with profit/loss\n/history - closed trades\n/pnl - profit/loss by day and week, all accounts\n"
    "/intraday - intraday account and open positions\n/intraday_history - intraday closed trades\n"
    "/gapgo - gap-and-go paper account\n/gapgo_history - gap-and-go closed trades\n"
    "/gaponly - gap-only control account\n/gaponly_history - gap-only closed trades\n"
    "/news - news-catalyst paper account\n/news_history - news-catalyst closed trades\n/pause - stop placing orders\n/resume - allow orders again\n"
    "/rebase - reset the peak-equity baseline (clears a drawdown halt)"
)

# The bot token is part of every request URL; never let the HTTP library log it.
logging.getLogger("httpx").setLevel(logging.WARNING)


class Html(str):
    """A reply that is already Telegram-HTML (bold, emoji); it is sent with parse_mode=HTML. Plain str is sent as is."""


def _payload(chat_id: str, text: str) -> dict:
    body = {"chat_id": chat_id, "text": text[:4000]}
    if isinstance(text, Html):
        body["parse_mode"] = "HTML"
    return body


def _e(value) -> str:
    return html.escape(str(value))


def _pl(amount: float, pct: float, cur: str) -> str:
    """Coloured marker plus signed percent and amount, e.g. '🟢 +2.9% (+₹1,431)'."""
    mark = "🟢" if amount > 0 else "🔴" if amount < 0 else "⚪"
    return f"{mark} {pct:+.1%} ({'+' if amount >= 0 else '-'}{cur}{abs(amount):,.0f})"


def format_positions(held: list, cur: str) -> Html:
    """Open positions as small cards, best first, with a total."""
    if not held:
        return Html("No open positions.")
    cards = [
        f"<b>{_e(h['symbol'])}</b>  {_pl(h['pnl'], h['pnl_pct'], cur)}\n"
        f"   {h['qty']} sh · bought {h['opened_at']:%d %b} at {cur}{h['avg_price']:,.2f}\n"
        f"   now {cur}{h['price']:,.2f} · stop {cur}{h['stop']:,.2f}"
        for h in held
    ]
    total, cost = sum(h["pnl"] for h in held), sum(h["qty"] * h["avg_price"] for h in held)
    up = sum(1 for h in held if h["pnl"] > 0)
    head = f"📊 <b>Open positions ({len(held)})</b>  ·  {up} up, {len(held) - up} down"
    return Html(
        head + "\n\n" + "\n\n".join(cards) + f"\n\n<b>Total unrealised</b>  {_pl(total, total / cost if cost else 0.0, cur)}"
    )


def format_history(trades: list, cur: str, limit: int = 15) -> Html:
    """Closed trades as small cards, newest first."""
    if not trades:
        return Html("No closed trades yet.")
    cards = [
        f"<b>{_e(t['symbol'])}</b>  {_pl(t['net_pnl'], t['exit_price'] / t['entry_price'] - 1, cur)}  · {_e(t['reason'])}\n"
        f"   {t['qty']} sh · {t['opened_at']:%d %b} → {t['closed_at']:%d %b}\n"
        f"   {cur}{t['entry_price']:,.2f} → {cur}{t['exit_price']:,.2f} · fees {cur}{t['fees']:,.0f}"
        for t in trades[:limit]
    ]
    total = sum(t["net_pnl"] for t in trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    head = f"📜 <b>Closed trades ({len(trades)})</b>  ·  {wins} won, {len(trades) - wins} lost"
    return Html(
        head + "\n\n" + "\n\n".join(cards) + f"\n\n<b>Realised net</b>  {'+' if total >= 0 else '-'}{cur}{abs(total):,.0f}"
    )


def _period_lines(rows: list, cur: str, keep: int = 8) -> str:
    """The last `keep` period rows as plain lines, newest first; empty periods are left out."""
    lines = [
        f"{r.label}  {'+' if r.net_pnl >= 0 else '-'}{cur}{abs(r.net_pnl):,.0f}  ({r.trades} trade{'s' if r.trades != 1 else ''})"
        for r in rows
        if r.trades
    ]
    return "\n".join(lines[-keep:]) if lines else "no closed trades yet"


def format_pnl(accounts: list, cur: str, tz=IST) -> Html:
    """Realized profit/loss of every paper account, [(name, closed trades)], and all of them combined. Each account shows its
    total and its last few days; the combined block shows the last days and weeks (Telegram cuts a message at 4,096 characters)."""
    parts, all_days, all_weeks = ["📈 <b>Profit / loss</b>"], [], []
    for name, trades in accounts:
        total = totals(trades, tz)
        days = group_by_day(trades, tz)
        parts.append(
            f"<b>{_e(name)}</b>  {'+' if total.net_pnl >= 0 else '-'}{cur}{abs(total.net_pnl):,.0f}"
            f"  ({total.trades} closed trade{'s' if total.trades != 1 else ''})\n{_period_lines(days, cur, 5)}"
        )
        all_days.append(days)
        all_weeks.append(group_by_week(trades, tz))
    if len(accounts) > 1:
        combined = sum(totals(trades, tz).net_pnl for _, trades in accounts)
        merge = lambda rows: reduce(merge_periods, rows, [])  # noqa: E731
        parts.append(
            f"<b>All accounts</b>  {'+' if combined >= 0 else '-'}{cur}{abs(combined):,.0f}\n"
            f"<u>by day</u>\n{_period_lines(merge(all_days), cur)}\n<u>by week</u>\n{_period_lines(merge(all_weeks), cur)}"
        )
    return Html("\n\n".join(parts))


def _worth_retrying(exc: BaseException) -> bool:
    """A network failure or a server error may pass; a 4xx (wrong token, chat not found) will not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return True


MAX_RATE_LIMIT_WAIT = 10.0  # seconds; a longer retry_after is not waited out inside a trading cycle


def _telegram_reply(exc: BaseException) -> dict:
    """Telegram's JSON error body ({"description": ..., "parameters": {"retry_after": n}}), or {}."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            return body if isinstance(body, dict) else {}
        except ValueError:  # the body is not JSON
            return {}
    return {}


def _describe(exc: BaseException) -> str:
    """Why a send failed, without the request URL (it contains the bot token)."""
    if isinstance(exc, httpx.HTTPStatusError):
        description = str(_telegram_reply(exc).get("description") or "")[:200]
        return f"HTTP {exc.response.status_code}" + (f": {description}" if description else "")
    return type(exc).__name__


def _retry_wait(exc: BaseException) -> float:
    """Seconds before the retry: what a 429 asks for (capped), else one second."""
    retry_after = (_telegram_reply(exc).get("parameters") or {}).get("retry_after")
    try:
        return min(MAX_RATE_LIMIT_WAIT, max(1.0, float(retry_after))) if retry_after is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


class Notifier:
    """Best-effort alerts: a Telegram outage must never break trading."""

    def __init__(
        self, token: str, chat_id: str, client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = time.sleep
    ):
        self.token, self.chat_id = token, str(chat_id)
        self.http = client or httpx.Client(timeout=15)
        self._sleep = sleep

    def send(self, text: str) -> bool:
        """Send an alert; never raises. One retry for a network blip, a server error or a rate limit (after the wait
        Telegram asks for). Not `retry_call`: its log line quotes the error text, whose URL contains the token."""
        for attempt in (1, 2):
            try:
                self.http.post(f"{API}/bot{self.token}/sendMessage", json=_payload(self.chat_id, text)).raise_for_status()
                return True
            except Exception as e:
                if attempt == 2 or not _worth_retrying(e):
                    log.warning("telegram send failed: %s", _describe(e))
                    return False
                self._sleep(_retry_wait(e))
        return False


def _command_name(text: str) -> str:
    """ "/status@MyBot extra words" -> "/status"; blank text -> ""."""
    words = (text or "").split()
    return words[0].split("@")[0].lower() if words else ""


def _status(control, broker, settings) -> Html:
    p = broker.portfolio()
    mode = "DRY RUN" if settings.dry_run else "PAPER ORDERS"
    state = "PAUSED" if control.is_paused() else "RUNNING"
    day = f"{(p.equity / p.start_of_day_equity - 1):+.2%} today" if p.start_of_day_equity else "n/a"
    cur = settings.currency
    icon = "⏸" if state == "PAUSED" else "▶️"
    return Html(
        f"{icon} <b>{state}</b>  ·  {mode}\n\n💰 Equity   <b>{cur}{p.equity:,.2f}</b> ({day})\n"
        f"💵 Cash       {cur}{p.cash:,.2f}\n📦 Positions  {len(p.positions)}"
    )


def _positions(broker, cur: str) -> str:
    if isinstance(broker, ListsHoldings):
        return format_positions(broker.holdings(), cur)
    p = broker.portfolio()
    if not p.positions:
        return "No open positions."
    return "\n".join(f"{s}: {p.position_qty.get(s, 0)} sh, {cur}{v:,.0f}" for s, v in sorted(p.positions.items()))


def _rebase(control, broker, cur: str) -> str:
    control.rebase_peak()
    return (
        f"Peak-equity baseline reset to {cur}{broker.portfolio().equity:,.2f}. "
        "A drawdown halt clears on the next cycle; the daily-loss limit is unaffected."
    )


def _set_paused(control, paused: bool) -> str:
    control.set_paused(paused)
    if paused:
        return "Trading PAUSED. No new orders will be placed. Existing broker stop/target orders stay active."
    return "Trading RESUMED."


def handle_command(
    text: str,
    control,
    broker,
    settings,
    intraday=None,
    extra: Optional[Mapping[str, Tuple[str, Callable[[], str]]]] = None,
    gapgo=None,
    gaponly=None,
    news=None,
) -> str:
    """Turn a command text into a reply, applying pause/resume/rebase as needed. `extra` adds commands:
    {"/name": (help line, handler)}. Anything unknown (or a command this broker cannot answer) gets the help text."""
    cmd = _command_name(text)
    if extra and cmd in extra:
        return extra[cmd][1]()
    cur = settings.currency
    handlers: dict[str, Callable[[], str]] = {
        "/pause": lambda: _set_paused(control, True),
        "/resume": lambda: _set_paused(control, False),
        "/rebase": lambda: _rebase(control, broker, cur),
        "/status": lambda: _status(control, broker, settings),
        "/positions": lambda: _positions(broker, cur),
    }
    if isinstance(broker, ListsTrades):
        handlers["/history"] = lambda: format_history(broker.trade_history(), cur)
        paper = (("Swing", broker), ("Intraday (ORB)", intraday), ("Gap-and-go", gapgo), ("Gap-only", gaponly), ("News", news))
        handlers["/pnl"] = lambda: format_pnl([(n, a.trade_history()) for n, a in paper if isinstance(a, ListsTrades)], cur)
    for name, account in (("intraday", intraday), ("gapgo", gapgo), ("gaponly", gaponly), ("news", news)):
        if account is not None:
            handlers[f"/{name}"] = partial(_account_view, account, name, False, cur)
            handlers[f"/{name}_history"] = partial(_account_view, account, name, True, cur)
    if cmd in handlers:
        return handlers[cmd]()
    return HELP + "".join(f"\n{name} - {line}" for name, (line, _) in (extra or {}).items())


def _account_view(account, name: str, history: bool, cur: str) -> str:
    """/intraday, /gapgo, /gaponly and /news (equity, cash, open positions) and their _history variants (closed trades)."""
    if history:
        return format_history(account.trade_history(), cur)
    p, held = account.valuation()  # a view: settles nothing and writes nothing (the engines own the database)
    day = f"{(p.equity / p.start_of_day_equity - 1):+.2%} today" if p.start_of_day_equity else "n/a"
    title = {
        "intraday": "INTRADAY account",
        "gapgo": "GAP-AND-GO paper account",
        "gaponly": "GAP-ONLY control account",
        "news": "NEWS-CATALYST paper account",
    }[name]
    head = f"⚡ <b>{title}</b>\n💰 Equity <b>{cur}{p.equity:,.2f}</b> ({day})\n💵 Cash {cur}{p.cash:,.2f}\n\n"
    return Html(head + format_positions(held, cur))


class CommandListener:
    """Long-polls Telegram and answers commands, but only from the one authorised chat."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        respond: Callable[[str], str],
        client: Optional[httpx.Client] = None,
        intercept: Optional[Callable[[str], Optional[str]]] = None,
    ):
        """`intercept` sees each authorised message first. It returns None to leave the message to `respond`, or a
        reply ("" for none) when it consumed the message; a consumed message is deleted from the chat (it may be a
        one-time password)."""
        self.token, self.chat_id, self.respond, self.intercept = token, str(chat_id), respond, intercept
        self.http = client or httpx.Client(timeout=40)
        self.offset = 0
        self._stop = threading.Event()
        self._conflict_logged_at: Optional[float] = None
        self._conflict_announced = False

    def poll_once(self, timeout: int = 25) -> int:
        """Fetch and answer pending commands from the authorised chat only; returns how many were handled."""
        r = self.http.get(f"{API}/bot{self.token}/getUpdates", params={"offset": self.offset, "timeout": timeout})
        r.raise_for_status()
        handled = 0
        for update in r.json().get("result", []):
            self.offset = update["update_id"] + 1
            msg = update.get("message") or {}
            if str(msg.get("chat", {}).get("id")) != self.chat_id:
                log.warning("ignored telegram message from unauthorised chat")
                continue
            if self.intercept is not None:
                taken = self.intercept(msg.get("text", ""))
                if taken is not None:
                    self._delete(msg.get("message_id"))
                    if taken:
                        self.http.post(f"{API}/bot{self.token}/sendMessage", json=_payload(self.chat_id, taken))
                    handled += 1
                    continue
            try:
                reply = self.respond(msg.get("text", ""))
            except Exception as e:
                log.exception("command failed")
                reply = f"Command failed: {type(e).__name__}"
            self.http.post(f"{API}/bot{self.token}/sendMessage", json=_payload(self.chat_id, reply))
            handled += 1
        return handled

    def _delete(self, message_id) -> None:
        try:
            self.http.post(f"{API}/bot{self.token}/deleteMessage", json={"chat_id": self.chat_id, "message_id": message_id})
        except httpx.HTTPError as e:
            log.warning("telegram delete failed: %s", type(e).__name__)

    CONFLICT_NOTICE = (
        "Another program is reading this bot's messages (Telegram reports a polling conflict), so your "
        "commands may reach it instead of this bot. If it is not yours, revoke the token in BotFather "
        "(/revoke) and put the new one in TELEGRAM_BOT_TOKEN."
    )

    def _poll_failed(self, e: Exception) -> None:
        """Log a failed poll usefully. A 409 means another program polls the same token: Telegram gives each update to
        whichever asked last. That repeats every few seconds, so it is logged at most every 10 minutes and the owner is
        told once per run, in the chat, since nobody reads the server logs."""
        if not isinstance(e, httpx.HTTPStatusError):
            log.warning("telegram poll failed: %s", type(e).__name__)
            return
        try:
            detail = e.response.json().get("description", "")
        except ValueError:
            detail = ""
        if e.response.status_code != 409:
            log.warning("telegram poll failed: HTTP %s %s", e.response.status_code, detail)
            return
        now = time.monotonic()
        if self._conflict_logged_at is None or now - self._conflict_logged_at >= 600:
            log.warning("telegram poll conflict (409): another program is polling this bot token (%s)", detail)
            self._conflict_logged_at = now
        if not self._conflict_announced:
            self._conflict_announced = True
            try:
                self.http.post(f"{API}/bot{self.token}/sendMessage", json=_payload(self.chat_id, self.CONFLICT_NOTICE))
            except httpx.HTTPError:
                log.warning("could not send the polling-conflict notice")

    def run(self) -> None:
        """Poll until stopped, surviving errors."""
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:
                self._poll_failed(e)
                self._stop.wait(10)

    def start(self) -> threading.Thread:
        """Run the listener in a background thread."""
        t = threading.Thread(target=self.run, daemon=True, name="telegram-listener")
        t.start()
        return t

    def stop(self) -> None:
        """Ask the listener to stop."""
        self._stop.set()
