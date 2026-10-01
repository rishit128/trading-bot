"""Telegram alerts and remote commands (/status, /positions, /history, /pnl, /pause, /resume, /rebase)."""
import html
import logging
import threading
import time
from typing import Callable, Mapping, Optional, Tuple

import httpx

from src.data.retry import retry_call
from src.engine.ports import ListsHoldings, ListsTrades
from src.data.india import IST
from src.engine.paper_report import group_by_day, group_by_week, merge_periods, totals

log = logging.getLogger(__name__)
API = "https://api.telegram.org"
HELP = ("/status - equity, cash, mode\n/positions - open positions with profit/loss\n/history - closed trades\n/pnl - profit/loss by day and week, both accounts\n"
        "/intraday - intraday account and open positions\n/intraday_history - intraday closed trades\n/pause - stop placing orders\n/resume - allow orders again\n"
        "/rebase - reset the peak-equity baseline (clears a drawdown halt)")

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
    cards = [f"<b>{_e(h['symbol'])}</b>  {_pl(h['pnl'], h['pnl_pct'], cur)}\n"
             f"   {h['qty']} sh · bought {h['opened_at']:%d %b} at {cur}{h['avg_price']:,.2f}\n"
             f"   now {cur}{h['price']:,.2f} · stop {cur}{h['stop']:,.2f}" for h in held]
    total, cost = sum(h["pnl"] for h in held), sum(h["qty"] * h["avg_price"] for h in held)
    up = sum(1 for h in held if h["pnl"] > 0)
    head = f"📊 <b>Open positions ({len(held)})</b>  ·  {up} up, {len(held) - up} down"
    return Html(head + "\n\n" + "\n\n".join(cards) + f"\n\n<b>Total unrealised</b>  {_pl(total, total / cost if cost else 0.0, cur)}")


def format_history(trades: list, cur: str, limit: int = 15) -> Html:
    """Closed trades as small cards, newest first."""
    if not trades:
        return Html("No closed trades yet.")
    cards = [f"<b>{_e(t['symbol'])}</b>  {_pl(t['net_pnl'], t['exit_price'] / t['entry_price'] - 1, cur)}  · {_e(t['reason'])}\n"
             f"   {t['qty']} sh · {t['opened_at']:%d %b} → {t['closed_at']:%d %b}\n"
             f"   {cur}{t['entry_price']:,.2f} → {cur}{t['exit_price']:,.2f} · fees {cur}{t['fees']:,.0f}" for t in trades[:limit]]
    total = sum(t["net_pnl"] for t in trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    head = f"📜 <b>Closed trades ({len(trades)})</b>  ·  {wins} won, {len(trades) - wins} lost"
    return Html(head + "\n\n" + "\n\n".join(cards) + f"\n\n<b>Realised net</b>  {'+' if total >= 0 else '-'}{cur}{abs(total):,.0f}")


def _period_lines(rows: list, cur: str) -> str:
    """Up to the last 8 period rows as plain lines, newest first; empty periods are left out."""
    lines = [f"{r.label}  {'+' if r.net_pnl >= 0 else '-'}{cur}{abs(r.net_pnl):,.0f}  ({r.trades} trade{'s' if r.trades != 1 else ''})"
             for r in rows if r.trades]
    return "\n".join(lines[-8:]) if lines else "no closed trades yet"


def format_pnl(swing_trades: list, intraday_trades: Optional[list], cur: str, tz=IST) -> Html:
    """Realized profit/loss by day and by week: the swing account, the intraday account (if given), and both combined."""
    swing_total = totals(swing_trades, tz)
    parts = [f"📈 <b>Profit / loss</b>\n\n<b>Swing</b>  {'+' if swing_total.net_pnl >= 0 else '-'}{cur}{abs(swing_total.net_pnl):,.0f}"
             f"  ({swing_total.trades} closed trade{'s' if swing_total.trades != 1 else ''})\n"
             f"<u>by day</u>\n{_period_lines(group_by_day(swing_trades, tz), cur)}\n"
             f"<u>by week</u>\n{_period_lines(group_by_week(swing_trades, tz), cur)}"]
    if intraday_trades is not None:
        intraday_total = totals(intraday_trades, tz)
        parts.append(f"<b>Intraday</b>  {'+' if intraday_total.net_pnl >= 0 else '-'}{cur}{abs(intraday_total.net_pnl):,.0f}"
                     f"  ({intraday_total.trades} closed trade{'s' if intraday_total.trades != 1 else ''})\n"
                     f"<u>by day</u>\n{_period_lines(group_by_day(intraday_trades, tz), cur)}\n"
                     f"<u>by week</u>\n{_period_lines(group_by_week(intraday_trades, tz), cur)}")
        combined = swing_total.net_pnl + intraday_total.net_pnl
        combined_day = merge_periods(group_by_day(swing_trades, tz), group_by_day(intraday_trades, tz))
        parts.append(f"<b>Combined</b>  {'+' if combined >= 0 else '-'}{cur}{abs(combined):,.0f}\n"
                     f"<u>by day</u>\n{_period_lines(combined_day, cur)}")
    return Html("\n\n".join(parts))


def _worth_retrying(exc: BaseException) -> bool:
    """A network failure or a server error may pass; a 4xx (wrong token, chat not found) will not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return True


class Notifier:
    """Best-effort alerts: a Telegram outage must never break trading."""

    def __init__(self, token: str, chat_id: str, client: Optional[httpx.Client] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.token, self.chat_id = token, str(chat_id)
        self.http = client or httpx.Client(timeout=15)
        self._sleep = sleep

    def send(self, text: str) -> bool:
        """Send an alert; never raises."""
        def post() -> None:
            self.http.post(f"{API}/bot{self.token}/sendMessage", json=_payload(self.chat_id, text)).raise_for_status()

        try:
            retry_call(post, attempts=2, delay=1.0, retry_if=_worth_retrying, sleep=self._sleep, what="telegram alert")
            return True
        except Exception as e:
            log.warning("telegram send failed: %s", type(e).__name__)
            return False


def handle_command(text: str, control, broker, settings, intraday=None,
                   extra: Optional[Mapping[str, Tuple[str, Callable[[], str]]]] = None) -> str:
    """Turn a command text into a reply, applying pause/resume/rebase as needed. `extra` adds commands:
    {"/name": (help line, handler)}."""
    cmd = (text or "").strip().split()[0].split("@")[0].lower() if (text or "").strip() else ""
    if extra and cmd in extra:
        return extra[cmd][1]()
    if cmd == "/pause":
        control.set_paused(True)
        return "Trading PAUSED. No new orders will be placed. Existing broker stop/target orders stay active."
    if cmd == "/resume":
        control.set_paused(False)
        return "Trading RESUMED."
    if cmd == "/rebase":
        control.rebase_peak()
        p = broker.portfolio()
        return (f"Peak-equity baseline reset to {settings.currency}{p.equity:,.2f}. "
                "A drawdown halt clears on the next cycle; the daily-loss limit is unaffected.")
    if cmd == "/status":
        p = broker.portfolio()
        mode = "DRY RUN" if settings.dry_run else "PAPER ORDERS"
        state = "PAUSED" if control.is_paused() else "RUNNING"
        day = f"{(p.equity / p.start_of_day_equity - 1):+.2%} today" if p.start_of_day_equity else "n/a"
        cur = settings.currency
        icon = "⏸" if state == "PAUSED" else "▶️"
        return Html(f"{icon} <b>{state}</b>  ·  {mode}\n\n💰 Equity   <b>{cur}{p.equity:,.2f}</b> ({day})\n"
                    f"💵 Cash       {cur}{p.cash:,.2f}\n📦 Positions  {len(p.positions)}")
    if cmd == "/positions":
        if isinstance(broker, ListsHoldings):
            return format_positions(broker.holdings(), settings.currency)
        p = broker.portfolio()
        if not p.positions:
            return "No open positions."
        return "\n".join(f"{s}: {p.position_qty.get(s, 0)} sh, {settings.currency}{v:,.0f}" for s, v in sorted(p.positions.items()))
    if cmd == "/history" and isinstance(broker, ListsTrades):
        return format_history(broker.trade_history(), settings.currency)
    if cmd == "/pnl" and isinstance(broker, ListsTrades):
        intraday_trades = intraday.trade_history() if isinstance(intraday, ListsTrades) else None
        return format_pnl(broker.trade_history(), intraday_trades, settings.currency)
    if cmd in ("/intraday", "/intraday_history") and intraday is not None:
        cur = settings.currency
        if cmd == "/intraday_history":
            return format_history(intraday.trade_history(), cur)
        p, held = intraday.portfolio(), intraday.holdings()
        day = f"{(p.equity / p.start_of_day_equity - 1):+.2%} today" if p.start_of_day_equity else "n/a"
        head = f"⚡ <b>INTRADAY account</b>\n💰 Equity <b>{cur}{p.equity:,.2f}</b> ({day})\n💵 Cash {cur}{p.cash:,.2f}\n\n"
        return Html(head + format_positions(held, cur))
    return HELP + "".join(f"\n{name} - {line}" for name, (line, _) in (extra or {}).items())


class CommandListener:
    """Long-polls Telegram and answers commands, but only from the one authorised chat."""

    def __init__(self, token: str, chat_id: str, respond: Callable[[str], str], client: Optional[httpx.Client] = None,
                 intercept: Optional[Callable[[str], Optional[str]]] = None):
        """`intercept` sees each authorised message first. It returns None to leave the message to `respond`, or a
        reply ("" for none) when it consumed the message; a consumed message is deleted from the chat (it may be a
        one-time password)."""
        self.token, self.chat_id, self.respond, self.intercept = token, str(chat_id), respond, intercept
        self.http = client or httpx.Client(timeout=40)
        self.offset = 0
        self._stop = threading.Event()

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

    def run(self) -> None:
        """Poll until stopped, surviving errors."""
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:
                log.warning("telegram poll failed: %s", type(e).__name__)
                self._stop.wait(10)

    def start(self) -> threading.Thread:
        """Run the listener in a background thread."""
        t = threading.Thread(target=self.run, daemon=True, name="telegram-listener")
        t.start()
        return t

    def stop(self) -> None:
        """Ask the listener to stop."""
        self._stop.set()
