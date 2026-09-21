"""Telegram alerts and remote commands (/status, /positions, /pause, /resume, /rebase)."""
import logging
import threading
from typing import Callable, Optional

import httpx

log = logging.getLogger(__name__)
API = "https://api.telegram.org"
HELP = ("/status - equity, cash, mode\n/positions - open positions\n/pause - stop placing orders\n/resume - allow orders again\n"
        "/rebase - reset the peak-equity baseline (clears a drawdown halt)")

# The bot token is part of every request URL; never let the HTTP library log it.
logging.getLogger("httpx").setLevel(logging.WARNING)


class Notifier:
    """Best-effort alerts: a Telegram outage must never break trading."""

    def __init__(self, token: str, chat_id: str, client: Optional[httpx.Client] = None):
        self.token, self.chat_id = token, str(chat_id)
        self.http = client or httpx.Client(timeout=15)

    def send(self, text: str) -> bool:
        """Send an alert; never raises."""
        try:
            r = self.http.post(f"{API}/bot{self.token}/sendMessage", json={"chat_id": self.chat_id, "text": text[:4000]})
            r.raise_for_status()
            return True
        except Exception as e:
            log.warning("telegram send failed: %s", type(e).__name__)
            return False


def handle_command(text: str, control, broker, settings) -> str:
    """Turn a command text into a reply, applying pause/resume/rebase as needed."""
    cmd = (text or "").strip().split()[0].split("@")[0].lower() if (text or "").strip() else ""
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
        return f"{state} | {mode}\nEquity {cur}{p.equity:,.2f} ({day})\nCash {cur}{p.cash:,.2f}\nPositions: {len(p.positions)}"
    if cmd == "/positions":
        p = broker.portfolio()
        if not p.positions:
            return "No open positions."
        return "\n".join(f"{s}: {p.position_qty.get(s, 0)} sh, {settings.currency}{v:,.0f}" for s, v in sorted(p.positions.items()))
    return HELP


class CommandListener:
    """Long-polls Telegram and answers commands, but only from the one authorised chat."""

    def __init__(self, token: str, chat_id: str, respond: Callable[[str], str], client: Optional[httpx.Client] = None):
        self.token, self.chat_id, self.respond = token, str(chat_id), respond
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
            try:
                reply = self.respond(msg.get("text", ""))
            except Exception as e:
                log.exception("command failed")
                reply = f"Command failed: {type(e).__name__}"
            self.http.post(f"{API}/bot{self.token}/sendMessage", json={"chat_id": self.chat_id, "text": reply[:4000]})
            handled += 1
        return handled

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
