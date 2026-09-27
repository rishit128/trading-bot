"""Getting the portal's one-time password from you on Telegram, through the trading bot's own Telegram bot.

A Telegram bot has one update queue, so exactly one reader may poll it:
* while the trading bot runs, its command listener is that reader and passes messages to an `OtpInbox` first;
* when it does not run, `TelegramOtp` polls the same bot itself (and stops with a clear message if the trading bot turns
  out to be polling too).
Either way only a message from the authorised chat, sent after the request, consisting of exactly 4 digits counts; it is
deleted from the chat as soon as it is read, and the OTP is never logged."""
import logging
import re
import threading
import time
from typing import Callable, Optional

import httpx

log = logging.getLogger(__name__)
API = "https://api.telegram.org"
OTP = re.compile(r"\s*(\d{4})\s*")
CANCEL = {"cancel", "stop", "/cancel"}
PROMPT = ("🔑 <b>OTP needed</b>\nReply here with the 4-digit OTP sent to your mobile, within {minutes} minutes.\n"
          "Send 'cancel' to stop. I delete your reply as soon as I read it.")
logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs request URLs, and the bot token is part of every URL


class TelegramError(Exception):
    """A Telegram API call failed. The message never includes the URL, which holds the bot token."""


class OtpTimeout(Exception):
    """No valid OTP arrived in time."""


class OtpCancelled(Exception):
    """The user replied 'cancel'."""


class OtpInbox:
    """Hands the OTP from the trading bot's Telegram listener to a waiting login. Plug `offer` into the listener's
    `intercept`: it takes a 4-digit message (always: an OTP must not linger in the chat) and, while a login waits,
    'cancel'; everything else goes on to the normal commands."""

    def __init__(self):
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._waiting, self._value, self._cancelled = False, "", False

    def offer(self, text: str) -> Optional[str]:
        """None if the message is not for us; else the reply to send ("" for none). The listener deletes taken messages."""
        text = (text or "").strip()
        with self._lock:
            if self._waiting and text.lower() in CANCEL:
                self._cancelled, self._waiting = True, False
                self._arrived.set()
                return "Portfolio login cancelled."
            match = OTP.fullmatch(text)
            if not match:
                return None
            if not self._waiting:
                return "No portfolio login is waiting for an OTP; I deleted that message."
            self._value, self._waiting = match.group(1), False
            self._arrived.set()
            return "OTP received, logging in."

    def wait(self, prompt: Callable[[str], object], timeout_s: float = 180) -> str:
        """Open the window, send the prompt, and block until the OTP arrives (OtpTimeout / OtpCancelled otherwise)."""
        with self._lock:
            self._waiting, self._value, self._cancelled = True, "", False
            self._arrived.clear()
        try:
            prompt(PROMPT.format(minutes=int(timeout_s // 60)))
            if not self._arrived.wait(timeout_s):
                raise OtpTimeout(f"no OTP within {timeout_s:.0f}s")
            if self._cancelled:
                raise OtpCancelled("login cancelled from Telegram")
            return self._value
        finally:
            with self._lock:
                self._waiting, self._value = False, ""


class TelegramOtp:
    """Polls the bot directly; for runs while the trading bot (and its listener) is not running."""

    def __init__(self, token: str, chat_id: str, client: Optional[httpx.Client] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._token, self.chat_id = token, str(chat_id)
        self.http = client or httpx.Client(timeout=40)
        self.clock = clock
        self.offset = 0

    def _call(self, method: str, **params):
        try:
            r = self.http.post(f"{API}/bot{self._token}/{method}", json=params)
        except httpx.HTTPError as e:
            raise TelegramError(f"{method}: {type(e).__name__}") from None
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 409:
            raise TelegramError("the trading bot is running and reading this Telegram bot: send /portfolio to it instead")
        if r.status_code != 200 or not body.get("ok"):
            raise TelegramError(f"{method}: HTTP {r.status_code} {body.get('description', '')}".strip())
        return body["result"]

    def send(self, text: str) -> None:
        """Sends with Telegram's HTML formatting on, matching the main bot's messages (text here has no untrusted
        content: it is either our own fixed prompts or the OTP-format hint, never a company name or an AI note)."""
        self._call("sendMessage", chat_id=self.chat_id, text=text, parse_mode="HTML")

    def begin(self) -> None:
        """Skip everything already waiting, so an old message can never be taken for this login's OTP."""
        pending = self._call("getUpdates", offset=-1, timeout=0)
        if pending:
            self.offset = pending[-1]["update_id"] + 1
            self._call("getUpdates", offset=self.offset, timeout=0)  # confirm: Telegram drops the skipped updates

    def wait_for_otp(self, timeout_s: float = 180) -> str:
        """Send the prompt, then return the first 4-digit reply from the authorised chat within timeout_s."""
        self.send(PROMPT.format(minutes=int(timeout_s // 60)))
        deadline = self.clock() + timeout_s
        while (left := deadline - self.clock()) > 0:
            for update in self._call("getUpdates", offset=self.offset, timeout=int(min(25, max(1, left)))):
                self.offset = update["update_id"] + 1
                msg = update.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != self.chat_id:
                    log.warning("ignored a telegram message from an unauthorised chat")
                    continue
                text = (msg.get("text") or "").strip()
                if text.lower() in CANCEL:
                    self._acknowledge()
                    raise OtpCancelled("login cancelled from Telegram")
                match = OTP.fullmatch(text)
                if not match:
                    self._quietly("sendMessage", chat_id=self.chat_id,
                                  text="That is not a 4-digit OTP. Reply with just the 4 digits, or 'cancel'.")
                    continue
                self._acknowledge()
                self._quietly("deleteMessage", chat_id=self.chat_id, message_id=msg["message_id"])
                return match.group(1)
        raise OtpTimeout(f"no OTP within {timeout_s:.0f}s")

    def _acknowledge(self) -> None:
        """Tell Telegram the updates so far are handled, so the OTP cannot be fetched from the queue again."""
        self._quietly("getUpdates", offset=self.offset, timeout=0)

    def _quietly(self, method: str, **params) -> None:
        try:
            self._call(method, **params)
        except TelegramError as e:
            log.warning("telegram %s failed: %s", method, e)
