"""Wires the read-only portfolio agent (src/portfolio) into the application: the trading bot's own Telegram bot for the
OTP and the report, and the free OpenRouter models for the optional plain-English note.

Two ways to run it:
* `/portfolio` to the running trading bot: `PortfolioCommand` starts the graph in the background, and the bot's command
  listener passes the OTP reply to it through an `OtpInbox` (one Telegram bot, one reader of its messages);
* `python main.py --portfolio` when the trading bot is not running: `run_standalone` polls the same bot itself."""
import json
import logging
import threading
from typing import Callable, Optional

from pydantic import BaseModel

from src.monitoring.telegram import Html
from src.portfolio import vault
from src.portfolio.graph import build_portfolio_graph, usable_note
from src.portfolio.report import strip_tags
from src.portfolio.runs import RunStore, fetch_portfolio
from src.portfolio.telegram_otp import OtpInbox, TelegramOtp

log = logging.getLogger(__name__)


def to_html(notify: Callable[[str], object]) -> Callable[[str], object]:
    """Wraps `notify` so every message it sends uses Telegram's HTML formatting (bold headers, safely escaped stock
    and sector names — see src/portfolio/report.py and session.py, which build the plain strings that carry that
    markup). Wrapping an already-wrapped notify is harmless: `Html` is just a marked `str`."""
    def wrapped(text: str) -> object:
        return notify(Html(str(text)))
    return wrapped


def strip_html(text) -> str:
    """The plain-text rendering of a report message, for a plain terminal (the standalone --portfolio fallback)."""
    return strip_tags(str(text))


NOTE_SCHEMA = {
    "name": "portfolio_note",
    "strict": True,
    "schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"],
               "additionalProperties": False},
}


class PortfolioNote(BaseModel):
    note: str


RULES = ("Plain, simple English, 4 to 6 short sentences, no headings or bullet points. Cover how spread out it is "
         "(stocks, sectors, market caps), which stocks drive the gains, which are weakest, and any flags. Weights and "
         "returns are percentages; return_on_cost_pct is the gain on the price paid. If a field is absent, do not "
         "mention it. Never talk about the data format or ask questions. Do NOT recommend buying, selling, holding or "
         "any other action, and do not invent facts, news, prices or targets.")


def note_prompt(data: dict, attempt: int = 1) -> str:
    """The prompt for the note. The second attempt is worded differently (a free model's odd first answer is cached
    under the exact prompt, and a reworded one also gives the model a fresh start)."""
    facts = json.dumps(data, ensure_ascii=False)
    if attempt == 1:
        return f"You describe an Indian stock portfolio to its owner. {RULES}\n\nPortfolio facts: {facts}"
    return f"Portfolio facts: {facts}\n\nDescribe this Indian stock portfolio for its owner. {RULES}"


def make_explainer(llm) -> Callable[[dict], str]:
    """The note, from the anonymised data only (stock names, sectors, weights, returns, flags). Free models sometimes
    answer with an error or a complaint; such a note gets one retry with a reworded prompt (the same prompt would be
    answered from the cache), and the graph drops it if it is still unusable."""
    def explain(data: dict) -> str:
        note = llm.structured_call(note_prompt(data), PortfolioNote, NOTE_SCHEMA, max_tokens=600).note
        if usable_note(note):
            return note
        return llm.structured_call(note_prompt(data, attempt=2), PortfolioNote, NOTE_SCHEMA, max_tokens=600).note
    return explain


def run_portfolio(get_otp: Callable[[], str], notify: Callable[[str], object],
                  explain: Optional[Callable[[dict], str]] = None, headless: bool = True,
                  store: Optional[RunStore] = None) -> dict:
    """One run of the portfolio graph; returns its final state. Never raises for an expected failure."""
    notify = to_html(notify)
    try:
        creds = vault.load()
    except (vault.MissingCredentials, vault.InsecureStore) as e:
        notify(f"Portfolio: {e}")
        return {"error": str(e)}
    except ImportError:  # e.g. inside the Docker image, which has no browser or credential store
        notify("Portfolio: not available here; install requirements-portfolio.txt (runs on the Windows PC)")
        return {"error": "portfolio requirements missing"}
    store = store or RunStore()
    graph = build_portfolio_graph(
        lambda: fetch_portfolio(creds, get_otp, store, headless=headless, progress=notify), notify, explain)
    return graph.invoke({})


def run_standalone(token: Optional[str], chat_id: Optional[str], notify: Callable[[str], object],
                   explain: Optional[Callable[[dict], str]], headless: bool, otp_from: str,
                   ask_terminal: Callable[[], str]) -> dict:
    """`main.py --portfolio`: the OTP from Telegram (same bot, polled directly) or typed in the terminal."""
    get_otp = ask_terminal
    if otp_from == "telegram":
        if not (token and chat_id):
            notify("Portfolio: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set; use --otp-from terminal")
            return {"error": "telegram not configured"}
        tg = TelegramOtp(token, chat_id)

        def from_telegram() -> str:
            tg.begin()
            print("Waiting for the OTP on Telegram...")
            return tg.wait_for_otp()

        get_otp = from_telegram
    return run_portfolio(get_otp, notify, explain, headless=headless)


class PortfolioCommand:
    """The trading bot's /portfolio command: one background run at a time. Plug `inbox.offer` into the listener."""

    HELP = "read the real Integrated account (read-only; asks for the OTP here)"

    def __init__(self, notify: Callable[[str], object], explain: Optional[Callable[[dict], str]] = None,
                 runner: Callable[..., dict] = run_portfolio):
        self.inbox = OtpInbox()
        self.notify, self.explain, self.runner = to_html(notify), explain, runner
        self._thread: Optional[threading.Thread] = None

    def __call__(self) -> str:
        if self._thread is not None and self._thread.is_alive():
            return "A portfolio run is already in progress."
        self._thread = threading.Thread(target=self._run, daemon=True, name="portfolio")
        self._thread.start()
        return "Portfolio run started (read-only). I will ask for the OTP here in a moment."

    def _run(self) -> None:
        try:
            self.runner(lambda: self.inbox.wait(self.notify), self.notify, self.explain)
        except Exception as e:
            log.exception("portfolio run failed")
            self.notify(f"Portfolio run failed: {type(e).__name__}")
