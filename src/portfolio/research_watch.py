"""Keep one portal session open through the trading day and re-read the Research list every few minutes (read-only).

Started by `/research_start` (one login, one OTP, see runs.watch_research). Every `every_s` seconds the Research page is
opened by address, the page's own data responses are collected, and handed to `on_bodies` (the application parses them:
this package never imports the trading code). It stops at `stop_at` (IST), when `stop` is set (`/research_stop`), or as
soon as the portal shows its login page (the session ended: never logged in again without a fresh OTP). The caller's
`PortalRun` then logs out, as after every run. The network guard stays on throughout."""

import logging
import re
import threading
from datetime import datetime, time as dtime, timezone
from typing import Any, Callable, List, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from src.portfolio.guard import AUTH_PATHS, is_static_file

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
RESEARCH_PAGE = "Dashboard/ResearchViewMore.html"
STOP_AT = dtime(15, 35)  # five minutes after the close: the last intraday calls are closed by then
LOGIN_PAGE = re.compile(r"/login/", re.I)
MAX_BODY = 2_000_000  # characters; the research list is ~30 KB


def read_research_once(page, base: str, settle_ms: int = 20_000) -> Optional[List[Any]]:
    """Open the Research page and return the portal's data responses (parsed JSON where possible), or None if the portal
    sent us to its login page instead (the session has ended)."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    host = (urlsplit(base).hostname or "").lower()
    finished: list = []

    def collect(request) -> None:
        finished.append(request)

    page.on("requestfinished", collect)
    try:
        page.goto(base + RESEARCH_PAGE, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=settle_ms)
        except PlaywrightTimeout:
            pass  # a page with a live price stream never goes idle: take what has loaded
    finally:
        page.remove_listener("requestfinished", collect)
    if LOGIN_PAGE.search(urlsplit(page.url).path):
        return None
    bodies: List[Any] = []
    for req in finished:
        parts = urlsplit(req.url)
        if req.resource_type not in ("xhr", "fetch") or (parts.hostname or "").lower() != host:
            continue
        if parts.path.lower().rstrip("/") in AUTH_PATHS or is_static_file(parts.path):
            continue
        response = req.response()
        if response is None:
            continue
        try:
            text = response.text()
        except Exception:
            continue  # aborted by the guard, or the page moved on
        if len(text) <= MAX_BODY:
            bodies.append(text)
    return bodies


def watch_research(
    page,
    base: str,
    on_bodies: Callable[[List[Any]], int],
    stop: threading.Event,
    every_s: float = 600,
    stop_at: dtime = STOP_AT,
    keep_alive: Callable[[], None] = lambda: None,
    progress: Optional[Callable[[str], object]] = None,
    read_once: Callable[..., Optional[List[Any]]] = read_research_once,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> str:
    """Read the Research list every `every_s` seconds until `stop_at` IST, `stop`, or the session ends. `on_bodies` gets
    each read's responses and returns how many calls it found in them. Returns why the watch ended."""
    tell = progress or (lambda _text: None)
    reads, warned = 0, False
    while True:
        if stop.is_set():
            return "stopped by /research_stop"
        if now_fn().astimezone(IST).time() >= stop_at:
            return f"day over ({stop_at.strftime('%H:%M')} IST)"
        bodies = read_once(page, base)
        if bodies is None:
            return "the portal ended the session (it showed its login page)"
        try:
            found = on_bodies(bodies)
        except Exception:
            log.exception("could not use the research read")
            found = -1
        reads += 1
        keep_alive()
        if reads == 1:
            tell(f"📡 Research watch on: {max(found, 0)} calls read; reading again every {every_s / 60:g} min "
                 f"until {stop_at.strftime('%H:%M')} IST")  # fmt: skip
        if found <= 0 and not warned:
            tell("⚠️ The Research page showed no calls this time (or the portal changed its data): the bot trades as "
                 "if there were none until a read finds them again.")  # fmt: skip
            warned = True
        elif found > 0:
            warned = False
        stop.wait(every_s)
