"""Log in to the Integrated portal in Chromium, read allow-listed pages, and always log out.

Login follows the portal's own pages: login.html (mobile number, "Get OTP") -> OtpVerify.html (4-digit OTP) ->
CustomerIDSelection.html (picks itself when the mobile has one customer ID) -> MpinPassword.html -> Dashboard. Each step
waits for the page it expects; anything else (an error box, a browser alert, a reset-MPIN page, a timeout) stops the
run at once. There is exactly one attempt per run: a retry could lock the account.
"""
import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from html import escape as esc
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from src.portfolio.guard import AUTH_PATHS, WRITE_WORDS, Blocked, action_of, is_static_file, verdict
from src.portfolio.vault import Credentials

log = logging.getLogger(__name__)
PORTAL = "https://webapp.integrated.investments/nv1/"


def _no_sandbox() -> bool:
    return os.getenv("PORTFOLIO_NO_SANDBOX", "false").strip().lower() == "true"


# The only pages ever opened after login. Opened by address, never by clicking the portal's menus.
READ_PAGES = {
    "dashboard": "Dashboard/DashboardV1.html",
    "portfolio": "PortfolioTracker/LiveSummaryDetailsV1.html",
    "analyzer": "Demat/portfolioanalyzer.html",
    "account": "Dashboard/MyAccount.html",
}
PAGE_NAMES = {"dashboard": "Dashboard (totals)", "portfolio": "Capital gains",
              "analyzer": "Portfolio analyser (holdings)", "account": "My account"}
# Read-only tabs inside the read pages, clicked by their exact visible label (and nothing else). Deliberately left out:
# Client Master (personal details), Download forms and eVoting (actions). Every click still passes the network guard.
READ_TABS = {
    "analyzer": ("Holdings", "Analyser", "Dividend", "Transaction", "Demat Charges"),
    "portfolio": ("Realised Gain/Loss", "Current Holdings", "Short Term /Long Term Holdings", "Transaction History"),
}
# Market feeds the dashboard loads (index quotes, movers, 52-week lists): not account data, and large; never saved.
MARKET_DATA = re.compile(r"MarketData|CMOTS|TopGain|TopLos|52Week|Movers|Basket", re.I)
STEPS = {
    "otp": re.compile(r"/login/OtpVerify\.html", re.I),
    "customer": re.compile(r"/login/CustomerIDSelection\.html", re.I),
    "mpin": re.compile(r"/login/MpinPassword\.html", re.I),
    "dashboard": re.compile(r"/Dashboard/Dashboard(V1)?\.html", re.I),
}
ALERT = ".mask-overlay .alert-box"  # the portal's CustomAlert error box
CUSTOMER_CHOICES = "input[name='CustomerId']"
TABLES_JS = """() => [...document.querySelectorAll('table')].filter(t => t.offsetParent !== null)
    .map(t => [...t.rows].map(r => [...r.cells].map(c => c.innerText.trim())))"""


class LoginFailed(Exception):
    """The login stopped at `stage` (mobile, otp, customer or mpin)."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class PortalRun:
    """`with PortalRun(...) as page:` gives a logged-in page; leaving the block always logs out and closes the browser.

    After the block, `logged_out` says whether the portal confirmed the logout (None: no session was ever started)
    and `blocked` lists every request the guard stopped."""

    def __init__(self, creds: Credentials, get_otp: Callable[[], str], profile_dir: Path, evidence_dir: Path,
                 base: str = PORTAL, headless: bool = True, step_timeout_s: float = 45,
                 progress: Optional[Callable[[str], object]] = None):
        self.creds, self.get_otp, self.base = creds, get_otp, base
        self.progress = Progress(progress)
        self.profile_dir, self.evidence_dir = profile_dir, evidence_dir
        self.headless, self.step_timeout_s = headless, step_timeout_s
        self.host = (urlsplit(base).hostname or "").lower()
        self.blocked: List[Blocked] = []
        self.dialogs: List[str] = []
        self.logged_out: Optional[bool] = None
        self._session_started = False  # the OTP was submitted: a half-login may exist
        self._logged_in = False  # the dashboard was reached: a full session exists and must be logged out

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            # A persistent profile keeps the portal's device ID, so each login is not a "new device". Service workers are
            # blocked because their requests would bypass the guard below. Chromium's own sandbox needs a capability
            # (CAP_SYS_ADMIN) that a container normally does not have; PORTFOLIO_NO_SANDBOX=true (set in the AWS
            # deployment's Dockerfile) trades that OS-level sandbox for Docker's own container isolation instead. Off
            # by default, so a local Windows/Mac run is unaffected.
            args = ["--no-sandbox", "--disable-setuid-sandbox"] if _no_sandbox() else []
            self._ctx = self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), headless=self.headless, service_workers="block", accept_downloads=False,
                viewport={"width": 1366, "height": 900}, args=args)
            self._ctx.route("**/*", self._route)
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self.page.set_default_timeout(self.step_timeout_s * 1000)
            self.page.on("dialog", self._dialog)
            self._login()
        except BaseException:
            self.__exit__(None, None, None, failed=True)
            raise
        return self.page

    def __exit__(self, exc_type, exc, tb, failed: bool = False) -> None:
        if failed or exc_type is not None:
            self._screenshot("failure.png")
        if self._logged_in:
            self.logged_out = self._logout()
            self.progress("🔒 <b>Logged out of Integrated.</b>" if self.logged_out else
                          "⚠️ <b>Logout NOT confirmed</b> — please log out on the Integrated portal yourself.")
        elif self._session_started:
            # The OTP was sent but the login never completed (no MPIN, so no dashboard session): there is nothing the
            # portal's logout applies to. Closing the browser and clearing its cookies below ends the half-login.
            self.progress("🔒 Login did not complete, so no session was left open — browser closed, cookies cleared.")
        try:
            self._ctx.clear_cookies()
            self._ctx.close()
        except Exception:
            pass
        self._pw.stop()

    # -- guard and dialogs ---------------------------------------------------------------------------------------
    def _route(self, route) -> None:
        req = route.request
        blocked = verdict(req.method, req.url, req.headers, self.host)
        if blocked is None:
            route.continue_()
            return
        self.blocked.append(blocked)
        log.warning("blocked %s %s%s [%s]: %s", blocked.method, blocked.host, blocked.path, blocked.action, blocked.reason)
        route.abort("blockedbyclient")

    def _dialog(self, dialog) -> None:
        self.dialogs.append(dialog.message[:200])
        dialog.dismiss()

    # -- login -----------------------------------------------------------------------------------------------------
    def _login(self) -> None:
        page, creds = self.page, self.creds
        self.progress("🔐 <b>Step 1/5</b> — Opening the Integrated login page...")
        page.goto(self.base + "login/login.html", wait_until="load")
        page.check("#MobileNoRadioBtn")
        self._type("#MobileNoTxt", creds.mobile)
        self._submit("mobile")  # "Get OTP": the portal texts the OTP to the mobile
        self._expect(("otp",), "mobile")
        self.progress("📱 <b>Step 2/5</b> — Mobile number accepted; the portal sent an OTP by SMS.")

        otp = self.get_otp()
        if not re.fullmatch(r"\d{4}", otp or ""):
            raise LoginFailed("otp", "the OTP must be 4 digits")
        page.wait_for_load_state("load")
        self._type("#OTPTxt", otp)
        self._submit("otp", opens_session=True)
        self.progress("🔑 <b>Step 3/5</b> — OTP submitted, checking...")
        step = self._expect(("customer", "mpin", "dashboard"), "otp")
        self.progress("✅ OTP accepted.")  # kept brief: the MPIN step (or the dashboard) follows immediately

        if step == "customer":
            step = self._expect(("mpin", "dashboard"), "customer", choices_shown=self._customer_choices_shown)
            if step == "choices":
                self._choose_customer()
                step = self._expect(("mpin", "dashboard"), "customer")
        if step == "mpin":
            page.wait_for_load_state("load")
            page.locator("#MpinTxt1").wait_for(state="visible")
            for i, digit in enumerate(creds.mpin, start=1):
                self._type(f"#MpinTxt{i}", digit)
            self._submit("mpin")
            self.progress("🔢 <b>Step 4/5</b> — MPIN entered, logging in...")
            self._expect(("dashboard",), "mpin")
        self._logged_in = True
        log.info("logged in")
        self.progress(f"✅ <b>Step 5/5</b> — Logged in (read-only). Reading {len(READ_PAGES)} pages; "
                     "nothing will be clicked or changed.")

    def _type(self, selector: str, text: str) -> None:
        """Type like a person, key by key: the portal enables its buttons from key events, which a bulk fill skips."""
        box = self.page.locator(selector)
        box.click()
        box.fill("")
        box.press_sequentially(text, delay=60)

    def _submit(self, stage: str, opens_session: bool = False) -> None:
        """Click Submit once the page has enabled it; a button that stays disabled stops the run with a clear reason."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            self.page.wait_for_function("() => { const b = document.querySelector('#Submitbtn'); return b && !b.disabled }",
                                        timeout=10_000)
        except PlaywrightTimeout:
            # "<stage> entry": nothing was sent, so this is not a rejection (and must not trigger the MPIN block)
            raise LoginFailed(f"{stage} entry", "the portal kept its Submit button disabled after typing") from None
        if opens_session:
            self._session_started = True  # from here a (partial) session may exist on the portal: always log out
        self.page.click("#Submitbtn")

    def _customer_choices_shown(self) -> bool:
        choices = self.page.locator(CUSTOMER_CHOICES)
        return choices.count() > 1 and choices.first.is_visible()

    def _choose_customer(self) -> None:
        count = self.page.locator(CUSTOMER_CHOICES).count()
        if not self.creds.customer_id:
            raise LoginFailed("customer", f"{count} customer IDs are linked to this mobile number; run setup again and "
                                          "enter the customer ID to use")
        choice = self.page.locator(f"{CUSTOMER_CHOICES}[value='{self.creds.customer_id}']")  # digits only (vault)
        if choice.count() != 1:
            raise LoginFailed("customer", "the saved customer ID is not one of those linked to this mobile number")
        choice.check()
        self.page.click("#Submitbtn")

    def _step_at(self, steps: Tuple[str, ...]) -> Optional[str]:
        """Which of `steps` the page's current address is, if any."""
        url = self.page.url
        return next((name for name in steps if STEPS[name].search(url)), None)

    def _expect(self, steps: Tuple[str, ...], stage: str, choices_shown: Optional[Callable[[], bool]] = None) -> str:
        """Wait until the page is one of `steps` (or, with choices_shown, until that is true: returns 'choices')."""
        from playwright.sync_api import Error as PlaywrightError

        page = self.page
        start = urlsplit(page.url).path.lower()
        deadline = time.monotonic() + self.step_timeout_s
        while time.monotonic() < deadline:
            if self.dialogs:
                raise LoginFailed(stage, f"portal said: {self.dialogs[-1]}")
            # The address is checked on its own first: the page checks below throw while a navigation is under way,
            # and a navigation to an expected page must never be mistaken for an unexpected one (it was, 2026-09-26).
            reached = self._step_at(steps)
            if reached:
                return reached
            try:
                alert = page.locator(ALERT)
                if alert.count() and alert.first.is_visible():
                    raise LoginFailed(stage, "portal said: " + alert.first.inner_text().strip()[:200])
                if choices_shown and choices_shown():
                    return "choices"
            except PlaywrightError:
                pass  # the page is navigating; look again
            reached = self._step_at(steps)
            if reached:
                return reached
            path = urlsplit(page.url).path
            if path.lower() != start:
                raise LoginFailed(stage, f"unexpected page {path}")
            page.wait_for_timeout(250)
        raise LoginFailed(stage, f"timed out waiting for {' or '.join(steps)} (still on {urlsplit(page.url).path})")

    # -- logout ----------------------------------------------------------------------------------------------------
    def _logout(self) -> bool:
        """Call the portal's own logout (the menu's Logoutfunc) and wait for the login page; False if unconfirmed."""
        page = self.page
        has_logout = "() => typeof Logoutfunc === 'function'"
        try:
            if not page.evaluate(has_logout):
                page.goto(self.base + READ_PAGES["dashboard"], wait_until="domcontentloaded")
                page.wait_for_function(has_logout, timeout=15_000)
            page.evaluate("() => Logoutfunc()")
            page.wait_for_url(re.compile(r"/login/login\.html", re.I), timeout=20_000)
            log.info("logged out")
            return True
        except Exception as e:
            log.warning("logout not confirmed: %s", type(e).__name__)
            return False

    def _screenshot(self, name: str) -> None:
        """Save what the page showed when a run failed, with the typed mobile/OTP/MPIN fields blanked first."""
        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            self.page.evaluate("() => document.querySelectorAll('input').forEach(i => { if (i.type !== 'button' && "
                               "i.type !== 'submit' && i.type !== 'radio') i.value = '' })")
            self.page.screenshot(path=str(self.evidence_dir / name), full_page=True)
        except Exception:
            pass


class Progress:
    """Step-by-step updates for the user (Telegram). Never raises: an update failing must not break the run."""

    def __init__(self, send: Optional[Callable[[str], object]]):
        self.send = send

    def __call__(self, text: str) -> None:
        if self.send is None:
            return
        try:
            self.send(text)
        except Exception as e:
            log.warning("progress update failed: %s", type(e).__name__)


def read_pages(page, base: str, out_dir: Path, pages: Dict[str, str] = READ_PAGES, settle_ms: int = 20_000,
               progress: Optional[Callable[[str], object]] = None, blocked: Optional[List[Blocked]] = None,
               tabs: Mapping[str, Sequence[str]] = READ_TABS) -> dict:
    """Open each read page by address and save what it shows: text, visible tables, a screenshot, and the portal's
    JSON data responses; then the same for each of its allow-listed read-only tabs (`tabs`). Returns a summary (also
    written to summary.json). After each page, `progress` gets one line, including any requests the guard blocked
    (`blocked` is the run's list) while it was open."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    tell = Progress(progress)
    blocked = blocked if blocked is not None else []

    out_dir.mkdir(parents=True, exist_ok=True)
    host = (urlsplit(base).hostname or "").lower()
    finished: list = []  # requests whose response has fully arrived: reading their body can never hang
    activity = [time.monotonic()]  # when the network last did anything (a request started, finished or failed)

    def collect(request) -> None:
        finished.append(request)
        activity[0] = time.monotonic()

    def touched(_request) -> None:
        activity[0] = time.monotonic()

    page.on("requestfinished", collect)
    page.on("request", touched)
    page.on("requestfailed", touched)
    summary: dict = {}
    try:
        for i, (name, rel) in enumerate(pages.items(), start=1):
            finished.clear()
            blocked_before = len(blocked)
            page.goto(base + rel, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=settle_ms)
            except PlaywrightTimeout:
                pass  # pages that poll prices never go idle; take what has loaded
            path = urlsplit(page.url).path
            entry: dict = {"page": path, "reached": path.lower().endswith(rel.lower())}
            if entry["reached"]:
                entry.update(_snapshot(page, out_dir, name))
            entry["data_calls"] = _save_responses(list(finished), host, out_dir / "api", name)
            if entry["reached"]:
                entry["tabs"] = {label: _read_tab(page, label, out_dir, name, host, finished, activity, settle_ms)
                                 for label in tabs.get(name, ())}
            summary[name] = entry
            tell(_page_line(i, len(pages), name, entry, blocked[blocked_before:]))
    finally:
        page.remove_listener("requestfinished", collect)
        page.remove_listener("request", touched)
        page.remove_listener("requestfailed", touched)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _snapshot(page, out_dir: Path, stem: str) -> dict:
    """Save the page as it is now: visible text, visible tables and a full-page screenshot."""
    text = page.inner_text("body")
    tables = page.evaluate(TABLES_JS)
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.tables.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False), encoding="utf-8")
    page.screenshot(path=str(out_dir / f"{stem}.png"), full_page=True)
    return {"text_chars": len(text), "tables": len(tables), "rows": sum(max(0, len(t) - 1) for t in tables)}


def _quiet(page, activity: list, settle_ms: int, quiet_ms: int = 800) -> None:
    """Wait until the network has done nothing for quiet_ms (at most settle_ms). Playwright's own 'networkidle' returns
    at once if the page was idle before a click, and never comes on a page with a live price stream."""
    deadline = time.monotonic() + settle_ms / 1000
    page.wait_for_timeout(200)  # let the click's requests start
    while time.monotonic() < deadline and time.monotonic() - activity[0] < quiet_ms / 1000:
        page.wait_for_timeout(100)


def _read_tab(page, label: str, out_dir: Path, name: str, host: str, finished: list, activity: list,
              settle_ms: int) -> dict:
    """Click one read-only tab by its exact visible label and save what it shows. A label that reads like an action is
    refused outright (a second check behind the allow-list); a tab that is missing is reported, not guessed at."""
    from playwright.sync_api import Error as PlaywrightError

    if WRITE_WORDS.search(label):
        return {"status": "refused: looks like an action"}
    target = page.get_by_text(label, exact=True)
    visible = [target.nth(i) for i in range(target.count()) if target.nth(i).is_visible()]
    if not visible:
        return {"status": "not found"}
    stem = f"{name}__{re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()}"
    finished.clear()
    try:
        visible[0].click()
    except PlaywrightError as e:
        return {"status": f"click failed: {type(e).__name__}"}
    _quiet(page, activity, settle_ms)
    return {"status": "read", **_snapshot(page, out_dir, stem),
            "data_calls": len(_save_responses(list(finished), host, out_dir / "api", stem))}


def _page_line(i: int, n: int, name: str, entry: dict, blocked: List[Blocked]) -> str:
    title = esc(PAGE_NAMES.get(name, name))
    if entry["reached"]:
        line = f"📄 <b>{i}/{n} {title}</b>: read ({entry['tables']} tables, {len(entry['data_calls'])} data responses)"
        tabs = entry.get("tabs") or {}
        if tabs:
            read = [f"{esc(label)} ({t['rows']} rows)" for label, t in tabs.items() if t["status"] == "read"]
            missed = [esc(label) for label, t in tabs.items() if t["status"] != "read"]
            line += f"\n   tabs read: {', '.join(read) or 'none'}" + (f"; not read: {', '.join(missed)}" if missed else "")
    else:
        line = f"⚠️ <b>{i}/{n} {title}</b>: could not open (the portal went to {esc(entry['page'])})"
    if blocked:
        actions = sorted({esc(b.action.rsplit("/", 1)[-1] or b.path.rsplit("/", 1)[-1]) for b in blocked})
        line += f"\n   🛡️ blocked {len(blocked)} write request(s) the page tried: {', '.join(actions)[:200]}"
    return line


def _save_responses(requests: list, host: str, out_dir: Path, prefix: str) -> List[dict]:
    """Save the portal's own data (XHR/fetch) responses, skipping login/logout calls and script/style files. Only
    requests that have finished are passed in: a live price stream never finishes, and waiting on its body would hang
    the run (it did on 2026-09-26). Request bodies are never saved."""
    saved: List[dict] = []
    for req in requests:
        parts = urlsplit(req.url)
        if req.resource_type not in ("xhr", "fetch") or (parts.hostname or "").lower() != host \
                or parts.path.lower().rstrip("/") in AUTH_PATHS or is_static_file(parts.path):
            continue
        r = req.response()
        if r is None:
            continue
        try:
            body = r.json()
        except Exception:
            try:
                body = r.text()[:200_000]
            except Exception:
                body = None  # aborted by the guard, or the page moved on
        action = action_of(req.headers)
        if MARKET_DATA.search(f"{parts.path} {action}"):
            continue
        label = re.sub(r"[^A-Za-z0-9]+", "_", action or parts.path.rsplit("/", 1)[-1]).strip("_")[:60]
        out_dir.mkdir(parents=True, exist_ok=True)
        file = out_dir / f"{prefix}_{len(saved):02d}_{label}.json"
        file.write_text(json.dumps({"method": req.method, "path": parts.path, "action": action, "status": r.status,
                                    "body": body}, indent=1, ensure_ascii=False), encoding="utf-8")
        saved.append({"file": file.name, "path": parts.path, "action": action, "status": r.status})
    return saved


def blocked_report(blocked: List[Blocked]) -> List[dict]:
    return [asdict(b) for b in blocked]
