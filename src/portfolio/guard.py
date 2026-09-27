"""The network guard: a second, independent line of defence behind "click nothing after login".

The portal sends every call as a POST, mostly to a few generic endpoints ("Common/Claim") that dispatch on the
App-Action-Name / App-Process-Name headers, so the guard checks the URL *and* those header values. Anything that looks
like a transaction is aborted before it leaves the browser. Over-blocking a read only loses that piece of data (and it
shows up in the run's report); under-blocking could move money, so the word list errs wide."""
import re
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlsplit

WRITE_WORDS = re.compile(
    r"order|buy|sell|payout|withdraw|pledge|transfer|payment|paygateway|redeem|purchase|mandate|addfund|sip|bid|apply"
    r"|switch|cancel|modify|delete|remove|update|insert|save|submit|register|credential|changempin|resetmpin|feedback",
    re.I)
# The login and logout calls themselves (they authenticate; none of them can trade or move money).
AUTH_PATHS = frozenset(p.lower() for p in (
    "/CIAMS/api/v1/TwoFA/init", "/CIAMS/api/v1/TwoFA/coidverify", "/CIAMS/api/v1/TwoFA/Verify",
    "/CIAMS/api/v1/TwoFA/coidlogin", "/CIAMS/api/v1/SSAuth/InitCustomer", "/CIAMS/api/v1/SSAuth/login",
    "/CIAMS/api/v1/SSAuth/logout", "/CIAMS/api/v1/Device/feature", "/CIAMS/api/v1/UserManagement/isvalidsession",
    "/StateManager/api/v1/State"))
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
FIRST_PARTY = ("integrated.investments", "integratedindia.in")


@dataclass(frozen=True)
class Blocked:
    method: str
    host: str
    path: str
    action: str
    reason: str


KNOWN = {  # what the portal's own automatic requests are, in plain words
    "ForceRemoveChild": "remove linked sessions/users (fired by the dashboard on load)",
    "ClientSettingsModify": "change your notification settings (fired by the dashboard on load)",
    "adkmfsip": "mutual-fund SIP advert",
}


def describe(b: "Blocked") -> str:
    """One line for the user: what was stopped, in plain words where known."""
    name = b.action.rsplit("/", 1)[-1] or b.path.rsplit("/", 1)[-1]
    plain = next((text for key, text in KNOWN.items() if key.lower() in f"{b.path} {b.action}".lower()), "")
    return f"{name}: {plain}" if plain else f"{name} ({b.reason})"


def action_of(headers: Dict[str, str]) -> str:
    """The portal's action/process header values, e.g. 'Holdings/Portfolio'."""
    return "/".join(v for k, v in sorted(headers.items()) if k.lower().startswith("app-") and v)


def verdict(method: str, url: str, headers: Dict[str, str], portal_host: str) -> Optional[Blocked]:
    """None if the request may go ahead, else why it is blocked."""
    parts = urlsplit(url)
    host, path, method = (parts.hostname or "").lower(), parts.path, method.upper()
    action = action_of(headers)
    first_party = host == portal_host or any(host == d or host.endswith("." + d) for d in FIRST_PARTY)
    if not first_party:
        if method in SAFE_METHODS:
            return None  # fonts, scripts, images from CDNs
        return Blocked(method, host, path, action, "writes to a third-party host")
    if path.lower().rstrip("/") in AUTH_PATHS:
        return None
    hit = WRITE_WORDS.search(f"{path}?{parts.query} {action}")
    if hit and not (method in SAFE_METHODS and not action and is_static_file(path)):
        return Blocked(method, host, path, action, f"looks like a transaction ({hit.group(0)!r})")
    return None


def is_static_file(path: str) -> bool:
    """A page, script, style or image fetched by GET: a file, not an action (e.g. Scripts/OrderWindow.js)."""
    return path.lower().endswith((".html", ".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".woff", ".woff2", ".ico"))
