"""Startup connectivity checks, so a bad key or a dead service is reported clearly at launch instead of on the first cycle."""
from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence

import httpx


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one startup check."""
    name: str
    ok: bool
    detail: str
    critical: bool


def run_checks(checks: Sequence[tuple]) -> List[CheckResult]:
    """Each check is (name, critical, fn); fn returns a short detail string or raises. Never raises itself."""
    results = []
    for name, critical, fn in checks:
        try:
            results.append(CheckResult(name, True, str(fn()), critical))
        except Exception as e:
            results.append(CheckResult(name, False, f"{type(e).__name__}: {str(e)[:120]}", critical))
    return results


def openrouter_check(key: str, get: Callable = httpx.get) -> str:
    """Validates the key without spending any model quota."""
    r = get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    if r.status_code == 401:
        raise RuntimeError("OpenRouter rejected the key (401)")
    r.raise_for_status()
    return "key accepted"


def market_data_check(download: Callable, ticker: str) -> str:
    """Fetch a few recent bars of a bellwether index."""
    df = download(ticker, period="5d", interval="1d", progress=False, timeout=15)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no data returned for {ticker}")
    return f"{ticker}: {len(df)} recent bars"


def broker_check(broker, label: str) -> str:
    """Read the account to prove the broker connection works."""
    p = broker.portfolio()
    return f"{label}: equity {p.equity:,.2f}, cash {p.cash:,.2f}, {len(p.positions)} positions"


def telegram_check(token: str, get: Callable = httpx.get) -> str:
    """Confirm the Telegram bot token is valid."""
    r = get(f"https://api.telegram.org/bot{token}/getMe", timeout=15)
    r.raise_for_status()
    return "bot @" + str((r.json().get("result") or {}).get("username"))


def format_results(results: Sequence[CheckResult]) -> str:
    """Human-readable list of check outcomes."""
    return "\n".join(f"  {'OK  ' if r.ok else ('FAIL' if r.critical else 'WARN')} {r.name}: {r.detail}" for r in results)


def critical_failures(results: Sequence[CheckResult]) -> List[CheckResult]:
    """The failed checks that should block startup."""
    return [r for r in results if not r.ok and r.critical]


def build_checks(settings, broker, env: Mapping[str, str], get: Callable = httpx.get, download: Callable = None) -> list:
    """The checks for the configured market. Keys and the broker are critical; market data and Telegram only warn."""
    if download is None:
        import yfinance as yf

        download = yf.download
    label = "built-in paper simulator"
    checks = [
        ("OpenRouter key", True, lambda: openrouter_check(env.get("OPENROUTER_API_KEY", ""), get)),
        ("Broker (" + label + ")", True, lambda: broker_check(broker, label)),
        ("Market data (Yahoo)", False, lambda: market_data_check(download, "^NSEI")),
    ]
    if env.get("TELEGRAM_BOT_TOKEN"):
        checks.append(("Telegram bot", False, lambda: telegram_check(env["TELEGRAM_BOT_TOKEN"], get)))
    return checks
