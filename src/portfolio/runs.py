"""One portal visit with its safety stops: one run at a time, no login after a failed MPIN step, and every run's
output in its own folder under portfolio_data/ (git-ignored: the account's data never leaves this PC)."""
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from src.portfolio.guard import describe
from src.portfolio.session import PORTAL, LoginFailed, PortalRun, blocked_report, read_pages
from src.portfolio.telegram_otp import OtpCancelled, OtpTimeout, TelegramError
from src.portfolio.vault import Credentials

log = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parents[2] / "portfolio_data"
STALE_LOCK_S = 20 * 60


class RunBusy(Exception):
    """Another run holds the lock: two logins at once would each ask for an OTP and confuse both."""


class RunStore:
    def __init__(self, root: Path = DATA):
        self.root = root
        self.profile, self.runs = root / "browser", root / "runs"
        self.state_file, self.lock_file = root / "state.json", root / "run.lock"

    def state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))

    def mpin_blocked(self) -> Optional[str]:
        return self.state().get("mpin_rejected")

    def block_mpin(self) -> None:
        self._save({**self.state(), "mpin_rejected": datetime.now().strftime("%Y-%m-%d %H:%M")})

    def clear_mpin_block(self) -> bool:
        state = self.state()
        cleared = state.pop("mpin_rejected", None) is not None
        self._save(state)
        return cleared

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            if time.time() - self.lock_file.stat().st_mtime > STALE_LOCK_S:
                self.lock_file.unlink()
        except FileNotFoundError:
            pass
        try:
            os.close(os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise RunBusy(f"another portfolio run is in progress (if none is, delete {self.lock_file})") from None
        try:
            yield
        finally:
            self.lock_file.unlink(missing_ok=True)

    def new_run_dir(self) -> Path:
        return self.runs / datetime.now().strftime("%Y-%m-%d_%H%M%S")


def fetch_portfolio(creds: Credentials, get_otp: Callable[[], str], store: RunStore, headless: bool = True,
                    base: str = PORTAL, step_timeout_s: float = 45,
                    progress: Optional[Callable[[str], object]] = None) -> dict:
    """Log in, save the read pages, log out. Never raises: failures come back as {'error': ...}."""
    blocked_since = store.mpin_blocked()
    if blocked_since:
        return {"error": f"not logging in: the MPIN step failed on {blocked_since} and another wrong MPIN could lock "
                         "the account; run `python -m src.portfolio setup` with the correct MPIN", "logged_out": None}
    run_dir = store.new_run_dir()
    result: dict = {"run_dir": str(run_dir)}
    started = time.monotonic()
    run = PortalRun(creds, get_otp, store.profile, run_dir, base=base, headless=headless, step_timeout_s=step_timeout_s,
                    progress=progress)
    try:
        with store.lock():
            try:
                with run as page:
                    result["pages"] = read_pages(page, run.base, run_dir, progress=progress, blocked=run.blocked)
            except LoginFailed as e:
                result["error"] = f"login stopped at {e}"
                if e.stage == "mpin":  # treated as a rejection even on a timeout: better to stop than risk a lockout
                    store.block_mpin()
            except (OtpTimeout, OtpCancelled, TelegramError) as e:
                result["error"] = f"no OTP: {e}"
            except Exception as e:
                log.exception("portfolio fetch failed")
                result["error"] = f"stopped by an unexpected {type(e).__name__}"  # no raw error text leaves the PC
    except RunBusy as e:
        result["error"] = str(e)
    if run.blocked:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "blocked.json").write_text(json.dumps(blocked_report(run.blocked), indent=2))
    result.update(blocked=len(run.blocked), logged_out=run.logged_out, seconds=round(time.monotonic() - started, 1),
                  blocked_list=[describe(b) for b in run.blocked])
    return result
