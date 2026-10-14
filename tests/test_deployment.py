"""The deployment rules that protect a running bot, pinned as tests: a deploy must not overwrite the server's settings, the
image is built once, and each container has a health check on its own heartbeat."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.aws.yml").read_text(encoding="utf-8"))["services"]
DEPLOY = (ROOT / "deploy" / "deploy.sh").read_text(encoding="utf-8")


def tracked_files():
    run = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return run.stdout.split() if run.returncode == 0 else None


def test_a_deploy_ships_the_committed_tree_so_no_local_env_or_database_can_travel():
    assert "git archive" in DEPLOY and "HEAD" in DEPLOY and "tar --exclude" not in DEPLOY  # not the working directory
    assert "names only" in DEPLOY  # differing settings are reported by name, never by value
    files = tracked_files()
    if files is None:
        pytest.skip("not a git checkout")
    risky = [
        f
        for f in files
        if f == ".env" or f.endswith((".pem", ".db", ".pkl")) or f.startswith(("research_cache/", "portfolio_data/"))
    ]
    assert risky == []  # nothing secret or bulky is tracked, so nothing of it can be archived


def test_a_deploy_refuses_a_dirty_tree_unless_told_to_leave_the_changes_out():
    assert "git status --porcelain --untracked-files=no" in DEPLOY and "DEPLOY_UNCOMMITTED" in DEPLOY
    assert "uncommitted changes were NOT deployed" in DEPLOY  # and the record says so when it was overridden


def committed(path: str) -> bytes | None:
    run = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True)
    return run.stdout if run.returncode == 0 else None


def test_the_scripts_that_run_on_the_server_are_committed_with_unix_line_endings():
    """The watchdog once reached the server with Windows line endings and bash could not parse it, so it never ran."""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes
    for path in ("deploy/watchdog.sh", "deploy/deploy.sh"):
        blob = committed(path)
        if blob is None:
            pytest.skip("not a git checkout")
        assert b"\r" not in blob, f"{path} is committed with CRLF line endings"
    assert "git show HEAD:deploy/watchdog.sh" in DEPLOY  # the watchdog is uploaded from the committed file


def working_bash() -> bool:
    """A usable bash (on some Windows PCs `bash` is a launcher for a WSL distribution that is not installed)."""
    if shutil.which("bash") is None:
        return False
    probe = subprocess.run(["bash", "-c", "echo ok"], capture_output=True)
    return probe.stdout.strip() == b"ok"


@pytest.mark.skipif(not working_bash(), reason="needs a working bash")
def test_the_watchdog_is_a_valid_shell_script_as_the_server_will_receive_it():
    blob = committed("deploy/watchdog.sh")
    if blob is None:
        pytest.skip("not a git checkout")
    run = subprocess.run(["bash", "-n"], input=blob, capture_output=True)
    assert run.returncode == 0, run.stderr.decode()


def test_the_image_is_built_once_and_shared_by_both_containers():
    assert [name for name, service in COMPOSE.items() if "build" in service] == ["trading-bot"]
    assert COMPOSE["trading-bot"]["image"] == COMPOSE["intraday-bot"]["image"]
    assert "trading-bot" in COMPOSE["intraday-bot"]["depends_on"]  # the build happens before the image is used


def test_each_container_is_health_checked_on_its_own_heartbeat():
    limits = {"trading-bot": 4500, "intraday-bot": 900}  # the swing loop reports every 30 min, the intraday one every 5
    for name, limit in limits.items():
        service = COMPOSE[name]
        path = service["environment"]["HEARTBEAT_FILE"]
        assert service["healthcheck"]["test"] == ["CMD", "python", "-m", "src.monitoring.heartbeat", "check", path, str(limit)]
        assert path.startswith("/data/")  # on the volume the bot user can write
    assert len({COMPOSE[n]["environment"]["HEARTBEAT_FILE"] for n in limits}) == 2


def test_the_watchdog_reports_an_unhealthy_container():
    assert "unhealthy" in (ROOT / "deploy" / "watchdog.sh").read_text(encoding="utf-8")


def test_a_deploy_records_which_commit_it_put_on_the_server():
    """So that production can be asked what it runs, and a rollback knows what to go back to."""
    assert "git rev-parse --short HEAD" in DEPLOY and ".deployed-commit" in DEPLOY
