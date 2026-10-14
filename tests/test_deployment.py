"""The deployment rules that protect a running bot, pinned as tests: a deploy must not overwrite the server's settings, the
image is built once, and each container has a health check on its own heartbeat."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.aws.yml").read_text(encoding="utf-8"))["services"]
DEPLOY = (ROOT / "deploy" / "deploy.sh").read_text(encoding="utf-8")


def test_a_deploy_never_ships_the_local_env_over_the_servers():
    tar_line = DEPLOY[DEPLOY.index("tar --exclude-vcs") : DEPLOY.index("-czf")]  # the whole multi-line command
    assert "--exclude=.env" in tar_line and "--exclude='*.db'" in tar_line and "--exclude=research_cache" in tar_line
    assert "names only" in DEPLOY  # differing settings are reported by name, never by value


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
    assert "+uncommitted-changes" in DEPLOY  # a deploy from a dirty checkout says so
