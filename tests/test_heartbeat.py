"""The heartbeat: what each loop records, and when the container health check calls the bot unhealthy."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from src.intraday.engine import run_together
from src.monitoring.heartbeat import MAX_CONSECUTIVE_FAILURES, Heartbeat, check, main
from src.runner import run_loop

NOW = 1_000_000.0


def beat(tmp_path, ok=True, detail="", at=NOW):
    hb = Heartbeat(str(tmp_path / "hb"), clock=lambda: at)
    hb.beat(ok, detail)
    return hb


def test_a_fresh_heartbeat_is_healthy_and_says_what_the_loop_did(tmp_path):
    beat(tmp_path, True, "market closed")
    healthy, why = check(str(tmp_path / "hb"), 4500, clock=lambda: NOW + 120)
    assert healthy and "120s ago" in why and "market closed" in why


def test_a_stale_heartbeat_means_the_loop_is_stuck_or_stopped(tmp_path):
    beat(tmp_path)
    healthy, why = check(str(tmp_path / "hb"), 900, clock=lambda: NOW + 901)
    assert not healthy and "stuck or stopped" in why


def test_no_file_is_unhealthy_not_a_crash(tmp_path):
    healthy, why = check(str(tmp_path / "missing"), 900, clock=lambda: NOW)
    assert not healthy and "no readable heartbeat" in why
    (tmp_path / "garbage").write_text("not json", encoding="utf-8")
    assert not check(str(tmp_path / "garbage"), 900)[0]


def test_one_failed_cycle_is_a_blip_but_repeated_failures_are_unhealthy(tmp_path):
    hb = Heartbeat(str(tmp_path / "hb"), clock=lambda: NOW)
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        hb.beat(False, "ConnectionError: yahoo down")
    assert check(str(tmp_path / "hb"), 900, clock=lambda: NOW)[0]
    hb.beat(False, "ConnectionError: yahoo down")
    healthy, why = check(str(tmp_path / "hb"), 900, clock=lambda: NOW)
    assert not healthy and "last 3 cycles failed" in why and "yahoo down" in why
    hb.beat(True, "recovered")  # a good cycle resets the count
    assert check(str(tmp_path / "hb"), 900, clock=lambda: NOW)[0]


def test_writing_is_atomic_and_an_unwritable_path_never_stops_trading(tmp_path):
    beat(tmp_path, True, "x" * 500)
    assert len(json.loads((tmp_path / "hb").read_text(encoding="utf-8"))["detail"]) == 200 and not list(tmp_path.glob("*.tmp"))
    Heartbeat(str(tmp_path / "no-such-dir" / "hb")).beat(True)  # raises nothing


def test_unconfigured_means_off():
    assert Heartbeat.from_setting("") is None and Heartbeat.from_setting("/data/hb") is not None


def test_the_swing_loop_records_every_kind_of_cycle(tmp_path):
    marks = []
    hb = SimpleNamespace(beat=lambda ok, detail="": marks.append((ok, detail)))
    closed = SimpleNamespace(broker=SimpleNamespace(is_market_open=lambda: False), notify=lambda t: None)
    run_loop(closed, 5, sleep=lambda s: None, max_cycles=1, out=lambda s: None, heartbeat=hb)
    opened = SimpleNamespace(broker=SimpleNamespace(is_market_open=lambda: True), run_once=lambda: [], notify=lambda t: None)
    run_loop(opened, 5, sleep=lambda s: None, max_cycles=1, out=lambda s: None, heartbeat=hb)

    def down():
        raise ConnectionError("network down")

    broken = SimpleNamespace(broker=SimpleNamespace(is_market_open=down), notify=lambda t: None)
    run_loop(broken, 5, sleep=lambda s: None, max_cycles=1, out=lambda s: None, heartbeat=hb)
    assert marks == [(True, "market closed"), (True, "cycle done"), (False, "ConnectionError: network down")]


def test_the_intraday_loop_reports_which_engine_failed_and_the_others_still_run(tmp_path):
    class Stop(BaseException):
        pass

    def broken():
        raise ConnectionError("yahoo down")

    marks, steps = [], []
    hb = SimpleNamespace(beat=lambda ok, detail="": marks.append((ok, detail)))
    engines = [SimpleNamespace(label="INTRADAY", step=broken), SimpleNamespace(label="GAP-GO", step=lambda: steps.append("ran"))]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        raise Stop()

    try:
        run_together(engines, sleep, heartbeat=hb)
    except Stop:
        pass
    assert marks == [(False, "failed: INTRADAY")] and steps == ["ran"]


def test_the_command_line_check_exits_0_healthy_and_1_unhealthy(tmp_path, capsys):
    path = tmp_path / "hb"
    Heartbeat(str(path)).beat(True, "ok")  # written now, with the real clock
    assert main(["x", "check", str(path), "900"]) == 0 and "healthy" in capsys.readouterr().out
    assert main(["x", "check", str(tmp_path / "none"), "900"]) == 1 and "UNHEALTHY" in capsys.readouterr().out
    assert main(["x"]) == 2


def test_the_module_runs_as_the_container_health_check_command(tmp_path):
    path = tmp_path / "hb"
    Heartbeat(str(path)).beat(True, "ok")
    root = Path(__file__).resolve().parent.parent
    run = subprocess.run(
        [sys.executable, "-m", "src.monitoring.heartbeat", "check", str(path), "900"], cwd=root, capture_output=True, text=True
    )
    assert run.returncode == 0 and run.stdout.startswith("healthy")
