from datetime import datetime
from zoneinfo import ZoneInfo

from src.ops.watchdog import Facts, decide

IST = ZoneInfo("Asia/Kolkata")


def at(hour, minute=0):
    return datetime(2026, 11, 3, hour, minute, tzinfo=IST)


def facts(hour=10, minute=0, **extra):
    base = dict(boot_id="b1", uptime_s=9999.0, trading_day=True, research_age_s=60.0)
    return Facts(now=at(hour, minute), **{**base, **extra})


def texts(plan):
    return " | ".join(text for _, text in plan.alerts)


def test_all_well_says_nothing():
    plan = decide(facts(), "b1", [])
    assert not plan.restart and plan.alerts == []


def test_a_stopped_bot_is_restarted_but_not_while_the_machine_is_still_booting():
    plan = decide(facts(not_running=("intraday",)), "b1", [])
    assert plan.restart and "intraday" in texts(plan)
    assert not decide(facts(not_running=("intraday",), uptime_s=60.0), "b1", []).restart


def test_a_machine_restart_is_reported_once_and_not_on_the_first_run():
    assert decide(facts(boot_id="b2"), None, []).alerts == []
    plan = decide(facts(boot_id="b2"), "b1", [])
    assert "restarted" in texts(plan) and "Entries may have been missed" in texts(plan) and plan.boot_id == "b2"
    assert "Entries" not in texts(decide(facts(20, boot_id="b2"), "b1", []))  # after hours nothing was missed


def test_a_silent_loop_is_reported_once_a_day_only_in_session_and_only_if_it_is_running():
    stuck = {"intraday": 30 * 60.0, "swing": 40 * 60.0}
    plan = decide(facts(heartbeat_age_s=stuck), "b1", [])
    assert "intraday loop" in texts(plan) and "swing" not in texts(plan)  # 40 minutes is normal for the swing loop
    assert decide(facts(heartbeat_age_s=stuck), "b1", [k for k, _ in plan.alerts]).alerts == []
    assert decide(facts(20, heartbeat_age_s=stuck), "b1", []).alerts == []
    assert "stuck" not in texts(decide(facts(heartbeat_age_s=stuck, not_running=("intraday",)), "b1", []))
    assert decide(facts(heartbeat_age_s={"intraday": None}), "b1", []).alerts == []  # no file yet


def test_the_research_watch_is_asked_for_twice_on_a_trading_day_and_never_on_a_holiday():
    assert decide(facts(9, 10, research_age_s=None), "b1", []).alerts == []  # before 09:20
    first = decide(facts(9, 21, research_age_s=None), "b1", [])
    assert "/research_start" in texts(first)
    sent = [k for k, _ in first.alerts]
    assert decide(facts(10, research_age_s=99999.0), "b1", sent).alerts == []
    assert "/research_start" in texts(decide(facts(11, 1, research_age_s=99999.0), "b1", sent))
    assert decide(facts(research_age_s=None, trading_day=False), "b1", []).alerts == []
    assert decide(facts(15, 5, research_age_s=None), "b1", ["2026-11-03 research 0", "2026-11-03 research 1"]).alerts == []


def test_positions_left_after_the_square_off_are_reported():
    assert decide(facts(15, 10, left_open={"ORB": 2}), "b1", []).alerts == []  # before the check
    plan = decide(facts(15, 21, left_open={"ORB": 2, "news": 0}), "b1", [])
    assert "ORB 2" in texts(plan) and "news" not in texts(plan)
