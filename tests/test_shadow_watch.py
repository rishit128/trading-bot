from datetime import datetime

from src.data.india import IST
from src.ops.shadow_watch import ShadowWatch, is_due, latest_summary, load_state, new_verdicts, save_state


def at(hour, minute=0, day=4):
    return datetime(2026, 11, day, hour, minute, tzinfo=IST)


def test_due_only_after_the_hour_on_a_trading_day_and_once():
    assert not is_due(at(16, 29), None, True)
    assert is_due(at(16, 30), None, True)
    assert is_due(at(21), "2026-11-03", True)  # a bot that was down catches up
    assert not is_due(at(21), "2026-11-04", True)
    assert not is_due(at(17), None, False)


def test_new_verdicts_only_when_a_test_first_reaches_one():
    running, passed, failed = "RUNNING (1 of 100)", "PASS: yes", "FAIL: no"
    assert new_verdicts({"a": running}, {"a": running}) == []
    assert new_verdicts({"a": running}, {"a": passed}) == ["a: PASS: yes"]
    assert new_verdicts({}, {"a": failed}) == ["a: FAIL: no"]
    assert new_verdicts({"a": failed}, {"a": failed}) == []
    assert new_verdicts({"a": failed}, {"a": passed}) == ["a: PASS: yes"]


def test_state_survives_a_bad_file(tmp_path):
    path = tmp_path / "state.json"
    assert load_state(path) == {}
    path.write_text("{not json")
    assert load_state(path) == {}
    assert "No test report" in latest_summary(path)
    save_state({"sent_day": "2026-11-04", "summary": "numbers"}, path)
    assert latest_summary(path) == "numbers"


def watch(tmp_path, results, trading=True):
    sent, runs = [], []

    def run():
        runs.append(1)
        return results.pop(0)

    return ShadowWatch(lambda now: trading, sent.append, run, tmp_path / "state.json"), sent, runs


def test_a_failed_report_says_so_once_and_is_not_repeated_that_day(tmp_path):
    w, sent, runs = watch(tmp_path, [False, True])
    assert w.check(at(16, 30)) and not w.check(at(16, 35))
    assert len(runs) == 1 and len(sent) == 1 and "failed" in sent[0]
    assert w.check(at(16, 30, day=5))  # tomorrow it tries again
    assert len(runs) == 2 and len(sent) == 1


def test_no_report_on_a_holiday_or_when_already_sent(tmp_path):
    w, _, runs = watch(tmp_path, [], trading=False)
    assert not w.check(at(17))
    w, _, runs = watch(tmp_path, [])
    save_state({"sent_day": "2026-11-04"}, tmp_path / "state.json")
    assert not w.check(at(17)) and not runs


def test_a_broken_tracker_does_not_reach_the_bot(tmp_path):
    def boom(now):
        raise RuntimeError("no")

    ShadowWatch(boom, print, lambda: True, tmp_path / "state.json").tick()
