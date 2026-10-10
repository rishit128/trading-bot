"""The command line: every flag still parses, `--live` stays an alias of `--paper-orders` (the server's compose file uses it),
and each command goes to the right place in the right order."""

import sys

import pytest

import main


def parse(*argv):
    return main.build_parser().parse_args(list(argv))


def test_paper_orders_and_the_older_live_flag_are_the_same_switch():
    assert parse().live is False  # the default is a dry run
    assert parse("--paper-orders").live is True and parse("--live").live is True
    args = parse("--loop", "30", "--live", "--holdout")  # the exact command the swing container runs
    assert (args.loop, args.live, args.holdout) == (30, True, True)
    assert parse("--intraday", "--live").intraday and parse("--intraday", "--paper-orders").live


def test_every_documented_flag_is_accepted():
    for flags in (
        ["--screen"], ["--graph"], ["--check"], ["--report"], ["--positions"], ["--intraday-report"],
        ["--portfolio", "--show", "--otp-from", "terminal", "--no-ai"],
    ):  # fmt: skip
        main.build_parser().parse_args(flags)
    with pytest.raises(SystemExit):
        parse("--otp-from", "carrier-pigeon")


@pytest.fixture
def calls(monkeypatch):
    """Replace every command with a recorder so a test sees which one `main` chose, and with what."""
    seen = []
    monkeypatch.setattr(main, "configure_logging", lambda *a: None)
    monkeypatch.setattr(main, "load_settings_or_exit", lambda: "settings")
    monkeypatch.setattr(main, "run_portfolio_agent", lambda *a: seen.append(("portfolio", a)) or 0)
    monkeypatch.setattr(main, "run_intraday", lambda live: seen.append(("intraday", live)))
    monkeypatch.setattr(main, "show_intraday_reports", lambda s: seen.append(("intraday-report", s)))
    monkeypatch.setattr(main, "preflight", lambda pipeline: seen.append(("check", pipeline)) or [])
    monkeypatch.setattr(main, "run_trading_bot", lambda *a: seen.append(("bot", a)))
    monkeypatch.setattr(
        main, "REPORT_COMMANDS", tuple((name, lambda s, n=name: seen.append((n, s))) for name, _ in main.REPORT_COMMANDS)
    )

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["main.py", *argv])
        main.main()
        return seen[-1] if seen else None

    return run


def test_the_swing_bot_gets_its_loop_paper_orders_and_holdout_settings(calls):
    assert calls("--loop", "30", "--live", "--holdout") == ("bot", (True, True, 30))
    assert calls() == ("bot", (False, False, None))  # no flags: one dry-run cycle


def test_each_command_goes_to_its_handler(calls):
    assert calls("--intraday", "--paper-orders") == ("intraday", True)
    assert calls("--intraday-report") == ("intraday-report", "settings")
    assert calls("--report") == ("report", "settings") and calls("--positions") == ("positions", "settings")
    with pytest.raises(SystemExit) as done:  # the portfolio run ends the process with its own exit code
        calls("--portfolio", "--no-ai")
    assert done.value.code == 0


def test_the_first_matching_report_wins_and_reports_come_before_the_check(calls):
    assert calls("--report", "--graph") == ("graph", "settings")  # graph is tried before report
    assert calls("--check", "--report") == ("report", "settings")


def test_check_exits_zero_when_healthy_and_one_when_a_critical_check_fails(calls, monkeypatch):
    with pytest.raises(SystemExit) as ok:
        calls("--check")
    assert ok.value.code == 0
    monkeypatch.setattr(main, "preflight", lambda pipeline: ["OpenRouter key"])
    with pytest.raises(SystemExit) as bad:
        calls("--check")
    assert bad.value.code == 1


def test_a_loop_shorter_than_a_minute_is_refused(calls, capsys):
    with pytest.raises(SystemExit) as refused:
        calls("--loop", "0")
    assert refused.value.code == 2 and "at least 1 minute" in capsys.readouterr().err
