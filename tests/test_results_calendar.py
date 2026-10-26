from datetime import date, datetime, timezone

from src.data.angel_quotes import EntryContext
from src.data.filings import Filing
from src.database import make_session_factory
from src.news.results_calendar import classify, record_notices, results_note, with_results

NOTICE = (
    "Raymond Ltdhas informed BSE that the meeting of the Board of Directors of the Company is scheduled on 27/10/2026 ,"
    "inter alia, to consider and approve Unaudited Financial Results"
)


def filing(subject, link, published=datetime(2026, 10, 20, 6, tzinfo=timezone.utc)):
    return Filing("BSE", "RAYMOND", "Raymond", subject, link, published)


def test_classify_reads_the_meeting_date_and_the_results_filings():
    assert classify(NOTICE) == ("MEETING", "2026-10-27")
    assert classify("Outcome of Board Meeting held on October 21, 2026") == ("RESULT", None)
    assert classify("Unaudited Financial Results for the quarter ended September 30, 2026") == ("RESULT", None)
    assert classify("Board meeting scheduled on 12/11/2026 to consider a dividend") == (None, None)  # not about results
    assert classify("scheduled on 31/02/2026 financial results") == (None, None)  # an impossible date
    assert classify("Appointment of a director") == (None, None)


def test_results_note_says_how_near_the_results_are_and_whether_they_were_just_filed(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'n.db'}")
    with sessions() as s:
        added = record_notices(s, [filing(NOTICE, "a"), filing(NOTICE, "a"), filing("Appointment", "b")])
        s.commit()
    assert added == 1  # the same link twice and an unrelated subject: one row
    assert results_note(sessions, "RAYMOND", date(2026, 10, 23)) == {"results_in_days": 4}
    assert results_note(sessions, "OTHER", date(2026, 10, 23)) == {}
    with sessions() as s:
        record_notices(s, [filing("Outcome of Board Meeting", "c", datetime(2026, 10, 28, 6, tzinfo=timezone.utc))])
        s.commit()
    assert results_note(sessions, "RAYMOND", date(2026, 10, 30)) == {"results_filed_days_ago": 1}  # the meeting has passed


def test_with_results_adds_to_the_note_and_never_fails_the_entry(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'n.db'}")
    with sessions() as s:
        record_notices(s, [filing(NOTICE, "a")])
        s.commit()
    base = lambda symbol, price: EntryContext(None, {"spread_pct": 0.1})  # noqa: E731
    out = with_results(base, sessions, lambda: date(2026, 10, 23))("RAYMOND", 100.0)
    assert out.note == {"spread_pct": 0.1, "results_in_days": 4} and out.skip is None

    def broken():
        raise RuntimeError("clock")

    assert with_results(base, sessions, broken)("RAYMOND", 100.0).note == {"spread_pct": 0.1}


def test_a_buy_within_three_days_of_announced_results_is_refused_and_a_later_one_is_not(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'n.db'}")
    with sessions() as s:
        record_notices(s, [filing(NOTICE, "a")])  # results meeting on 27 Oct
        s.commit()
    base = lambda symbol, price: EntryContext(None, {})  # noqa: E731
    near = with_results(base, sessions, lambda: date(2026, 10, 24))("RAYMOND", 100.0)  # 3 days before
    assert "results due in 3 day" in near.skip and near.note["results_in_days"] == 3
    assert with_results(base, sessions, lambda: date(2026, 10, 23))("RAYMOND", 100.0).skip is None  # 4 days: still allowed
    assert with_results(base, sessions, lambda: date(2026, 10, 24))("OTHER", 100.0).skip is None  # no meeting announced
    other = lambda symbol, price: EntryContext("near the circuit", {})  # noqa: E731
    assert with_results(other, sessions, lambda: date(2026, 10, 24))("RAYMOND", 100.0).skip == "near the circuit"
