"""Fetching 5-minute bars and archiving them: idempotent, round-trips exactly, and keeps sessions apart."""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.india import IST
from src.data.intraday_bars import BarArchive, fetch_5m_bars, fetch_today_bars
from src.database import make_session_factory


def frame(day, n=4, base=100.0, start=(9, 15)):
    idx = pd.DatetimeIndex([datetime(*day, *start, tzinfo=IST) + timedelta(minutes=5 * i) for i in range(n)])
    px = [base + i for i in range(n)]
    return pd.DataFrame({"Open": px, "High": [p + 0.5 for p in px], "Low": [p - 0.5 for p in px], "Close": [p + 0.25 for p in px],
                         "Volume": [1000.0 + i for i in range(n)]}, index=idx)


@pytest.fixture
def archive(tmp_path):
    return BarArchive(make_session_factory(f"sqlite:///{tmp_path / 'a.db'}"))


# -- fetching -----------------------------------------------------------------------------------------------------------
def test_fetch_returns_ist_frames_per_symbol_and_omits_symbols_without_data():
    wide = pd.concat({"AAA.NS": frame((2026, 9, 21)).tz_convert("UTC")}, axis=1)     # BBB.NS is missing entirely
    out = fetch_5m_bars(["AAA", "BBB"], "1d", download=lambda tickers, **kw: wide)
    assert set(out) == {"AAA"} and str(out["AAA"].index.tz) == str(IST) and len(out["AAA"]) == 4
    assert fetch_5m_bars(["AAA"], download=lambda tickers, **kw: pd.DataFrame()) == {}
    assert fetch_5m_bars(["AAA"], download=lambda tickers, **kw: None) == {}


def test_fetch_asks_yahoo_for_the_period_in_batches_and_drops_bars_without_a_close():
    calls = []
    bad = frame((2026, 9, 21))
    bad.iloc[1, bad.columns.get_loc("Close")] = float("nan")

    def download(tickers, **kw):
        calls.append((list(tickers), kw))
        return pd.concat({t: bad for t in tickers}, axis=1)

    out = fetch_5m_bars(["A", "B", "C"], "59d", download=download, batch=2)
    assert [c[0] for c in calls] == [["A.NS", "B.NS"], ["C.NS"]] and calls[0][1] == {"period": "59d", "interval": "5m"}
    assert len(out["A"]) == 3 and set(out) == {"A", "B", "C"}


def test_fetch_today_is_the_one_day_period():
    seen = []
    fetch_today_bars(["A"], download=lambda tickers, **kw: seen.append(kw["period"]))
    assert seen == ["1d"]


# -- the archive --------------------------------------------------------------------------------------------------------
def test_bars_round_trip_exactly_with_an_ist_index(archive):
    original = frame((2026, 9, 21))
    assert archive.save({"AAA": original}) == 4
    loaded = archive.load()["AAA"]
    pd.testing.assert_frame_equal(loaded, original, check_freq=False)
    assert str(loaded.index.tz) == str(IST)


def test_saving_the_same_session_twice_adds_nothing(archive):
    bars = {"AAA": frame((2026, 9, 21)), "BBB": frame((2026, 9, 21), base=200.0)}
    assert archive.save(bars) == 8 and archive.save(bars) == 0
    assert len(archive.load()["AAA"]) == 4


def test_a_new_symbol_on_a_stored_day_and_a_new_day_for_a_stored_symbol_are_both_added(archive):
    archive.save({"AAA": frame((2026, 9, 21))})
    assert archive.save({"AAA": frame((2026, 9, 22)), "BBB": frame((2026, 9, 21))}) == 8
    assert archive.days() == ["2026-09-21", "2026-09-22"]
    assert len(archive.load()["AAA"]) == 8 and len(archive.load()["BBB"]) == 4


def test_a_frame_spanning_several_sessions_is_split_by_day_and_only_missing_days_are_added(archive):
    archive.save({"AAA": frame((2026, 9, 22))})
    window = pd.concat([frame((2026, 9, 21)), frame((2026, 9, 22)), frame((2026, 9, 23))])
    assert archive.save({"AAA": window}) == 8                        # the 22nd was already stored
    assert archive.days() == ["2026-09-21", "2026-09-22", "2026-09-23"]
    assert len(archive.load()["AAA"]) == 12


def test_days_lists_the_stored_sessions_and_load_honours_the_date_range(archive):
    assert archive.days() == []
    archive.save({"AAA": pd.concat([frame((2026, 9, d)) for d in (21, 22, 23)])})
    assert archive.days() == ["2026-09-21", "2026-09-22", "2026-09-23"]
    assert len(archive.load(start="2026-09-22")["AAA"]) == 8
    assert len(archive.load(end="2026-09-22")["AAA"]) == 8
    assert len(archive.load(start="2026-09-22", end="2026-09-22")["AAA"]) == 4
    assert archive.load(start="2027-01-01") == {}


def test_saving_nothing_is_a_no_op(archive):
    assert archive.save({}) == 0 and archive.days() == []


def test_a_session_is_keyed_by_its_ist_date_not_its_utc_date(archive):
    """09:15 IST is 03:45 UTC the same day; a bar at 00:30 IST belongs to the IST date even though UTC says the day before."""
    late = frame((2026, 9, 22), n=1, start=(0, 30))
    archive.save({"AAA": late})
    assert archive.days() == ["2026-09-22"]
