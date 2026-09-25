"""Outcome labelling: the arithmetic (pure), and the database job around it (idempotent, matured-only, no lookahead)."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.database import DecisionRecord, OutcomeRecord, make_session_factory
from src.learning.outcomes import HORIZONS, backfill_bar_dates, forward_outcome, label_outcomes

DAYS = pd.bdate_range("2026-01-05", periods=40)  # 2026-01-05 is a Monday


def bars(opens=None, closes=None, lows=None, highs=None):
    n = len(DAYS)
    o = np.full(n, 100.0) if opens is None else np.asarray(opens, dtype=float)
    c = np.full(n, 100.0) if closes is None else np.asarray(closes, dtype=float)
    lo = np.minimum(o, c) - 1 if lows is None else np.asarray(lows, dtype=float)
    hi = np.maximum(o, c) + 1 if highs is None else np.asarray(highs, dtype=float)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c}, index=DAYS)


def day(i):
    return str(DAYS[i].date())


# ---------------------------------------------------------------- the arithmetic
def test_entry_is_the_next_open_and_exit_is_the_close_horizon_sessions_later():
    o, c = np.full(40, 100.0), np.full(40, 100.0)
    c[10] = 50.0    # the analysed session closed at 50: irrelevant, the bot cannot trade that close
    o[11] = 100.0   # the next open is the entry
    c[15] = 110.0   # 5 sessions after the analysed one
    out = forward_outcome("X", bars(o, c), day(10), 5, 0.15)
    assert out.entry_price == 100.0 and out.exit_price == 110.0 and out.gross_return == pytest.approx(0.10)
    assert out.entry_date == day(11) and out.exit_date == day(15) and out.horizon == 5 and out.symbol == "X"


def test_the_worst_return_and_stop_touch_come_from_the_lows_during_the_hold_only():
    lows = np.full(40, 99.0)
    lows[13] = 80.0    # inside the window (sessions 11..15): a 20% dip
    lows[30] = 1.0     # long after: must not count
    lows[9] = 1.0      # before the entry: must not count
    lows[10] = 1.0     # the analysed session itself: the bot cannot trade it, so it must not count either
    out = forward_outcome("X", bars(lows=lows), day(10), 5, 0.15)
    assert out.worst_return == pytest.approx(-0.20) and out.hit_stop is True
    tight = forward_outcome("X", bars(lows=lows), day(10), 5, 0.25)
    assert tight.hit_stop is False and tight.worst_return == pytest.approx(-0.20)


def test_a_session_that_has_not_matured_or_is_not_in_the_data_gives_no_label():
    b = bars()
    assert forward_outcome("X", b, day(36), 5, 0.15) is None      # only 3 sessions after: not matured
    assert forward_outcome("X", b, day(34), 5, 0.15) is not None  # exactly 5 sessions after: matured
    assert forward_outcome("X", b, day(35), 5, 0.15) is None      # the exit session would be one past the last bar
    assert forward_outcome("X", b, day(39), 1, 0.15) is None      # the newest bar has no next session to enter on
    assert forward_outcome("X", b, "2026-01-10", 5, 0.15) is None  # a Saturday: not a session, never guessed
    assert forward_outcome("X", b, "2025-12-01", 5, 0.15) is None  # before the data starts


def test_the_label_never_depends_on_bars_after_the_exit():
    base = bars()
    later = base.copy()
    later.iloc[20:, :] = 1e6  # wildly different data AFTER the 5-session window ending at index 15
    a, b = forward_outcome("X", base, day(10), 5, 0.15), forward_outcome("X", later, day(10), 5, 0.15)
    assert a == b


def test_a_timezone_aware_index_works_and_a_bad_horizon_is_rejected():
    b = bars()
    b.index = b.index.tz_localize("Asia/Kolkata")
    assert forward_outcome("X", b, day(10), 5, 0.15) is not None
    with pytest.raises(ValueError, match="horizon"):
        forward_outcome("X", bars(), day(10), 0, 0.15)


# ---------------------------------------------------------------- the database job
def decision(symbol, bar_date, action="BUY", conf=0.7, with_column=True):
    return DecisionRecord(symbol=symbol, price=100.0, final_action=action, final_confidence=conf, reasoning="r",
                          risk_approved=False, risk_quantity=0, risk_reason="x", bar_date=bar_date if with_column else None,
                          snapshot_json=json.dumps({"symbol": symbol, "price": 100.0, "ma50": 98.0, "ma200": 95.0, "rsi": 60.0,
                                                    "volume": 1000, "bar_date": bar_date}))


def seeded(tmp_path, *rows):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'o.db'}")
    with sessions() as s:
        s.add_all(rows)
        s.commit()
    return sessions


def test_every_analysed_session_is_labelled_once_at_each_matured_horizon(tmp_path):
    sessions = seeded(tmp_path, decision("AAA", day(0)), decision("AAA", day(0), action="HOLD"),  # the same session twice
                      decision("BBB", day(5)))
    fetched = []

    def fetch(symbol):
        fetched.append(symbol)
        return bars()

    summary = label_outcomes(sessions, fetch, horizons=(5, 20), today=date(2026, 6, 1))
    assert summary.labelled == 4 and summary.not_mature == 0 and sorted(fetched) == ["AAA", "BBB"]  # 2 sessions x 2 horizons
    with sessions() as s:
        rows = list(s.scalars(select(OutcomeRecord)))
    assert {(r.symbol, r.bar_date, r.horizon) for r in rows} == {("AAA", day(0), 5), ("AAA", day(0), 20),
                                                                 ("BBB", day(5), 5), ("BBB", day(5), 20)}
    again = label_outcomes(sessions, fetch, horizons=(5, 20), today=date(2026, 6, 1))
    assert again.labelled == 0 and again.already_done == 4  # idempotent: nothing is labelled twice


def test_only_matured_sessions_are_labelled_and_immature_symbols_are_not_even_fetched(tmp_path):
    sessions = seeded(tmp_path, decision("NEW", day(38)))
    fetched = []
    summary = label_outcomes(sessions, lambda s: fetched.append(s) or bars(), today=date(2026, 3, 2))
    assert summary.labelled == 0 and summary.not_mature == len(HORIZONS) and fetched == []


def test_a_symbol_with_no_price_data_is_reported_not_guessed(tmp_path):
    sessions = seeded(tmp_path, decision("GONE", day(0)), decision("OK", day(0)))
    summary = label_outcomes(sessions, lambda s: None if s == "GONE" else bars(), horizons=(5,), today=date(2026, 3, 30))
    assert summary.labelled == 1 and summary.symbols_without_bars == ("GONE",) and summary.no_data == 1
    broken = label_outcomes(sessions, lambda s: (_ for _ in ()).throw(ConnectionError("down")), horizons=(5,), today=date(2026, 3, 30))
    assert broken.labelled == 0 and "GONE" in broken.symbols_without_bars  # a failing fetch is survived, and retried next run


def test_decisions_from_before_the_bar_date_column_are_backfilled_from_their_stored_snapshot(tmp_path):
    sessions = seeded(tmp_path, decision("OLD", day(0), with_column=False))
    assert backfill_bar_dates(sessions) == 1
    with sessions() as s:
        assert s.scalar(select(DecisionRecord.bar_date)) == day(0)
    assert backfill_bar_dates(sessions) == 0  # nothing left to fill
    summary = label_outcomes(sessions, lambda s: bars(), horizons=(5,), today=date(2026, 3, 30))
    assert summary.labelled == 1


def test_a_full_pipeline_cycle_records_the_bar_date_that_outcomes_attach_to(tmp_path):
    from tests.characterization import build, observe

    pipeline, broker, sessions, messages = build(tmp_path)
    observe(pipeline, broker, sessions, messages)
    with sessions() as s:
        dates = {d.bar_date for d in s.scalars(select(DecisionRecord))}
    assert dates == {"2026-09-24"}  # the completed session the scripted snapshots were built from
