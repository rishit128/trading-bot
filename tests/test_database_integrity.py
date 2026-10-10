"""What the database itself guarantees, not just the application: one order per decision, one recorded signal per stock per
session. A new database has the rules from the start, an older one gains them on startup, and one whose rows already break a
rule keeps working with a warning (nothing is ever deleted or rewritten)."""

import logging
import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError

from src.database import IntradaySignalRecord, OrderRecord, make_session_factory
from src.database.migrations import apply_migrations
from src.intraday.engine import IntradayEngine
from tests.intraday_helpers import BREAKOUT, VOLS, make_bars
from tests.test_intraday_engine import at, engine, signals

INDEXES = {"ux_orders_decision", "ux_intraday_signals_day_symbol"}


def order(decision_id=1):
    return OrderRecord(decision_id=decision_id, symbol="AAA", side="BUY", quantity=1, status="filled")


def signal(day="2026-09-21", symbol="AAA"):
    return IntradaySignalRecord(
        day=day, symbol=symbol, bar_time="t", signal_price=1, stop=1, target=1, range_high=1, range_low=1,
        volume_ratio=1, strength=1, outcome="ENTERED", qty=1, system_version="v",
    )  # fmt: skip


def indexes(path):
    with sqlite3.connect(path) as con:
        return {row[1] for row in con.execute("SELECT * FROM sqlite_master WHERE type='index'")} & INDEXES


def test_a_new_database_enforces_both_rules(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'n.db'}")
    assert indexes(tmp_path / "n.db") == INDEXES
    with sessions() as s:
        s.add(order(1))
        s.add(signal())
        s.commit()
        s.add(order(1))
        with pytest.raises(IntegrityError):
            s.commit()
        s.rollback()
        s.add(order(2))  # another decision is fine
        s.add(signal(symbol="BBB"))  # another stock is fine
        s.add(signal(day="2026-09-22"))  # another session is fine
        s.commit()
        s.add(signal())
        with pytest.raises(IntegrityError):
            s.commit()


def drop_indexes(path):
    with sqlite3.connect(path) as con:
        for name in INDEXES:
            con.execute(f"DROP INDEX {name}")


def test_an_older_database_gains_the_rules_when_it_starts(tmp_path):
    path = tmp_path / "old.db"
    make_session_factory(f"sqlite:///{path}")
    drop_indexes(path)
    assert indexes(path) == set()
    make_session_factory(f"sqlite:///{path}")
    assert indexes(path) == INDEXES


def test_the_migration_script_path_creates_them_too_and_a_dry_run_changes_nothing(tmp_path):
    path = tmp_path / "old.db"
    make_session_factory(f"sqlite:///{path}")
    drop_indexes(path)
    apply_migrations(f"sqlite:///{path}", dry_run=True)
    assert indexes(path) == set()
    apply_migrations(f"sqlite:///{path}")
    assert indexes(path) == INDEXES


def test_existing_duplicates_keep_the_database_working_nothing_is_deleted_and_the_other_rule_still_applies(tmp_path, caplog):
    path = tmp_path / "dupes.db"
    make_session_factory(f"sqlite:///{path}")
    drop_indexes(path)
    with sqlite3.connect(path) as con:
        for _ in range(2):  # the same decision recorded twice, as an older version could have done
            con.execute("INSERT INTO orders (created_at, decision_id, symbol, side, quantity, status) "
                        "VALUES ('2026-01-01', 7, 'AAA', 'BUY', 1, 'filled')")  # fmt: skip
    with caplog.at_level(logging.WARNING, logger="src.database.migrations"):
        sessions = make_session_factory(f"sqlite:///{path}")  # must not raise
    assert "ux_orders_decision" in caplog.text and "NOT created" in caplog.text
    assert indexes(path) == {"ux_intraday_signals_day_symbol"}  # the rule the data respects is still enforced
    with sessions() as s:
        assert s.query(OrderRecord).count() == 2  # untouched


def test_a_restart_does_not_record_or_trade_the_same_signal_again(tmp_path, caplog):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS)}
    first, broker, *_ = engine(tmp_path, bars, at(9, 45), live=False)  # dry run: no position or trade to remember it by
    first.step()
    assert [r.outcome for r in signals(broker)] == ["DRY_RUN"]
    restarted = IntradayEngine(
        broker, ["AAA"], first.clock, fetch_bars=first.fetch_bars, live=False, now_fn=first.now_fn, notify=None
    )  # fmt: skip
    with caplog.at_level(logging.ERROR, logger="src.intraday.engine"):
        restarted.step()  # a new process on the same database, the same day
    assert "could not record" not in caplog.text  # the engine itself knew better: the database rule was not even needed
    assert [r.outcome for r in signals(broker)] == ["DRY_RUN"]  # not recorded twice


def test_the_startup_path_never_runs_in_a_way_that_breaks_a_current_database(tmp_path):
    path = f"sqlite:///{tmp_path / 'c.db'}"
    for _ in range(3):  # idempotent: starting again and again changes nothing
        make_session_factory(path)
    assert indexes(tmp_path / "c.db") == INDEXES
