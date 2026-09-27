"""The sqlite database layer's operational guarantees: WAL mode and safe concurrent access from two OS processes (e.g.
the swing bot's /intraday command reading intraday.db while the intraday engine's own process writes it), and that
schema migrations are explicit, idempotent, and never invent NOT NULL columns.

An older database (a subset of today's `decisions`) must be *reported* out of date before anything touches it, dry-run
must not write, becoming-current-only-via-apply must converge, and a brand-new database must already count as current
(create_all is a create, not an upgrade)."""
import multiprocessing
import sqlite3
import time

import pytest
from sqlalchemy import text

from src.database import PaperAccountRecord, make_session_factory
from src.database.migrations import apply_migrations, is_current, schema_status


def test_new_databases_use_wal_journal_mode(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'w.db'}")
    with sessions() as s:
        assert s.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"


def _writer(path, stop_after):
    sessions = make_session_factory(f"sqlite:///{path}")
    end = time.time() + stop_after
    while time.time() < end:
        with sessions() as s:
            acct = s.get(PaperAccountRecord, 1) or PaperAccountRecord(id=1, cash=0.0, initial_cash=0.0)
            acct.cash += 1.0
            s.add(acct)
            s.commit()


def _reader(path, stop_after, errors):
    sessions = make_session_factory(f"sqlite:///{path}")
    end = time.time() + stop_after
    try:
        while time.time() < end:
            with sessions() as s:
                s.get(PaperAccountRecord, 1)
    except Exception as e:  # pragma: no cover - only hit on a real regression
        errors.put(f"{type(e).__name__}: {e}")


def test_a_concurrent_reader_process_is_never_blocked_or_rejected_by_the_writer_process(tmp_path):
    path = tmp_path / "shared.db"
    make_session_factory(f"sqlite:///{path}")  # create it (and set WAL) before either process opens it
    errors = multiprocessing.Queue()
    duration = 1.5
    w = multiprocessing.Process(target=_writer, args=(path, duration))
    r = multiprocessing.Process(target=_reader, args=(path, duration, errors))
    w.start(); r.start()
    w.join(timeout=duration + 5); r.join(timeout=duration + 5)
    assert not w.is_alive() and not r.is_alive()
    assert errors.empty(), errors.get()


# -- schema migrations -------------------------------------------------------------------------------------------
def _old_decisions_db(path):
    """The decisions table as it looked before the feature columns existed."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY,
            created_at DATETIME, symbol VARCHAR(16), price FLOAT,
            final_action VARCHAR(8), final_confidence FLOAT,
            reasoning TEXT, risk_approved BOOLEAN, risk_quantity INTEGER, risk_reason TEXT);
        """)
    conn.commit()
    conn.close()


@pytest.fixture
def old_db(tmp_path):
    path = tmp_path / "legacy.db"
    _old_decisions_db(path)
    return f"sqlite:///{path}"


def test_old_database_is_reported_out_of_date_without_any_change(old_db):
    status = schema_status(old_db)
    assert "decisions" in status
    missing = status["decisions"]["missing"]
    assert "universe_size" in missing and "prompt_version" in missing  # the feature columns
    assert is_current(old_db) is False


def test_dry_run_reports_and_writes_nothing(old_db):
    before = schema_status(old_db)
    report = apply_migrations(old_db, dry_run=True)
    assert report["ok"] is False and report["columns_to_add"]
    assert schema_status(old_db) == before  # dry run really did not touch the file


def test_apply_brings_legacy_database_to_current(old_db):
    applied = apply_migrations(old_db, dry_run=False)
    assert len(applied["columns_added"]) == 1 and "universe_size" in applied["columns_added"][0][1]
    assert is_current(old_db) is True
    assert apply_migrations(old_db, dry_run=False)["ok"] is True  # a second apply is a no-op (idempotent)


def test_brand_new_database_counts_as_current():
    url = "sqlite:///:memory:"
    assert is_current(url) is True  # no tables -> nothing missing; create_all builds it fresh, not via upgrade
    report = apply_migrations(url, dry_run=True)
    assert report["ok"] is True
    assert schema_status(url) == {}
