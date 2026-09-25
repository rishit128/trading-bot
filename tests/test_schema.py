"""P0, prod-DB: schema migrations are explicit, idempotent, and never invent NOT NULL columns.

An older database (a subset of today's `decisions`) must be *reported* out of date before anything
touches it, dry-run must not write, becoming-current-only-via-apply must converge, and a brand-new
database must already count as current (create_all is a create, not an upgrade)."""
import sqlite3

import pytest

from src.schema import apply_migrations, is_current, schema_status


def _old_decisions_db(path):
    """The decisions table as it looked before the P0 feature columns existed."""
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
    assert "universe_size" in missing and "prompt_version" in missing  # the P0 feature columns
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