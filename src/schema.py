"""The explicit, rootable schema layer between `create_all` and a live database.

`make_session_factory` has always applied migrations idempotently at startup - create_all is a no-op
against existing tables, `_add_missing_columns` adds only *nullable* columns, `_ensure_indexes` is
IF NOT EXISTS. This module makes that explicit and *rooted*: it can answer "is this database up to
date?" without touching it, print exactly what an older database is missing, and apply the same
migration the startup path would. It refuses to invent a NOT NULL column (that is a manual,
schema-breaking change and must be a real migration), so `apply_migrations` never leaves production
in a half-built state.

The migration order is always: create tables (create_all) -> add missing columns (nullable only) ->
add indexes (IF NOT EXISTS). Run it through `scripts/migrate_db.py`, never raw DDL."""
from __future__ import annotations

from typing import Dict, List, Tuple

from sqlalchemy import create_engine, inspect

from src.database import Base, _add_missing_columns, _ensure_indexes


def schema_status(database_url: str) -> Dict[str, Dict[str, List[str]]]:
    """Per existing table: the columns the schema wants but the live database does not have yet.

    Reports *existing* tables only - a brand-new database has no tables and is therefore fully
    buildable by create_all, which is a different (create) path from upgrade."""
    engine = create_engine(database_url)
    status: Dict[str, Dict[str, List[str]]] = {}
    try:
        inspector = inspect(engine)
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            columns = [c["name"] for c in inspector.get_columns(table.name)]
            missing = [c.name for c in table.columns if c.name not in columns]
            status[table.name] = {"existing": sorted(columns), "missing": sorted(missing)}
    finally:
        engine.dispose()
    return status


def _missing(database_url: str) -> List[Tuple[str, List[str]]]:
    return [(table, info["missing"]) for table, info in schema_status(database_url).items()
            if info["missing"]]


def is_current(database_url: str) -> bool:
    """True when every existing table already has every schema column (a new DB counts, too)."""
    return not _missing(database_url)


def apply_migrations(database_url: str, dry_run: bool = False) -> dict:
    """Bring an older database up to date, or report exactly what that would change.

    Dry-run never touches the database; a run applies the same idempotent, nullable-only migration the
    startup path uses and returns what was actually added. A NOT NULL schema addition raises instead of
    being guessed - that is a manual migration and is never applied silently."""
    missing = _missing(database_url)
    if dry_run:
        return {"ok": not missing, "columns_to_add": missing}

    engine = create_engine(database_url)
    try:
        _add_missing_columns(engine)  # nullable-only by contract; raises on NOT NULL additions
        _ensure_indexes(engine)
    finally:
        engine.dispose()
    added = missing
    return {"ok": not _missing(database_url), "columns_added": added}