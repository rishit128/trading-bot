"""Opening a database: the engine, the tables, the migration of an older file, and the session factory."""
from typing import Any, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.database.migrations import add_missing_columns, ensure_indexes
from src.database.models import Base


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    """Create the engine and tables, add any missing columns to an older database, and return a session factory."""
    kwargs: Dict[str, Any] = {"pool_pre_ping": True}  # a dropped connection is replaced instead of failing the cycle
    is_sqlite = database_url.startswith("sqlite")
    if is_sqlite:
        kwargs["connect_args"] = {"timeout": 30}  # wait for a lock instead of failing with "database is locked"
    engine = create_engine(database_url, **kwargs)
    if is_sqlite and ":memory:" not in database_url:
        # WAL: a reader (e.g. the swing bot's /intraday Telegram command) never blocks or is blocked by the writer
        # (the intraday engine's own process saving every 5 minutes) sharing the same file. The default journal mode
        # occasionally surfaced "attempt to write a readonly database" under that two-process access pattern.
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
    Base.metadata.create_all(engine)
    add_missing_columns(engine)
    ensure_indexes(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
