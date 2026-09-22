"""Two OS processes sharing one sqlite file (e.g. the swing bot's /intraday command reading intraday.db while the
intraday engine's own process writes it) must not hit "database is locked" / "attempt to write a readonly database"."""
import multiprocessing
import time

from sqlalchemy import text

from src.database import PaperAccountRecord, make_session_factory


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
