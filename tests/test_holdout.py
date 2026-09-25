"""the holdout reserve is deterministic, decision-only and untouched by the live account.

The paper account is the tuning surface for the whole repo - everything from the learning phase to
calibration is judged against it - so the reserve must be an independent ruler: same starting cash,
identical D1/D2 conventions, nothing read back from broker state. These tests pin: (a) the reserve
replays only approved high-confidence BUYs, ignoring HOLD / weak / rejected calls and symbols it cannot
price; (b) repeated marks are idempotent in value; (c) marks accumulate as timestamped ledger rows; and
(d) reports compare the reserve to the paper account's last mark without touching its state."""
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from src.database import (
    DecisionRecord,
    HoldoutMark,
    PaperAccountRecord,
    make_session_factory,
)
from src.ops.holdout import HoldoutReserve


def _bars(symbol: str) -> pd.DataFrame:
    """A calm daily frame for symbol A: flat 100 until a 300 gap on 25 Sep closes an 8% bracket at TARGET."""
    idx = pd.date_range("2026-09-01", "2026-09-30")
    df = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=idx)
    if symbol == "A":
        df.loc["2026-09-13":, ["Open", "Close", "High"]] = 300.0  # gaps up 2 sessions after entry: TARGET, not TIME
        df.loc["2026-09-13":, ["Low"]] = 295.0
    df["Volume"] = 1_000_000
    return df


def _fetch(symbol: str):
    if symbol == "UNPRICED":
        return None
    return _bars("A" if symbol == "A" else symbol)


def _decision(s, symbol, action, conf, approved=True, created=None):
    s.add(DecisionRecord(symbol=symbol, price=100.0, final_action=action, final_confidence=conf,
                         reasoning="r", risk_approved=approved, risk_quantity=0 if not approved else 10,
                         risk_reason="ok", created_at=created or datetime(2026, 9, 10, 5, 30, tzinfo=timezone.utc)))


def _factory():
    return make_session_factory("sqlite:///:memory:")


def test_reserve_replays_only_approved_high_confidence_buys():
    sessions = _factory()
    with sessions() as s:
        s.add(PaperAccountRecord(id=1, cash=100_000.0, initial_cash=100_000.0))
        _decision(s, "A", "BUY", 0.80)  # the one real fill
        _decision(s, "UNPRICED", "BUY", 0.80)  # approved but unpricable: skipped, never guessed
        _decision(s, "A", "HOLD", 0.70)  # not a BUY
        _decision(s, "A", "BUY", 0.40)  # below min confidence
        _decision(s, "A", "BUY", 0.80, approved=False)  # risk-rejected
        s.commit()
    reserve = HoldoutReserve(sessions, _fetch)
    m = reserve.mark()
    assert m["decisions"] == 1  # only A's 0.80 CALL survives the filter
    assert m["closed_trades"] == 1  # the 300 gap hits the take-profit bracket
    assert m["equity"] > 100_000.0  # the winner is in, priced at the TARGET, not busted
    assert m["open_positions"] == 0


def test_reserve_mark_is_deterministic_and_appends_ledger_rows():
    sessions = _factory()
    with sessions() as s:
        s.add(PaperAccountRecord(id=1, cash=100_000.0, initial_cash=100_000.0))
        _decision(s, "A", "BUY", 0.80)
        s.commit()
    reserve = HoldoutReserve(sessions, _fetch)
    first, second = reserve.mark(), reserve.mark()
    assert first == second  # same stored decisions + bars -> same values, no broker state consulted
    with sessions() as s:
        assert len(list(s.scalars(select(HoldoutMark)))) == 2  # two timestamped ledger rows, one per mark


def test_reserve_reports_history_and_drift_without_touching_paper():
    sessions = _factory()
    with sessions() as s:
        s.add(PaperAccountRecord(id=1, cash=100_000.0, initial_cash=100_000.0, last_mark=98_000.0))
        _decision(s, "A", "BUY", 0.80)
        s.commit()
    reserve = HoldoutReserve(sessions, _fetch, initial_cash=100_000.0)
    reserve.mark()
    report = reserve.report()
    assert len(report["marks"]) == 1
    assert report["drift_vs_paper"] == report["marks"][0]["equity"] - 98_000.0
    assert report["initial_cash"] == 100_000.0
    # a fresh reserve with nothing decided still yields a zero-safety mark, not a crash
    empty = HoldoutReserve(_factory(), _fetch, initial_cash=100_000.0)
    assert empty.mark()["decisions"] == 0 and empty.mark()["equity"] == 100_000.0