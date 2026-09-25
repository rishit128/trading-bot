"""P0 broker reconciliation: the paper account's cash must be rebuildable from its recorded fills, every filled order
must land in a position or a trade, and an approved trade decision must have an order record. A ledger that drifts
(cash edited, a fill lost) fails the check instead of silently compounding."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.database import DecisionRecord, OrderRecord, make_session_factory
from src.engine.paper_broker import PaperBroker
from src.reconciliation import reconcile_paper


class Feed:
    """First fills at the per-symbol price; for stops, bars that gap through the stop level."""

    def __init__(self):
        self.prices = {"A": 100.0, "B": 200.0}
        self.after_buy = {}

    def last_price(self, symbol):
        return self.prices.get(symbol, 100.0)

    def bars_since(self, symbol, since):
        return self.after_buy.get(symbol, pd.DataFrame())


class Clock:
    def is_open(self, now=None):
        return True

    def today_ist(self, now=None):
        return "2024-01-03"


def _broker(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'r.db'}")
    feed, clock = Feed(), Clock()
    broker = PaperBroker(sessions, feed, clock,
                         now_fn=lambda: datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc))
    return sessions, broker, feed


def test_a_clean_ledger_reconciles_to_zero_findings(tmp_path):
    sessions, broker, _ = _broker(tmp_path)
    broker.buy_with_bracket("A", 100, 100.5, 0.08, 1.0)
    broker.buy_with_bracket("B", 80, 200.5, 0.08, 1.0)
    broker.sell("A", 100)  # a SIGNAL exit -> a closed trade
    r = reconcile_paper(sessions)
    assert r["ok"] and r["findings"] == []
    assert r["closed_trades"] == 1 and r["positions"] == 1
    assert r["expected_cash"] == pytest.approx(r["cash"], abs=0.01)


def test_the_stop_path_reconciles_too(tmp_path):
    sessions, broker, feed = _broker(tmp_path)
    broker.buy_with_bracket("B", 80, 200.5, 0.08, 1.0)  # stop level 184.46 (8% below the 200.5 signal close)
    feed.after_buy["B"] = pd.DataFrame({"Open": [180.0], "High": [181.0], "Low": [179.0], "Close": [180.5]})
    broker.holdings()  # settlement run: the gap-open below the stop closes the position at 180.0
    r = reconcile_paper(sessions)
    assert r["ok"], r["findings"]
    assert r["closed_trades"] == 1 and r["positions"] == 0
    assert r["expected_cash"] == pytest.approx(r["cash"], abs=0.01)


def test_a_drifted_cash_balance_is_caught(tmp_path):
    sessions, broker, _ = _broker(tmp_path)
    broker.buy_with_bracket("A", 100, 100.5, 0.08, 1.0)
    with sessions() as s:
        from src.database import PaperAccountRecord

        acct = s.get(PaperAccountRecord, 1)
        acct.cash += 50.0
        s.commit()
    r = reconcile_paper(sessions)
    assert not r["ok"]
    assert any("cash does not tie" in f for f in r["findings"])


def test_approved_decisions_and_filled_orders_tie(tmp_path):
    sessions, broker, _ = _broker(tmp_path)
    broker.buy_with_bracket("A", 100, 100.5, 0.08, 1.0)  # needs an approved BUY decision + a filled BUY order
    with sessions() as s:
        d1 = DecisionRecord(symbol="A", price=100.5, final_action="BUY", final_confidence=0.9, reasoning="r",
                            risk_approved=True, risk_quantity=100, risk_reason="ok",
                           created_at=datetime(2024, 1, 3, tzinfo=timezone.utc))
        s.add(d1)
        s.flush()
        s.add(OrderRecord(decision_id=d1.id, symbol="A", side="BUY", quantity=100, status="filled",
                          filled_qty=100, fill_price=100.05, created_at=datetime(2024, 1, 3, tzinfo=timezone.utc)))
        s.commit()
    r = reconcile_paper(sessions)
    assert r["ok"], r["findings"]
    assert (r["positions"], r["closed_trades"]) == (1, 0)


def test_a_filled_order_without_any_fill_is_flagged(tmp_path):
    sessions, broker, feed = _broker(tmp_path)
    broker.buy_with_bracket("A", 100, 100.5, 0.08, 1.0)
    with sessions() as s:
        from src.database import PaperAccountRecord, PaperPositionRecord

        s.get(PaperPositionRecord, "A").qty = 0  # the order filled but the fill was then lost from the position
        d = DecisionRecord(symbol="A", price=100.5, final_action="BUY", final_confidence=0.9, reasoning="r",
                           risk_approved=True, risk_quantity=100, risk_reason="ok",
                           created_at=datetime(2024, 1, 3, tzinfo=timezone.utc))
        s.add(d)
        s.flush()
        s.add(OrderRecord(decision_id=d.id, symbol="A", side="BUY", quantity=100, status="filled",
                          filled_qty=100, fill_price=100.05, created_at=datetime(2024, 1, 3, tzinfo=timezone.utc)))
        s.commit()
        acct = s.get(PaperAccountRecord, 1)
        acct.cash += 100 * 100.05  # and the money came back, hiding it from the simple cash check
        s.commit()
    r = reconcile_paper(sessions)
    assert not r["ok"]
    assert any("A" in f and "positions+trades" in f for f in r["findings"])