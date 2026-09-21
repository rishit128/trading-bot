import json
import sqlite3
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.indicators import StaleDataError, build_snapshot
from src.data.market_data import fetch_snapshot
from src.data.universe import ScreenConfig, screen_bars
from src.database import DecisionRecord, OrderRecord, make_session_factory
from src.engine.paper_broker import Fill
from src.engine.risk_engine import Portfolio
from tests.test_llm_and_pipeline import FakeBroker, make_pipeline


# ---------------------------------------------------------------- pipeline: protection sweep
class ProtectBroker(FakeBroker):
    def __init__(self, *lists, protect_fails=False):
        super().__init__(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0))
        self.lists, self.checks, self.protected, self.protect_fails = list(lists), 0, [], protect_fails

    def unprotected_positions(self):
        result = self.lists[min(self.checks, len(self.lists) - 1)]
        self.checks += 1
        if isinstance(result, Exception):
            raise result
        return result

    def protect(self, symbol, qty, stop):
        if self.protect_fails:
            raise RuntimeError("rejected")
        self.protected.append((symbol, qty, round(stop, 2)))


def sweep_pipeline(tmp_path, broker, dry_run=False, **settings_kw):
    pipe, _, _ = make_pipeline(tmp_path, dry_run=dry_run, broker=broker, technical_action="HOLD")
    pipe.settings = pipe.settings.__class__(**{**pipe.settings.__dict__, **settings_kw})
    messages, naps = [], []
    pipe._notify, pipe._sleep = messages.append, naps.append
    return pipe, messages, naps


def test_naked_position_is_alerted_once_and_protected_at_the_configured_stop(tmp_path):
    broker = ProtectBroker([("AAPL", 50, 100.0)])
    pipe, messages, naps = sweep_pipeline(tmp_path, broker)
    pipe.run_once()
    pipe.run_once()
    assert broker.protected == [("AAPL", 50, 92.0), ("AAPL", 50, 92.0)]  # 8% below the entry, retried each cycle
    assert sum("UNPROTECTED POSITION" in m for m in messages) == 1        # but only announced once
    assert naps and naps[0] == 3.0


def test_a_position_that_gets_its_stop_during_the_grace_pause_is_left_alone(tmp_path):
    broker = ProtectBroker([("AAPL", 50, 100.0)], [])  # naked on the first look, protected on the second
    pipe, messages, _ = sweep_pipeline(tmp_path, broker)
    pipe.run_once()
    assert broker.protected == [] and not any("UNPROTECTED" in m for m in messages)


def test_dry_run_alerts_but_never_places_orders(tmp_path):
    broker = ProtectBroker([("AAPL", 50, 100.0)])
    pipe, messages, naps = sweep_pipeline(tmp_path, broker, dry_run=True)
    pipe.run_once()
    assert broker.protected == [] and naps == [] and any("UNPROTECTED POSITION" in m for m in messages)


def test_auto_protect_can_be_switched_off_leaving_only_the_alert(tmp_path):
    broker = ProtectBroker([("AAPL", 50, 100.0)])
    pipe, messages, _ = sweep_pipeline(tmp_path, broker, auto_protect=False)
    pipe.run_once()
    assert broker.protected == [] and any("UNPROTECTED POSITION" in m for m in messages)


def test_failure_to_protect_is_reported_loudly(tmp_path):
    pipe, messages, _ = sweep_pipeline(tmp_path, ProtectBroker([("AAPL", 50, 100.0)], protect_fails=True))
    pipe.run_once()
    assert any("COULD NOT PROTECT AAPL" in m and "manually" in m for m in messages)


def test_a_broken_protection_check_alerts_but_does_not_stop_trading(tmp_path):
    pipe, messages, _ = sweep_pipeline(tmp_path, ProtectBroker(ConnectionError("broker down")))
    assert len(pipe.run_once()) == 1 and any("PROTECTION CHECK FAILED" in m for m in messages)


def test_alert_fires_again_if_a_repaired_position_loses_its_stop_later(tmp_path):
    broker = ProtectBroker([("AAPL", 50, 100.0)], [("AAPL", 50, 100.0)], [], [("AAPL", 50, 100.0)], [("AAPL", 50, 100.0)])
    pipe, messages, _ = sweep_pipeline(tmp_path, broker)
    pipe.run_once()  # naked (checks 0,1)
    pipe.run_once()  # protected (check 2)
    pipe.run_once()  # naked again (checks 3,4)
    assert sum("UNPROTECTED POSITION" in m for m in messages) == 2


def test_a_failed_sell_triggers_an_immediate_protection_repair(tmp_path):
    broker = ProtectBroker([], [("AAPL", 50, 100.0)])
    broker._portfolio = Portfolio(95_000.0, 100_000.0, {"AAPL": 5_000.0}, {"AAPL": 50}, 100_000.0)

    def failing_sell(symbol, qty):
        raise RuntimeError("sell failed after cancelling its stop/target orders; the position may be UNPROTECTED")

    broker.sell = failing_sell
    pipe, _, _ = make_pipeline(tmp_path, technical_action="SELL", dry_run=False, broker=broker)
    messages = []
    pipe._notify, pipe._sleep = messages.append, lambda s: None
    [result] = pipe.run_once()
    assert result.order_status == "FAILED" and broker.protected == [("AAPL", 50, 92.0)]
    assert any("Protective stop placed" in m for m in messages)


# ---------------------------------------------------------------- pipeline: recording what the broker filled
class ConfirmingBroker(FakeBroker):
    def __init__(self, filled, price=100.5):
        super().__init__(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0))
        self.filled, self.price = filled, price

    def confirm(self, fill, qty):
        return Fill(fill.broker_order_id, "filled" if self.filled == qty else "partially_filled", self.filled, self.price)


def test_actual_fill_quantity_and_price_are_stored_and_a_full_fill_is_quiet(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path, dry_run=False, broker=ConfirmingBroker(filled=50))
    messages = []
    pipe._notify = messages.append
    pipe.run_once()
    with sessions() as s:
        row = s.scalar(select(OrderRecord))
        assert (row.status, row.filled_qty, row.fill_price) == ("filled", 50, 100.5)
    assert not any("PARTIAL" in m for m in messages)


def test_a_partial_fill_is_reported_with_both_quantities(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path, dry_run=False, broker=ConfirmingBroker(filled=20))
    messages = []
    pipe._notify = messages.append
    pipe.run_once()
    assert any("PARTIAL/UNFILLED BUY AAPL: ordered 50, broker filled 20" in m for m in messages)
    with sessions() as s:
        assert s.scalar(select(OrderRecord.filled_qty)) == 20


def test_decision_inputs_are_stored_for_replay(tmp_path):
    pipe, _, sessions = make_pipeline(tmp_path)
    pipe.run_once()
    with sessions() as s:
        snap = json.loads(s.scalar(select(DecisionRecord.snapshot_json)))
    assert snap["symbol"] == "AAPL" and snap["price"] == 100.0 and snap["ma200"] == 95.0 and "bar_date" in snap


# ---------------------------------------------------------------- stale data
def bars(n=260, end="2026-09-18", volume=1000.0):
    idx = pd.date_range(end=end, periods=n, freq="B")
    return pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(n)], "Volume": volume}, index=idx)


def test_snapshot_records_the_date_of_its_last_bar():
    assert build_snapshot("X", bars()).bar_date == "2026-09-18"
    assert build_snapshot("X", bars().reset_index(drop=True)).bar_date is None  # no dates to record


def test_a_stale_last_bar_is_refused():
    with pytest.raises(StaleDataError, match="suspended"):
        fetch_snapshot("X", download=lambda *a, **k: bars(end="2026-08-20"), today=lambda: date(2026, 9, 21))


def test_a_long_weekend_gap_is_not_stale():
    fetch_snapshot("X", download=lambda *a, **k: bars(end="2026-09-16"), today=lambda: date(2026, 9, 21))


def test_staleness_is_measured_against_the_expected_completed_session():
    fetch_snapshot("X", as_of=lambda: date(2026, 9, 18), download=lambda *a, **k: bars(end="2026-09-18"))
    with pytest.raises(StaleDataError):
        fetch_snapshot("X", as_of=lambda: date(2026, 9, 18), download=lambda *a, **k: bars(end="2026-08-28"))


def test_zero_volume_on_the_last_bar_is_refused():
    with pytest.raises(StaleDataError, match="zero volume"):
        fetch_snapshot("X", download=lambda *a, **k: bars(volume=0.0), today=lambda: date(2026, 9, 21))


def test_stale_data_is_an_ordinary_value_error_so_existing_handlers_catch_it():
    assert issubclass(StaleDataError, ValueError)


def test_yahoo_downloads_carry_a_timeout():
    seen = {}

    def download(*a, **k):
        seen.update(k)
        return bars()

    fetch_snapshot("X", download=download, today=lambda: date(2026, 9, 21), timeout=12.0)
    assert seen["timeout"] == 12.0


def test_screener_skips_stocks_whose_data_stops_early():
    def frame(symbol, end):
        # up 1% / down 0.6% alternately: a healthy uptrend with RSI ~62, so the ONLY thing that can exclude it is its dates
        prices = 50 * pd.Series([1.01 if i % 2 == 0 else 0.994 for i in range(260)]).cumprod().to_numpy()
        idx = pd.MultiIndex.from_product([[symbol], pd.date_range(end=end, periods=260, freq="B")], names=["symbol", "timestamp"])
        return pd.DataFrame({"close": prices, "volume": 5e6}, index=idx)

    cfg = ScreenConfig(min_price=1.0, min_traded_value=1.0, max_daily_volatility=0.5)
    both = pd.concat([frame("LIVE", "2026-09-18"), frame("HALTED", "2026-08-01")])
    assert {c.symbol for c in screen_bars(both, cfg)} == {"LIVE"}
    assert {c.symbol for c in screen_bars(frame("HALTED", "2026-08-01"), cfg)} == {"HALTED"}  # alone it IS the newest, so it passes


# ---------------------------------------------------------------- database
def test_older_orders_table_gains_the_fill_columns(tmp_path):
    path = tmp_path / "old_orders.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE orders (id INTEGER PRIMARY KEY, created_at DATETIME, decision_id INTEGER, symbol VARCHAR(16),
        side VARCHAR(8), quantity INTEGER, status VARCHAR(24), broker_order_id VARCHAR(64), error TEXT)""")
    con.commit()
    con.close()
    sessions = make_session_factory(f"sqlite:///{path}")
    with sessions() as s:
        s.add(OrderRecord(decision_id=1, symbol="X", side="BUY", quantity=1, status="filled", filled_qty=1, fill_price=9.5))
        s.commit()
        assert s.scalar(select(OrderRecord.fill_price)) == 9.5
