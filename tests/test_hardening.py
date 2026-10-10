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
from tests.test_llm_and_pipeline import make_pipeline


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
