from datetime import date

import pandas as pd

from src.research.delivery import load_delivery, parse_bhavcopy

CSV = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
       "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
       "AAA, EQ, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 60, 60.00\n"
       "BBB, EQ, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 30, 30.00\n"
       "CCC, BE, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, -, -\n")


def test_parse_keeps_only_eq_series_and_reads_delivery_percent():
    s = parse_bhavcopy(CSV)
    assert s.to_dict() == {"AAA": 60.0, "BBB": 30.0}


def test_load_skips_weekends_and_holidays_and_caches(tmp_path):
    calls = []

    def fetch(day):
        calls.append(day)
        return CSV if day == date(2026, 9, 18) else None  # every other day is a "holiday"

    df = load_delivery(0.02, tmp_path, fetch_day=fetch, today=lambda: date(2026, 9, 21))
    assert list(df.index) == [pd.Timestamp("2026-09-18")] and df.loc["2026-09-18", "AAA"] == 60.0
    assert all(d.weekday() < 5 for d in calls)
    calls.clear()
    again = load_delivery(0.02, tmp_path, fetch_day=fetch, today=lambda: date(2026, 9, 21))
    assert date(2026, 9, 18) not in calls and again.equals(df)  # cached day is not fetched again
