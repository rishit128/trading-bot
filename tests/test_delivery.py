from datetime import date

import pandas as pd

from src.data.delivery import load_delivery, parse_bhavcopy

CSV = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
       "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
       "AAA, EQ, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 60, 60.00\n"
       "BBB, EQ, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 30, 30.00\n"
       "CCC, BE, 18-Sep-2026, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, -, -\n")


def test_parse_keeps_only_eq_series_and_reads_delivery_percent():
    s = parse_bhavcopy(CSV)
    assert s.to_dict() == {"AAA": 60.0, "BBB": 30.0}


def test_load_reads_hand_downloaded_files_in_the_window_and_skips_everything_else(tmp_path):
    (tmp_path / "sec_bhavdata_full_18092026.csv").write_text(CSV)
    (tmp_path / "sec_bhavdata_full_17092026.csv").write_text(CSV.replace("60.00", "70.00"))
    (tmp_path / "sec_bhavdata_full_01012020.csv").write_text(CSV)  # outside the window
    (tmp_path / "sec_bhavdata_full_16092026.csv").write_text("not a bhavcopy")  # unreadable: skipped, not guessed
    (tmp_path / "notes.txt").write_text("ignored")
    df = load_delivery(0.05, tmp_path, today=lambda: date(2026, 9, 21))
    assert list(df.index) == [pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")]
    assert df.loc["2026-09-17", "AAA"] == 70.0 and df.loc["2026-09-18", "AAA"] == 60.0


def test_no_folder_or_no_files_gives_an_empty_frame(tmp_path):
    assert load_delivery(1, tmp_path / "missing").empty
    assert load_delivery(1, tmp_path).empty
