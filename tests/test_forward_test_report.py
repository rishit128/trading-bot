"""The forward-test report judges recorded trades at today's fee schedule and sets the swing account next to the index."""

import sqlite3

import pandas as pd
import pytest

from scripts import forward_test_report as report
from scripts import research_etf_rotation as rotation
from src.engine.costs import india_delivery_fees, intraday_fees, net_return


def account_db(tmp_path, monkeypatch, trades=(), equity=(), initial_cash=25_000.0):
    """A paper account's database with the tables the report reads, under a temporary project root."""
    with sqlite3.connect(tmp_path / "acct.db") as con:
        con.execute("create table paper_trades (symbol, qty, entry_price, exit_price, opened_at, net_pnl)")
        con.execute("create table intraday_signals (system_version, outcome, day, final_signal)")
        con.execute("create table equity_history (created_at, equity)")
        con.execute("create table paper_account (initial_cash)")
        con.execute("create table decisions (strategy_version, final_action, created_at)")
        con.executemany("insert into paper_trades values (?, ?, ?, ?, ?, ?)", trades)
        con.executemany("insert into equity_history values (?, ?)", equity)
        con.execute("insert into paper_account values (?)", (initial_cash,))
    monkeypatch.setattr(report, "ROOT", tmp_path)
    return tmp_path / "acct.db"


def test_a_trade_booked_at_another_fee_schedule_is_judged_at_todays(tmp_path, monkeypatch, capsys):
    booked_wrong = -999.0  # the booked net is not read at all
    account_db(tmp_path, monkeypatch, trades=[("AAA", 50, 100.0, 101.0, "2026-11-09 04:05:00", booked_wrong)])
    report.report("orb", report.Account("acct.db", "orb-v6", 0.0025, None, "", True, "2026-11-09"))
    expected = net_return(100.0, 101.0, 50, intraday_fees)
    assert f"mean net {expected:+.3%} a trade" in capsys.readouterr().out
    assert expected == pytest.approx(0.0089, abs=0.0002)  # 1% less the 0.106% fees


def test_the_swing_account_is_costed_at_delivery_fees_and_set_next_to_the_index(tmp_path, monkeypatch, capsys):
    days = pd.bdate_range("2026-09-28", periods=30)
    nifty = pd.Series([100.0 + i for i in range(30)], index=days.tz_localize("Asia/Kolkata"))  # +29% over the period
    monkeypatch.setattr(report, "load_index_close", lambda symbol, years: nifty)
    equity = [(f"{d.date()} 04:00:00", 20_000.0) for d in days[:10]]
    equity += [(f"{d.date()} 04:00:00", 25_000.0) for d in days[10:]]  # Rs 5,000 added on the 11th day; no profit, no loss
    db = account_db(tmp_path, monkeypatch, trades=[("AAA", 22, 100.0, 100.0, "2026-10-01 04:05:00", 0.0)], equity=equity)
    report.against_the_index(db)
    out = capsys.readouterr().out
    in_index = 20_000 * 129 / 100 + 5_000 * 129 / 110 - 25_000
    assert "on Rs 25,000 put in: account Rs +0 (+0.0%)" in out and f"Nifty 50 Rs {in_index:+,.0f}" in out
    report.report("swing", report.Account("acct.db", "swing-v6", 0.03, None, "", False, "2026-12-01"))
    flat = net_return(100.0, 100.0, 22, india_delivery_fees)
    assert f"1 trades, mean net {flat:+.3%}" in capsys.readouterr().out and flat == pytest.approx(-0.02, abs=0.0002)


# -- the ETF rotation research script: its pure parts -------------------------------------------------------------------------
def test_a_unit_split_is_taken_out_of_the_price_history():
    days = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.DataFrame({"Open": [1000.0, 1010, 101, 102], "Close": [1000.0, 1010, 101.5, 102]}, index=days)
    frame["High"], frame["Low"] = frame["Close"], frame["Close"]
    assert rotation.adjust_splits(frame)["Close"].tolist() == pytest.approx([100.0, 101.0, 101.5, 102.0])


def test_the_rotation_holds_the_strongest_etfs_and_cash_when_none_beats_cash():
    days = pd.bdate_range("2020-01-01", periods=rotation.LOOKBACK + 2)
    grow = pd.Series(range(len(days)), index=days)
    close = pd.DataFrame({"NIFTYBEES": 100 + 0.1 * grow, "JUNIORBEES": 100 + 0.2 * grow, "GOLDBEES": 100 - 0.05 * grow})
    assert rotation.targets(close, days[-1], 1) == ["JUNIORBEES"]
    assert rotation.targets(close, days[-1], 2) == ["JUNIORBEES", "NIFTYBEES"]
    assert rotation.targets(close[["GOLDBEES"]], days[-1], 2) == [None, None]  # falling: cash
    assert rotation.targets(close, days[10], 1) == [None]  # under a year of history: no ranking yet


def test_the_rotation_simulation_pays_fees_and_never_spends_more_than_it_has():
    days = pd.bdate_range("2020-01-01", periods=60)
    close = pd.DataFrame({s: [100.0] * 60 for s in rotation.ETFS}, index=days)
    curve, fills = rotation.simulate(close, close, lambda day: ["NIFTYBEES"], india_delivery_fees, 0.0005, ["NIFTYBEES"])
    assert fills == 1  # bought once; an unchanged holding is not traded again
    assert rotation.CAPITAL * 0.99 < curve.iloc[0] < rotation.CAPITAL  # the buy's fees and slippage, nothing more
    switched, fills = rotation.simulate(
        close, close, lambda day: ["GOLDBEES"], india_delivery_fees, 0.0005, ["NIFTYBEES"]
    )  # sold at the first month end
    assert fills == 3 and switched.iloc[-1] < curve.iloc[-1] and switched.min() > 0
