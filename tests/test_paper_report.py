"""The combined P&L report's aggregation: grouping by day/week, totals, and merging two accounts' periods."""
from datetime import datetime, timezone

import pytest

from src.ops.paper_report import PeriodPnl, group_by_day, group_by_week, merge_periods, totals


def trade(closed_at, net_pnl, fees=1.0):
    return {"closed_at": datetime.fromisoformat(closed_at).replace(tzinfo=timezone.utc), "net_pnl": net_pnl, "fees": fees}


def test_group_by_day_sums_each_calendar_day_and_counts_wins():
    trades = [trade("2026-09-29T04:00", 100.0), trade("2026-09-29T09:00", -30.0), trade("2026-09-30T04:00", 50.0)]
    rows = group_by_day(trades)
    assert [r.label for r in rows] == ["2026-09-29", "2026-09-30"]
    assert rows[0].trades == 2 and rows[0].wins == 1 and rows[0].net_pnl == pytest.approx(70.0) and rows[0].fees == pytest.approx(2.0)
    assert rows[1].win_rate == 1.0


def test_a_utc_time_near_midnight_groups_by_its_ist_date():
    # 2026-09-29 19:00 UTC is 2026-09-30 00:30 IST: the next day
    rows = group_by_day([trade("2026-09-29T19:00", 10.0)])
    assert rows[0].label == "2026-09-30"


def test_group_by_week_uses_iso_week_numbers():
    trades = [trade("2026-09-28T04:00", 10.0), trade("2026-10-02T04:00", 20.0), trade("2026-10-05T04:00", 30.0)]
    rows = group_by_week(trades)
    assert [r.label for r in rows] == ["2026-W40", "2026-W41"]
    assert rows[0].net_pnl == pytest.approx(30.0) and rows[1].net_pnl == pytest.approx(30.0)


def test_win_rate_is_none_for_an_empty_period_and_a_losing_trade_is_not_a_win():
    assert PeriodPnl("x", 0, 0, 0.0, 0.0).win_rate is None
    rows = group_by_day([trade("2026-09-29T04:00", 0.0)])
    assert rows[0].wins == 0 and rows[0].win_rate == 0.0


def test_totals_sums_every_trade_regardless_of_date():
    trades = [trade("2026-09-29T04:00", 100.0, fees=2.0), trade("2026-10-05T04:00", -40.0, fees=3.0)]
    t = totals(trades)
    assert (t.label, t.trades, t.wins, t.net_pnl, t.fees) == ("total", 2, 1, pytest.approx(60.0), pytest.approx(5.0))


def test_totals_of_no_trades_is_a_zero_row():
    t = totals([])
    assert (t.trades, t.wins, t.net_pnl, t.fees, t.win_rate) == (0, 0, 0.0, 0.0, None)


def test_merge_periods_sums_shared_labels_and_keeps_labels_only_one_side_has():
    a = group_by_day([trade("2026-09-29T04:00", 100.0, fees=1.0)])
    b = group_by_day([trade("2026-09-29T04:00", -20.0, fees=2.0), trade("2026-09-30T04:00", 5.0, fees=0.5)])
    merged = merge_periods(a, b)
    assert [r.label for r in merged] == ["2026-09-29", "2026-09-30"]
    assert merged[0].trades == 2 and merged[0].net_pnl == pytest.approx(80.0) and merged[0].fees == pytest.approx(3.0)
    assert merged[1].net_pnl == pytest.approx(5.0)


def test_merge_periods_with_either_side_empty_returns_the_other():
    rows = group_by_day([trade("2026-09-29T04:00", 10.0)])
    assert merge_periods(rows, []) == rows and merge_periods([], rows) == rows


def test_group_by_week_uses_the_iso_week_year_not_the_calendar_year_at_the_boundary():
    # 2029-12-31 falls in ISO week 2030-W01, not calendar year 2029
    rows = group_by_week([trade("2029-12-31T04:00", 10.0)])
    assert rows[0].label == "2030-W01"


def test_merge_periods_returns_rows_sorted_by_label_regardless_of_input_order():
    a = group_by_day([trade("2026-10-02T04:00", 5.0)])
    b = group_by_day([trade("2026-09-29T04:00", 10.0)])
    assert [r.label for r in merge_periods(a, b)] == ["2026-09-29", "2026-10-02"]
