from datetime import date

import pandas as pd

from src.intraday import strategy as st

DAY = date(2026, 11, 5)
IST = "Asia/Kolkata"


def bars(prior_close, last_close, opened=None):
    """A stock's closes on 3 and 4 November, then today's first bar at `opened` (None: nothing today)."""
    rows = [("2026-11-03 15:25", prior_close), ("2026-11-04 15:25", last_close)]
    if opened is not None:
        rows.append(("2026-11-05 09:15", opened))
    index = pd.DatetimeIndex([t for t, _ in rows]).tz_localize(IST)
    return pd.DataFrame({"Open": [p for _, p in rows], "Close": [p for _, p in rows]}, index=index)


def scan(gappers, others=20, **extra):
    stocks = {f"G{i}": bars(100, 100, 103) for i in range(gappers)}
    stocks.update({f"F{i}": bars(100, 100, 100) for i in range(others)})
    stocks.update(extra)
    return st.gap_only_refusals(stocks, DAY)


def test_the_limit_is_on_the_number_of_gappers():
    assert scan(10)("G0") is None
    assert "crowded gap day: 11 stocks" in scan(11)("G0")


def test_a_gap_of_exactly_two_percent_counts_and_less_does_not():
    just = {f"J{i}": bars(100, 100, 102) for i in range(11)}
    assert "crowded" in scan(0, **just)("J0")
    under = {f"U{i}": bars(100, 100, 101.9) for i in range(11)}
    assert scan(0, **under)("U0") is None


def test_a_stock_up_over_three_percent_the_session_before_is_refused():
    refusal = scan(3, RAN=bars(100, 103.1, 106), EDGE=bars(100, 102.9, 106))
    assert "already up 3.1%" in refusal("RAN")
    assert refusal("EDGE") is None


def test_a_stock_without_two_earlier_closes_is_not_refused_for_its_rise():
    one = bars(100, 100, 103).iloc[1:]
    assert scan(3, NEW=one)("NEW") is None


def test_no_entries_when_the_crowd_cannot_be_read():
    alone = {f"N{i}": bars(100, 100, 103).iloc[2:] for i in range(30)}
    refusal = st.gap_only_refusals({**alone, "G": bars(100, 100, 103)}, DAY)
    assert "cannot be counted" in refusal("G")


def test_stocks_with_no_bars_today_are_not_counted():
    idle = {f"I{i}": bars(100, 100) for i in range(11)}
    assert scan(2, **idle)("G0") is None
