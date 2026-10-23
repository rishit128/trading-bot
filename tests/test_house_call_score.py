from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.house_call_score import COST, Call, score_call
from src.data.india import IST


def bars(rows, day=(2026, 10, 21), start=(9, 15)):
    first = datetime(*day, *start, tzinfo=IST)
    index = pd.DatetimeIndex([first + timedelta(minutes=5 * i) for i in range(len(rows))])
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def call(side="buy", kind="Intraday", stop=99.0, target=102.0, valid_to=None):
    published = datetime(2026, 10, 21, 9, 14, tzinfo=IST).astimezone(timezone.utc)
    return Call("AAA", kind, side, published, 100.0, 100.1, stop, target, valid_to)


def test_a_buy_call_that_reaches_its_target_is_a_win_net_of_costs():
    out = score_call(call(), bars([(100, 100.5, 99.8, 100.2), (100.2, 102.5, 100.1, 102.0)]))
    assert (
        out.result == "TARGET"
        and out.entry == 100
        and out.gross == pytest.approx(0.02)
        and out.net == pytest.approx(0.02 - COST)
        and out.in_zone
    )


def test_a_bar_touching_both_stop_and_target_counts_as_the_stop():
    out = score_call(call(), bars([(100, 102.5, 98.5, 100.0)]))
    assert out.result == "STOP" and out.exit == 99.0


def test_a_sell_call_profits_when_the_price_falls():
    out = score_call(call("sell", stop=101.0, target=98.0), bars([(100, 100.5, 97.5, 98.0)]))
    assert out.result == "TARGET" and out.gross == pytest.approx(0.02)


def test_an_intraday_call_is_left_at_the_square_off():
    flat = [(100, 100.4, 99.6, 100.0)] * 71  # 09:15 to 15:05; the 15:10 bar ends at the 15:15 square-off
    out = score_call(call(), bars(flat + [(100, 100.4, 99.6, 101.0)] + [(101, 105, 90, 95.0)]))
    assert out.result == "TIME" and out.exit == 101.0 and out.drift == pytest.approx(0.01)


def test_a_call_with_time_left_is_open_and_one_with_no_later_bars_has_no_data():
    assert score_call(call(), bars([(100, 100.4, 99.6, 100.0)] * 5)).result == "OPEN"
    assert score_call(call(), bars([(100, 100.4, 99.6, 100.0)], day=(2026, 10, 20))).result == "NO_DATA"


def test_a_swing_call_runs_to_its_valid_date():
    two_days = bars([(100, 100.4, 99.6, 100.0)] * 75)
    second = bars([(100, 100.4, 99.6, 101.0)] * 75, day=(2026, 10, 22))
    out = score_call(call(kind="Swing Trade", valid_to=date(2026, 10, 22)), pd.concat([two_days, second]))
    assert out.result == "TIME" and out.exit == 101.0
