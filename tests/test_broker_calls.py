"""The broker-calls account: what it buys, what it refuses, and how it leaves."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from src.data.house_calls import HouseCall
from src.data.india import IST
from src.database import IntradaySignalRecord, make_session_factory
from src.engine.paper_broker import PaperBroker
from src.intraday.calls import CALLS_CASH, CallsEngine
from src.monitoring.telegram import HELP, handle_command
from tests.test_paper_broker import FakeClock, FakeFeed

DAY = "2026-11-05"
NOW = datetime(2026, 11, 5, 10, 0, tzinfo=IST)


def bars(close=100.0):
    index = pd.date_range(f"{DAY} 09:15", f"{DAY} 09:50", freq="5min", tz=IST)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000.0}, index=index)


def call(symbol="AAA", side="buy", kind="Intraday", is_open=True, published=None, stop=99.0, target=102.0, comment=""):
    if side == "sell":
        stop, target = target, stop
    return HouseCall(
        symbol,
        kind,
        side,
        100.0,
        100.5,
        stop,
        target,
        published or datetime(2026, 11, 5, 9, 40, tzinfo=IST),
        None,
        is_open,
        comment,
    )


def setup(tmp_path, calls, data=None, cash=CALLS_CASH, **kw):
    data = {"AAA": bars()} if data is None else data
    feed, clock = FakeFeed({s: float(d["Close"].iloc[-1]) for s, d in data.items()}), FakeClock()
    clock.day = DAY
    now = [NOW.astimezone(timezone.utc)]
    broker = PaperBroker(
        make_session_factory(f"sqlite:///{tmp_path / 'c.db'}"),
        feed,
        clock,
        initial_cash=cash,
        fees=lambda s, v: 0.0,
        slippage=0.0,
        now_fn=lambda: now[0],
    )
    messages = []
    box = {"calls": calls}
    eng = CallsEngine(
        broker,
        [],
        clock,
        fetch_bars=lambda symbols: {s: data[s] for s in symbols if s in data},
        notify=messages.append,
        live=True,
        now_fn=lambda: now[0],
        house_calls_fn=lambda: box["calls"],
        **kw,
    )
    return eng, broker, box, messages


def signals(broker):
    with broker.sessions() as s:
        return {r.symbol: r for r in s.query(IntradaySignalRecord).all()}


def test_an_open_intraday_buy_call_is_bought_at_the_calls_stop_and_target(tmp_path):
    eng, broker, _, messages = setup(tmp_path, [call()])
    assert "entered 1" in eng.step()
    held = broker.holdings()[0]
    assert held["symbol"] == "AAA" and held["qty"] == 93 and held["stop"] == 99.0  # Rs 28,000 / 3 at Rs 100
    row = signals(broker)["AAA"]
    assert (
        row.outcome == "ENTERED" and row.system_version == "calls-v3" and row.target == 102.0 and row.house_call == "Intraday buy"
    )
    assert "broker's Intraday call" in messages[0] and "breakout" not in messages[0]


def test_sell_calls_other_kinds_earlier_days_and_closed_calls_are_not_bought(tmp_path):
    calls = [
        call(side="sell"),
        call("BBB", kind="Swing Trade"),
        call("CCC", published=datetime(2026, 11, 4, 10, 0, tzinfo=IST)),
        call("DDD", is_open=False, comment="TARGET_HIT"),
    ]
    data = {s: bars() for s in ("AAA", "BBB", "CCC", "DDD")}
    eng, broker, _, _ = setup(tmp_path, calls, data)
    assert eng.step() == "no new broker calls"
    assert not broker.holdings() and not signals(broker)


def test_a_price_already_past_the_target_or_stop_is_refused_and_recorded(tmp_path):
    eng, broker, _, _ = setup(tmp_path, [call(), call("BBB")], {"AAA": bars(103.0), "BBB": bars(98.0)})
    eng.step()
    rows = signals(broker)
    assert not broker.holdings()
    assert rows["AAA"].outcome == "SKIPPED" and "outside the call's stop" in rows["AAA"].reason
    assert rows["BBB"].outcome == "SKIPPED"


def test_calls_past_the_position_limit_are_recorded(tmp_path):
    eng, broker, _, _ = setup(tmp_path, [call(), call("BBB")], {"AAA": bars(), "BBB": bars()}, max_positions=1)
    eng.step()
    assert [h["symbol"] for h in broker.holdings()] == ["AAA"]
    assert "position limit" in signals(broker)["BBB"].reason


def test_a_stock_with_no_price_yet_is_tried_again(tmp_path):
    data = {}
    eng, broker, _, _ = setup(tmp_path, [call()], data)
    eng.broker.feed.prices["AAA"] = 100.0
    eng.step()
    assert not signals(broker)
    data["AAA"] = bars()
    eng.step()
    assert broker.holdings()


def test_it_sells_when_the_broker_closes_the_call(tmp_path):
    eng, broker, box, messages = setup(tmp_path, [call()])
    eng.step()
    box["calls"] = [call(is_open=False, comment="BOOK PROFIT @101")]
    eng.step()
    assert not broker.holdings()
    assert any("BROKER EXIT AAA (BOOK PROFIT @101)" in m for m in messages)
    box["calls"] = [call()]  # one attempt per stock per day
    eng.step()
    assert not broker.holdings()


def test_a_stale_read_sells_nothing_and_alerts_once(tmp_path):
    eng, broker, box, messages = setup(tmp_path, [call()])
    eng.step()
    box["calls"] = []
    eng.step()
    eng.step()
    assert broker.holdings()
    assert sum("/research_start" in m for m in messages) == 1


def test_the_telegram_view_names_the_account(tmp_path):
    _, broker, _, _ = setup(tmp_path, [])
    view = handle_command("/calls", None, None, SimpleNamespace(currency="Rs "), calls=broker)
    assert "BROKER-CALLS paper account" in view and "30,000" in view and "/calls - broker-calls paper account" in HELP


def test_the_account_holds_three_calls_sharing_rs_28000(tmp_path):
    names = ["AAA", "BBB", "CCC", "DDD"]
    eng, broker, _, _ = setup(tmp_path, [call(n) for n in names], {n: bars() for n in names})
    assert "entered 3" in eng.step()
    assert [h["qty"] for h in broker.holdings()] == [93, 93, 93]
    assert broker.portfolio().cash >= 2_000  # the rest stays as cash
    assert "position limit reached (3 open)" in signals(broker)["DDD"].reason


def test_the_risk_rule_still_caps_a_call_with_a_wide_stop(tmp_path):
    eng, broker, _, _ = setup(tmp_path, [call(stop=95.0)])
    eng.step()
    assert broker.holdings()[0]["qty"] == 30  # 0.5% of Rs 30,000 = Rs 150 at risk over a Rs 5 stop distance


def test_a_call_is_not_followed_in_a_weak_market_and_is_when_the_market_cannot_be_measured(tmp_path):
    eng, broker, *_ = setup(tmp_path, [call()], breadth_fn=lambda: 0.39)
    assert "entered 0" in eng.step() and not broker.holdings()
    assert "weak market: 39% of stocks are up" in signals(broker)["AAA"].reason
    for i, breadth in enumerate((0.40, None)):
        (tmp_path / str(i)).mkdir()
        eng, broker, *_ = setup(tmp_path / str(i), [call()], breadth_fn=lambda b=breadth: b)
        assert "entered 1" in eng.step()
