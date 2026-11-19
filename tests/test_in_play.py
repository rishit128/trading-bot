"""orb-v8: the whole NSE market's liquid movers join the breakout scan, and the market checks read every stock."""

from datetime import datetime, timezone

import pytest

from src.data import in_play as ip
from src.data.angel import AngelStream
from src.database import make_session_factory
from src.intraday import strategy as st
from tests.intraday_helpers import BREAKOUT, VOLS, make_bars
from tests.test_intraday_engine import at, engine
from tests.test_orb_leaders import DAY, bars, market


def quote(token, ltp, close=100.0, volume=1_000_000, upper=None):
    return {"symbolToken": token, "ltp": ltp, "close": close, "tradeVolume": volume, "upperCircuit": upper}


def test_quotes_are_read_and_a_stock_without_a_trade_or_a_previous_close_is_dropped():
    got = ip.parse_quotes(
        [quote("1", 103), quote("2", 0), quote("3", 101, close=0), quote("4", 101, volume=0), quote("9", 101)],
        {"1": "AAA", "2": "BBB", "3": "CCC", "4": "DDD"},
    )
    assert got == [ip.Quote("AAA", 103.0, 100.0, 103_000_000.0, None)]  # token 9 is not an NSE equity we know


def test_a_mover_is_liquid_up_two_percent_priced_above_fifty_and_clear_of_its_upper_circuit():
    q = ip.Quote
    quotes = [
        q("SMALL", 103, 100, 3e7, None),  # Rs 3 crore traded: too thin
        q("FLAT", 101, 100, 9e8, None),  # +1%
        q("PENNY", 41, 40, 9e8, 50),  # under Rs 50
        q("PINNED", 109.5, 100, 9e8, 110),  # within 1% of the upper circuit
        q("LIST", 105, 100, 9e8, None),  # already in the fixed list
        q("BIG", 102, 100, 8e8, 120),
        q("BIGGER", 104, 100, 9e8, None),
    ]
    assert ip.movers(quotes, exclude=["LIST"]) == ["BIGGER", "BIG"]  # the most traded first


class FakeApi:
    def __init__(self, prices):
        self.prices, self.calls = prices, 0

    def getMarketData(self, mode, wanted):
        self.calls += 1
        rows = [quote(t, self.prices[t], volume=10_000_000) for t in wanted["NSE"] if t in self.prices]
        return {"data": {"fetched": rows}}


def scanner(prices, base=(), now=datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc), clock=lambda: 0.0):
    added = []
    tokens = {f"S{t}": t for t in prices}
    api = FakeApi(prices)
    scan = ip.InPlay(lambda: api, tokens, list(base), added.extend, now_fn=lambda: now, clock=clock, pace=0)
    return scan, added, api


def test_a_scan_adds_the_movers_once_and_reads_the_whole_markets_breadth(monkeypatch):
    monkeypatch.setattr(ip, "MIN_QUOTED", 4)
    prices = {str(i): 99.0 for i in range(120)} | {"200": 103.0, "201": 105.0, "202": 104.0}
    scan, added, api = scanner(prices, base=["S202"])
    assert scan.scan() == 2 and added == ["S201", "S200"] and scan.symbols() == ["S201", "S200"]
    assert api.calls == 3  # 123 stocks, 50 a request
    assert scan.breadth() == pytest.approx(3 / 123)
    assert scan.scan() == 0 and added == ["S201", "S200"]  # not added twice


def test_a_scan_that_priced_too_few_stocks_is_not_a_reading(monkeypatch):
    scan, added, _ = scanner({"1": 105.0})
    assert scan.scan() == 0 and scan.gains() is None and scan.breadth() is None and added == []


def test_an_old_scan_is_not_used_and_the_added_stocks_are_forgotten_the_next_day(monkeypatch):
    monkeypatch.setattr(ip, "MIN_QUOTED", 1)
    now, t = [datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc)], [0.0]
    scan, *_ = scanner({"1": 105.0}, clock=lambda: t[0])
    scan.now_fn = lambda: now[0]
    scan.scan()
    assert scan.breadth() == 1.0
    t[0] = ip.IN_PLAY_MAX_AGE + 1
    assert scan.breadth() is None and scan.symbols() == ["S1"]
    now[0] = datetime(2026, 11, 17, 5, 0, tzinfo=timezone.utc)
    assert scan.symbols() == []


def test_no_more_than_the_days_limit_is_added(monkeypatch):
    monkeypatch.setattr(ip, "MIN_QUOTED", 1)
    monkeypatch.setattr(ip, "IN_PLAY_MAX", 2)
    scan, added, _ = scanner({str(i): 103.0 + i for i in range(5)})
    scan.scan()
    assert len(added) == 2


def test_the_stream_subscribes_an_added_stock_and_fetches_its_earlier_bars(tmp_path, monkeypatch):
    stream = AngelStream(make_session_factory(f"sqlite:///{tmp_path / 'x.db'}"), ["SBIN"], {"SBIN": "3045", "TCS": "11536"})
    subscribed, fetched = [], []

    class Ws:
        def subscribe(self, cid, mode, tokens):
            subscribed.append(tokens[0]["tokens"])

    stream.ws = Ws()
    monkeypatch.setattr(stream, "backfill", lambda only=None: fetched.append(only))
    stream.add(["TCS", "SBIN", "UNKNOWN"])
    for thread in __import__("threading").enumerate():
        if thread.name == "angel-backfill-added":
            thread.join(5)
    assert subscribed == [["11536"]] and fetched == [{"TCS": "11536"}]
    assert stream.symbol_of == {"3045": "SBIN", "11536": "TCS"} and stream.base == ["SBIN"]


def test_the_market_check_reads_the_whole_market_when_it_is_given():
    scan = market(up=8, down=2, LEAD=bars(100, 103))  # the list alone: 82% up
    weak_whole_market = {f"W{i}": (0.01 if i < 3 else -0.01) for i in range(10)}
    assert "weak market: 30% of stocks are up" in st.orb_leaders(scan, DAY, market=weak_whole_market)("LEAD")
    assert st.orb_leaders(scan, DAY)("LEAD") is None


def test_the_breakout_scan_includes_the_added_stocks(tmp_path):
    stocks = {s: make_bars(BREAKOUT, volumes=VOLS) for s in ("AAA", "NEW")}
    eng, broker, *_ = engine(tmp_path, stocks, at(9, 40), prices={s: 101.5 for s in stocks}, extra_fn=lambda: ["NEW", "AAA"])
    eng.universe = ["AAA"]
    assert "scanned 2 stocks, entered 2" in eng.step()
    assert {h["symbol"] for h in broker.holdings()} == {"AAA", "NEW"}


def test_the_whole_market_line_is_fifty_five_percent_and_the_lists_stays_sixty():
    whole = {f"W{i}": (0.01 if i < 56 else -0.01) for i in range(100)}  # 56% of the whole market up
    scan = market(up=5, down=5, LEAD=bars(100, 103))  # the list alone: 55% up
    assert st.orb_leaders(scan, DAY, market=whole)("LEAD") is None
    thinner = {**whole, **{f"W{i}": -0.01 for i in range(50, 56)}}  # 50% up
    assert "needs 55%" in st.orb_leaders(scan, DAY, market=thinner)("LEAD")
    assert "needs 60%" in st.orb_leaders(scan, DAY)("LEAD")
