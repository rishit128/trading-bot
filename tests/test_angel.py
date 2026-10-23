from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from src.data.angel import AngelStream, BarBuilder, bar_row, eq_tokens
from src.data.angel_bars import MIN_CLEAN_DAYS, AngelBars, AngelPriceFeed, AngelTrust, agreement, clean_days
from src.data.angel_quotes import AngelQuotes
from src.data.india import IST
from src.database import AngelBarRecord, IntradayBarRecord, make_session_factory

DAY = "2026-10-22"


def at(h, m, s=0, day=22):
    return datetime(2026, 10, day, h, m, s, tzinfo=IST).astimezone(timezone.utc)


def test_eq_tokens_keeps_only_nse_equity():
    master = [
        {"exch_seg": "NSE", "symbol": "SBIN-EQ", "name": "SBIN", "token": "3045"},
        {"exch_seg": "NSE", "symbol": "XYZ-BE", "name": "XYZ", "token": "9"},
        {"exch_seg": "BSE", "symbol": "SBIN", "name": "SBIN", "token": "500112"},
    ]
    assert eq_tokens(master) == {"SBIN": "3045"}


def test_ticks_become_five_minute_bars_with_volume_deltas():
    b = BarBuilder()
    b.tick("A", at(9, 15, 2), 100.0, 1000)
    b.tick("A", at(9, 17), 103.0, 1500)
    b.tick("A", at(9, 19, 59), 101.0, 1800)
    b.tick("A", at(9, 20, 1), 102.0, 2000)  # opens the next bar and finishes the first
    first = b.take(at(9, 20, 2))[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (100.0, 103.0, 100.0, 101.0, 1800)
    b.tick("A", at(9, 24), 104.0, 2600)
    second = b.take(at(9, 25, 1))[0]  # a quiet stock's bar is finished by the clock
    assert (second["ts"], second["volume"]) == (at(9, 20), 800)


def test_ticks_outside_the_session_are_ignored():
    b = BarBuilder()
    b.tick("A", at(9, 7), 100.0, 10)  # pre-open
    b.tick("A", at(15, 31), 100.0, 10)  # closing session
    assert b.take(at(16, 0)) == []


def test_a_bar_started_mid_session_is_dropped():
    b = BarBuilder()
    b.tick("A", at(13, 37), 100.0, 5_000_000)
    assert b.take(at(13, 41)) == []
    b.tick("A", at(13, 41, 5), 101.0, 5_100_000)
    assert b.take(at(13, 46))[0]["volume"] == 100_000


def tick(token, when, price, volume=500):
    return {
        "token": f'"{token}"',
        "exchange_timestamp": int(when.timestamp() * 1000),
        "last_traded_price": int(price * 100),
        "volume_trade_for_the_day": volume,
    }


def test_stream_saves_a_finished_bar_once_and_remembers_the_price(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    stream = AngelStream(sessions, ["SBIN"], {"SBIN": "3045", "TCS": "11536"})
    stream.on_tick(None, tick("3045", at(9, 15, 1), 100.0))
    stream.on_tick(None, tick("3045", at(9, 18), 102.0))
    stream.on_tick(None, tick("11536", at(9, 18), 5.0))  # not in the universe
    assert stream.last_price("SBIN") == 102.0 and stream.last_price("TCS") is None
    assert stream.save(at(9, 20, 1)) == 1 and stream.save(at(9, 20, 2)) == 0
    with sessions() as s:
        row = s.query(AngelBarRecord).one()
    assert (row.symbol, row.day, row.open, row.close, row.volume) == ("SBIN", DAY, 100.0, 102.0, 500)


def test_stream_is_ready_only_with_fresh_ticks_and_the_backfill(tmp_path):
    stream = AngelStream(make_session_factory(f"sqlite:///{tmp_path / 'x.db'}"), ["SBIN"], {"SBIN": "3045"})
    assert not stream.ready()
    stream.on_tick(None, tick("3045", at(9, 15, 1), 100.0))
    assert not stream.ready()  # ticks, but the day's earlier bars are not in
    stream.backfilled = True
    assert stream.ready()
    stream.last_tick -= 1000
    assert not stream.ready()


def put_bars(sessions, day, symbols, close=100.0, offset=0.0, source=None):
    start = datetime.fromisoformat(f"{day}T09:15:00+05:30").astimezone(timezone.utc)
    angel, yahoo = [], []
    for symbol in symbols:
        for i in range(4):
            ts = start + timedelta(minutes=5 * i)
            angel.append(bar_row(symbol, ts, close, close + 1, close - 1, close + offset, 1000.0, ts))
            yahoo.append(
                dict(
                    symbol=symbol,
                    ts=ts,
                    day=day,
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    volume=900.0,
                    source=source,
                )
            )
    with sessions() as s:
        s.bulk_insert_mappings(AngelBarRecord, angel)
        s.bulk_insert_mappings(IntradayBarRecord, yahoo)
        s.commit()


def test_agreement_and_clean_days(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    put_bars(sessions, "2026-10-19", ["A", "B"])
    put_bars(sessions, "2026-10-20", ["A", "B"], offset=5.0)  # Angel One's closes are 5% off
    assert clean_days(sessions) == ["2026-10-19"]
    a = pd.DataFrame({"symbol": ["A"], "ts": [1], "close": [100.0]})
    assert agreement(a, pd.DataFrame({"symbol": ["A", "B"], "ts": [1, 1], "close": [100.05, 7.0]}))["coverage"] == 0.5


class FakeStream:
    def __init__(self, ready=True):
        self.up = ready

    def ready(self):
        return self.up


def yahoo(symbols):
    idx = pd.DatetimeIndex([datetime(2026, 10, 22, 9, 15, tzinfo=IST)])
    return {
        s: pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]}, index=idx)
        for s in symbols
        if s != "NONE"
    }


def test_angel_bars_serve_angel_with_a_yahoo_fallback(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    put_bars(sessions, "2026-10-21", ["A"])
    put_bars(sessions, DAY, ["A"])
    now = lambda: at(10, 0)  # noqa: E731
    stream = FakeStream()
    bars = AngelBars(sessions, stream, yahoo, mode="on", now_fn=now)
    out = bars(["A", "B"])
    assert list(out) == ["A", "B"]
    assert set(out["A"]["Source"]) == {"angel"} and len(out["A"]) == 8  # an earlier session and today
    assert set(out["B"]["Source"]) == {"yahoo"}  # no Angel One bars for it
    stream.up = False
    assert set(bars(["A"])["A"]["Source"]) == {"yahoo"}  # stream down: Yahoo for everything


def test_auto_mode_waits_for_enough_clean_sessions(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    bars = AngelBars(sessions, FakeStream(), yahoo, mode="auto", now_fn=lambda: at(10, 0))
    put_bars(sessions, DAY, ["A"], source="yahoo")
    assert not bars.trusted()
    assert not AngelBars(sessions, FakeStream(), yahoo, mode="off").trusted()
    for day in range(MIN_CLEAN_DAYS - 1):
        put_bars(sessions, f"2026-10-{12 + day:02d}", ["A"], source="yahoo")
    assert AngelBars(sessions, FakeStream(), yahoo, mode="auto", now_fn=lambda: at(10, 0)).trusted()


def full_quote(upper=110.0, bid=99.9, ask=100.1):
    return {
        "upperCircuit": upper,
        "lowerCircuit": 90.0,
        "totBuyQuan": 300,
        "totSellQuan": 100,
        "52WeekLow": 50.0,
        "52WeekHigh": 150.0,
        "depth": {"buy": [{"price": bid}], "sell": [{"price": ask}]},
    }


def make_quotes(upper=110.0, fno_fails=False):
    def put_call():
        if fno_fails:
            raise RuntimeError("rate")
        return {"status": True, "data": [{"pcr": 0.8, "tradingSymbol": "SBIN27OCT26FUT"}]}

    api = SimpleNamespace(
        getMarketData=lambda mode, tokens: {"status": True, "data": {"fetched": [{**full_quote(upper), "symbolToken": "3045"}]}},
        putCallRatio=put_call,
        oIBuildup=lambda p: {
            "status": True,
            "data": [{"tradingSymbol": "SBIN27OCT26FUT", "percentChange": "4.5"}] if p["datatype"] == "Long Built Up" else [],
        },
    )
    quotes = AngelQuotes(lambda: api, {"SBIN": "3045"}, clock=lambda: 1e9)
    return quotes


def test_context_logs_spread_and_fno_and_skips_near_the_upper_circuit(monkeypatch):
    monkeypatch.setattr("src.data.angel_quotes.time.sleep", lambda s: None)
    ok = make_quotes().entry_context("SBIN", 100.0)
    assert ok.skip is None and ok.note["spread_pct"] == 0.2
    assert ok.note["book_imbalance"] == 0.5 and ok.note["range_52w"] == 0.5
    assert ok.note["fno"] == {"pcr": 0.8, "build_up": "Long Built Up", "oi_change_pct": 4.5}
    assert "upper circuit" in make_quotes(upper=100.5).entry_context("SBIN", 100.0).skip


def test_context_survives_an_fno_failure(monkeypatch):
    monkeypatch.setattr("src.data.angel_quotes.time.sleep", lambda s: None)
    context = make_quotes(fno_fails=True).entry_context("SBIN", 100.0)
    assert context.skip is None and "unavailable" in context.note["fno"]


def test_backfill_fetches_a_week_for_a_stock_with_no_history_and_only_today_for_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.angel.time.sleep", lambda s: None)
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    put_bars(sessions, "2026-10-21", ["OLD"])
    asked = {}

    def candles(params):
        asked[params["symboltoken"]] = params["fromdate"]
        return {
            "status": True,
            "data": [["2026-10-21T09:15:00+05:30", 1, 2, 0.5, 1.5, 10], ["2026-10-21T09:20:00+05:30", 1, 2, 0.5, 1.6, 20]],
        }

    stream = AngelStream(sessions, ["OLD", "NEW"], {"OLD": "1", "NEW": "2"})
    stream.api = SimpleNamespace(getCandleData=candles)
    stream.backfill()
    assert asked["1"].startswith(datetime.now(IST).date().isoformat()) and asked["2"] < asked["1"]
    assert stream.backfilled
    with sessions() as s:
        assert s.query(AngelBarRecord).filter_by(symbol="NEW").count() == 2


def test_quotes_are_fetched_fifty_to_a_request_and_shared(monkeypatch):
    monkeypatch.setattr("src.data.angel_quotes.time.sleep", lambda s: None)
    symbols = [f"S{i}" for i in range(120)]
    tokens = {s: str(i) for i, s in enumerate(symbols)}
    calls = []

    def market(mode, request):
        calls.append(len(request["NSE"]))
        return {"status": True, "data": {"fetched": [{**full_quote(), "symbolToken": t} for t in request["NSE"]]}}

    api = SimpleNamespace(
        getMarketData=market, putCallRatio=lambda: {"status": True, "data": []}, oIBuildup=lambda p: {"status": True, "data": []}
    )
    quotes = AngelQuotes(lambda: api, tokens, symbols, clock=lambda: 1e9)
    quotes.entry_context("S3", 100.0)
    quotes.entry_context("S99", 100.0)  # served from the refresh the first one caused
    assert calls == [50, 50, 20]


class Yahoo:
    def last_price(self, symbol):
        return 1.0

    def last_quote(self, symbol):
        return 1.0, at(9, 0)

    def bars_since(self, symbol, since):
        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]}, index=pd.DatetimeIndex([at(9, 0)])
        )


def test_the_price_feed_uses_a_recent_angel_bar_and_falls_back_to_yahoo(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    put_bars(sessions, DAY, ["A"])  # bars 09:15 to 09:30, the last one closing at 09:35
    feed = AngelPriceFeed(sessions, Yahoo(), lambda: True, now_fn=lambda: at(9, 40))
    assert feed.last_price("A") == 100.0 and feed.last_quote("A")[1] == at(9, 30)
    assert feed.last_price("B") == 1.0  # no Angel One bars for it
    late = AngelPriceFeed(sessions, Yahoo(), lambda: True, now_fn=lambda: at(11, 0))
    assert late.last_price("A") == 1.0  # the newest bar is stale
    assert AngelPriceFeed(sessions, Yahoo(), lambda: False, now_fn=lambda: at(9, 40)).last_price("A") == 1.0
    bars = feed.bars_since("A", at(9, 20))
    assert len(bars) == 2 and bars.index[0] == at(9, 25).astimezone(IST)
    assert len(feed.bars_since("A", at(9, 20) - timedelta(days=2))) == 1  # reaching back before today: Yahoo's


def test_the_trust_check_follows_the_mode(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    assert AngelTrust(sessions, "on")() and not AngelTrust(sessions, "off")() and not AngelTrust(sessions, "auto")()


def test_the_login_is_announced_once_a_day(tmp_path):
    sent = []
    stream = AngelStream(make_session_factory(f"sqlite:///{tmp_path / 'x.db'}"), ["SBIN"], {"SBIN": "3045"}, notify=sent.append)
    stream._tell_login()
    stream._tell_login()
    assert len(sent) == 1 and "logged in" in sent[0] and "1 stocks" in sent[0]
