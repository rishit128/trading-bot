"""A whole intraday session against the real paper broker: entries, exits from the 5-minute bars, the position cap, the
daily-loss halt, a restart in the middle of the day, the bar downloader and the run loop's resilience."""
from datetime import timedelta, timezone

import pandas as pd
import pytest

from src.data.india import IST
from src.database import make_session_factory
from src.engine.paper_broker import PaperBroker
from src.intraday.engine import IntradayEngine, fetch_today_bars
from tests.test_intraday import BREAKOUT, DAY, VOLS, at, make_bars
from tests.test_paper_broker import FakeClock, FakeFeed


def session(tmp_path, symbols=("AAA",), prices=None, now_ist=None, **engine_kw):
    feed, clock = FakeFeed(prices or {s: 101.5 for s in symbols}), FakeClock()
    now = [(now_ist or at(9, 45)).astimezone(timezone.utc)]
    broker = PaperBroker(make_session_factory(f"sqlite:///{tmp_path / 'day.db'}"), feed, clock, initial_cash=1_000_000.0,
                         fees=lambda s, v: 0.0, slippage=0.0, now_fn=lambda: now[0])
    messages = []
    bars = {s: make_bars(BREAKOUT, volumes=VOLS) for s in symbols}
    eng = IntradayEngine(broker, list(symbols), clock, fetch_bars=lambda syms: {s: bars[s] for s in syms if s in bars},
                         notify=messages.append, live=True, now_fn=lambda: now[0], **engine_kw)
    return eng, broker, feed, now, messages, bars


def tape(feed, symbol, when_ist, o, h, l, c):
    """Add one 5-minute bar to the broker's feed so a stop or target can be settled from it."""
    idx = pd.DatetimeIndex([when_ist.astimezone(timezone.utc)])
    frame = pd.DataFrame({"Open": [o], "High": [h], "Low": [l], "Close": [c]}, index=idx)
    feed.bars[symbol] = frame if symbol not in feed.bars else pd.concat([feed.bars[symbol], frame])


def test_a_stop_hit_is_reported_with_its_loss_and_the_stock_is_not_re_entered_the_same_day(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    assert "entered 1" in eng.step()
    tape(feed, "AAA", at(10, 30), 100.0, 100.2, 99.0, 99.2)  # trades through the 99.90 stop
    now[0] = at(10, 35).astimezone(timezone.utc)
    eng.step()
    assert broker.holdings() == []
    assert any(m.startswith("[INTRADAY] STOP AAA") and "net" in m for m in messages)
    [trade] = broker.trade_history()
    assert trade["reason"] == "STOP" and trade["exit_price"] == pytest.approx(99.9) and trade["net_pnl"] < 0
    assert "entered 0" in eng.step()  # one trade per stock per day, even after a stop-out


def test_a_target_hit_books_the_profit(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    eng.step()
    target = 101.5 + 2 * (101.5 - 99.9)
    tape(feed, "AAA", at(11, 0), 102.0, target + 0.5, 101.9, target)
    now[0] = at(11, 5).astimezone(timezone.utc)
    eng.step()
    [trade] = broker.trade_history()
    assert trade["reason"] == "TARGET" and trade["exit_price"] == pytest.approx(target) and trade["net_pnl"] > 0
    assert any("TARGET AAA" in m for m in messages)


def test_the_position_cap_stops_further_entries(tmp_path):
    eng, broker, *_ = session(tmp_path, symbols=("AAA", "BBB", "CCC"), max_positions=2)
    assert "entered 2" in eng.step()
    assert len(broker.holdings()) == 2
    assert eng.step() == "max positions open"


def test_the_daily_loss_limit_blocks_new_entries_after_a_losing_stop(tmp_path):
    eng, broker, feed, now, _, bars = session(tmp_path, symbols=("AAA", "BBB"), max_daily_loss_pct=0.0001,
                                              min_position_pct=0.5, max_position_pct=0.5)
    eng.universe = ["AAA"]  # enter only AAA first, leaving BBB for after the loss
    eng.step()
    tape(feed, "AAA", at(10, 0), 100.0, 100.1, 98.5, 98.6)
    now[0] = at(10, 5).astimezone(timezone.utc)
    eng.universe = ["AAA", "BBB"]
    assert eng.step() == "daily loss limit reached: no new entries"
    assert broker.holdings() == []


def test_a_restart_in_the_middle_of_the_day_remembers_what_was_already_traded(tmp_path):
    eng, broker, feed, now, messages, bars = session(tmp_path, symbols=("AAA", "BBB"))
    eng.universe = ["AAA"]
    eng.step()
    tape(feed, "AAA", at(10, 30), 100.0, 100.2, 99.0, 99.2)
    now[0] = at(10, 35).astimezone(timezone.utc)
    eng.step()  # AAA is stopped out and closed
    reborn = IntradayEngine(broker, ["AAA", "BBB"], eng.clock, fetch_bars=eng.fetch_bars, notify=messages.append, live=True,
                            now_fn=lambda: now[0])  # a fresh process with no memory of the morning
    assert "entered 1" in reborn.step()
    assert [h["symbol"] for h in broker.holdings()] == ["BBB"]  # AAA was traded today, so it is not bought again


def test_the_bar_downloader_returns_ist_indexed_frames_and_omits_symbols_without_data():
    idx = pd.DatetimeIndex([DAY.replace(hour=9, minute=15) + timedelta(minutes=5 * i) for i in range(3)]).tz_convert("UTC")
    frame = pd.DataFrame({"Open": [1.0, 2, 3], "High": [1.1, 2, 3], "Low": [0.9, 2, 3], "Close": [1.0, 2, 3], "Volume": [10, 20, 30]},
                         index=idx)
    wide = pd.concat({"AAA.NS": frame}, axis=1)  # BBB.NS is missing from the download entirely
    out = fetch_today_bars(["AAA", "BBB"], download=lambda tickers, **kw: wide)
    assert set(out) == {"AAA"} and str(out["AAA"].index.tz) == str(IST) and len(out["AAA"]) == 3
    assert fetch_today_bars(["AAA"], download=lambda tickers, **kw: pd.DataFrame()) == {}


def test_the_run_loop_survives_a_failing_cycle_and_paces_itself(tmp_path):
    eng, *_ = session(tmp_path)
    calls, sleeps = [], []

    class Stop(BaseException):
        pass

    def step():
        calls.append(1)
        raise ConnectionError("yahoo down")

    eng.step = step

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise Stop()

    with pytest.raises(Stop):
        eng.run_loop(sleep=sleep)
    assert len(calls) == 3 and all(s >= 30 for s in sleeps)  # kept going after each failure, never a busy loop
