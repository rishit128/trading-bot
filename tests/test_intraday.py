from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.india import IST
from src.engine.paper_broker import PaperBroker
from src.database import make_session_factory
from src.intraday import strategy as st
from src.intraday.engine import IntradayEngine
from tests.test_paper_broker import FakeClock, FakeFeed

DAY = datetime(2026, 9, 21, tzinfo=IST)


def make_bars(closes, highs=None, lows=None, volumes=None, start=(9, 15)):
    """5-minute bars from 09:15 IST; open = previous close."""
    idx = pd.DatetimeIndex([DAY.replace(hour=start[0], minute=start[1]) + timedelta(minutes=5 * i) for i in range(len(closes))])
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame({"Open": opens, "High": highs or [max(o, c) + 0.1 for o, c in zip(opens, closes)],
                         "Low": lows or [min(o, c) - 0.1 for o, c in zip(opens, closes)], "Close": closes,
                         "Volume": volumes or [1000.0] * len(closes)}, index=idx)


BREAKOUT = [100, 100.4, 100.2, 100.3, 101.5]  # range 99.9..100.5, then a close above it on the 5th bar
VOLS = [1000, 1000, 1000, 1000, 3000]


def test_breakout_bar_above_range_vwap_and_volume_is_a_setup():
    s = st.find_setup("AAA", make_bars(BREAKOUT, volumes=VOLS))
    assert s and s.signal_price == 101.5 and s.stop == pytest.approx(99.9) and s.range_high == pytest.approx(100.5)
    assert s.target == pytest.approx(101.5 + 2 * (101.5 - 99.9))


def test_no_setup_without_a_volume_surge_or_with_a_too_narrow_range():
    assert st.find_setup("AAA", make_bars(BREAKOUT)) is None  # volume never surges
    flat = make_bars([100, 100.01, 100.0, 100.0, 100.6], highs=[100.02] * 5, lows=[99.99] * 5, volumes=VOLS)
    assert st.find_setup("AAA", flat) is None  # opening range far below 0.3%


def test_position_size_risks_half_a_percent_and_caps_notional():
    assert st.position_size(1_000_000, 100.0, 99.0) == 2000  # risk 5,000 / 1 per share, notional 200,000 = the 20% cap
    assert st.position_size(1_000_000, 100.0, 90.0) == 500   # risk 5,000 / 10 per share
    assert st.position_size(1_000_000, 100.0, 100.0) == 0


def test_intraday_fees_have_stt_on_sells_only_and_flat_brokerage():
    buy, sell = st.intraday_fees("BUY", 200_000), st.intraday_fees("SELL", 200_000)
    assert sell > buy and 20 < buy < 40 and 60 < sell < 90


def engine(tmp_path, bars, now_ist, live=True, prices=None):
    feed, clock = FakeFeed(prices or {"AAA": 101.5}), FakeClock()
    now = [now_ist.astimezone(timezone.utc)]
    broker = PaperBroker(make_session_factory(f"sqlite:///{tmp_path / 'i.db'}"), feed, clock, initial_cash=1_000_000.0,
                         fees=lambda s, v: 0.0, slippage=0.0, now_fn=lambda: now[0])
    messages = []
    eng = IntradayEngine(broker, ["AAA"], clock, fetch_bars=lambda syms: {s: bars[s] for s in syms if s in bars},
                         notify=messages.append, live=live, now_fn=lambda: now[0])
    return eng, broker, feed, now, messages


def at(h, m):
    return DAY.replace(hour=h, minute=m, second=20)


def test_engine_buys_a_fresh_breakout_once_with_stop_and_target(tmp_path):
    eng, broker, *_ , messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    assert "entered 1" in eng.step() and "entered 0" in eng.step()  # second cycle: one trade per stock per day
    [h] = broker.holdings()
    assert h["symbol"] == "AAA" and h["stop"] == pytest.approx(99.9) and any("BUY AAA" in m and "[INTRADAY]" in m for m in messages)


def test_dry_run_only_reports_the_signal(tmp_path):
    eng, broker, *_, messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45), live=False)
    eng.step()
    assert broker.holdings() == [] and any("DRY RUN signal" in m for m in messages)


def test_a_stale_breakout_from_earlier_in_the_day_is_not_chased(tmp_path):
    bars = make_bars(BREAKOUT + [101.4, 101.3, 101.4], volumes=VOLS + [1000] * 3)
    eng, broker, *_ = engine(tmp_path, {"AAA": bars}, at(9, 15).replace(minute=15) + timedelta(minutes=45))
    eng.step()
    assert broker.holdings() == []


def test_everything_is_squared_off_at_1515_and_no_new_entries_after(tmp_path):
    eng, broker, feed, now, messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.step()
    assert len(broker.holdings()) == 1
    feed.prices["AAA"] = 103.0
    now[0] = at(15, 16).astimezone(timezone.utc)
    assert eng.step() == "square-off: closed 1"
    assert broker.holdings() == [] and any("SQUARE-OFF AAA" in m for m in messages)
    assert eng.step() == "square-off: closed 0"


def test_no_entries_before_the_opening_range_completes_or_when_the_market_is_closed(tmp_path):
    eng, broker, _, _, _ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 20))
    assert eng.step() == "waiting for the opening range"
    eng.clock.open = False
    assert eng.step() == "closed"
