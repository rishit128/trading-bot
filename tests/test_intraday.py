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
    assert s.strength == pytest.approx(1.0)  # 3x average volume = double the 1.5x minimum -> full strength


def test_a_breakout_that_barely_clears_the_volume_bar_has_zero_strength():
    s = st.find_setup("AAA", make_bars(BREAKOUT, volumes=[1000, 1000, 1000, 1000, 1500]))  # exactly 1.5x average
    assert s and s.strength == pytest.approx(0.0)


def test_no_setup_without_a_volume_surge_or_with_a_too_narrow_range():
    assert st.find_setup("AAA", make_bars(BREAKOUT)) is None  # volume never surges
    flat = make_bars([100, 100.01, 100.0, 100.0, 100.6], highs=[100.02] * 5, lows=[99.99] * 5, volumes=VOLS)
    assert st.find_setup("AAA", flat) is None  # opening range far below 0.3%


def test_position_size_scales_with_breakout_strength_between_the_floor_and_ceiling():
    assert st.position_size(1_000_000, 100.0, 99.0, strength=0.0, min_pct=0.02, max_pct=0.05) == 200   # floor: 2%
    assert st.position_size(1_000_000, 100.0, 99.0, strength=1.0, min_pct=0.02, max_pct=0.05) == 500   # ceiling: 5%
    assert st.position_size(1_000_000, 100.0, 99.0, strength=0.5, min_pct=0.02, max_pct=0.05) == 350   # halfway: 3.5%
    assert st.position_size(1_000_000, 100.0, 100.0) == 0   # no risk (stop == entry) is refused
    assert st.position_size(1_000_000, 100.0, 105.0) == 0   # stop above entry is refused


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
    assert any("volume strength 100%" in m for m in messages)
    # 1,000,000 equity, full strength -> the engine's default max_position_pct (5%) at 101.5/share
    assert h["qty"] == int(1_000_000 * 0.05 / 101.5)


def test_engine_uses_its_own_configured_min_and_max_position_pct(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.min_position_pct, eng.max_position_pct = 0.10, 0.10  # flat 10%, easy to check
    eng.step()
    [h] = broker.holdings()
    assert h["qty"] == int(1_000_000 * 0.10 / 101.5)


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
