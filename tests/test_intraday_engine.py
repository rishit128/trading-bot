"""The intraday engine against the real paper broker: entries, exits from the 5-minute bars, the position cap, the daily-loss
halt, a restart mid-day, what it records and refuses, stale positions, and archiving each finished session."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.india import IST
from src.engine.paper_broker import PaperBroker
from src.database import IntradaySignalRecord, IntradayUniverseRecord, make_session_factory
from src.intraday.engine import IntradayEngine
from tests.intraday_helpers import BREAKOUT, DAY, VOLS, make_bars
from tests.test_paper_broker import FakeClock, FakeFeed


def engine(tmp_path, bars, now_ist, live=True, prices=None, cash=1_000_000.0, db="i.db", **engine_kw):
    feed, clock = FakeFeed({"AAA": 101.5} if prices is None else prices), FakeClock()
    now = [now_ist.astimezone(timezone.utc)]
    broker = PaperBroker(make_session_factory(f"sqlite:///{tmp_path / db}"), feed, clock, initial_cash=cash,
                         fees=lambda s, v: 0.0, slippage=0.0, now_fn=lambda: now[0])
    messages = []
    eng = IntradayEngine(broker, ["AAA"], clock, fetch_bars=lambda syms: {s: bars[s] for s in syms if s in bars},
                         notify=messages.append, live=live, now_fn=lambda: now[0], **engine_kw)
    return eng, broker, feed, now, messages


def signals(broker):
    with broker.sessions() as s:
        return list(s.scalars(select(IntradaySignalRecord).order_by(IntradaySignalRecord.id)))


def at(h, m):
    return DAY.replace(hour=h, minute=m, second=20)


def test_engine_buys_a_fresh_breakout_once_with_stop_and_target(tmp_path):
    eng, broker, *_ , messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    assert "entered 1" in eng.step() and "entered 0" in eng.step()  # second cycle: one trade per stock per day
    [h] = broker.holdings()
    assert h["symbol"] == "AAA" and h["stop"] == pytest.approx(99.9)
    assert any("BUY AAA" in m and "[INTRADAY]" in m for m in messages)
    assert any("volume 3.0x average" in m for m in messages)
    # Rs 1,000,000: the 0.5% risk budget (Rs 5,000 / Rs 1.60 stop distance = 3,125 shares) is larger than the 20% cap
    # (Rs 200,000 / 101.5 = 1,970 shares), so the cap sets the size
    assert h["qty"] == int(1_000_000 * 0.20 / 101.5)


def test_engine_sizes_from_its_own_risk_and_cap_settings(tmp_path):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS)}
    eng, broker, *_ = engine(tmp_path, bars, at(9, 45), max_position_pct=0.10)   # a tighter cap
    eng.step()
    assert broker.holdings()[0]["qty"] == int(1_000_000 * 0.10 / 101.5)
    eng, broker, *_ = engine(tmp_path, bars, at(9, 45), risk_pct=0.0005, db="risk.db")
    eng.step()                                                                     # a tighter risk budget: Rs 500 / Rs 1.60
    assert broker.holdings()[0]["qty"] == int(1_000_000 * 0.0005 / 1.6)


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


# -- a whole session, against the real paper broker -----------------------------------------------------------------
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
                                              risk_pct=0.5, max_position_pct=0.5)
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


# -- what the engine records, and what it refuses ---------------------------------------------------------------------------
def test_every_entry_is_recorded_with_its_levels_size_and_the_system_version(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.step()
    [row] = signals(broker)
    assert (row.symbol, row.day, row.outcome, row.reason) == ("AAA", "2026-09-21", "ENTERED", None)
    assert row.qty == broker.holdings()[0]["qty"] and row.signal_price == pytest.approx(101.5)
    assert row.stop == pytest.approx(99.9) and row.target == pytest.approx(104.7) and row.volume_ratio == pytest.approx(3.0)
    assert row.range_high == pytest.approx(100.5) and row.range_low == pytest.approx(99.9)
    assert row.bar_time.startswith("2026-09-21T09:35")
    from src.versions import INTRADAY_VERSION
    assert row.system_version == INTRADAY_VERSION


def test_a_stock_too_dear_for_the_account_is_skipped_visibly_and_not_retried(tmp_path):
    dear = make_bars([c * 100 for c in BREAKOUT], volumes=VOLS)          # ~Rs 10,150 a share
    eng, broker, *_ = engine(tmp_path, {"AAA": dear}, at(9, 45), prices={"AAA": 10150.0}, cash=20_000.0)
    assert eng.step().endswith("entered 0, skipped 1")
    assert broker.holdings() == []
    [row] = signals(broker)
    assert row.outcome == "SKIPPED" and row.qty == 0 and "costs more than the largest position" in row.reason
    assert "skipped" not in eng.step()                                    # one attempt per stock per day: not counted again
    assert len(signals(broker)) == 1


def test_a_dry_run_signal_is_recorded_as_such(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45), live=False)
    eng.step()
    [row] = signals(broker)
    assert row.outcome == "DRY_RUN" and row.qty > 0 and broker.holdings() == []


def test_a_rejected_order_is_recorded_with_the_brokers_reason(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45), prices={})  # no live price
    assert "entered 0" in eng.step() and "skipped" not in eng.step()
    [row] = signals(broker)
    assert row.outcome == "FAILED" and "no price" in row.reason and row.qty > 0


def test_cash_spent_on_one_entry_is_not_offered_to_the_next_in_the_same_cycle(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path, symbols=("AAA", "BBB"), risk_pct=1.0, max_position_pct=1.0,
                                                  max_positions=2)
    # without the running total, BBB would ask for ~all the cash again
    assert "entered 2" in eng.step()
    first, second = sorted(broker.holdings(), key=lambda h: h["symbol"])
    assert first["qty"] > 9000 and 0 < second["qty"] < 100                # BBB got only what was left
    assert [r.outcome for r in signals(broker)] == ["ENTERED", "ENTERED"]


def test_the_universe_is_recorded_once_per_day_with_its_source(tmp_path):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS)}
    eng, broker, _, now, _ = engine(tmp_path, bars, at(9, 45), universe_source="a test list")

    def rows():
        with broker.sessions() as s:
            return list(s.scalars(select(IntradayUniverseRecord).order_by(IntradayUniverseRecord.id)))

    eng.step(); eng.step()
    [row] = rows()
    assert (row.day, row.source, row.size, row.symbols_json) == ("2026-09-21", "a test list", 1, '["AAA"]')
    now[0] = (at(9, 45) + timedelta(days=1)).astimezone(timezone.utc)
    eng.step()
    assert [r.day for r in rows()] == ["2026-09-21", "2026-09-22"]


def test_bars_from_a_previous_session_are_never_traded(tmp_path):
    yesterday = make_bars(BREAKOUT, volumes=VOLS)
    yesterday.index = yesterday.index - timedelta(days=1)                # Yahoo still serving the last session at 09:45
    eng, broker, *_ = engine(tmp_path, {"AAA": yesterday}, at(9, 45))
    assert "entered 0" in eng.step() and broker.holdings() == [] and signals(broker) == []


def test_a_position_left_over_from_an_earlier_day_is_closed_at_the_next_sessions_first_cycle(tmp_path):
    eng, broker, feed, now, messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.step()
    # bought Monday; the process then missed the 15:15 square-off
    assert len(broker.holdings()) == 1
    feed.prices["AAA"] = 103.0
    now[0] = datetime(2026, 9, 22, 9, 20, 20, tzinfo=IST).astimezone(timezone.utc)
    assert eng.step() == "waiting for the opening range"
    assert broker.holdings() == []
    assert any(m.startswith("[INTRADAY] STALE AAA") and "2026-09-21" in m for m in messages)
    assert broker.trade_history()[0]["reason"] == "SIGNAL"


def test_a_position_bought_today_is_not_treated_as_stale(tmp_path):
    eng, broker, feed, now, messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45))
    eng.step()
    now[0] = at(10, 0).astimezone(timezone.utc)
    eng.step()
    assert len(broker.holdings()) == 1 and not any("STALE" in m for m in messages)


# -- archiving each finished session ------------------------------------------------------------------------------------------
def archiving_engine(tmp_path, bars, now_ist, **kw):
    from src.data.intraday_bars import BarArchive

    eng, broker, feed, now, messages = engine(tmp_path, bars, now_ist, **kw)
    eng.clock.open = False                                   # after the close
    eng.archive = BarArchive(broker.sessions)
    return eng, now


def test_a_finished_session_is_archived_once_after_the_close(tmp_path):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS)}
    calls = []
    eng, now = archiving_engine(tmp_path, bars, at(15, 40))
    eng.fetch_bars = lambda syms: calls.append(syms) or bars
    assert eng.step() == "closed" and eng.archive.days() == ["2026-09-21"] and calls == [["AAA"]]
    assert len(eng.archive.load()["AAA"]) == 5
    eng.step()
    assert calls == [["AAA"]]                                # already done today: no second download


def test_nothing_is_archived_before_the_last_bar_is_published_on_weekends_or_without_an_archive(tmp_path):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS)}
    eng, now = archiving_engine(tmp_path, bars, at(15, 34))
    calls = []
    eng.fetch_bars = lambda syms: calls.append(1) or bars
    eng.step()
    assert eng.archive.days() == [] and calls == []          # 15:34: the 15:25 bar may not be out yet, so nothing is downloaded
    now[0] = datetime(2026, 9, 26, 15, 40, tzinfo=IST).astimezone(timezone.utc)   # a Saturday
    eng.step()
    assert eng.archive.days() == [] and calls == []          # no session on a weekend: not even a download
    eng.archive = None
    now[0] = at(15, 40).astimezone(timezone.utc)
    assert eng.step() == "closed"                            # no archive configured: nothing to do, nothing breaks


def test_a_holiday_saves_nothing_and_is_not_retried_but_a_failed_download_is(tmp_path, caplog):
    yesterday = make_bars(BREAKOUT, volumes=VOLS)
    yesterday.index = yesterday.index - timedelta(days=1)    # Yahoo still serving the previous session
    eng, now = archiving_engine(tmp_path, {"AAA": yesterday}, at(15, 40))
    calls = []
    eng.fetch_bars = lambda syms: calls.append(1) or {"AAA": yesterday}
    with caplog.at_level("INFO", logger="src.intraday.engine"):
        eng.step(); eng.step()
    assert eng.archive.days() == [] and len(calls) == 1      # not a session: nothing stored, and not asked again
    assert "archived 0 bars for 0 stocks" in caplog.text    # yesterday's frame is not counted as a stock of today
    eng2, _ = archiving_engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(15, 40), db="second.db")
    attempts = []

    def flaky(syms):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("yahoo down")
        return {"AAA": make_bars(BREAKOUT, volumes=VOLS)}

    eng2.fetch_bars = flaky
    eng2.step()
    assert eng2.archive.days() == []                         # failed: nothing stored
    eng2.step()
    assert eng2.archive.days() == ["2026-09-21"] and len(attempts) == 2   # retried at the next cycle


def test_a_bar_still_forming_is_never_acted_on_only_a_finished_one(tmp_path):
    eng, broker, feed, now, messages = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 37))
    # the 09:35 breakout bar has 2 minutes left: not decided on yet
    assert "entered 0" in eng.step() and broker.holdings() == []
    now[0] = at(9, 40).astimezone(timezone.utc)
    assert "entered 1" in eng.step()                                      # 09:40:20: the bar has finished
