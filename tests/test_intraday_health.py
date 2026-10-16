"""Intraday reliability: an alert for each fault (a failed square-off or stop check, a stale fill price, a thin scan, a
position valued at cost), none on a normal day, one summary line each evening, views that write nothing, the 09:30 start
deadline, and exit alerts that survive a restart."""

import logging
from datetime import time as dtime
from datetime import timedelta, timezone

import pytest
from sqlalchemy import func, select

from src.config import Settings
from src.database import PaperAccountRecord, PaperPositionRecord, PaperTradeRecord
from src.intraday.engine import IntradayEngine, run_together
from tests.intraday_helpers import make_bars
from tests.test_intraday_engine import at, session, tape
from tests.test_paper_broker import T0, bar, make


def utc(ist_time):
    return ist_time.astimezone(timezone.utc)


def alerts(messages):
    return [m for m in messages if "ALERT" in m]


# ---------------------------------------------------------------- a normal day says nothing, then one summary line
def test_a_normal_day_raises_no_alert_and_ends_with_one_summary_line(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    eng.step()  # 09:45: the breakout is bought
    now[0] = utc(at(15, 20))
    eng.step()  # square-off
    now[0] = utc(at(15, 40))
    eng.clock.open = False
    assert eng.step() == "closed"
    assert alerts(messages) == []
    [line] = [m for m in messages if "net P&L" in m]
    assert line.startswith("[INTRADAY] 2026-09-21: universe 1, bars for 1; signals: ENTERED 1")
    assert "entries 1; exits 1 (SIGNAL 1); net P&L " in line
    eng.step()
    assert len([m for m in messages if "net P&L" in m]) == 1  # once a day


def test_no_summary_on_a_day_the_engine_never_ran(tmp_path):
    eng, _, _, now, messages, _ = session(tmp_path, now_ist=at(16, 0))
    eng.clock.open = False
    eng.step()
    assert not [m for m in messages if "net P&L" in m]


def test_the_summary_groups_skipped_signals_by_reason(tmp_path):
    eng, _, _, now, messages, _ = session(tmp_path, symbols=("AAA", "BBB", "CCC"), max_positions=1, record_capped=True)
    eng.step()
    now[0] = utc(at(15, 40))
    eng.clock.open = False
    eng.step()
    [line] = [m for m in messages if "net P&L" in m]
    assert "ENTERED 1, SKIPPED 2 (position limit reached (1 open) x2)" in line


# ---------------------------------------------------------------- faults
def test_a_scan_that_returns_too_few_stocks_alerts_once_and_again_after_it_recovers(tmp_path):
    eng, _, _, now, messages, bars = session(tmp_path)
    quiet = make_bars([100.0] * 5)  # a normal download with nothing breaking out
    bars.clear()  # the download came back empty
    eng.step()
    [a] = alerts(messages)
    assert "bars for only 0 of 1 stocks (under 80%)" in a
    now[0] = utc(at(9, 50))
    eng.step()
    assert len(alerts(messages)) == 1  # still down: no second alert
    bars["AAA"] = quiet
    now[0] = utc(at(9, 55))
    eng.step()
    bars.clear()
    now[0] = utc(at(10, 0))
    eng.step()
    assert len(alerts(messages)) == 2  # it came back and failed again: that is news


def test_a_download_that_raises_alerts_and_still_fails_the_cycle(tmp_path):
    eng, _, _, _, messages, _ = session(tmp_path)

    def boom(symbols):
        raise ConnectionError("yahoo down")

    eng.fetch_bars = boom
    with pytest.raises(ConnectionError):  # the cycle still fails, so the heartbeat shows it
        eng.step()
    [a] = alerts(messages)
    assert "the 5-minute download failed (ConnectionError: yahoo down)" in a


def test_a_failed_stop_check_alerts(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    eng.step()  # AAA bought
    feed.bars_since = lambda symbol, since: (_ for _ in ()).throw(RuntimeError("feed down"))
    now[0] = utc(at(9, 50))
    eng.step()
    [a] = alerts(messages)
    assert "could not check the stop/target of AAA" in a and "unprotected" in a
    assert broker.unchecked_exits == {"AAA"}


def test_a_position_valued_at_cost_alerts(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    eng.step()
    del feed.prices["AAA"]  # the feed has no price any more
    now[0] = utc(at(9, 50))
    eng.step()
    [a] = alerts(messages)
    assert "no price for AAA: valued at cost" in a and broker.valued_at_cost == {"AAA"}


def test_a_failed_square_off_alerts_and_is_retried(tmp_path):
    eng, broker, _, now, messages, _ = session(tmp_path)
    eng.step()
    real_sell = broker.sell

    def refuse(symbol, qty):
        raise RuntimeError("no price")

    broker.sell = refuse
    now[0] = utc(at(15, 20))
    eng.step()
    [a] = alerts(messages)
    assert "square-off of AAA" in a and "FAILED (no price)" in a
    now[0] = utc(at(15, 25))
    eng.step()
    assert len(alerts(messages)) == 1  # still failing: one alert
    broker.sell = real_sell
    now[0] = utc(at(15, 28))
    assert eng.step() == "square-off: closed 1" and broker.holdings() == []


@pytest.mark.parametrize("age_minutes,alerted", [(12, True), (4, False)])
def test_a_fill_priced_from_an_old_bar_alerts(tmp_path, age_minutes, alerted):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    bar_start = now[0] - timedelta(minutes=age_minutes)
    feed.last_quote = lambda symbol: (101.5, bar_start)
    eng.step()
    assert bool(alerts(messages)) is alerted
    if alerted:
        assert "AAA was filled at 101.50 from a bar that started 12 minutes ago" in alerts(messages)[0]


def test_a_naive_quote_time_is_read_as_utc(tmp_path):
    eng, _, feed, now, messages, _ = session(tmp_path)
    feed.last_quote = lambda symbol: (101.5, (now[0] - timedelta(minutes=30)).replace(tzinfo=None))
    eng.step()
    assert "30 minutes ago" in alerts(messages)[0]


# ---------------------------------------------------------------- R2: a view writes nothing
def test_the_valuation_view_settles_and_writes_nothing(tmp_path):
    broker, feed, clock, now, sessions = make(tmp_path, cash=100_000.0, slippage=0.0)
    broker.cash_yield = 0.05
    broker.buy_with_bracket("AAA", 10, 100.0, 0.05, 0.5)  # stop 95
    feed.bars["AAA"] = bar(5, 100.0, 100.0, 90.0, 92.0)  # a bar through the stop that portfolio() would settle
    feed.prices["AAA"] = 92.0
    now[0] = T0 + timedelta(hours=3)

    def state():
        with sessions() as s:
            acct = s.get(PaperAccountRecord, 1)
            pos = s.scalars(select(PaperPositionRecord)).all()
            return (
                acct.cash,
                acct.interest_accrued_at,
                acct.interest_earned,
                acct.day,
                acct.day_start_equity,
                acct.last_mark,
                [(p.symbol, p.qty, p.last_checked) for p in pos],
                s.scalar(select(func.count()).select_from(PaperTradeRecord)),
            )

    before = state()
    pf, held = broker.valuation()
    assert state() == before  # not a row changed
    assert pf.positions == {"AAA": 920.0} and pf.cash == 99_000.0 and pf.equity == pytest.approx(99_920.0)
    assert [h["symbol"] for h in held] == ["AAA"] and held[0]["price"] == 92.0 and held[0]["pnl"] == pytest.approx(-80.0)
    broker.portfolio()  # the engine's own call is what settles the stop
    assert state() != before and broker.holdings() == []


def test_the_view_shows_a_priceless_position_at_cost_and_remembers_it(tmp_path):
    broker, feed, _, _, _ = make(tmp_path)
    broker.buy_with_bracket("AAA", 10, 100.0, 0.05, 0.5)
    del feed.prices["AAA"]
    pf, held = broker.valuation()
    assert held[0]["price"] == held[0]["avg_price"] and broker.valued_at_cost == {"AAA"}
    assert pf.positions == {"AAA": 1000.0}


# ---------------------------------------------------------------- R3: gap-only's 09:30 step
def test_a_late_start_of_the_0930_step_alerts_once_a_timely_one_does_not(tmp_path):
    for name in ("timely", "late", "none"):
        (tmp_path / name).mkdir()
    timely, _, _, _, told, _ = session(tmp_path / "timely", now_ist=at(9, 31), step_deadline=dtime(9, 33))
    timely.step()
    assert alerts(told) == []
    late, _, _, now, told, _ = session(tmp_path / "late", now_ist=at(9, 36), step_deadline=dtime(9, 33))
    late.step()
    [a] = alerts(told)
    assert "09:30 step started late, at 09:36:20 (it must start by 09:33)" in a
    now[0] = utc(at(9, 41))
    late.step()
    assert len(alerts(told)) == 1  # once a day
    none, *_, told, _ = session(tmp_path / "none", now_ist=at(9, 36))
    none.step()
    assert alerts(told) == []  # an engine without a deadline never alerts


def test_the_run_loop_logs_when_each_step_started(tmp_path, caplog):
    eng, *_ = session(tmp_path)

    class Stop(BaseException):
        pass

    def sleep(seconds):
        raise Stop()

    with caplog.at_level(logging.INFO, logger="src.intraday.engine"), pytest.raises(Stop):
        run_together([eng], sleep=sleep)
    assert any("intraday cycle (started " in r.getMessage() and r.getMessage().count(":") >= 3 for r in caplog.records)


# ---------------------------------------------------------------- R5: configuration and restarts
def test_the_intraday_daily_loss_limit_is_a_validated_setting(monkeypatch):
    assert Settings().intraday_max_daily_loss_pct == 0.02
    with pytest.raises(ValueError, match="intraday_max_daily_loss_pct"):
        Settings(intraday_max_daily_loss_pct=0.0)
    from src.config import load_settings

    monkeypatch.setenv("INTRADAY_MAX_DAILY_LOSS_PCT", "0.05")
    assert load_settings().intraday_max_daily_loss_pct == 0.05


def test_an_exit_after_a_restart_is_still_reported(tmp_path):
    eng, broker, feed, now, messages, _ = session(tmp_path)
    eng.step()  # AAA bought at 09:45 by the old process
    reborn_messages = []
    reborn = IntradayEngine(
        broker, ["AAA"], eng.clock, fetch_bars=eng.fetch_bars, notify=reborn_messages.append, live=True, now_fn=lambda: now[0]
    )  # a new process: it never saw the buy
    tape(feed, "AAA", at(10, 30), 100.0, 100.2, 99.0, 99.2)  # the stop is hit while it starts up
    now[0] = utc(at(10, 35))
    reborn.step()
    assert any(m.startswith("[INTRADAY] STOP AAA") for m in reborn_messages)
