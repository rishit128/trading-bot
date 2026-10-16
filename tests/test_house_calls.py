"""The broker's research calls ("house calls"): parsing the portal's response, the one-snapshot store, and how the swing
pipeline, the breakout engine and the all-day research watch use them."""

import json
import threading
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.agents.base import LEAD
from src.app.portfolio_agent import PortfolioCommand, make_research_sink
from src.config import Settings
from src.data.house_calls import (
    MAX_AGE,
    HouseCall,
    HouseCallStore,
    call_for,
    intraday_buys,
    looks_like_calls,
    open_sells,
    parse_calls,
    prompt_line,
    recently_stopped,
    swing_buys,
)
from src.data.india import IST
from src.data.indicators import Snapshot
from src.database import DecisionRecord, HouseCallRecord, make_session_factory
from src.engine.risk_engine import Portfolio
from src.llm import AgentSignal
from src.pipeline import TradingPipeline
from src.portfolio.research_watch import watch_research
from tests.intraday_helpers import BREAKOUT, DAY, VOLS, make_bars
from tests.test_intraday_engine import at, engine, signals
from tests.test_llm_and_pipeline import FakeBroker

# The three calls exactly as the portal sent them on 2026-10-15 (dashboard Research feed), plus edge cases.
CAMS = {"Scrip": "COMPUTER AGE MNGT SER LTD", "ScripCode": "CAMS", "ScripSegment": "1", "ScripToken": "342", "PDC": 694.0,
        "ISIN": "INE596I01020", "CMPFrom": "706.10", "CMPTo": "0.00", "StopLoss": "698.00", "TargetPrice": "715.00",
        "ValidFrom": "", "ValidTo": "", "RecomTypeCode": "1", "RecomSubTypeCode": "1", "RecomType": "Equity",
        "CallCategory": "buy", "RecomSubType": "Intraday", "CallMode": "Open", "CallType": "buy", "AppCallType": "buy",
        "CurrentGain": 0.0, "LTP": 705.65, "PublishedDtm": "15 Oct 2026 09:36 AM", "PotentialGain": 1.26,
        "ExitComment": "", "ExitDtm": "1900-01-01T00:00:00", "ExitUser": ""}  # fmt: skip
RBL = {**CAMS, "ScripCode": "RBLBANK", "CMPFrom": "387.65", "CMPTo": "387.50", "StopLoss": "376.50",
       "TargetPrice": "404.50", "ValidFrom": "2026-10-28", "ValidTo": "2026-10-28", "RecomSubType": "Swing Trade",
       "PublishedDtm": "14 Oct 2026 02:00 PM"}  # fmt: skip
ABC = {**CAMS, "ScripCode": "ABCAPITAL", "CMPFrom": "383.80", "CMPTo": "384.00", "StopLoss": "373.00",
       "TargetPrice": "400.00", "ValidTo": "2026-10-30", "RecomSubType": "Stellar Pick", "PublishedDtm": "14 Oct 2026 11:19 AM"}  # fmt: skip
DMART_SELL = {**CAMS, "ScripCode": "DMART", "CMPFrom": "3391.80", "CMPTo": "3392.00", "StopLoss": "3424.00",
              "TargetPrice": "3355.00", "CallType": "sell", "AppCallType": "sell", "PublishedDtm": "15 Oct 2026 10:48 AM"}  # fmt: skip
BSE_STOPPED = {**CAMS, "ScripCode": "BSE", "CallMode": "Closed", "ExitComment": "Stoploss Hit",
               "ExitDtm": "2026-10-15T11:20:00", "PublishedDtm": "15 Oct 2026 10:18 AM"}  # fmt: skip
OPTION = {**CAMS, "ScripCode": "BANKNIFTY27OCT2656500PE", "RecomType": "F&O"}
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=IST)


def body(*rows):
    return json.dumps(list(rows))  # the portal sends the list as a JSON string


# ---------------------------------------------------------------- parsing
def test_the_portals_calls_are_parsed_with_their_entry_range_in_order():
    cams, rbl, abc = parse_calls([body(CAMS, RBL, ABC)])
    assert (cams.symbol, cams.kind, cams.side, cams.entry_low, cams.entry_high) == ("CAMS", "Intraday", "buy", 706.1, 706.1)
    assert (rbl.entry_low, rbl.entry_high, rbl.stop, rbl.target) == (387.5, 387.65, 376.5, 404.5)  # given high-low
    assert rbl.valid_to == date(2026, 10, 28) and cams.valid_to is None
    assert abc.published == datetime(2026, 10, 14, 11, 19, tzinfo=IST) and all(c.is_open for c in (cams, rbl, abc))


def test_closed_calls_keep_their_outcome_and_exit_time_and_options_are_left_out():
    [bse] = parse_calls([body(BSE_STOPPED, OPTION)])
    assert not bse.is_open and bse.exit_comment == "Stoploss Hit" and bse.exit_at == datetime(2026, 10, 15, 11, 20, tzinfo=IST)


def test_a_response_is_recognised_by_its_fields_and_broken_rows_are_skipped():
    assert looks_like_calls(body(CAMS)) and looks_like_calls([CAMS])
    assert not looks_like_calls(json.dumps([{"Symbol": "X"}])) and not looks_like_calls("not json") and not looks_like_calls([])
    assert parse_calls([body({**CAMS, "StopLoss": "0.00"}, {**CAMS, "PublishedDtm": "soon"}, {**CAMS, "CallType": "hold"})]) == []
    assert len(parse_calls([body(CAMS), body(CAMS)])) == 1  # the same call in two responses counts once


def test_the_prompt_line_quotes_the_portal_numbers_only():
    [rbl] = parse_calls([body(RBL)])
    assert prompt_line(rbl) == (
        "the broker's research desk published a Swing Trade BUY call on 14 Oct 2026 14:00 IST: entry 387.50-387.65, "
        "stop 376.50, target 404.50, valid to 2026-10-28; status open"
    )
    assert prompt_line(None) == "none"
    assert HouseCall.from_dict(json.loads(rbl.to_json())) == rbl


def test_which_calls_feed_which_strategy():
    calls = parse_calls(
        [body(CAMS, RBL, ABC, DMART_SELL, BSE_STOPPED, {**CAMS, "ScripCode": "OLD", "PublishedDtm": "14 Oct 2026 09:30 AM"})]
    )
    assert swing_buys(calls) == ["RBLBANK", "ABCAPITAL"]  # open Swing / Stellar buys; never intraday, sells or closed
    assert intraday_buys(calls, date(2026, 10, 15)) == ["CAMS"]  # today's only: yesterday's intraday call is history
    assert list(open_sells(calls)) == ["DMART"]
    assert call_for(calls, "BSE", NOW).exit_comment == "Stoploss Hit"  # closed 40 minutes ago: still shown
    assert call_for(calls, "BSE", NOW + timedelta(days=4)) is None and call_for(calls, "NONE", NOW) is None
    assert recently_stopped(calls, "BSE", NOW) is not None and recently_stopped(calls, "CAMS", NOW) is None


# ---------------------------------------------------------------- the snapshot store
def test_the_store_keeps_one_snapshot_and_ignores_a_stale_one(tmp_path):
    now = [NOW]
    store = HouseCallStore(make_session_factory(f"sqlite:///{tmp_path / 'h.db'}"), now_fn=lambda: now[0])
    assert store.current() == []
    store.replace(parse_calls([body(CAMS, RBL)]))
    store.replace(parse_calls([body(ABC, BSE_STOPPED)]))  # the snapshot is replaced, never appended to
    with store.sessions() as s:
        assert sorted(s.scalars(select(HouseCallRecord.symbol))) == ["ABCAPITAL", "BSE"]
    assert {c.symbol for c in store.current()} == {"ABCAPITAL", "BSE"}
    assert store.current()[0].published.tzinfo is not None
    assert next(c for c in store.current() if c.symbol == "BSE").exit_at == datetime(2026, 10, 15, 11, 20, tzinfo=IST)
    now[0] = NOW + MAX_AGE + timedelta(seconds=1)  # the watch stopped: the calls are no longer trusted
    assert store.current() == []


def events(store):
    from src.database import HouseCallEvent

    with store.sessions() as s:
        return [(e.symbol, e.is_open, e.exit_comment, e.stop, e.first_seen, e.last_seen) for e in s.scalars(select(HouseCallEvent).order_by(HouseCallEvent.id))]  # fmt: skip


def test_the_history_adds_a_row_only_when_a_call_changes(tmp_path):
    store = HouseCallStore(make_session_factory(f"sqlite:///{tmp_path / 'h.db'}"), now_fn=lambda: NOW)
    t1, t2, t3, t4 = (NOW + timedelta(minutes=10 * i) for i in range(4))
    store.replace(parse_calls([body(CAMS, RBL)]), fetched_at=t1)
    store.replace(parse_calls([body(CAMS, RBL)]), fetched_at=t2)  # the same calls read again: nothing added
    rows = events(store)
    assert [r[0] for r in rows] == ["CAMS", "RBLBANK"]
    utc = lambda t: t.astimezone(timezone.utc).replace(tzinfo=None)  # noqa: E731 - SQLite hands times back naive
    assert all(r[4] == utc(t1) and r[5] == utc(t2) for r in rows)  # first seen at the first read, last seen at the second
    closed = {**CAMS, "CallMode": "Closed", "ExitComment": "Target Hit", "ExitDtm": "2026-10-15T12:25:00"}
    store.replace(parse_calls([body(closed, RBL)]), fetched_at=t3)  # CAMS hit its target: one row
    store.replace(parse_calls([body(closed, {**RBL, "StopLoss": "380.00"})]), fetched_at=t4)  # RBL's stop raised: one row
    rows = events(store)
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [
        ("CAMS", True, "", 698.0),
        ("RBLBANK", True, "", 376.5),
        ("CAMS", False, "Target Hit", 698.0),
        ("RBLBANK", True, "", 380.0),
    ]
    assert rows[0][5] == utc(t2) and rows[2][4] == utc(t3) and rows[2][5] == utc(t4)  # the old state is kept as it ended
    store.replace([], fetched_at=t4 + timedelta(minutes=10))  # a call gone from the list is not a change: nothing added
    assert len(events(store)) == 4 and store.current() == []


def test_a_failing_history_never_costs_the_snapshot(tmp_path, monkeypatch):
    store = HouseCallStore(make_session_factory(f"sqlite:///{tmp_path / 'h.db'}"), now_fn=lambda: NOW)

    def broken(*a):
        raise OSError("disk full")

    monkeypatch.setattr(store, "record_history", broken)
    store.replace(parse_calls([body(CAMS)]))
    assert [c.symbol for c in store.current()] == ["CAMS"] and events(store) == []


def test_a_broken_store_means_no_calls_not_a_crash(tmp_path):
    def broken():
        raise OSError("disk gone")

    assert HouseCallStore(broken).current() == []


# ---------------------------------------------------------------- the research watch and its sink
def test_the_sink_stores_each_read_and_announces_only_new_calls(tmp_path):
    store = HouseCallStore(make_session_factory(f"sqlite:///{tmp_path / 'h.db'}"), now_fn=lambda: NOW)
    told = []
    sink = make_research_sink(store, told.append)
    assert sink(['{"other": 1}', body(CAMS, RBL)]) == 2 and told == []  # the first read announces nothing
    assert sink([body(CAMS, RBL, ABC)]) == 3
    assert len(told) == 1 and "ABCAPITAL" in told[0] and "Stellar Pick BUY" in told[0]
    assert sink(['{"other": 1}']) == 0  # no call list in this read: the snapshot is left alone (and goes stale)
    assert {c.symbol for c in store.current()} == {"CAMS", "RBLBANK", "ABCAPITAL"}


def clock(*times):
    it = iter(times)
    last = [times[-1]]

    def now():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return now


def test_the_watch_reads_until_the_day_is_over_and_keeps_its_lock_alive():
    reads, alive, told = [], [], []
    stop = threading.Event()
    stop.wait = lambda seconds: None  # no real waiting in a test
    t = [datetime(2026, 10, 15, 9, 5, tzinfo=IST), datetime(2026, 10, 15, 9, 15, tzinfo=IST),
         datetime(2026, 10, 15, 15, 36, tzinfo=IST)]  # fmt: skip
    ended = watch_research(
        "page", "base/", lambda b: len(b), stop, keep_alive=lambda: alive.append(1), progress=told.append,
        read_once=lambda page, base: reads.append(1) or ["x", "y"], now_fn=clock(*t),
    )  # fmt: skip
    assert ended.startswith("day over") and len(reads) == 2 and len(alive) == 2
    assert "Research watch on: 2 calls" in told[0] and len(told) == 1


def test_the_watch_stops_when_the_portal_ends_the_session_or_on_request():
    stop = threading.Event()
    morning = lambda: datetime(2026, 10, 15, 10, 0, tzinfo=IST)  # noqa: E731
    assert "ended the session" in watch_research("p", "b", len, stop, read_once=lambda p, b: None, now_fn=morning)
    stop.set()
    assert "stopped" in watch_research("p", "b", len, stop, read_once=lambda p, b: ["x"], now_fn=morning)


def test_a_read_with_no_calls_warns_once_and_a_failing_sink_does_not_end_the_watch():
    stop, told, n = threading.Event(), [], [0]

    def sink(bodies):
        n[0] += 1
        if n[0] == 3:
            raise ValueError("bad data")
        return 0

    def wait(seconds):
        if n[0] >= 4:
            stop.set()

    stop.wait = wait
    watch_research("p", "b", sink, stop, read_once=lambda p, b: [], progress=told.append,
                   now_fn=lambda: datetime(2026, 10, 15, 10, 0, tzinfo=IST))  # fmt: skip
    assert n[0] == 4 and sum("no calls" in m for m in told) == 1


def test_research_start_and_stop_share_the_otp_inbox_and_run_one_at_a_time():
    started, release = [], threading.Event()

    def runner(get_otp, notify, sink, stop):
        started.append(sink)
        stop.wait(5)

    cmd = PortfolioCommand(lambda text: None, research_sink=lambda bodies: 0, research_runner=runner)
    assert "No research watch" in cmd.research_stop()
    assert "starting" in cmd.research_start()
    assert "already in progress" in cmd.research_start() and "already in progress" in cmd()
    assert "Stopping" in cmd.research_stop()
    cmd._thread.join(2)
    assert not cmd._thread.is_alive() and len(started) == 1
    release.set()
    assert "not set up" in PortfolioCommand(lambda text: None).research_start()


# ---------------------------------------------------------------- the swing pipeline
class SeeingAgent:
    """A lead agent that records which research call it was shown and always says BUY."""

    name, role = "technical", LEAD

    def __init__(self):
        self.shown = {}

    def analyze(self, ctx):
        self.shown[ctx.symbol] = ctx.house()
        return AgentSignal(action="BUY", confidence=0.9, reasoning="t")


def swing(tmp_path, calls, held=None, watchlist=("TCS",)):
    agent = SeeingAgent()
    held = held or {}
    broker = FakeBroker(Portfolio(100_000.0, 100_000.0, {s: 1000.0 for s in held}, dict(held), 100_000.0))
    sessions = make_session_factory(f"sqlite:///{tmp_path / 't.db'}")
    told = []
    pipe = TradingPipeline(
        Settings(watchlist=watchlist, dry_run=True),
        [agent],
        broker,
        sessions,
        lambda sym: Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000, bar_date="2026-10-14"),
        notify=told.append,
        house_calls_fn=lambda: calls[0],
    )
    return pipe, agent, sessions, told


def decisions(sessions):
    with sessions() as s:
        return list(s.scalars(select(DecisionRecord).order_by(DecisionRecord.id)))


def test_open_swing_buy_calls_join_the_candidates_and_each_decision_records_why(tmp_path):
    calls = [parse_calls([body(RBL, ABC, CAMS, {**RBL, "ScripCode": "TCS"})])]
    pipe, agent, sessions, _ = swing(tmp_path, calls)
    results = pipe.run_once()
    assert [r.symbol for r in results] == ["TCS", "RBLBANK", "ABCAPITAL"]  # never CAMS: an intraday call
    sources = {d.symbol: d.candidate_source for d in decisions(sessions)}
    assert sources == {"TCS": "watchlist+research", "RBLBANK": "research", "ABCAPITAL": "research"}
    assert agent.shown["RBLBANK"].kind == "Swing Trade"
    rbl = next(d for d in decisions(sessions) if d.symbol == "RBLBANK")
    assert HouseCall.from_dict(json.loads(rbl.house_call_json)) == agent.shown["RBLBANK"]


def test_a_research_only_stock_skips_the_filters_and_what_they_would_say_is_recorded_not_acted_on(tmp_path):
    """Record-only (C2): RBLBANK is below its 50-day average, so the rule says HOLD and the screener filtered it out; the
    AI's BUY still stands. ABCAPITAL's screen check fails: recorded as unknown, and the cycle carries on."""
    calls = [parse_calls([body(RBL, ABC, {**RBL, "ScripCode": "TCS"})])]
    pipe, _, sessions, _ = swing(tmp_path, calls)
    below_ma50 = {"RBLBANK": Snapshot("RBLBANK", 90.0, 98.0, 95.0, 60.0, 1000, bar_date="2026-10-14")}
    pipe.snapshot_fn = lambda sym: below_ma50.get(sym) or Snapshot(sym, 100.0, 98.0, 95.0, 60.0, 1000, bar_date="2026-10-14")
    asked = []

    def screen(symbol, equity):
        asked.append((symbol, equity))
        if symbol == "ABCAPITAL":
            raise ValueError("scan failed")
        return "filtered out"

    pipe.screen_verdict_fn = screen
    pipe.run_once()
    rows = {d.symbol: (d.final_action, d.rule_action, d.screen_verdict) for d in decisions(sessions)}
    assert rows == {"TCS": ("BUY", "BUY", None), "RBLBANK": ("BUY", "HOLD", "filtered out"), "ABCAPITAL": ("BUY", "BUY", None)}
    assert sorted(asked) == [("ABCAPITAL", 100_000.0), ("RBLBANK", 100_000.0)]  # only research-only stocks are checked

    from src.learning.jobs import research_gate_line

    line = research_gate_line(sessions)
    assert "2 stock-sessions, 2 AI BUYs" in line and "turned 1 of those BUYs into HOLDs" in line and "filtered out 1" in line


def test_the_screeners_verdict_on_one_stock():
    from src.data.universe import Candidate, ScreenConfig, UniverseScreener

    day = date(2026, 10, 16)
    screener = UniverseScreener(list, lambda s: None, ScreenConfig(max_candidates=1, affordable_pct=0.1), today=lambda: day)
    screener._day, screener._ranked = day, [Candidate("A", 50.0, 0.1, 60, 1e8, 0.01), Candidate("B", 50.0, 0.1, 60, 1e8, 0.01), Candidate("C", 5_000.0, 0.1, 60, 1e8, 0.01)]  # fmt: skip
    verdicts = {s: screener.verdict(s, equity=10_000.0) for s in ("A", "B", "C", "D")}
    assert verdicts == {"A": "candidate", "B": "below the top 1", "C": "unaffordable", "D": "filtered out"}


def test_without_calls_the_pipeline_runs_exactly_as_before(tmp_path):
    pipe, agent, sessions, told = swing(tmp_path, [[]])
    assert [r.symbol for r in pipe.run_once()] == ["TCS"]
    [d] = decisions(sessions)
    assert d.candidate_source == "watchlist" and d.house_call_json is None and agent.shown == {"TCS": None} and told == []


def test_a_remembered_answer_is_reused_only_under_the_same_research_call(tmp_path):
    calls = [[]]
    pipe, agent, sessions, _ = swing(tmp_path, calls)
    pipe.run_once()
    pipe.run_once()
    assert len(agent.shown) == 1 and len(decisions(sessions)) == 2  # same bar, same (no) call: the AI is asked once
    agent.shown.clear()
    calls[0] = parse_calls([body({**RBL, "ScripCode": "TCS"})])  # a call on TCS is published mid-session
    pipe.run_once()
    assert "TCS" in agent.shown and agent.shown["TCS"] is not None  # asked again, once, now with the call
    agent.shown.clear()
    pipe.run_once()
    assert "TCS" not in agent.shown


def test_the_user_is_warned_once_when_the_house_turns_against_a_held_stock(tmp_path):
    calls = [parse_calls([body({**DMART_SELL, "ScripCode": "TCS"}, {**BSE_STOPPED, "ScripCode": "INFY"})])]
    pipe, _, _, told = swing(tmp_path, calls, held={"TCS": 10, "INFY": 5}, watchlist=("TCS", "INFY"))
    pipe.run_once()
    pipe.run_once()
    warnings = [m for m in told if m.startswith("HOUSE WARNING")]
    assert len(warnings) == 2 and "TCS" in warnings[1] and "open SELL call" in warnings[1]
    assert "INFY" in warnings[0] and "Stoploss Hit" in warnings[0]


# ---------------------------------------------------------------- the breakout engine
def test_todays_intraday_buy_calls_join_the_breakout_scan_and_must_still_break_out(tmp_path):
    bars = {"AAA": make_bars(BREAKOUT, volumes=VOLS), "CALLED": make_bars(BREAKOUT, volumes=VOLS), "FLAT": make_bars([100] * 5)}
    day = DAY.strftime("%d %b %Y")
    calls = parse_calls([body({**CAMS, "ScripCode": "CALLED", "PublishedDtm": f"{day} 09:20 AM"},
                              {**CAMS, "ScripCode": "FLAT", "PublishedDtm": f"{day} 09:20 AM"})])  # fmt: skip
    eng, broker, *_ = engine(tmp_path, bars, at(9, 45), prices={"AAA": 101.5, "CALLED": 101.5}, house_calls_fn=lambda: calls)
    assert "entered 2" in eng.step()
    assert sorted(h["symbol"] for h in broker.holdings()) == ["AAA", "CALLED"]  # FLAT has a call but no breakout
    tags = {r.symbol: r.house_call for r in signals(broker)}
    assert tags == {"AAA": None, "CALLED": "Intraday buy"}


def test_a_stock_the_house_says_to_sell_is_not_bought_on_a_breakout(tmp_path):
    day = DAY.strftime("%d %b %Y")
    calls = parse_calls([body({**DMART_SELL, "ScripCode": "AAA", "PublishedDtm": f"{day} 09:20 AM"})])
    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45), house_calls_fn=lambda: calls)
    assert "entered 0, skipped 1" in eng.step() and broker.holdings() == []
    [rec] = signals(broker)
    assert rec.outcome == "SKIPPED" and "SELL call" in rec.reason and rec.house_call == "Intraday sell"


def test_a_failing_call_reader_never_stops_the_breakout_engine(tmp_path):
    def broken():
        raise RuntimeError("db locked")

    eng, broker, *_ = engine(tmp_path, {"AAA": make_bars(BREAKOUT, volumes=VOLS)}, at(9, 45), house_calls_fn=broken)
    assert "entered 1" in eng.step()


@pytest.mark.parametrize("label", ["GAP-GO", "GAP-ONLY"])
def test_the_gap_engines_are_not_given_the_calls(label):
    import inspect

    import src.app.intraday_cli as cli

    source = inspect.getsource(cli.run_intraday)
    assert source.count("house_calls_fn") == 1 and label in inspect.getsource(cli)  # only the breakout engine gets them


def test_snapshot_times_are_handled_in_utc_and_ist(tmp_path):
    store = HouseCallStore(make_session_factory(f"sqlite:///{tmp_path / 'h.db'}"), now_fn=lambda: datetime.now(timezone.utc))
    store.replace(parse_calls([body(CAMS)]))
    [c] = store.current()
    assert c.published == datetime(2026, 10, 15, 9, 36, tzinfo=IST)


# ---------------------------------------------------------------- the browser read, the run and the command (no browser)
class FakeResponse:
    def __init__(self, text, fails=False):
        self._text, self.fails = text, fails

    def text(self):
        if self.fails:
            raise RuntimeError("aborted by the guard")
        return self._text


class FakeRequest:
    def __init__(self, url, text="[]", kind="xhr", response=True, fails=False):
        self.url, self.resource_type = url, kind
        self._response = FakeResponse(text, fails) if response else None

    def response(self):
        return self._response


class FakePage:
    """Just enough of a Playwright page: goto fires the given requests, then the page sits at `lands_on`."""

    def __init__(self, requests, lands_on="https://p.in/nv1/Dashboard/ResearchViewMore.html", idle_fails=False):
        self.requests, self.url, self.lands_on, self.idle_fails = requests, "", lands_on, idle_fails
        self.listeners, self.visited = [], []

    def on(self, event, fn):
        self.listeners.append(fn)

    def remove_listener(self, event, fn):
        self.listeners.remove(fn)

    def goto(self, url, wait_until=None):
        self.visited.append(url)
        self.url = self.lands_on
        for r in self.requests:
            for fn in list(self.listeners):
                fn(r)

    def wait_for_timeout(self, ms):
        pass

    def wait_for_load_state(self, state, timeout=None):
        if self.idle_fails:
            raise TimeoutError("never idle")


def test_one_read_returns_the_portals_own_data_responses_only():
    from src.portfolio.research_watch import read_research_once

    page = FakePage(
        [
            FakeRequest("https://p.in/SMDService/api/v1/SMD/ClaimV3", body(CAMS)),
            FakeRequest("https://other.site/track", "{}"),  # another site
            FakeRequest("https://p.in/CIAMS/api/v1/UserManagement/isvalidsession", "{}"),  # a login call
            FakeRequest("https://p.in/nv1/app.js", "code", kind="script"),
            FakeRequest("https://p.in/SMDService/api/v1/SMD/ClaimV3", response=False),
            FakeRequest("https://p.in/SMDService/api/v1/SMD/ClaimV3", fails=True),
        ],
        idle_fails=True,
    )
    assert read_research_once(page, "https://p.in/nv1/") == [body(CAMS)]
    assert page.visited == ["https://p.in/nv1/Dashboard/ResearchViewMore.html"] and page.listeners == []
    assert read_research_once(FakePage([], lands_on="https://p.in/nv1/login/login.html"), "https://p.in/nv1/") is None


class FakeRun:
    """Stands in for PortalRun: entering it is the login (or its failure), leaving it the logout."""

    fail = None

    def __init__(self, *args, **kwargs):
        self.base, self.blocked, self.logged_out = "https://p.in/nv1/", [], None

    def __enter__(self):
        if FakeRun.fail is not None:
            raise FakeRun.fail
        return "page"

    def __exit__(self, *exc):
        self.logged_out = True


@pytest.fixture
def portal_run(monkeypatch, tmp_path):
    import src.portfolio.runs as runs

    monkeypatch.setattr(runs, "PortalRun", FakeRun)
    monkeypatch.setattr(runs, "watch_research", lambda page, base, on_bodies, stop, **kw: f"day over ({on_bodies([body(CAMS)])})")
    FakeRun.fail = None
    yield runs, runs.RunStore(tmp_path / "pd")
    FakeRun.fail = None


def test_the_watch_run_logs_in_once_and_out_and_reports_how_it_ended(portal_run):
    runs, store = portal_run
    result = runs.watch_portfolio_research(None, lambda: "1234", store, len, threading.Event())
    assert result["ended"] == "day over (1)" and result["logged_out"] is True and not store.lock_file.exists()


def test_the_watch_run_keeps_the_portfolio_runs_safety_stops(portal_run):
    from src.portfolio.session import LoginFailed
    from src.portfolio.telegram_otp import OtpTimeout

    runs, store = portal_run
    FakeRun.fail = OtpTimeout("no OTP within 180s")
    assert "no OTP" in runs.watch_portfolio_research(None, lambda: "", store, len, threading.Event())["error"]
    FakeRun.fail = RuntimeError("boom")
    assert "unexpected RuntimeError" in runs.watch_portfolio_research(None, lambda: "", store, len, threading.Event())["error"]
    FakeRun.fail = LoginFailed("mpin", "Invalid MPIN")
    assert "login stopped at mpin" in runs.watch_portfolio_research(None, lambda: "", store, len, threading.Event())["error"]
    assert "MPIN step failed" in runs.watch_portfolio_research(None, lambda: "", store, len, threading.Event())["error"]
    store.clear_mpin_block()
    with store.lock():  # another run in progress
        assert "in progress" in runs.watch_portfolio_research(None, lambda: "", store, len, threading.Event())["error"]
    store.touch_lock()  # no lock file: harmless


def test_the_research_command_reports_the_end_of_the_day_or_why_it_could_not_run(monkeypatch):
    import src.app.portfolio_agent as agent

    told = []
    monkeypatch.setattr(agent.vault, "load", lambda: "creds")
    monkeypatch.setattr(agent, "watch_portfolio_research", lambda *a, **k: {"ended": "day over (15:35 IST)"})
    agent.run_research_watch(lambda: "", told.append, len, threading.Event())
    monkeypatch.setattr(agent, "watch_portfolio_research", lambda *a, **k: {"error": "no OTP: timeout"})
    agent.run_research_watch(lambda: "", told.append, len, threading.Event())

    def missing():
        raise agent.vault.MissingCredentials("no login saved")

    monkeypatch.setattr(agent.vault, "load", missing)
    assert agent.run_research_watch(lambda: "", told.append, len, threading.Event()) == {"error": "no login saved"}
    assert "ended: day over" in told[0] and "did not run: no OTP" in told[1] and "no login saved" in told[2]


def test_a_crashing_watch_is_reported_not_raised():
    told = []

    def runner(*args):
        raise RuntimeError("boom")

    cmd = PortfolioCommand(told.append, research_sink=len, research_runner=runner)
    cmd.research_start()
    cmd._thread.join(2)
    assert told and "Research watch failed: RuntimeError" in told[-1]


def test_a_response_that_is_a_json_string_inside_a_json_string_is_still_read():
    assert looks_like_calls(json.dumps(body(CAMS)))
