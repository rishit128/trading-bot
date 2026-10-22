"""The news-catalyst strategy: reading the exchange feeds, the AI label's materiality tests, the entry rules A, B and C, the
filing watcher, and the engine against the real paper broker (entries, the market filter, the trade cap, the minimum
position, the break-even stop and the after-close measurement of every filing's move)."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.angel_quotes import NewsContext
from src.data.filings import Filing, bse_nse_map, excerpt, nse_symbol_from_link, parse_bse, parse_nse
from src.data.india import IST
from src.database import IntradaySignalRecord, NewsFilingRecord, make_session_factory
from src.engine.paper_broker import PaperBroker
from src.news import strategy as ns
from src.news.engine import NewsEngine, filing_moves
from src.news.labels import FilingLabel, first_session, materiality, profit_change_pct, routine_subject
from src.news.watcher import FilingWatcher
from tests.test_paper_broker import FakeClock, FakeFeed

DAY = datetime(2026, 9, 21, tzinfo=IST)  # a Monday
PREV = datetime(2026, 9, 18, tzinfo=IST)  # the Friday before


def bars(day, closes, volumes=None, start=(9, 15), first_open=None):
    """5-minute bars from `start`; open = previous close (the first bar's open is `first_open`, default its close)."""
    idx = pd.DatetimeIndex([day.replace(hour=start[0], minute=start[1]) + timedelta(minutes=5 * i) for i in range(len(closes))])
    opens = [first_open if first_open is not None else closes[0]] + list(closes[:-1])
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) + 0.1 for o, c in zip(opens, closes)],
            "Low": [min(o, c) - 0.1 for o, c in zip(opens, closes)],
            "Close": closes,
            "Volume": volumes or [1000.0] * len(closes),
        },
        index=idx,
    )


def history(today_closes, prev_close=100.0, volumes=None, first_open=None, prev_volumes=None):
    """The previous session (ending at `prev_close`) and today, as one frame, like the 5-day download."""
    n = len(today_closes)
    prev = bars(PREV, [prev_close] * n, volumes=prev_volumes)
    return pd.concat([prev, bars(DAY, today_closes, volumes=volumes, first_open=first_open)])


# Gap of +3% (100 -> 103), opening range 102.9..103.6 (09:15-09:30), then a close above it on the 09:30 bar.
GAP_UP = [103.0, 103.5, 103.3, 104.0, 104.2]

# ------------------------------------------------------------------ feeds


BSE_XML = (
    b"\xef\xbb\xbf<?xml version='1.0' encoding='utf-8'?><rss><channel>"
    b"<item><title>Kajaria Ceramics Ltd (500233)</title><link>https://bse/a.pdf</link><scripcode>500233</scripcode>"
    b"<description>Financial Results for the quarter</description><pubDate>21-Oct-2026 16:24:42</pubDate></item>"
    b"<item><title>Some Fund (999999)</title><link>https://bse/b.pdf</link><scripcode>999999</scripcode>"
    b"<description>NAV as on 19-10-2026</description><pubDate>21-Oct-2026 16:20:00</pubDate></item>"
    b"</channel></rss>"
)
NSE_XML = (
    b"<rss><channel><item><title>E2E Networks Limited</title>"
    b"<link>https://nsearchives.nseindia.com/corporate/E2E_21102026162502_E2E_Trans_Final.pdf</link>"
    b"<description>E2E Networks Limited has informed the Exchange about Transcript |SUBJECT: Con. Call Updates</description>"
    b"<pubDate>21-Oct-2026 16:25:25</pubDate></item></channel></rss>"
)


def test_the_bse_feed_keeps_only_items_that_map_to_an_nse_stock_with_their_ist_time_in_utc():
    [f] = parse_bse(BSE_XML, {"500233": "KAJARIACER"})
    assert (f.source, f.symbol, f.company, f.link) == ("BSE", "KAJARIACER", "Kajaria Ceramics Ltd", "https://bse/a.pdf")
    assert f.published == datetime(2026, 10, 21, 16, 24, 42, tzinfo=IST).astimezone(timezone.utc)


def test_the_nse_feed_takes_the_symbol_from_the_attachment_name_and_keeps_the_filing_category():
    [f] = parse_nse(NSE_XML)
    assert f.symbol == "E2E" and f.subject.endswith("about Transcript (Con. Call Updates)")
    assert routine_subject(f.subject) == "transcript"
    assert nse_symbol_from_link("https://x/other/file.pdf") is None


def test_the_symbol_map_pairs_a_bse_code_with_the_nse_stock_of_the_same_symbol_only():
    master = [
        {"exch_seg": "NSE", "symbol": "RELIANCE-EQ", "name": "RELIANCE", "token": "2885"},
        {"exch_seg": "BSE", "symbol": "RELIANCE", "name": "RELIANCE", "token": "500325"},
        {"exch_seg": "BSE", "symbol": "920DCBBL34", "name": "920DCBBL34", "token": "976178"},  # a bond
        {"exch_seg": "NSE", "symbol": "NIFTY", "name": "NIFTY", "token": "1"},  # not an -EQ stock
    ]
    assert bse_nse_map(master) == {"500325": "RELIANCE"}


def test_the_excerpt_keeps_the_opening_and_puts_the_results_table_before_other_figures():
    deal = "\n".join(f"the acquisition consideration of Rs {i} crore" for i in range(2000))
    table = "Particulars  30.09.2026  30.06.2026  30.09.2025\nProfit for the period  747  983  644"
    text = excerpt("COVER LETTER " + "x" * 3000 + "\n" + deal + "\n" + table)
    assert text.startswith("COVER LETTER") and "Profit for the period  747  983  644" in text
    assert len(text) <= 12_000


# ------------------------------------------------------------------ labels


def label(**kw):
    base = dict(event_type="results", direction="up", surprise="large", key_numbers="", reason="")
    return FilingLabel(**{**base, **kw})


def test_routine_subjects_are_recognised_and_news_is_not():
    assert routine_subject("Closure of Trading Window") == "trading window"
    assert routine_subject("Financial Results for the quarter ended September 30, 2026") is None


def test_the_profit_change_is_computed_from_the_two_figures_including_losses():
    assert profit_change_pct(label(profit_quarter=747, profit_year_ago=644)) == pytest.approx(15.99, abs=0.01)
    assert profit_change_pct(label(profit_quarter=50, profit_year_ago=-10)) == 100.0
    assert profit_change_pct(label(profit_quarter=-5, profit_year_ago=40)) == -100.0
    assert profit_change_pct(label(profit_quarter=-30, profit_year_ago=-60)) == pytest.approx(50.0)  # a smaller loss
    assert profit_change_pct(label(profit_quarter=10)) is None


def test_results_are_material_only_with_20_percent_and_an_agreeing_direction_and_surprise():
    no_cap = lambda: pytest.fail("market value is only asked for an order win")  # noqa: E731
    assert materiality(label(profit_quarter=130, profit_year_ago=100), no_cap).material
    assert not materiality(label(profit_quarter=116, profit_year_ago=100), no_cap).material  # +16%
    assert not materiality(label(profit_quarter=130, profit_year_ago=100, direction="down"), no_cap).material
    assert not materiality(label(profit_quarter=130, profit_year_ago=100, surprise="none"), no_cap).material
    assert not materiality(label(profit_quarter=130, profit_year_ago=100, surprise="unclear"), no_cap).material
    assert materiality(label(profit_quarter=60, profit_year_ago=100, direction="down", surprise="moderate"), no_cap).material


def test_an_order_win_needs_two_percent_of_the_market_value_and_other_events_a_large_surprise():
    order = label(event_type="order_win", surprise="moderate", order_value_crore=300)
    assert materiality(order, lambda: 10_000.0).material  # 3%
    assert not materiality(order, lambda: 20_000.0).material  # 1.5%
    assert not materiality(order, lambda: None).material
    assert materiality(label(event_type="approval", surprise="large"), lambda: None).material
    assert not materiality(label(event_type="approval", surprise="moderate"), lambda: None).material
    assert not materiality(label(event_type="routine", surprise="large"), lambda: None).material


def test_a_filing_after_the_close_or_on_a_closed_day_belongs_to_the_next_session():
    weekday = lambda when: when.weekday() < 5  # noqa: E731
    friday_evening = datetime(2026, 9, 18, 18, 0, tzinfo=IST)
    assert first_session(friday_evening, weekday) == "2026-09-21"  # Monday
    assert first_session(datetime(2026, 9, 21, 8, 0, tzinfo=IST), weekday) == "2026-09-21"
    assert first_session(datetime(2026, 9, 21, 15, 45, tzinfo=IST), weekday) == "2026-09-22"


# ------------------------------------------------------------------ rules


def today(df):
    return df[pd.DatetimeIndex(df.index).date == DAY.date()]


def test_rule_a_buys_the_first_close_above_the_range_and_vwap_with_the_range_low_as_stop():
    check = ns.rule_a("AAA", today(history(GAP_UP)), 100.0)
    assert check.status == ns.FIRED and check.setup.signal_price == 104.0
    assert check.setup.stop == pytest.approx(102.9) and check.setup.bar_time.time().strftime("%H:%M") == "09:30"


def test_rule_a_refuses_a_gap_outside_2_to_8_percent_and_waits_for_the_breakout():
    assert ns.rule_a("AAA", today(history([101.0, 101.2, 101.1, 102.0])), 100.0).reason.startswith("gap +1.0%")
    assert ns.rule_a("AAA", today(history([109.0, 109.2, 109.1, 110.0])), 100.0).status == ns.NO
    waiting = ns.rule_a("AAA", today(history([103.0, 103.5, 103.3, 103.4])), 100.0)
    assert waiting.status == ns.WAITING  # 09:30 bar closed inside the range: may still break out before 10:30


def test_rule_a_skips_a_stop_too_far_or_too_close():
    wide = [103.0, 105.5, 102.0, 106.0]  # range low 101.9: stop 3.9% below 106
    assert "stop" in ns.rule_a("AAA", today(history(wide)), 100.0).reason


def test_rule_b_needs_a_075_percent_move_on_3x_the_usual_volume_and_stops_at_the_filing_price():
    df = history([100.0] * 10 + [101.0], volumes=[1000.0] * 10 + [5000.0], prev_volumes=[1000.0] * 11)
    day, filed = today(df), DAY.replace(hour=10, minute=6)  # inside the 10:05-10:10 bar, the last one
    usual = ns.normal_volume(df, day.index[-1], DAY.date())
    assert usual == 1000.0  # the previous session's 10:05 bar
    check = ns.rule_b("AAA", day, filed, 100.0, usual)
    assert check.status == ns.FIRED and check.setup.stop == 100.0 and check.ref_price == 100.0
    quiet = history([100.0] * 10 + [101.0], volumes=[1000.0] * 11)
    assert "volume" in ns.rule_b("AAA", today(quiet), filed, 100.0, 1000.0).reason
    small = history([100.0] * 10 + [100.5], volumes=[1000.0] * 10 + [5000.0])
    assert "needs +0.75%" in ns.rule_b("AAA", today(small), filed, 100.0, 1000.0).reason
    assert ns.rule_b("AAA", day.iloc[:10], filed, 100.0, 1000.0).status == ns.WAITING  # that bar has not closed


def test_the_market_filter_is_the_share_above_the_previous_close_at_0930():
    up, down = today(history(GAP_UP)), today(history([99.0, 99.0, 98.0, 98.0]))
    assert ns.breadth({"A": up, "B": down, "C": down}, {"A": 100, "B": 100, "C": 100}) == pytest.approx(1 / 3)
    assert ns.breadth({"A": up.iloc[:2], "B": up.iloc[:2]}, {"A": 100, "B": 100}) is None  # no 09:25 bar yet


def test_the_stop_moves_to_the_entry_after_one_risk():
    assert ns.breakeven_due(104.0, 102.0, 106.0) and not ns.breakeven_due(104.0, 102.0, 105.9)


def test_moves_are_measured_from_the_filing_price_or_for_overnight_news_from_the_previous_close():
    day = today(history([100.0] * 40 + [110.0]))
    during = filing_moves(day, 100.0, DAY.replace(hour=9, minute=40), True)
    assert during["ref_price"] == 100.0 and during["move_5m"] == 0.0 and during["move_close"] == pytest.approx(0.10)
    overnight = filing_moves(day, 95.0, DAY - timedelta(hours=12), False)
    assert overnight["ref_price"] == 95.0 and overnight["move_5m"] == pytest.approx(100 / 95 - 1)


# ------------------------------------------------------------------ the watcher


class FakeReader:
    def __init__(self, filings, text="PAT 30.09.2026: 130 vs 30.09.2025: 100"):
        self.filings, self.text = filings, text

    def read(self):
        return list(self.filings)

    def attachment_text(self, link):
        return self.text


class FakeLLM:
    def __init__(self, answer=None, fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def structured_call(self, prompt, model_class, schema, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no model answered")
        return self.answer


def filing(symbol="AAA", subject="Financial Results", minutes=0, source="BSE", link=None):
    when = (DAY.replace(hour=8, minute=0) + timedelta(minutes=minutes)).astimezone(timezone.utc)
    return Filing(source, symbol, f"{symbol} Ltd", subject, link or f"https://{source}/{symbol}{minutes}.pdf", when)


def watcher(tmp_path, filings, llm, universe=("AAA", "BBB")):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'w.db'}")
    notes = []
    w = FilingWatcher(
        sessions, FakeReader(filings), llm, lambda: set(universe), lambda when: when.weekday() < 5, lambda s: 5000.0, notes.append
    )
    return w, sessions, notes


def rows(sessions):
    with sessions() as s:
        return list(s.scalars(select(NewsFilingRecord).order_by(NewsFilingRecord.id)))


def test_the_watcher_stores_each_universe_filing_once_marks_routine_ones_and_labels_the_rest(tmp_path):
    llm = FakeLLM(label(profit_quarter=130, profit_year_ago=100, raw_model="m1"))
    feed = [filing(), filing("BBB", "Closure of Trading Window", 1), filing("ZZZ", minutes=2)]
    w, sessions, notes = watcher(tmp_path, feed, llm)
    assert w.poll_once() == 2 and w.poll_once() == 0  # ZZZ is outside the universe; nothing is stored twice
    news, routine = rows(sessions)
    assert (news.label_status, news.material, news.session, news.ai_model) == ("LABELLED", True, "2026-09-21", "m1")
    assert json.loads(news.numbers_json)["profit_change_pct"] == pytest.approx(30.0)
    assert (routine.label_status, routine.material) == ("ROUTINE", False) and llm.calls == 1
    assert any("material filing: AAA results" in n for n in notes)


def test_the_live_price_is_stored_when_a_filing_is_first_seen(tmp_path):
    w, sessions, _ = watcher(tmp_path, [filing()], FakeLLM())
    w.price_fn = lambda symbol: 123.4
    w.poll_once()
    assert rows(sessions)[0].seen_price == 123.4


def test_the_same_filing_from_the_other_exchange_is_a_duplicate_and_an_ai_failure_is_recorded(tmp_path):
    feed = [filing(), filing(source="NSE", minutes=1)]
    w, sessions, _ = watcher(tmp_path, feed, FakeLLM(fail=True))
    w.poll_once()
    [only] = rows(sessions)
    assert only.source == "BSE" and only.label_status == "FAILED" and only.material is False


def test_the_watcher_polls_every_minute_in_the_plans_window_and_every_five_minutes_otherwise(tmp_path):
    w, *_ = watcher(tmp_path, [], FakeLLM())
    w.now_fn = lambda: DAY.replace(hour=10)
    assert w.interval() == 60.0
    w.now_fn = lambda: DAY.replace(hour=20)
    assert w.interval() == 300.0


# ------------------------------------------------------------------ the engine


def engine(tmp_path, data, now_ist, universe=("AAA",), cash=25_000.0, price=104.0, **kw):
    feed, clock = FakeFeed({s: price for s in data}), FakeClock()
    now = [now_ist.astimezone(timezone.utc)]
    broker = PaperBroker(
        make_session_factory(f"sqlite:///{tmp_path / 'n.db'}"),
        feed,
        clock,
        initial_cash=cash,
        fees=lambda s, v: 0.0,
        slippage=0.0,
        now_fn=lambda: now[0],
    )
    messages = []

    def fetch(symbols):
        cut = pd.Timestamp(now[0])
        return {s: data[s][pd.DatetimeIndex(data[s].index) <= cut] for s in symbols if s in data}

    eng = NewsEngine(
        broker, list(universe), clock, fetch_bars=fetch, notify=messages.append, live=True, now_fn=lambda: now[0], **kw
    )
    return eng, broker, feed, clock, now, messages


def add_filing(broker, symbol="AAA", published=None, **kw):
    fields = dict(
        key=f"k-{symbol}-{published}",
        source="BSE",
        symbol=symbol,
        company=symbol,
        subject="Financial Results",
        link="https://x",
        published=published or (PREV.replace(hour=18)).astimezone(timezone.utc),
        session="2026-09-21",
        label_status="LABELLED",
        event_type="results",
        direction="up",
        surprise="large",
        material=True,
    )
    fields.update(kw)
    with broker.sessions() as s:
        row = NewsFilingRecord(**fields)
        s.add(row)
        s.commit()
        return row.id


def filing_row(broker, row_id):
    with broker.sessions() as s:
        return s.get(NewsFilingRecord, row_id)


def at(h, m):
    return DAY.replace(hour=h, minute=m, second=20)


def test_overnight_news_with_a_gap_and_a_breakout_is_bought_with_the_range_low_as_stop(tmp_path):
    eng, broker, *_, messages = engine(tmp_path, {"AAA": history(GAP_UP)}, at(9, 35))
    row = add_filing(broker)
    assert "entered 1" in eng.step()
    [h] = broker.holdings()
    assert h["symbol"] == "AAA" and h["stop"] == pytest.approx(102.9)
    r = filing_row(broker, row)
    assert (r.rule_status, r.entry_kind, r.ref_price) == (ns.FIRED, "A", 100.0) and "ENTERED" in r.rule_reason
    assert any("[NEWS] BUY AAA" in m for m in messages)
    # 0.5% of Rs 25,000 = Rs 125 at a 1.10 stop distance = 113 shares, but the 20% cap (Rs 5,000) allows 48
    assert h["qty"] == int(25_000 * 0.20 / 104.0)


def test_angel_one_context_is_stored_and_a_buy_near_the_upper_circuit_is_refused(tmp_path):
    eng, broker, *_ = engine(
        tmp_path, {"AAA": history(GAP_UP)}, at(9, 35), context_fn=lambda s, p: NewsContext(None, {"spread_pct": 0.1})
    )
    row = add_filing(broker)
    assert "entered 1" in eng.step()
    assert json.loads(filing_row(broker, row).context_json) == {"spread_pct": 0.1}
    (tmp_path / "x").mkdir()
    eng, broker, *_ = engine(
        tmp_path / "x", {"AAA": history(GAP_UP)}, at(9, 35), context_fn=lambda s, p: NewsContext("near the upper circuit")
    )
    row = add_filing(broker)
    assert "entered 0" in eng.step() and broker.holdings() == []
    assert filing_row(broker, row).rule_reason == "near the upper circuit"


def test_a_failing_angel_one_quote_does_not_stop_the_entry(tmp_path):
    def broken(symbol, price):
        raise RuntimeError("rate limit")

    eng, broker, *_ = engine(tmp_path, {"AAA": history(GAP_UP)}, at(9, 35), context_fn=broken)
    add_filing(broker)
    assert "entered 1" in eng.step()


def test_no_long_is_taken_when_under_40_percent_of_the_universe_is_up_at_0930(tmp_path):
    weak = history([99.0, 99.0, 98.0, 98.0, 98.0])
    data = {"AAA": history(GAP_UP), "BBB": weak, "CCC": weak}
    eng, broker, *_ = engine(tmp_path, data, at(9, 35), universe=("AAA", "BBB", "CCC"))
    row = add_filing(broker)
    eng.step()
    assert broker.holdings() == [] and filing_row(broker, row).rule_reason.startswith("market filter: 33%")


def test_down_routine_and_immaterial_filings_get_a_final_no_with_the_reason(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": history(GAP_UP)}, at(9, 35))
    down = add_filing(broker, direction="down")
    routine = add_filing(broker, published=PREV.replace(hour=19).astimezone(timezone.utc), label_status="ROUTINE",
                         material=False, material_reason="routine by subject (trading window)")  # fmt: skip
    pending = add_filing(broker, published=PREV.replace(hour=20).astimezone(timezone.utc), label_status="PENDING", material=None)
    eng.step()
    assert "long only" in filing_row(broker, down).rule_reason and filing_row(broker, down).rule_status == ns.NO
    assert filing_row(broker, routine).rule_reason == "routine by subject (trading window)"
    assert filing_row(broker, pending).rule_status is None  # the AI has not read it yet


def test_at_most_the_daily_trade_cap_is_entered_and_the_rest_are_recorded(tmp_path):
    data = {"AAA": history(GAP_UP), "BBB": history(GAP_UP)}
    eng, broker, *_ = engine(tmp_path, data, at(9, 35), universe=("AAA", "BBB"), max_trades_per_day=1)
    add_filing(broker, "AAA")
    second = add_filing(broker, "BBB")
    eng.step()
    assert len(broker.holdings()) == 1 and "1 trades already today" in filing_row(broker, second).rule_reason


def test_a_position_under_rs_3000_is_skipped_and_recorded(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": history(GAP_UP)}, at(9, 35), cash=10_000.0)  # 20% cap = Rs 2,000
    add_filing(broker)
    eng.step()
    assert broker.holdings() == []
    with broker.sessions() as s:
        [sig] = s.scalars(select(IntradaySignalRecord))
    assert sig.outcome == "SKIPPED" and "under the Rs 3,000 minimum" in sig.reason


def test_a_rule_that_fired_on_an_earlier_bar_is_not_chased(tmp_path):
    eng, broker, *_ = engine(tmp_path, {"AAA": history(GAP_UP + [104.3, 104.4])}, at(9, 45))  # label arrived late
    row = add_filing(broker)
    eng.step()
    assert broker.holdings() == [] and "seen too late" in filing_row(broker, row).rule_reason


def test_the_stop_moves_to_the_entry_once_the_trade_is_up_one_risk(tmp_path):
    closes = GAP_UP + [104.5, 106.0]  # entry 104.0, stop 102.9: 1R = 1.1, first reached by the 09:45 bar's high (106.1)
    eng, broker, feed, clock, now, messages = engine(tmp_path, {"AAA": history(closes)}, at(9, 35))
    add_filing(broker)
    eng.step()
    now[0] = at(9, 45).astimezone(timezone.utc)
    eng.step()
    assert broker.holdings()[0]["stop"] == pytest.approx(102.9)  # the 09:40 bar's high 104.6: not yet 1R
    now[0] = at(9, 50).astimezone(timezone.utc)
    eng.step()
    assert broker.holdings()[0]["stop"] == pytest.approx(104.0) and any("stop moved to the entry" in m for m in messages)


def test_after_the_close_every_filing_gets_its_moves_and_undecided_ones_a_final_verdict(tmp_path):
    eng, broker, feed, clock, now, messages = engine(tmp_path, {"AAA": history(GAP_UP * 15)}, at(9, 35))
    row = add_filing(broker, label_status="PENDING", material=None)
    clock.open = False
    now[0] = at(15, 40).astimezone(timezone.utc)
    eng._day = "2026-09-21"
    eng.step()
    r = filing_row(broker, row)
    assert r.ref_price == 100.0 and r.move_close == pytest.approx(0.042) and r.rule_status == ns.NO
    assert any("2026-09-21 filings: 1 on universe stocks" in m for m in messages)


def test_news_filed_during_the_session_is_traded_by_rule_b_with_the_stored_time_read_as_utc(tmp_path):
    # SQLite returns the stored UTC time without its zone: read as IST it would be 04:36, an "overnight" filing for rule A
    closes, volumes = [100.0] * 10 + [101.0], [1000.0] * 10 + [5000.0]
    data = {"AAA": history(closes, prev_close=99.5, volumes=volumes, prev_volumes=[1000.0] * 11)}  # up at 09:30: filter passes
    eng, broker, *_ = engine(tmp_path, data, at(10, 10), price=101.0)
    row = add_filing(broker, published=DAY.replace(hour=10, minute=6).astimezone(timezone.utc))
    eng.step()
    r = filing_row(broker, row)
    assert (r.entry_kind, r.rule_status, r.ref_price) == ("B", ns.FIRED, 100.0)
    assert broker.holdings()[0]["stop"] == pytest.approx(100.0)


def test_the_news_models_default_to_the_benchmarked_list_and_can_be_set(monkeypatch):
    from src.config import NEWS_MODELS, load_settings

    monkeypatch.delenv("NEWS_MODELS", raising=False)
    assert load_settings().news_models == NEWS_MODELS and NEWS_MODELS[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    monkeypatch.setenv("NEWS_MODELS", " a/b:free , c/d:free ")
    assert load_settings().news_models == ("a/b:free", "c/d:free")


def test_an_order_win_is_judged_on_its_size_even_when_the_surprise_is_unclear():
    order = label(event_type="order_win", surprise="unclear", order_value_crore=300)
    assert materiality(order, lambda: 10_000.0).material
