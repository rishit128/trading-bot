"""orb-v7: a breakout is bought only in a leader while the market is rising, and a scan tries its strongest stocks first."""

from datetime import date

import pandas as pd
import pytest

from src.intraday import strategy as st
from src.intraday.engine import IntradayEngine
from tests.intraday_helpers import BREAKOUT, VOLS, make_bars
from tests.test_intraday_engine import at, engine

DAY = date(2026, 11, 5)
IST = "Asia/Kolkata"


def bars(prev_close, now):
    """A stock's last bar of 4 November and its latest bar today (None: nothing today)."""
    rows = [("2026-11-04 15:25", prev_close)] + ([("2026-11-05 09:20", now)] if now is not None else [])
    index = pd.DatetimeIndex([t for t, _ in rows]).tz_localize(IST)
    return pd.DataFrame({"Open": [p for _, p in rows], "Close": [p for _, p in rows]}, index=index)


def market(up, down, **extra):
    stocks = {f"U{i}": bars(100, 101) for i in range(up)}
    stocks.update({f"D{i}": bars(100, 99) for i in range(down)})
    stocks.update(extra)
    return stocks


def test_the_days_gain_is_the_latest_price_over_the_previous_close():
    gains = st.day_gains({"UP": bars(100, 103), "NEW": bars(100, None).iloc[:0], "QUIET": bars(100, None)}, DAY)
    assert gains == {"UP": pytest.approx(0.03)}  # no bar today, or no earlier session: not counted


def test_a_breakout_is_refused_in_a_weak_market_and_in_a_stock_that_is_not_leading():
    weak = st.orb_leaders(market(up=5, down=5, LEAD=bars(100, 103)), DAY)
    assert "weak market: 55% of stocks are up" in weak("LEAD")
    rising = st.orb_leaders(market(up=6, down=3, LEAD=bars(100, 102), LAG=bars(100, 101.9)), DAY)
    assert rising("LEAD") is None  # exactly 2% up, 8 of 11 stocks up
    assert "not a leader: +1.9%" in rising("LAG")
    assert "not a leader: unknown" in rising("ABSENT")


def test_nothing_is_bought_when_most_stocks_have_no_previous_close():
    blind = {f"N{i}": bars(100, None) for i in range(6)}
    assert "cannot be measured (previous close for only 1 of 7)" in st.orb_leaders({**blind, "LEAD": bars(100, 103)}, DAY)("LEAD")


def test_a_scan_gives_its_free_positions_to_the_highest_scores(tmp_path):
    stocks = {s: make_bars(BREAKOUT, volumes=VOLS) for s in ("AAA", "BBB", "CCC")}
    eng, broker, *_ = engine(
        tmp_path,
        stocks,
        at(9, 40),
        prices={s: 101.5 for s in stocks},
        max_positions=2,
        priority_fn=lambda scan, day: {"CCC": 0.05, "BBB": 0.03},  # AAA has no score: tried last
    )
    eng.universe = list(stocks)
    assert "entered 2" in eng.step()
    assert {h["symbol"] for h in broker.holdings()} == {"CCC", "BBB"}


def test_the_market_check_still_sees_stocks_already_handled_today(tmp_path):
    stocks = {s: make_bars(BREAKOUT, volumes=VOLS) for s in ("AAA", "BBB", "CCC")}
    seen = []

    def refusals(scan, day):
        seen.append(sorted(scan))
        return lambda symbol: None

    eng, *_ = engine(tmp_path, stocks, at(9, 40), prices={s: 101.5 for s in stocks}, refusals_fn=refusals)
    eng.universe = list(stocks)
    eng._new_day(at(9, 40).date().isoformat())
    eng._done_today.update({"AAA", "BBB"})  # skipped earlier today: not scanned again, but still part of the market
    assert "scanned 1 stocks" in eng.step()
    assert seen == [["AAA", "BBB", "CCC"]]


def test_the_breakout_account_is_wired_to_the_leaders_rule():
    import inspect

    from src.app import intraday_cli

    source = inspect.getsource(intraday_cli.run_intraday)
    assert "refusals_fn=partial(orb_leaders, sectors=sectors)" in source and "priority_fn=day_gains" in source
    assert IntradayEngine.__dataclass_fields__["priority_fn"].default is None


# -- the industry groups and the weak market (orb-v7, gaponly-v6) --------------------------------------------------------------
BANKS = {f"B{i}": "Banks" for i in range(3)}


def test_a_groups_gain_is_its_median_stock_and_a_small_group_is_not_read():
    gains = {"B0": 0.01, "B1": -0.02, "B2": 0.03, "T0": 0.05, "T1": 0.04, "LONE": 0.02}
    sectors = {**BANKS, "T0": "Tech", "T1": "Tech"}
    assert st.sector_gains(gains, sectors) == {"Banks": 0.01}  # Tech has two stocks; LONE has no group
    assert st.weak_sector("B0", sectors, {"Banks": -0.004}) == "weak sector: the median Banks stock is -0.4% on the day"
    assert st.weak_sector("B0", sectors, {"Banks": 0.001}) is None
    assert st.weak_sector("T0", sectors, {"Banks": -0.01}) is None and st.weak_sector("LONE", sectors, {}) is None


def test_a_leader_in_a_falling_group_is_not_bought():
    falling = {"B0": bars(100, 103), "B1": bars(100, 99), "B2": bars(100, 98)}
    refusal = st.orb_leaders(market(up=8, down=1, **falling), DAY, BANKS)
    assert "weak sector: the median Banks stock is -1.0%" in refusal("B0")
    assert st.orb_leaders(market(up=8, down=1, **falling), DAY)("B0") is None  # no sector list: no sector check


def test_gap_only_buys_nothing_in_a_weak_market_or_a_falling_group():
    def gapper(now):
        frame = bars(100, now)
        return pd.concat([bars(100, None).set_axis(pd.DatetimeIndex(["2026-11-03 15:25"]).tz_localize(IST)), frame])

    assert st.gap_only_refusals(market(up=3, down=6, G=gapper(103)), DAY)("G") is None  # 4 of 10 up: just enough
    assert "weak market" in st.gap_only_refusals(market(up=2, down=7, G=gapper(103)), DAY)("G")  # 3 of 10 up
    falling = {"B0": gapper(103), "B1": bars(100, 99), "B2": bars(100, 98)}
    assert "weak sector" in st.gap_only_refusals(market(up=8, down=1, **falling), DAY, BANKS)("B0")
    assert st.gap_only_refusals(market(up=8, down=1, **falling), DAY)("B0") is None


def test_the_sector_list_is_cached_and_only_the_missing_stocks_are_looked_up(tmp_path):
    from src.data.sectors import load_sectors

    asked = []

    def lookup(symbol):
        asked.append(symbol)
        return {"ABB": "Industrials", "TCS": "Technology"}.get(symbol)  # Yahoo has no sector for NEW

    first = load_sectors(tmp_path, ["ABB", "TCS", "NEW"], lookup, background=False)
    assert first == {"ABB": "Industrials", "TCS": "Technology"} and asked == ["ABB", "TCS", "NEW"]
    again = load_sectors(tmp_path, ["ABB", "TCS"], lookup, background=False)
    assert again == first and asked == ["ABB", "TCS", "NEW"]  # from the cache: nothing asked
    (tmp_path / "sectors.json").write_text("not json")
    assert load_sectors(tmp_path, ["ABB"], lookup, background=False) == {"ABB": "Industrials"}  # a broken cache is rebuilt
