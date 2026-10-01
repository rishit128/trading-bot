from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data import india
from src.data.india import (IndiaClock, IntradayFeed, NseFileMissing, _parse_symbols, load_index_symbols,
                            load_nse_symbols, yf_bar_fetcher)

UTC = timezone.utc


def nse_csv(n=150, extra=""):
    rows = ["SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING"]
    rows += [f"STK{i:03d},Company {i} Limited,EQ,01-JAN-2000" for i in range(n)]
    rows += ["TTRADE,Trade To Trade Ltd,BE,01-JAN-2000", "JUNK,Junk Ltd,BZ,01-JAN-2000", extra]
    return "\n".join(r for r in rows if r)


def nifty_csv(n):
    return "Company Name,Industry,Symbol,Series,ISIN Code\n" + "\n".join(f"C{i},Ind{i % 3},SYM{i:03d},EQ,{i}" for i in range(n))


def test_parser_keeps_only_eq_series_and_strips_column_names():
    symbols = _parse_symbols(nse_csv(3))
    assert symbols == ["STK000", "STK001", "STK002"]  # BE/BZ (trade-to-trade) excluded


def test_parser_handles_nifty_style_csv_without_series_filter_needed():
    csv = "Company Name,Industry,Symbol,Series,ISIN Code\nA,X,ABB,EQ,1\nB,Y,TCS,EQ,2\n"
    assert _parse_symbols(csv) == ["ABB", "TCS"]


def yahoo_list(n):
    return lambda min_traded_value: [f"Y{i:04d}" for i in range(n)]


def yahoo_down(min_traded_value):
    raise ConnectionError("yahoo down")


def test_a_hand_saved_equity_file_wins_over_yahoo(tmp_path):
    (tmp_path / "EQUITY_L.csv").write_text(nse_csv())
    assert len(load_nse_symbols(tmp_path, yahoo=yahoo_list(3000))) == 150


def test_without_files_the_list_comes_from_yahoo_automatically(tmp_path):
    seen = []
    assert len(load_nse_symbols(tmp_path, min_traded_value=1e8, yahoo=lambda m: seen.append(m) or yahoo_list(2500)(m))) == 2500
    assert seen == [1e8]  # the liquidity floor is passed on for Yahoo's pre-cut


def test_yahoo_failure_falls_back_to_a_saved_nifty500_file(tmp_path):
    (tmp_path / "ind_nifty500list.csv").write_text(nifty_csv(120))
    assert len(load_nse_symbols(tmp_path, yahoo=yahoo_down)) == 120
    assert len(load_nse_symbols(tmp_path, yahoo=yahoo_list(5))) == 120  # a suspiciously tiny Yahoo list is not trusted


def test_nothing_usable_raises_a_message_saying_what_failed(tmp_path):
    with pytest.raises(NseFileMissing, match="Yahoo's list was unavailable"):
        load_nse_symbols(tmp_path, yahoo=yahoo_down)
    (tmp_path / "EQUITY_L.csv").write_text(nse_csv(5))  # a half-saved file is not trusted either
    with pytest.raises(RuntimeError, match="NSE stock list"):
        load_nse_symbols(tmp_path, yahoo=yahoo_down)


def test_yahoo_list_keeps_main_board_symbols_and_drops_other_segments_and_illiquid_ones():
    from src.data.india import yahoo_nse_symbols

    quotes = [{"symbol": s + ".NS", "averageDailyVolume3Month": v, "regularMarketPrice": 100.0, "financialCurrency": "INR"}
              for s, v in (
        ("RELIANCE", 5e6), ("BAJAJ-AUTO", 1e6), ("KLBRENG-B", 1e6),  # real symbols, hyphens included
        ("TINY", 1e3),  # Rs 1 lakh a day: well under half the floor
        ("NOVOL", None),  # no volume from Yahoo: kept, not guessed away
        ("ABC-SM", 5e6), ("XYZ-IV", 5e6), ("011NSETEST", 5e6))] + [{"symbol": "AAPL"}]
    quotes.append({"symbol": "GOLDBEES.NS", "averageDailyVolume3Month": 5e6, "regularMarketPrice": 100.0})  # an ETF
    pages = [{"quotes": quotes[:5], "total": len(quotes)}, {"quotes": quotes[5:], "total": len(quotes)}]
    got = yahoo_nse_symbols(min_traded_value=1e8, screen=lambda offset: pages[0 if offset == 0 else 1], pause=0)
    assert got == ["RELIANCE", "BAJAJ-AUTO", "KLBRENG-B", "NOVOL"]


def test_the_folder_comes_from_nse_files_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NSE_FILES_DIR", str(tmp_path))
    (tmp_path / "EQUITY_L.csv").write_text(nse_csv())
    assert india.nse_files_dir() == tmp_path and len(load_nse_symbols()) == 150


def test_index_lists_give_symbols(tmp_path):
    (tmp_path / "ind_nifty50list.csv").write_text(nifty_csv(4))
    assert load_index_symbols(50, tmp_path) == ["SYM000", "SYM001", "SYM002", "SYM003"]


def test_a_present_but_unreadable_index_file_raises_nsefilemissing_not_a_raw_parsing_error(tmp_path):
    """A caller (run_intraday) catches only NseFileMissing to fall back to Yahoo; a half-saved or wrong file (e.g. an
    HTML error page saved with a .csv extension, or one missing the SYMBOL column) must not bypass that fallback with
    an uncaught pandas/KeyError."""
    (tmp_path / "ind_nifty100list.csv").write_text("<html>not a csv</html>")
    with pytest.raises(NseFileMissing):
        load_index_symbols(100, tmp_path)


def test_no_code_downloads_from_nse_websites():
    """NSE's and NSE Indices' terms prohibit automated collection: no module may call their sites (comments may name them)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"https?://[^\s\"']*(nseindia\.com|niftyindices\.com)")
    offenders = [str(p.relative_to(root)) for p in [root / "main.py", *(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def wide(tickers, days=5):
    idx = pd.date_range("2026-09-10", periods=days, freq="B")
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    data = {(t, f): [100.0 + i for i in range(days)] for t in tickers for f in ["Open", "High", "Low", "Close"]}
    data.update({(t, "Volume"): [1000.0] * days for t in tickers})
    return pd.DataFrame(data, index=idx, columns=cols)


def test_bar_fetcher_returns_long_format_with_plain_symbols_and_skips_unresolvable():
    seen = {}

    def download(tickers, **kw):
        seen.update(tickers=tickers, **kw)
        return wide(["AAA.NS", "BBB.NS"])  # CCC.NS could not be resolved by Yahoo

    fetch = yf_bar_fetcher(330, download, today=lambda: datetime(2026, 9, 21, 11, 0))
    df = fetch(["AAA", "BBB", "CCC"])
    assert seen["tickers"] == ["AAA.NS", "BBB.NS", "CCC.NS"]
    assert seen["end"] == "2026-09-21" and seen["start"] == (date(2026, 9, 21) - timedelta(days=330)).isoformat() and seen["interval"] == "1d"
    assert sorted(set(df.index.get_level_values(0))) == ["AAA", "BBB"]
    assert list(df.columns) == ["Close", "High", "Low", "Volume"] and len(df) == 10


def test_bar_fetcher_handles_empty_and_all_nan_results():
    assert yf_bar_fetcher(330, lambda t, **kw: pd.DataFrame(), lambda: datetime(2026, 9, 21))(["AAA"]).empty
    nan = wide(["AAA.NS"]) * float("nan")
    assert yf_bar_fetcher(330, lambda t, **kw: nan, lambda: datetime(2026, 9, 21))(["AAA"]).empty


def test_screener_can_consume_the_indian_bar_format():
    from src.data.universe import ScreenConfig, screen_bars

    fetch = yf_bar_fetcher(330, lambda t, **kw: wide(["AAA.NS"], days=5), lambda: datetime(2026, 9, 21))
    assert screen_bars(fetch(["AAA"]), ScreenConfig()) == []  # only 5 bars: rejected for short history, no crash


# ---- market clock, checked against real 2026 sessions (the calendar matched Yahoo's Nifty days exactly) ----
CLOCK = IndiaClock()


@pytest.mark.parametrize("when,expected", [
    (datetime(2026, 9, 18, 5, 0, tzinfo=UTC), True),     # Friday 10:30 IST
    (datetime(2026, 9, 18, 3, 40, tzinfo=UTC), False),   # 09:10 IST: pre-open, market not trading yet
    (datetime(2026, 9, 18, 10, 30, tzinfo=UTC), False),  # 16:00 IST: after the close
    (datetime(2026, 9, 19, 5, 0, tzinfo=UTC), False),    # Saturday
    (datetime(2026, 9, 20, 5, 0, tzinfo=UTC), False),    # Sunday
    (datetime(2026, 1, 26, 5, 0, tzinfo=UTC), False),    # Republic Day (a Monday holiday)
])
def test_clock_matches_real_nse_sessions(when, expected):
    assert CLOCK.is_open(when) is expected


def test_clock_open_and_close_boundaries():
    assert CLOCK.is_open(datetime(2026, 9, 18, 3, 45, tzinfo=UTC)) is True    # 09:15 IST
    assert CLOCK.is_open(datetime(2026, 9, 18, 9, 59, tzinfo=UTC)) is True    # 15:29 IST
    assert CLOCK.is_open(datetime(2026, 9, 18, 10, 0, tzinfo=UTC)) is False   # 15:30 IST


def test_clock_falls_back_to_weekday_hours_past_the_calendars_last_date():
    assert CLOCK.is_open(datetime(2027, 6, 1, 5, 0, tzinfo=UTC)) is True     # Tuesday 10:30 IST
    assert CLOCK.is_open(datetime(2027, 6, 5, 5, 0, tzinfo=UTC)) is False    # Saturday
    assert CLOCK.is_open(datetime(2027, 6, 1, 12, 0, tzinfo=UTC)) is False   # 17:30 IST


def test_today_ist_uses_india_date_not_utc_date():
    assert CLOCK.today_ist(datetime(2026, 9, 20, 20, 0, tzinfo=UTC)) == "2026-09-21"


def five_min_bars():
    idx = pd.date_range("2026-09-18 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": [100, 101, 102, 103.0], "High": [101, 102, 103, 104.0],
                         "Low": [99, 100, 101, 102.0], "Close": [100.5, 101.5, 102.5, 103.5]}, index=idx)


def test_intraday_feed_last_price_and_bars_since():
    feed = IntradayFeed(download=lambda ticker, **kw: five_min_bars())
    assert feed.last_price("RELIANCE") == 103.5
    since = datetime(2026, 9, 18, 3, 47, tzinfo=UTC)  # 09:17 IST -> bars starting 09:20 and later
    assert len(feed.bars_since("RELIANCE", since)) == 3


def test_intraday_feed_requests_ns_ticker_5m_and_errors_when_empty():
    seen = {}

    def download(ticker, **kw):
        seen.update(ticker=ticker, **kw)
        return pd.DataFrame()

    feed = IntradayFeed(download=download)
    with pytest.raises(ValueError, match="no price data"):
        feed.last_price("TCS")
    assert seen["ticker"] == "TCS.NS" and seen["interval"] == "5m"
    assert feed.bars_since("TCS", datetime(2026, 9, 18, tzinfo=UTC)).empty


def test_intraday_feed_accepts_naive_timestamps_as_read_back_from_sqlite():
    feed = IntradayFeed(download=lambda ticker, **kw: five_min_bars())
    naive_utc = datetime(2026, 9, 18, 3, 47)  # SQLite returns datetimes without tzinfo; stored values are UTC
    assert len(feed.bars_since("RELIANCE", naive_utc)) == 3


def test_top_n_ranks_by_traded_value_so_a_stock_missing_its_market_cap_is_not_lost():
    """Regression: Yahoo left RELIANCE's market cap empty (2026-09-26), which sorted it near the end of the list."""
    from src.data.india import yahoo_nse_symbols

    def q(sym, value):
        return {"symbol": sym + ".NS", "averageDailyVolume3Month": value / 100.0, "regularMarketPrice": 100.0,
                "financialCurrency": "INR"}

    quotes = [q("MID1", 5e8), q("MID2", 4e8), q("SMALL", 1e7), q("RELIANCE", 1.4e10)]  # RELIANCE last, as Yahoo sent it
    got = yahoo_nse_symbols(top=2, screen=lambda offset: {"quotes": quotes if offset == 0 else [], "total": 4}, pause=0)
    assert got == ["RELIANCE", "MID1"]


def test_yahoo_corporate_actions_become_ist_ex_dates_with_zero_rows_dropped():
    from datetime import date
    from types import SimpleNamespace

    import pandas as pd

    from src.data.india import yahoo_corporate_actions

    idx = pd.DatetimeIndex(["2024-10-28 00:00", "2025-01-01 00:00"]).tz_localize("Asia/Kolkata")
    seen = []

    def ticker(name):
        seen.append(name)
        return SimpleNamespace(splits=pd.Series([2.0, 0.0], index=idx), dividends=pd.Series([5.5], index=idx[:1]))

    splits, dividends = yahoo_corporate_actions("RELIANCE", ticker=ticker)
    assert seen == ["RELIANCE.NS"]
    assert splits == {date(2024, 10, 28): 2.0} and dividends == {date(2024, 10, 28): 5.5}
