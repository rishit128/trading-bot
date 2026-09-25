from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import india
from src.data.india import IndiaClock, IntradayFeed, _parse_symbols, fetch_nse_symbols, yf_bar_fetcher

UTC = timezone.utc


def nse_csv(n=150, extra=""):
    rows = ["SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING"]
    rows += [f"STK{i:03d},Company {i} Limited,EQ,01-JAN-2000" for i in range(n)]
    rows += ["TTRADE,Trade To Trade Ltd,BE,01-JAN-2000", "JUNK,Junk Ltd,BZ,01-JAN-2000", extra]
    return "\n".join(r for r in rows if r)


def resp(text, status=200):
    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")

    return SimpleNamespace(text=text, status_code=status, raise_for_status=raise_for_status)


def test_parser_keeps_only_eq_series_and_strips_column_names():
    symbols = _parse_symbols(nse_csv(3))
    assert symbols == ["STK000", "STK001", "STK002"]  # BE/BZ (trade-to-trade) excluded


def test_parser_handles_nifty_style_csv_without_series_filter_needed():
    csv = "Company Name,Industry,Symbol,Series,ISIN Code\nA,X,ABB,EQ,1\nB,Y,TCS,EQ,2\n"
    assert _parse_symbols(csv) == ["ABB", "TCS"]


def test_fetch_uses_nse_list_first():
    calls = []

    def get(url, **kw):
        calls.append(url)
        return resp(nse_csv())

    assert len(fetch_nse_symbols(get)) == 150 and calls == [india.NSE_EQUITY_LIST]


def test_fetch_falls_back_to_nifty500_when_nse_archive_fails():
    def get(url, **kw):
        if url == india.NSE_EQUITY_LIST:
            raise ConnectionError("blocked")
        return resp("Company Name,Industry,Symbol,Series,ISIN Code\n" +
                    "\n".join(f"C{i},X,SYM{i:03d},EQ,{i}" for i in range(120)))

    assert len(fetch_nse_symbols(get)) == 120


def test_fetch_rejects_suspiciously_small_lists_and_raises_if_nothing_works():
    with pytest.raises(RuntimeError, match="NSE stock list"):
        fetch_nse_symbols(lambda url, **kw: resp(nse_csv(5)))
    with pytest.raises(RuntimeError):
        fetch_nse_symbols(lambda url, **kw: resp("", status=503))


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
