"""The research price loaders: NSE index lists, the cached index series and the cached OHLC universe."""


def test_index_constituents_come_from_the_hand_downloaded_list(tmp_path):
    from src.data.india import load_index_symbols

    (tmp_path / "ind_nifty50list.csv").write_text("Company Name,Industry,Symbol,Series,ISIN Code\nA,X,ABB,EQ,1\nB,Y,TCS,EQ,2\n")
    assert load_index_symbols(50, tmp_path) == ["ABB", "TCS"]


def test_index_cache_expires_and_falls_back_to_the_stale_copy(tmp_path):
    import os, time
    from datetime import date
    import pandas as pd
    from src.data.price_history import load_index_close

    def frame(last):
        idx = pd.bdate_range(end=last, periods=5)
        return pd.DataFrame({"Close": range(5)}, index=idx)

    calls = []

    def fresh(tickers, start, end):
        calls.append(1)
        return frame("2026-09-25")

    old = load_index_close("^X", 1, tmp_path, download=lambda *a: frame("2026-09-23"), today=lambda: date(2026, 9, 23))
    assert old.index[-1] == pd.Timestamp("2026-09-23")
    path = tmp_path / "index_X_1y.pkl"
    # a fresh cache is reused, and a research caller without max_age_hours never refreshes
    assert load_index_close("^X", 1, tmp_path, download=fresh, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-23")
    os.utime(path, (time.time() - 48 * 3600,) * 2)
    assert load_index_close("^X", 1, tmp_path, download=fresh).index[-1] == pd.Timestamp("2026-09-23")
    assert not calls
    # an expired cache is re-downloaded
    assert load_index_close("^X", 1, tmp_path, download=fresh, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-25")
    # ...and if that fails the stale copy is used rather than crashing the cycle
    os.utime(path, (time.time() - 48 * 3600,) * 2)

    def broken(*a):
        raise RuntimeError("yahoo down")

    assert load_index_close("^X", 1, tmp_path, download=broken, max_age_hours=12).index[-1] == pd.Timestamp("2026-09-25")


def test_ohlc_universe_downloads_in_chunks_caches_to_disk_and_skips_unresolvable_symbols(tmp_path):
    import numpy as np
    import pandas as pd
    from datetime import date
    from src.data.price_history import load_ohlc_universe

    idx = pd.bdate_range("2024-01-01", periods=30)

    def frame(base):
        close = pd.Series(np.linspace(base, base + 10, 30), index=idx)
        return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0}, index=idx)

    calls = []

    def download(tickers, start, end):
        calls.append(list(tickers))
        known = {"AAA.NS": frame(100), "BBB.NS": frame(200), "CCC.NS": frame(300)}
        return pd.concat({t: known[t] for t in tickers if t in known}, axis=1)  # ZZZ.NS is not in the download

    kwargs = dict(index=500, years=1, cache_dir=tmp_path, symbols=["AAA", "BBB", "CCC", "ZZZ"], chunk=2,
                  today=lambda: date(2024, 3, 1))
    out = load_ohlc_universe(download=download, **kwargs)
    assert set(out) == {"AAA", "BBB", "CCC"} and calls == [["AAA.NS", "BBB.NS"], ["CCC.NS", "ZZZ.NS"]]  # chunked, ZZZ skipped
    assert list(out["AAA"].columns) == ["Open", "High", "Low", "Close", "Volume"] and len(out["AAA"]) == 30
    assert (tmp_path / "ohlc_nifty500_1y.pkl").exists()
    again = load_ohlc_universe(download=lambda *a: (_ for _ in ()).throw(AssertionError("must read the cache")), **kwargs)
    assert set(again) == set(out)  # the second call is served from disk with no download
    refreshed = load_ohlc_universe(download=download, refresh=True, **kwargs)
    assert len(calls) == 4 and set(refreshed) == set(out)  # refresh=True downloads again
