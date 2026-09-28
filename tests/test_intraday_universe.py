"""One universe selector for the live engine and every backtest: explicit list, else the saved Nifty 100 file, else Yahoo."""
from src.data.india import NseFileMissing
from src.intraday.universe import select_universe


def no_file(size):
    raise NseFileMissing("no file")


def test_an_explicit_list_wins_and_is_cleaned_and_deduplicated():
    called = []
    symbols, label = select_universe([" reliance", "TCS", "tcs", ""], from_file=lambda n: called.append(n) or ["X"],
                                     from_yahoo=lambda top: called.append(top) or ["Y"])
    assert symbols == ["RELIANCE", "TCS"] and "explicit list" in label and "2 symbols" in label
    assert called == []                                    # neither the file nor Yahoo was consulted


def test_the_saved_nifty_file_is_used_when_there_is_no_explicit_list():
    symbols, label = select_universe((), size=100, from_file=lambda n: ["A", "B"] if n == 100 else [],
                                     from_yahoo=lambda top: ["Y"])
    assert symbols == ["A", "B"] and label == "Nifty 100 (hand-saved NSE file)"


def test_yahoo_is_the_fallback_and_says_it_is_not_the_nifty():
    symbols, label = select_universe((), size=100, from_file=no_file, from_yahoo=lambda top: [f"S{i}" for i in range(top)])
    assert len(symbols) == 100 and label.startswith("Yahoo top 100") and "not the Nifty 100" in label


def test_the_size_is_passed_through_to_both_sources():
    seen = []
    select_universe((), size=50, from_file=lambda n: seen.append(("file", n)) or (_ for _ in ()).throw(NseFileMissing("x")),
                    from_yahoo=lambda top: seen.append(("yahoo", top)) or [])
    assert seen == [("file", 50), ("yahoo", 50)]
