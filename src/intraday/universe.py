"""Which stocks the intraday engine scans: one selector, shared by the live engine and every backtest, so a result is always
about the universe that was traded.

In order: an explicit list (INTRADAY_UNIVERSE), then the hand-saved Nifty 100 file, then Yahoo's top 100 by average daily
traded value. The last is NOT the Nifty 100: on 2026-09-28 it shared only 44 names with it, adding recently listed stocks
whose IPO-period volume inflates a 3-month average. It is the only automatic choice that makes no request to NSE's sites
(the AWS deployment has no saved file), so it is allowed but reported with a label that says what it is."""
from typing import Callable, List, Sequence, Tuple

from src.data.india import NseFileMissing, load_index_symbols, yahoo_nse_symbols


def select_universe(explicit: Sequence[str] = (), size: int = 100,
                    from_file: Callable[[int], List[str]] = load_index_symbols,
                    from_yahoo: Callable[..., List[str]] = yahoo_nse_symbols) -> Tuple[List[str], str]:
    """(symbols, a label saying where they came from)."""
    if explicit:
        symbols = list(dict.fromkeys(s.strip().upper() for s in explicit if s.strip()))
        return symbols, f"explicit list (INTRADAY_UNIVERSE, {len(symbols)} symbols)"
    try:
        return from_file(size), f"Nifty {size} (hand-saved NSE file)"
    except NseFileMissing:
        return from_yahoo(top=size), f"Yahoo top {size} by average daily traded value (not the Nifty {size})"
