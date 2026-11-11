"""Monthly ETF rotation against holding the Nifty ETF (declared 2026-11-11, before any result).

Why it might work: asset classes trend for months (time-series momentum: Moskowitz, Ooi and Pedersen 2012; dual momentum:
Antonacci 2014), because large investors move money slowly; the loser is whoever holds the falling asset out of habit. Why here:
the account's delivery brokerage is 0.3% a side, so a rule that trades a few times a year in one or two large positions pays far
less in costs than fifteen small stock positions do (2.0% a round trip on Rs 2,200).

Data: Angel One daily bars (the owner's account), NIFTYBEES (Nifty 50), JUNIORBEES (Nifty Next 50) and GOLDBEES (gold), from
2010-01-01 or each ETF's first bar; a one-day fall of more than 40% is read as a unit split and the earlier prices are divided by
the whole-number ratio. No survivorship question: three ETFs, chosen before looking, all still trading.

Rules (fixed):
  R1  On the last session of each month rank the three ETFs by their 12-month return. Hold the best one if its return beats cash
      (3.5% a year), else hold cash. Trade at the next session's open.
  R2  The same ranking, the best two, half the account each; one that does not beat cash is left as cash.
  Both trade only when the holdings change (no monthly re-balancing between two unchanged holdings).
  Benchmark: NIFTYBEES bought on the first day and held.
Account: Rs 25,000, whole units, idle cash earns 3.5% a year, this account's delivery fees on every fill
(src.engine.costs.india_delivery_fees, which charges the stock rate of STT: a little more than an ETF really pays) and 0.05%
slippage a side.

Gate (fixed): the period is cut in two halves by date. A rule passes only if in BOTH halves its CAGR is above the benchmark's, its
Sharpe is higher and its worst drawdown is shallower, and over the whole period its CAGR is still above the benchmark's with fees
and slippage doubled. If both pass, R2 is taken (two holdings instead of one). A pass earns a paper account, nothing more.

    python scripts/research_etf_rotation.py download     # Angel One daily bars into research_cache/angel_daily/
    python scripts/research_etf_rotation.py test"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402

ETFS = ("NIFTYBEES", "JUNIORBEES", "GOLDBEES")
BENCHMARK = "NIFTYBEES"
CACHE = Path(__file__).resolve().parent.parent / "research_cache" / "angel_daily"
START = datetime(2010, 1, 1)
CHUNK_DAYS, PACE = 1900, 3.0  # Angel One returns at most 2,000 daily bars a request
LOOKBACK = 252  # sessions: 12 months
CAPITAL, CASH_YIELD = 25_000.0, 0.035
SPLIT_FALL = 0.6  # a close under this share of the previous close is a unit split, not a market move
Fees = Callable[[str, float], float]


def adjust_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Divide the prices before each unit split by its whole-number ratio, so the series is continuous."""
    out = frame.copy()
    ratio = out["Close"] / out["Close"].shift(1)
    for day in ratio.index[ratio < SPLIT_FALL]:
        factor = round(1 / float(ratio[day]))
        out.loc[out.index < day, ["Open", "High", "Low", "Close"]] /= factor
    return out


def download() -> None:
    import src.config  # noqa: F401  (loads ../.env)
    from src.data.angel import AngelSession, load_tokens

    api, tokens = AngelSession().api(), load_tokens(CACHE.parent)
    CACHE.mkdir(parents=True, exist_ok=True)
    for symbol in ETFS:
        rows: List[list] = []
        start = START
        while start < datetime.now():
            end = min(start + timedelta(days=CHUNK_DAYS), datetime.now())
            params = {"exchange": "NSE", "symboltoken": tokens[symbol], "interval": "ONE_DAY"}
            params.update(fromdate=f"{start:%Y-%m-%d} 09:15", todate=f"{end:%Y-%m-%d} 15:30")
            for attempt in range(5):
                try:
                    rows += api.getCandleData(params).get("data") or []
                    break
                except Exception as e:  # the broker's rate limit answers with text, not JSON
                    print(f"{symbol} {start:%Y-%m-%d}: {type(e).__name__}, retrying")
                    time.sleep(PACE * 5 * (attempt + 1))
            else:
                raise SystemExit(f"{symbol}: Angel One refused the request five times; nothing saved for it")
            time.sleep(PACE)
            start = end + timedelta(days=1)
        frame = pd.DataFrame(rows, columns=["at", "Open", "High", "Low", "Close", "Volume"]).drop_duplicates("at")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("at"))).tz_localize(None).normalize()
        frame.sort_index().to_pickle(CACHE / f"{symbol}.pkl")
        print(f"{symbol}: {len(frame)} daily bars, {frame.index[0].date()} .. {frame.index[-1].date()}")


def targets(close: pd.DataFrame, day: pd.Timestamp, top: int) -> List[Optional[str]]:
    """The `top` holdings chosen on `day`: the ETFs with the best 12-month return, None (cash) for one that does not beat cash."""
    past = close.loc[:day]
    if len(past) <= LOOKBACK:
        return [None] * top
    gain = (past.iloc[-1] / past.iloc[-1 - LOOKBACK] - 1).dropna().sort_values(ascending=False)
    picks: List[Optional[str]] = [s if g > CASH_YIELD else None for s, g in gain.head(top).items()]
    return picks + [None] * (top - len(picks))


def simulate(
    opens: pd.DataFrame,
    close: pd.DataFrame,
    choose: Callable[[pd.Timestamp], List[Optional[str]]],
    fees: Fees,
    slippage: float,
    first: Optional[List[Optional[str]]] = None,
) -> Tuple[pd.Series, int]:
    """Equity by day and the number of fills. `choose(month_end)` gives the holdings wanted; they are traded at the next open,
    sells first, each new holding getting an equal share of the equity. `first` is bought at the first open."""
    days = close.index
    month_end = set(close.groupby([days.year, days.month]).tail(1).index)
    cash, units, fills = CAPITAL, {}, 0
    wanted = first
    equity = []
    for i, day in enumerate(days):
        cash *= (1 + CASH_YIELD) ** (1 / 252)
        if wanted is not None:
            keep = {s for s in wanted if s}
            for symbol in [s for s in units if s not in keep]:
                value = units.pop(symbol) * opens.at[day, symbol] * (1 - slippage)
                cash += value - fees("SELL", value)
                fills += 1
            total = cash + sum(n * opens.at[day, s] for s, n in units.items())
            for symbol in [s for s in wanted if s and s not in units]:
                price = opens.at[day, symbol] * (1 + slippage)
                budget = min(total / len(wanted), cash)
                n = int(budget / (price * 1.005))  # room for the fees
                if n >= 1:
                    cash -= n * price + fees("BUY", n * price)
                    units[symbol] = n
                    fills += 1
            wanted = None
        equity.append(cash + sum(n * close.at[day, s] for s, n in units.items()))
        if day in month_end and i + 1 < len(days):
            wanted = choose(day)
    return pd.Series(equity, index=days), fills


def stats(curve: pd.Series) -> Tuple[float, float, float]:
    """(CAGR, Sharpe, worst drawdown) of an equity curve."""
    daily = curve.pct_change().dropna()
    years = len(curve) / 252
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    return cagr, sharpe, float((curve / curve.cummax() - 1).min())


def test() -> None:
    frames = {s: adjust_splits(pd.read_pickle(CACHE / f"{s}.pkl")) for s in ETFS}
    close = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).dropna()
    close = close.loc[close.index < pd.Timestamp(datetime.now().date())]  # today's bar is not finished
    opens = pd.DataFrame({s: f["Open"] for s, f in frames.items()}).reindex(close.index)
    opens = opens.where(opens > 0, close)  # a missing open: use the close
    print(
        f"{', '.join(ETFS)}: {len(close)} sessions, {close.index[0].date()} .. {close.index[-1].date()} (first year only ranks)"
    )

    def doubled(side: str, value: float) -> float:
        return 2 * india_delivery_fees(side, value)

    rules: Dict[str, Callable[[pd.Timestamp], List[Optional[str]]]] = {
        "R1 best one": lambda day: targets(close, day, 1),
        "R2 best two": lambda day: targets(close, day, 2),
        "benchmark": lambda day: [BENCHMARK],
    }
    ranked, first = close.index[LOOKBACK], close.index[LOOKBACK + 1]  # the first ranking, and the open it is traded at
    live_opens, live_close = opens.loc[first:], close.loc[first:]
    curves, at_double = {}, {}
    for name, choose in rules.items():
        curve, fills = simulate(live_opens, live_close, choose, india_delivery_fees, SLIPPAGE, choose(ranked))
        curves[name] = curve
        at_double[name] = stats(simulate(live_opens, live_close, choose, doubled, 2 * SLIPPAGE, choose(ranked))[0])[0]
        print(f"{name:12s} fills {fills:4d}, final Rs {curve.iloc[-1]:,.0f}")

    middle = live_close.index[len(live_close) // 2]
    windows = {"full": slice(None), "1st half": slice(None, middle), "2nd half": slice(middle, None)}
    print(f"\n{'':12s}" + "".join(f"{w:>30s}" for w in windows) + f"{'CAGR at 2x costs':>20s}")
    table = {name: {w: stats(curve.loc[cut]) for w, cut in windows.items()} for name, curve in curves.items()}
    for name, row in table.items():
        cells = "".join(f"{c:+9.1%} /{s:5.2f} /{d:7.1%}      " for c, s, d in row.values())
        print(f"{name:12s}  {cells}{at_double[name]:+14.1%}")
    print("(each cell: CAGR / Sharpe / worst drawdown)")

    base = table["benchmark"]
    passed = [
        name
        for name in ("R1 best one", "R2 best two")
        if all(
            table[name][w][0] > base[w][0] and table[name][w][1] > base[w][1] and table[name][w][2] > base[w][2]
            for w in ("1st half", "2nd half")
        )
        and at_double[name] > at_double["benchmark"]
    ]
    print("\nPASSED: " + (", ".join(passed) if passed else "none"))


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    elif sys.argv[1:] == ["test"]:
        test()
    else:
        raise SystemExit(__doc__)
