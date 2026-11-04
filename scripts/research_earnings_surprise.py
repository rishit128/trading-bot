"""Does beating analysts' estimates keep paying for weeks after costs? (pre-declared 2026-11-04, before any result was seen.)

    python scripts/research_earnings_surprise.py all    # waits for the close, then downloads both and tests
    python scripts/research_earnings_surprise.py test

Prices are Angel One's daily candles. Dates, reported EPS and estimates are Yahoo's, used with the owner's agreement: nothing else here gives
estimates. Its hours are unreliable, so the entry is the open two sessions after the date. Surprise is (reported - estimate) / |estimate|. Its
size counts only against the past year's surprises across all stocks, which were known at the time. A stock must close at Rs 100 or more and
trade Rs 10 crore a day. A window holding an overnight move beyond 30% is dropped (the candles are not adjusted). Today's listings only, and
estimates may be revised later, which flatters the returns. Delivery costs on Rs 2,200; excess is net of them, less the median stock.

Data check, first: stocks in the top tenth must react better than the bottom tenth from the close before the date to the close after it, with
t >= 2.6 by month. If not, Yahoo's estimates do not describe what the market expected, and the result is INVALID: neither pass nor fail.

The test is one cell: top 10% held 60 sessions (the published drift's window). All of: at least 300 trades; mean net > 0; mean excess > 0 with
t >= 2.6, clustered by calendar month; both halves of the months > 0; positive at twice the slippage; beats ordinary results (the 40th to 60th
percentile) held as long; and its neighbours (top 10% held 40, top 20% held 60) keep mean excess > 0. A pass makes the rule a candidate for
paper trading in the ORB account, no more. A fail deletes both downloaded files, keeping only the result in the trial register."""

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from datetime import time as clock
from pathlib import Path

import numpy as np
import pandas as pd
from research_results_hold import CACHE, DAILY, DATES, POSITION, TRIALS, clustered, download, frames

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.india import IST  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.significance import deflated_t  # noqa: E402

EARNINGS = CACHE / "earnings_history.pkl"
AFTER_CLOSE = clock(
    15, 45
)  # Yahoo is the live bots' backup feed and the Angel One account is theirs: neither is touched while they trade
PACE, PAST_DAYS, MIN_PAST, SAME_QUARTER_DAYS = 0.6, 365, 50, 20
MIN_PRICE, MIN_VALUE, SPLIT_MOVE = 100.0, 1e8, 0.30
MIN_TRADES, T_MIN = 300, 2.6
GROUPS = {"top 10%": (90, 101), "top 20%": (80, 101), "middle": (40, 60), "bottom 10%": (-1, 10)}
HOLDS = (20, 40, 60)
JUDGED, NEIGHBOURS = ("top 10%", 60), (("top 10%", 40), ("top 20%", 60))


def wait_for_close() -> None:
    while datetime.now(IST).time() < AFTER_CLOSE:
        time.sleep(60)


def fetch_earnings(symbols) -> None:
    """Saved every 25 stocks; a run resumes. A stock Yahoo cannot answer for is kept empty, so that one bad name is not asked for forever."""
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    have = pickle.loads(EARNINGS.read_bytes()) if EARNINGS.exists() else {}  # nosec B301
    for n, symbol in enumerate([s for s in symbols if s not in have]):
        try:
            frame = yf.Ticker(symbol + ".NS").get_earnings_dates(limit=60)
            have[symbol] = pd.DataFrame() if frame is None else frame[["EPS Estimate", "Reported EPS"]].dropna()
        except Exception as e:
            print(f"{symbol}: {type(e).__name__}", flush=True)
            have[symbol] = pd.DataFrame()
        time.sleep(PACE)
        if n % 25 == 24:
            EARNINGS.write_bytes(pickle.dumps(have))
            print(f"{len(have)} of {len(symbols)} earnings histories", flush=True)
    EARNINGS.write_bytes(pickle.dumps(have))
    print(f"earnings: {sum(len(f) > 0 for f in have.values())} of {len(symbols)} stocks have data", flush=True)


def events(history: dict) -> pd.DataFrame:
    """One row per stock and result: date (IST), surprise %, and its percentile against the past year's, known at that date."""
    rows = []
    for symbol, frame in history.items():
        if not len(frame):
            continue
        at = pd.DatetimeIndex(frame.index).tz_convert(IST).tz_localize(None).normalize()
        table = pd.DataFrame(
            {"date": at, "estimate": frame["EPS Estimate"].to_numpy(), "reported": frame["Reported EPS"].to_numpy()}
        )
        table = table[table["estimate"] != 0].sort_values("date")
        table = table[table["date"].diff().dt.days.fillna(999) > SAME_QUARTER_DAYS]
        table = table.assign(symbol=symbol, surprise=(table["reported"] - table["estimate"]) / table["estimate"].abs() * 100)
        rows.append(table)
    out = pd.concat(rows).sort_values("date").reset_index(drop=True)
    dates, values = out["date"].to_numpy(), out["surprise"].to_numpy()
    ranks = np.full(len(out), np.nan)
    for i in range(len(out)):
        start = np.searchsorted(dates, dates[i] - np.timedelta64(PAST_DAYS, "D"))
        end = np.searchsorted(dates, dates[i])
        if end - start >= MIN_PAST:
            ranks[i] = (values[start:end] < values[i]).mean() * 100
    return out.assign(rank=ranks).dropna(subset=["rank"])


def outcomes(table: pd.DataFrame, px: dict) -> pd.DataFrame:
    """Each event with a tradable entry: its reaction and its gross and market-relative % return for each hold."""
    opens, closes = px["Open"], px["Close"]
    days = opens.index
    value = (closes * px["Volume"]).rolling(20).mean()
    o, c, v = opens.to_numpy(), closes.to_numpy(), value.to_numpy()
    jump = np.vstack([np.zeros((1, o.shape[1]), bool), np.abs(o[1:] / c[:-1] - 1) > SPLIT_MOVE])
    columns = {s: j for j, s in enumerate(opens.columns)}
    move = closes.div(closes.shift(2)).sub(1)
    excess_move = move.sub(move.median(axis=1), axis=0).to_numpy() * 100
    rows = []
    for e in table.itertuples():
        j = columns.get(e.symbol)
        at = days.searchsorted(e.date)
        entry = at + 2
        if j is None or at < 1 or entry >= len(days) or not np.isfinite(o[entry, j]):
            continue
        if not (c[entry - 1, j] >= MIN_PRICE and v[entry - 1, j] >= MIN_VALUE):
            continue
        row = {"symbol": e.symbol, "rank": e.rank, "day": days[entry], "reaction": excess_move[at + 1, j]}
        for hold in HOLDS:
            leave = entry + hold - 1
            if leave < len(days) and np.isfinite(c[leave, j]) and not jump[entry + 1 : leave + 1, j].any():
                row[f"gross_{hold}"] = (c[leave, j] / o[entry, j] - 1) * 100
                row[f"market_{hold}"] = float(np.nanmedian(c[leave] / o[entry] - 1)) * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    out["month"] = out["day"].dt.strftime("%Y-%m")
    return out


def group(done: pd.DataFrame, name: str) -> pd.DataFrame:
    low, high = GROUPS[name]
    return done[(done["rank"] >= low) & (done["rank"] < high)]


def cell(done: pd.DataFrame, name: str, hold: int, cost: float) -> dict:
    part = group(done, name).dropna(subset=[f"gross_{hold}"])
    net = part[f"gross_{hold}"] - cost
    excess = net - part[f"market_{hold}"]
    return {
        "n": len(part),
        "net": net.mean(),
        "excess": excess,
        "months": part["month"],
        "summary": clustered(excess, part["month"]),
    }


def verdict(done: pd.DataFrame, cost: float, cost2: float) -> list:
    """The reasons the judged cell fails; empty when it passes."""
    name, hold = JUDGED
    main, twice = cell(done, name, hold, cost), cell(done, name, hold, cost2)
    s = main["summary"]
    tests = {
        f"at least {MIN_TRADES} trades": main["n"] >= MIN_TRADES,
        "mean net > 0": main["net"] > 0,
        "mean excess > 0": s["mean"] > 0,
        f"t >= {T_MIN}": s["t"] >= T_MIN,
        "both halves > 0": s["h1"] > 0 and s["h2"] > 0,
        "positive at twice the slippage": twice["summary"]["mean"] > 0,
        "beats ordinary results": s["mean"] > cell(done, "middle", hold, cost)["summary"]["mean"],
        "neighbours' excess > 0": all(cell(done, a, b, cost)["summary"]["mean"] > 0 for a, b in NEIGHBOURS),
    }
    return [label for label, ok in tests.items() if not ok]


def test() -> None:
    px = frames(pickle.loads(DAILY.read_bytes()))  # nosec B301
    done = outcomes(events(pickle.loads(EARNINGS.read_bytes())), px)  # nosec B301
    fees = (india_delivery_fees("BUY", POSITION) + india_delivery_fees("SELL", POSITION)) / POSITION * 100
    cost, cost2 = fees + 2 * SLIPPAGE * 100, fees + 4 * SLIPPAGE * 100
    trials = int(pd.read_csv(TRIALS)["variants"].fillna(1).sum())
    print(
        f"{done['symbol'].nunique()} stocks, {len(done)} tradable results, {done['day'].min().date()} to {done['day'].max().date()}"
    )
    print(f"costs {cost:.2f}% a round trip; every variant tried in this project so far: {trials}")

    top, bottom = group(done, "top 10%"), group(done, "bottom 10%")
    per = top.groupby("month")["reaction"].mean() - bottom.groupby("month")["reaction"].mean()
    per = per.dropna()
    check = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
    print(
        f"\ndata check: top 10% reacted {top['reaction'].mean():+.2f}%, bottom 10% {bottom['reaction'].mean():+.2f}%; t {check:+.1f}"
    )
    if not check >= T_MIN:
        print("INVALID: Yahoo's estimates do not describe what the market expected. The table below decides nothing.")

    print(f"\n{'group':<12}{'hold':>5}{'n':>6}{'net':>8}{'excess':>8}{'t':>6}{'halves':>16}{'deflated':>9}")
    for name in GROUPS:
        for hold in HOLDS:
            c = cell(done, name, hold, cost)
            s = c["summary"]
            print(
                f"{name:<12}{hold:>5}{c['n']:>6}{c['net']:>+7.2f}%{s['mean']:>+7.2f}%{s['t']:>+6.1f}"
                f"{s['h1']:>+7.2f}%{s['h2']:>+7.2f}%{deflated_t(s['t'], trials).probability:>9.2f}"
            )
    if not check >= T_MIN:
        return
    failed = verdict(done, cost, cost2)
    print(f"\nthe test, {JUDGED[0]} held {JUDGED[1]}:", "PASS" if not failed else "FAIL: " + "; ".join(failed))


def main() -> None:
    if sys.argv[1:] == ["test"]:
        test()
        return
    wait_for_close()
    fetch_earnings(list(json.loads(DATES.read_text())))
    download()
    test()


if __name__ == "__main__":
    main()
