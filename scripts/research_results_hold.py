"""Does a strong reaction to results keep paying for days after costs? (pre-declared 2026-11-03, before any result was seen.)

    python scripts/research_results_hold.py download   # Angel One daily candles to research_cache/angel_daily.pkl (resumes)
    python scripts/research_results_hold.py test

Prices are Angel One's. Dates come from research_cache/results_dates.json (Yahoo's, the one history on hand, used for dates only and
with the owner's agreement). Yahoo's times are unreliable, so nothing depends on the hour: the reaction is measured from the close
before the date to the close after it, and the entry is the open of the session after that, when the news is out either way.

A stock qualifies if its reaction beats the median stock's by the threshold, it closed at Rs 100 or more, and it traded at least Rs 10
crore a day over the last 20 sessions (the swing screener's floor). A window holding an overnight move beyond 30% is dropped: Angel One's
candles are not adjusted, so that is probably a split or bonus. Today's listings only, which flatters the returns.

Cells: thresholds 2%, 5%, 10% x holds of 5, 10, 20 sessions. Net means after delivery costs on a live Rs 2,200 position; excess is net
less the median stock over the same days. A placebo takes the same reactions on days at least 5 sessions from any results date.
Gate for a cell, all of them: at least 300 trades; mean net > 0; mean excess > 0 with t >= 2.6, clustered by calendar month since holds
overlap; both halves of the months > 0; still positive at twice the slippage; beats its placebo; and the neighbouring cells (the next
threshold and the next hold) keep mean excess > 0. A pass makes the rule a candidate for paper trading in the ORB account, no more. A fail
deletes the downloaded file, keeping only this script's result in the trial register."""

import json
import logging
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config  # noqa: E402, F401  (loads .env: the Angel One keys)
from src.data.angel import AngelSession, load_tokens  # noqa: E402
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.significance import deflated_t  # noqa: E402

CACHE = ROOT / "research_cache"
DAILY, DATES, TRIALS = CACHE / "angel_daily.pkl", CACHE / "results_dates.json", ROOT / "docs" / "trading" / "research-trials.csv"
PACE, TRIES, CHUNK_DAYS, CHUNKS = 1.5, 5, 1900, 2  # the broker refuses bursts; about 2,000 days fit in one request
THRESHOLDS, HOLDS = (2.0, 5.0, 10.0), (5, 10, 20)
MIN_PRICE, MIN_VALUE, SPLIT_MOVE, PLACEBO_GAP = 100.0, 1e8, 0.30, 5
MIN_TRADES, T_MIN, POSITION = 300, 2.6, 2200.0


def fetch(api, token: str) -> list:
    """All chunks of one stock, oldest first; raises if a chunk cannot be read, so that a stock is never saved half done."""
    rows: list = []
    end = date.today()
    for _ in range(CHUNKS):
        start = end - timedelta(days=CHUNK_DAYS)
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": f"{start} 09:15",
            "todate": f"{end} 15:30",
        }
        for attempt in range(TRIES):
            try:
                rows = (api.getCandleData(params).get("data") or []) + rows
                break
            except Exception:
                if attempt == TRIES - 1:
                    raise
                time.sleep(PACE * 5 * (attempt + 1))
        time.sleep(PACE)
        end = start - timedelta(days=1)
    return rows


def download() -> None:
    """In passes until every stock is saved or a pass adds none: the broker refuses requests while the live backfill runs."""
    tokens = load_tokens(CACHE)
    wanted = [s for s, v in json.loads(DATES.read_text()).items() if v.get("dates") and s in tokens]
    have = pickle.loads(DAILY.read_bytes()) if DAILY.exists() else {}  # nosec B301
    api = AngelSession().api()
    while True:
        missing = [s for s in wanted if s not in have]
        print(f"{len(have)} of {len(wanted)} saved", flush=True)
        added = 0
        for symbol in missing:
            try:
                rows = fetch(api, tokens[symbol])
            except Exception as e:
                print(f"{symbol}: {type(e).__name__}", flush=True)
                continue
            added += 1
            if rows:
                frame = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
                frame.index = pd.to_datetime(frame.pop("ts")).dt.tz_localize(None).dt.normalize()
                have[symbol] = frame[~frame.index.duplicated()].sort_index()
            else:
                have[symbol] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            if added % 25 == 0:
                DAILY.write_bytes(pickle.dumps(have))
                print(f"{len(have)} of {len(wanted)} saved", flush=True)
        DAILY.write_bytes(pickle.dumps(have))
        if not added or len(have) >= len(wanted):
            print(f"finished: {len(have)} of {len(wanted)}", flush=True)
            return


def clustered(values: pd.Series, months: pd.Series) -> dict:
    per = values.groupby(months.values).mean().sort_index()
    half = len(per) // 2
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 2 and per.std(ddof=1) > 0 else float("nan")
    return {"mean": values.mean(), "t": t, "h1": per.iloc[:half].mean(), "h2": per.iloc[half:].mean(), "months": len(per)}


def frames(daily):
    return {
        k: pd.DataFrame({s: df[k] for s, df in daily.items() if len(df)}).sort_index().astype(float)
        for k in ("Open", "Close", "Volume")
    }


def trades(px, flags, hold, placebo_days=None):
    """Long entries the session after a flagged reaction: (flag frame of excess reaction, already filtered) -> one row each."""
    opens, closes = px["Open"].to_numpy(), px["Close"].to_numpy()
    jump = np.abs(opens[1:] / closes[:-1] - 1) > SPLIT_MOVE
    jump = np.vstack([np.zeros((1, opens.shape[1]), bool), jump])
    rows = []
    dates = px["Open"].index
    median_open_close = {}
    for i, j in zip(*np.nonzero(flags.to_numpy())):
        entry, exit_i = i + 1, i + hold
        if (
            exit_i >= len(dates)
            or not np.isfinite(opens[entry, j])
            or not np.isfinite(closes[exit_i, j])
            or jump[i - 2 : exit_i + 1, j].any()
        ):
            continue
        key = (entry, exit_i)
        if key not in median_open_close:
            median_open_close[key] = float(np.nanmedian(closes[exit_i] / opens[entry] - 1))
        rows.append((dates[entry], closes[exit_i, j] / opens[entry, j] - 1, median_open_close[key]))
    out = pd.DataFrame(rows, columns=["day", "gross", "market"])
    out["month"] = out["day"].dt.strftime("%Y-%m")
    return out


def test() -> None:
    daily = pickle.loads(DAILY.read_bytes())  # nosec B301
    px = frames(daily)
    close, dates = px["Close"], px["Close"].index
    raw = json.loads(DATES.read_text())
    is_results = pd.DataFrame(False, index=dates, columns=close.columns)
    near = is_results.copy()
    for symbol in close.columns:
        for stamp in raw[symbol]["dates"]:
            at = dates.searchsorted(pd.Timestamp(stamp[:10]))
            if 1 <= at < len(dates) - 2:
                is_results.iat[at, is_results.columns.get_loc(symbol)] = True
                near.iloc[max(0, at - PLACEBO_GAP) : at + PLACEBO_GAP + 1, near.columns.get_loc(symbol)] = True
    print(
        f"{len(close.columns)} stocks, {dates[0].date()} to {dates[-1].date()}, {int(is_results.to_numpy().sum())} results dates in range"
    )
    # The reaction ending at the close after each date, stamped on the session the entry follows.
    reaction = close / close.shift(2) - 1
    excess = (reaction.sub(reaction.median(axis=1), axis=0)) * 100
    value = (px["Close"] * px["Volume"]).rolling(20).mean()
    tradable = (close >= MIN_PRICE) & (value >= MIN_VALUE)
    on_results = is_results.shift(1, fill_value=False)  # d + 1: the close after the date
    off_results = ~near.shift(1, fill_value=True)
    round_trip = (india_delivery_fees("BUY", POSITION) + india_delivery_fees("SELL", POSITION)) / POSITION
    cost, cost2 = round_trip + 2 * SLIPPAGE, round_trip + 4 * SLIPPAGE
    n_trials = int(pd.read_csv(TRIALS)["variants"].fillna(1).sum())
    cells = {}
    print(f"\ncosts {cost:.2%} a round trip; every variant tried in this project so far: {n_trials}")
    print(
        f"{'react':>6}{'hold':>5}{'n':>6}{'net':>8}{'excess':>8}{'t':>6}{'halves':>15}{'2x slip':>8}{'placebo':>9}{'p-n':>6}{'deflated':>9}"
    )
    for threshold in THRESHOLDS:
        for hold in HOLDS:
            strong = (excess >= threshold) & tradable
            t = trades(px, strong & on_results, hold)
            p = trades(px, strong & off_results, hold)
            if t.empty:
                continue
            net, ex = t["gross"] - cost, t["gross"] - t["market"] - cost
            s = clustered(ex, t["month"])
            placebo = float((p["gross"] - p["market"] - cost).mean()) if len(p) else float("nan")
            cells[(threshold, hold)] = dict(n=len(t), net=net.mean(), placebo=placebo, slip2=(t["gross"] - cost2).mean(), **s)
            d = deflated_t(float(s["t"]), n_trials) if s["t"] == s["t"] else None
            print(
                f"{threshold:6.0f}{hold:5d}{len(t):6d}{net.mean():+8.2%}{s['mean']:+8.2%}{s['t']:+6.1f}{s['h1']:+8.2%}{s['h2']:+7.2%}"
                f"{cells[(threshold, hold)]['slip2']:+8.2%}{placebo:+9.2%}{len(p):6d}{getattr(d, 'probability', float('nan')):9.2f}"
            )

    def neighbours(threshold, hold):
        a, b = THRESHOLDS.index(threshold), HOLDS.index(hold)
        keys = [(THRESHOLDS[a + k], hold) for k in (-1, 1) if 0 <= a + k < len(THRESHOLDS)]
        keys += [(threshold, HOLDS[b + k]) for k in (-1, 1) if 0 <= b + k < len(HOLDS)]
        return all(k in cells and cells[k]["mean"] > 0 for k in keys)

    print("\ngate:")
    for (threshold, hold), c in cells.items():
        checks = {
            "n": c["n"] >= MIN_TRADES, "net": c["net"] > 0, "excess": c["mean"] > 0, "t": c["t"] >= T_MIN, "halves": c["h1"] > 0 and c["h2"] > 0,
            "2x": c["slip2"] > 0, "placebo": c["mean"] > c["placebo"], "neighbours": neighbours(threshold, hold),
        }  # fmt: skip
        failed = [k for k, ok in checks.items() if not ok]
        print(f"  react {threshold:.0f}% hold {hold}: {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    {"download": download, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "test"]()
