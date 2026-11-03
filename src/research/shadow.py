"""Record-only comparisons that need no account and change no trade (pre-declared 2026-11-03; scripts/shadow_report.py).

S1 asks whether the breakout rule works on stocks with real news, the only lead the research left. Every breakout the ORB engine
records, taken or skipped, is scored on the final bars with the live exits, and split by whether a material good-news filing
was published before the signal. S2 asks whether the swing AI's decisions add anything where they differ from the rule. S3 and S4
follow each filing from the first bar after it was seen: which kinds of news move, and whether buying good results and holding for
days beats the costs that a same-day trade cannot (S4 is the test; the moves are larger). Prices are Angel One's bars, dates the
exchange filings' own: no free history of either exists before 2026-10-21, so these run forward only."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

COST_PCT = 0.21  # round trip: fees plus 0.05% slippage a side
BAR = pd.Timedelta(minutes=5)
SQUARE_OFF = "15:15"
MIN_SIGNALS, MIN_DATES, T_GO = 100, 30, 2.0
MIN_DISAGREEMENTS, AI_EDGE = 100, 0.01
HOLDS = (1, 3, 5)  # sessions after the entry session's close; S4 judges 5 only, so that one test is run, not three
JUDGED_HOLD, MIN_RESULTS, T_RESULTS = 5, 100, 2.6


def score_signal(
    bars: pd.DataFrame, bar_time: pd.Timestamp, entry: float, stop: float, target: float
) -> Optional[Tuple[str, float]]:
    """(exit, net % after costs) of one long entry at the signal bar's close: the stop first when a bar touches both, as the paper
    broker does, else the last bar before the square-off. `bars` is one stock's session (columns t, high, low, close; t is
    aware IST). None when no bar follows the signal."""
    after = bars[(bars["t"] >= bar_time + BAR) & (bars["t"].dt.strftime("%H:%M") < SQUARE_OFF)]
    if after.empty or entry <= 0:
        return None
    for low, high in zip(after["low"], after["high"]):
        if low <= stop:
            return "STOP", (stop / entry - 1) * 100 - COST_PCT
        if high >= target:
            return "TARGET", (target / entry - 1) * 100 - COST_PCT
    return "CLOSE", (float(after["close"].iloc[-1]) / entry - 1) * 100 - COST_PCT


def with_catalyst(signals: pd.DataFrame, filings: pd.DataFrame) -> pd.Series:
    """Per signal (columns symbol, day, bar_time): a material good-news filing of that stock for that session published by the
    time the signal bar closed. Only what was known at the time counts. `filings` has symbol, session, published (aware)."""
    first = filings.groupby(["symbol", "session"])["published"].min()
    known = []
    for symbol, day, bar_time in zip(signals["symbol"], signals["day"], signals["bar_time"]):
        at = first.get((symbol, day))
        known.append(at is not None and at <= bar_time + BAR)
    return pd.Series(known, index=signals.index, dtype=bool)


@dataclass(frozen=True)
class Stats:
    n: int
    dates: int
    mean: float
    win: float
    t: float
    first_half: float
    second_half: float


def stats(frame: pd.DataFrame, column: str = "net") -> Stats:
    """Mean per signal, and t and halves from the per-date means (signals on one date are not independent)."""
    if frame.empty:
        return Stats(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    per_date = frame.groupby("day")[column].mean().sort_index()
    sd = per_date.std(ddof=1) if len(per_date) > 1 else 0.0
    half = len(per_date) // 2
    return Stats(
        len(frame),
        len(per_date),
        float(frame[column].mean()),
        float((frame[column] > 0).mean()),
        float(per_date.mean() / (sd / np.sqrt(len(per_date)))) if sd and sd > 0 else float("nan"),
        float(per_date.iloc[:half].mean()) if half else float("nan"),
        float(per_date.iloc[half:].mean()),
    )


def catalyst_verdict(news: Stats, rest: Stats) -> str:
    """S1's gate, fixed before any of its evidence."""
    if news.n < MIN_SIGNALS or news.dates < MIN_DATES:
        return f"RUNNING ({news.n} of {MIN_SIGNALS} signals, {news.dates} of {MIN_DATES} dates)"
    passed = news.mean > 0 and news.t >= T_GO and news.first_half > 0 and news.second_half > 0 and news.mean > rest.mean
    return "PASS: a candidate for the next ORB version" if passed else "FAIL: news does not rescue the breakout"


def ai_cells(decisions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Every new-entry decision with a rule verdict and an outcome, one per stock and bar (the last of the day), labelled by
    who said what. `decisions`: symbol, bar_date, rule_action, final_action, candidate_source; `outcomes`: symbol, bar_date,
    gross_return."""
    fresh = decisions[decisions["candidate_source"].fillna("") != "held"].dropna(subset=["rule_action", "bar_date"])
    fresh = fresh.drop_duplicates(["symbol", "bar_date"], keep="last")
    merged = fresh.merge(outcomes, on=["symbol", "bar_date"])
    merged["cell"] = [
        f"{rule.lower()} / AI {final.lower()}" for rule, final in zip(merged["rule_action"], merged["final_action"])
    ]
    merged["day"] = merged["bar_date"]
    return merged


def ai_verdict(cells: pd.DataFrame) -> str:
    """S2's gate, fixed before any of its evidence: where they disagree, the AI's side must beat the rule's by 1% over 20
    sessions (t >= 2). Otherwise the AI is not adding anything to the buy decision."""
    overruled = cells[cells["rule_action"] != cells["final_action"]]
    if len(overruled) < MIN_DISAGREEMENTS:
        return f"RUNNING ({len(overruled)} of {MIN_DISAGREEMENTS} disagreements)"
    frame = overruled.assign(edge=overruled["gross_return"].where(overruled["final_action"] == "BUY", -overruled["gross_return"]))
    edge = stats(frame, "edge")
    passed = edge.mean >= AI_EDGE and edge.t >= T_GO
    return "PASS: the AI adds something" if passed else "FAIL: rule-only is as good; a candidate for dropping the AI from buys"


def unrecorded_entries(opened: Iterable[Tuple[str, str]], ordered: Iterable[Tuple[str, str]]) -> list:
    """(symbol, date) of positions with no filled BUY order on that date: bought by hand or by the core sleeve."""
    known = set(ordered)
    return [entry for entry in opened if entry not in known]


def session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per stock and session from 5-minute bars (symbol, day, t, open, close), up to the square-off: `close` is the last bar
    before 15:15."""
    kept = bars[bars["t"].dt.strftime("%H:%M") < SQUARE_OFF].sort_values("t")
    return kept.groupby(["symbol", "day"], as_index=False).agg(close=("close", "last"))


def follow_filings(filings: pd.DataFrame, bars: pd.DataFrame, today: str) -> pd.DataFrame:
    """Each stock's first filing of a session, bought at the open of the first bar starting after we saw it (a filing seen after the
    last bar goes to the next session). `filings`: symbol, session, seen (aware), event_type, direction, material; `bars`: symbol,
    day, t, open, close. Adds entry_day, entry, ret_0 (to that session's close) and ret_N (N sessions later), as % moves, and the same
    for the median stock (mkt_0, mkt_N). A session still running is not used as an exit."""
    done = sorted(day for day in bars["day"].unique() if day < today)
    closes = session_bars(bars[bars["day"].isin(done)]).pivot(index="day", columns="symbol", values="close").sort_index()
    by_stock = {symbol: frame.sort_values("t") for symbol, frame in bars[bars["day"].isin(done)].groupby("symbol")}
    position = {day: i for i, day in enumerate(closes.index)}
    open_at = bars[bars["day"].isin(done)].pivot_table(index="t", columns="symbol", values="open")
    rows = []
    for f in filings.sort_values("seen").drop_duplicates(["symbol", "session"]).itertuples():
        stock = by_stock.get(f.symbol)
        if stock is None:
            continue
        later = stock[(stock["t"] >= f.seen) & (stock["t"].dt.strftime("%H:%M") < SQUARE_OFF)]
        if later.empty:
            continue
        t0, entry, day = later["t"].iloc[0], float(later["open"].iloc[0]), later["day"].iloc[0]
        row = dict(symbol=f.symbol, session=f.session, event_type=f.event_type, direction=f.direction, material=bool(f.material))
        row.update(day=day, entry=entry)
        base = open_at.loc[t0]
        for hold in (0, *HOLDS):
            i = position[day] + hold
            if i < len(closes):
                exit_close = closes.iloc[i]
                row[f"ret_{hold}"] = (float(exit_close[f.symbol]) / entry - 1) * 100
                row[f"mkt_{hold}"] = float((exit_close / base - 1).median()) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def results_verdict(s: Stats, excess: Stats) -> str:
    """S4's gate, fixed before any of its evidence: net of delivery costs and beyond the median stock over the same days."""
    if s.n < MIN_RESULTS:
        return f"RUNNING ({s.n} of {MIN_RESULTS} results)"
    passed = s.mean > 0 and excess.mean > 0 and excess.t >= T_RESULTS and excess.first_half > 0 and excess.second_half > 0
    return "PASS: a candidate design for the ORB account" if passed else "FAIL: good results do not pay after costs"
