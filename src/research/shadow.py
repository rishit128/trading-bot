"""Record-only comparisons that need no account and change no trade (pre-declared 2026-11-03; scripts/shadow_report.py).

S1 asks whether the breakout rule works on stocks with real news, the only lead the research left. Every breakout the ORB engine
records, taken or skipped, is scored on the final bars with the live exits, and split by whether a material good-news filing
was published before the signal. S2 asks whether the swing AI's decisions add anything where they differ from the rule. S3 and S4
follow each filing from the first bar after it was seen: which kinds of news move, and whether buying good results and holding for
days beats the costs that a same-day trade cannot (S4 is the test; the moves are larger). S5 asks whether a same-day trade on a filing
beats its costs when the rule looks at how far the price has already moved. S6 asks whether a large rise in profit keeps paying for
about three months, as an Indian study found. Prices are Angel One's bars, dates the exchange filings'
own: no free history of either exists before 2026-10-21, so these run forward only."""

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

COST_PCT = 0.21  # round trip: fees plus 0.05% slippage a side
BAR = pd.Timedelta(minutes=5)
SQUARE_OFF = "15:15"
MIN_SIGNALS, MIN_DATES, T_GO = 100, 30, 2.0
MIN_DISAGREEMENTS, AI_EDGE = 100, 0.01
HOLDS = (1, 3, 5)  # sessions after the entry session's close; S4 judges 5 only, so that one test is run, not three
JUDGED_HOLD, MIN_RESULTS, T_RESULTS = 5, 100, 2.6
COST_2X_PCT = COST_PCT + 0.10  # twice the slippage: 0.05% more a side
FIRST_ENTRY, LAST_ENTRY = "09:20", "14:30"  # S5: past the opening auction; late enough news goes to the next session
FRESH_BELOW, FADE_FROM = 1.0, 3.0  # S5: % ahead of the median stock since the previous close, at entry
MIN_FILINGS, MIN_FILING_DATES, T_FILINGS = 100, 30, 2.6
DRIFT_FROM, DRIFT_MIN_SCORE, DRIFT_HOLD = "2026-11-05", 1.0, 60  # S6: first filing session; % of market value; sessions held
DRIFT_SPLIT_MOVE = 0.30  # Angel One's candles are not adjusted: an overnight move past this is probably a split or bonus
MIN_DRIFT, MIN_DRIFT_WEEKS, T_DRIFT = 100, 20, 2.6
CRORE_PER = (
    (r"crore|\bcrs?\b", 1.0),
    (r"lakh|\blacs?\b", 0.01),
    (r"million|\bmn\b", 0.1),
    (r"billion|\bbn\b", 100.0),
)
FILING_RULES = ("F1 not priced in, long", "F2 overreaction, short", "F3 bad news, short")


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


def filing_rule(direction: str, ahead: float) -> Optional[Tuple[str, int]]:
    """S5's rule and side for a material filing, given how far the stock is ahead of the median stock (%)."""
    if direction == "down":
        return FILING_RULES[2], -1
    if direction != "up":
        return None
    if ahead < FRESH_BELOW:
        return FILING_RULES[0], 1
    return (FILING_RULES[1], -1) if ahead >= FADE_FROM else None


def filing_trades(filings: pd.DataFrame, bars: pd.DataFrame, today: str) -> pd.DataFrame:
    """S5's same-day trades. Each stock's first material filing of a session enters at the open of the first bar from FIRST_ENTRY to
    LAST_ENTRY starting after we saw it, in that session or the next, and exits at the close of the last bar before the square-off.
    A bar with no range is locked at a price limit, so no fill. `filings`: symbol, session, seen (aware), direction, material; `bars`:
    symbol, day, t, open, high, low, close. Returns rule, side, ahead, and the % results gross, net, net_2x and excess (net of costs,
    beyond the median stock over the same bars)."""
    bars = bars[bars["day"] < today].sort_values("t")
    clock = bars["t"].dt.strftime("%H:%M")
    days = sorted(bars["day"].unique())
    position = {day: i for i, day in enumerate(days)}
    previous_close = bars.groupby(["day", "symbol"])["close"].last().unstack()
    last_close = bars[clock < SQUARE_OFF].groupby(["day", "symbol"])["close"].last().unstack()
    open_at = bars.pivot_table(index="t", columns="symbol", values="open")
    tradable = bars[(clock >= FIRST_ENTRY) & (clock <= LAST_ENTRY)]
    by_stock = {symbol: frame for symbol, frame in tradable.groupby("symbol")}
    material = filings[filings["material"].astype(bool)].sort_values("seen").drop_duplicates(["symbol", "session"])
    rows = []
    for symbol, seen, direction in zip(material["symbol"], material["seen"], material["direction"]):
        stock = by_stock.get(symbol)
        if stock is None:
            continue
        later = stock[stock["t"] >= seen]
        if later.empty:
            continue
        bar = later.iloc[0]
        at = position[bar["day"]]
        first_possible = next(i for i, day in enumerate(days) if day >= seen.date().isoformat())
        if at == 0 or at > first_possible + 1 or bar["high"] == bar["low"] or bar["open"] <= 0:
            continue
        before, entry = previous_close.iloc[at - 1], float(bar["open"])
        if not np.isfinite(before.get(symbol, np.nan)):
            continue
        opens = open_at.loc[bar["t"]]
        ahead = ((entry / before[symbol] - 1) - float((opens / before - 1).median())) * 100
        rule = filing_rule(direction, ahead)
        close = last_close.loc[bar["day"]]
        if rule is None or not np.isfinite(close.get(symbol, np.nan)):
            continue
        gross = rule[1] * (float(close[symbol]) / entry - 1) * 100
        market = rule[1] * float((close / opens - 1).median()) * 100
        row = dict(symbol=symbol, day=bar["day"], rule=rule[0], side=rule[1], ahead=ahead, gross=gross, net=gross - COST_PCT)
        row.update(net_2x=gross - COST_2X_PCT, excess=gross - market - COST_PCT)
        rows.append(row)
    return pd.DataFrame(rows)


def filing_verdict(net: Stats, excess: Stats, doubled: Stats) -> str:
    """S5's gate, fixed before any of its evidence, for each rule on its own."""
    if net.n < MIN_FILINGS or net.dates < MIN_FILING_DATES:
        return f"RUNNING ({net.n} of {MIN_FILINGS} filings, {net.dates} of {MIN_FILING_DATES} dates)"
    passed = (
        net.mean > 0
        and doubled.mean > 0
        and excess.mean > 0
        and excess.t >= T_FILINGS
        and excess.first_half > 0
        and excess.second_half > 0
    )
    return "PASS: a candidate design for the ORB account, on paper" if passed else "FAIL: this does not beat its costs"


def profit_rise_crore(numbers_json: Optional[str]) -> Optional[float]:
    """The quarter's profit less the year-ago quarter's, in Rs crore, from a filing's labelled numbers. Companies report in crore, lakh or
    million, and the unit is only in the words as written; None when the numbers or the unit are missing."""
    numbers = json.loads(numbers_json) if numbers_json else {}
    now, before = numbers.get("profit_quarter"), numbers.get("profit_year_ago")
    if now is None or before is None:
        return None
    written = str(numbers.get("as_written") or "").lower()
    for pattern, crore in CRORE_PER:
        if re.search(pattern, written):
            return (float(now) - float(before)) * crore
    return None


def drift_candidates(filings: pd.DataFrame, caps: Dict[str, float]) -> pd.DataFrame:
    """S6's filings: each stock's first results filing of a session from DRIFT_FROM with a known profit rise of at least DRIFT_MIN_SCORE
    % of market value. `filings`: symbol, session, seen, event_type, numbers_json. `caps`: "symbol|session" -> Rs crore."""
    results = filings[(filings["event_type"] == "results") & (filings["session"] >= DRIFT_FROM)].copy()
    results["rise"] = [profit_rise_crore(n) for n in results["numbers_json"]]
    results = results.dropna(subset=["rise"]).sort_values("seen").drop_duplicates(["symbol", "session"])
    results["cap"] = [caps.get(f"{a}|{b}") for a, b in zip(results["symbol"], results["session"])]
    results = results.dropna(subset=["cap"])
    return results[results["rise"] / results["cap"] * 100 >= DRIFT_MIN_SCORE]


def drift_trades(candidates: pd.DataFrame, bars: pd.DataFrame, today: str, cost: float) -> Tuple[pd.DataFrame, int]:
    """(matured trades, how many are still running). The first session that could react is the one the filing was seen before the close
    of; the entry is the next session's open and the exit the close DRIFT_HOLD sessions on. A window with a split-sized overnight move
    is dropped. `day` is the entry's week, since holds overlap. `bars`: symbol, day, t, open, close."""
    bars = bars[bars["day"] < today].sort_values("t")
    if candidates.empty or bars.empty:
        return pd.DataFrame(), 0
    opens = bars.groupby(["day", "symbol"])["open"].first().unstack().sort_index()
    closes = bars.groupby(["day", "symbol"])["close"].last().unstack().sort_index()
    days = list(opens.index)
    rows, running = [], 0
    for symbol, seen in zip(candidates["symbol"], candidates["seen"]):
        date = seen.date().isoformat()
        react = next((i for i, day in enumerate(days) if day > date or (day == date and seen.strftime("%H:%M") < "15:30")), None)
        if react is None or react + 1 >= len(days):
            running += 1
            continue
        entry_at, exit_at = react + 1, react + DRIFT_HOLD
        if exit_at >= len(days):
            running += 1
            continue
        if symbol not in opens or not np.isfinite(opens[symbol].iloc[entry_at]) or not np.isfinite(closes[symbol].iloc[exit_at]):
            continue
        overnight = (
            opens[symbol].iloc[entry_at + 1 : exit_at + 1].to_numpy() / closes[symbol].iloc[entry_at:exit_at].to_numpy() - 1
        )
        if np.nanmax(np.abs(overnight), initial=0.0) > DRIFT_SPLIT_MOVE:
            continue
        gross = (float(closes[symbol].iloc[exit_at]) / float(opens[symbol].iloc[entry_at]) - 1) * 100
        market = float((closes.iloc[exit_at] / opens.iloc[entry_at] - 1).median()) * 100
        week = pd.Timestamp(days[entry_at]).to_period("W").start_time.date().isoformat()
        rows.append(dict(symbol=symbol, day=week, gross=gross, net=gross - cost, excess=gross - market - cost))
    return pd.DataFrame(rows), running


def drift_verdict(net: Stats, excess: Stats, running: int) -> str:
    """S6's gate, fixed before any of its evidence; `dates` here counts entry weeks."""
    if net.n < MIN_DRIFT or net.dates < MIN_DRIFT_WEEKS:
        return f"RUNNING ({net.n} of {MIN_DRIFT} finished, {net.dates} of {MIN_DRIFT_WEEKS} weeks, {running} still held)"
    passed = net.mean > 0 and excess.mean > 0 and excess.t >= T_DRIFT and excess.first_half > 0 and excess.second_half > 0
    return "PASS: a candidate design for the ORB account, on paper" if passed else "FAIL: profit drift does not pay after costs"
