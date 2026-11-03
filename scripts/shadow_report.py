"""The record-only tests: news-backed breakouts (S1), the swing AI against the rule (S2), the 09:30 feed fix, and swing positions
opened outside the normal order path. Read only. Evidence counts from EVIDENCE_FROM; what came before was seen while the tests
were designed, so it is shown apart. Fixed in advance in src/research/shadow.py and STRATEGY.md.

    python scripts/shadow_report.py"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.research.shadow import (  # noqa: E402
    HOLDS,
    JUDGED_HOLD,
    Stats,
    ai_cells,
    ai_verdict,
    catalyst_verdict,
    follow_filings,
    results_verdict,
    score_signal,
    stats,
    unrecorded_entries,
    with_catalyst,
)
from src.versions import INTRADAY_VERSION  # noqa: E402

EVIDENCE_FROM = "2026-11-04"  # the first session after the tests were fixed
FEED_FIX_FROM = "2026-11-04"  # the first session whose 09:30 step reads Angel One's bars
IST = "Asia/Kolkata"
SWING_HORIZON = 20
S3_MIN = 5  # a kind of news with fewer filings is not shown
LIVE_POSITION = 2200.0  # the swing account's 8.8% of Rs 25,000


def read(db: str, query: str, params=()) -> pd.DataFrame:
    with sqlite3.connect(f"file:{ROOT / db}?mode=ro", uri=True) as con:
        return pd.read_sql(query, con, params=params)


def line(name: str, s: Stats) -> str:
    if not s.n:
        return f"  {name:<22} none yet"
    return f"  {name:<22} n={s.n:<5} dates={s.dates:<3} mean {s.mean:+.3f}% win {s.win:.0%} t {s.t:+.2f} halves {s.first_half:+.2f} / {s.second_half:+.2f}"


def scored_signals() -> pd.DataFrame:
    signals = read(
        "intraday.db",
        "select day, symbol, bar_time, signal_price, stop, target, outcome, final_signal from intraday_signals "
        "where system_version in ('orb-v4', ?) and signal_price > 0 and coalesce(reason, '') not like '%trade-to-trade%'",
        (INTRADAY_VERSION,),
    )
    if signals.empty:
        return signals
    signals["bar_time"] = pd.to_datetime(signals["bar_time"], utc=True).dt.tz_convert(IST)
    days = sorted(signals["day"].unique())
    marks = ",".join("?" * len(days))
    bars = pd.concat(
        read("intraday.db", f"select symbol, day, ts, high, low, close from {table} where day in ({marks})", days)
        for table in ("intraday_bars", "angel_bars")
    ).drop_duplicates(["symbol", "ts"], keep="last")
    bars["t"] = pd.to_datetime(bars["ts"]).dt.tz_localize("UTC").dt.tz_convert(IST)
    sessions = {key: frame.sort_values("t") for key, frame in bars.groupby(["symbol", "day"])}
    rows = []
    for r in signals.itertuples():
        session = sessions.get((r.symbol, r.day))
        scored = score_signal(session, r.bar_time, r.signal_price, r.stop, r.target) if session is not None else None
        rows.append(scored[1] if scored else float("nan"))
    signals["net"] = rows
    return signals.dropna(subset=["net"])


def report_news_breakouts(signals: pd.DataFrame) -> None:
    print(
        f"\nS1 breakouts in stocks with material good news (orb-v4 and {INTRADAY_VERSION}, net of costs, taken or skipped, not trade-to-trade):"
    )
    if signals.empty:
        print("  no signals")
        return
    filings = read("news.db", "select symbol, session, published from news_filings where material = 1 and direction = 'up'")
    filings["published"] = pd.to_datetime(filings["published"], utc=True)
    filings["session"] = filings["session"].astype(str).str[:10]
    signals = signals.assign(news=with_catalyst(signals, filings))
    for label, part in (
        ("seen, not evidence", signals[signals["day"] < EVIDENCE_FROM]),
        ("evidence", signals[signals["day"] >= EVIDENCE_FROM]),
    ):
        print(f" {label}:")
        print(line("with news", stats(part[part["news"]])))
        print(line("the rest", stats(part[~part["news"]])))
    evidence = signals[signals["day"] >= EVIDENCE_FROM]
    print(" verdict:", catalyst_verdict(stats(evidence[evidence["news"]]), stats(evidence[~evidence["news"]])))


def report_ai() -> None:
    print(f"\nS2 the swing AI against the rule (new entries, {SWING_HORIZON}-session gross return):")
    decisions = read(
        "trading.db",
        "select symbol, bar_date, rule_action, final_action, candidate_source from decisions where rule_action is not null order by id",
    )
    outcomes = read(
        "trading.db", "select symbol, bar_date, gross_return from decision_outcomes where horizon = ?", (SWING_HORIZON,)
    )
    cells = ai_cells(decisions, outcomes)
    if cells.empty:
        print("  no labelled decisions yet (an outcome needs 20 sessions to mature)")
        return
    for label, part in (
        ("seen, not evidence", cells[cells["day"] < EVIDENCE_FROM]),
        ("evidence", cells[cells["day"] >= EVIDENCE_FROM]),
    ):
        print(f" {label}:")
        for cell, group in part.groupby("cell"):
            print(line(cell, stats(group.assign(net=group["gross_return"] * 100))))
    print(" verdict:", ai_verdict(cells[cells["day"] >= EVIDENCE_FROM]))


def filings_followed() -> pd.DataFrame:
    filings = read(
        "news.db",
        "select symbol, session, created_at, event_type, direction, material from news_filings where label_status = 'LABELLED'",
    )
    filings["seen"] = pd.to_datetime(filings["created_at"], utc=True).dt.tz_convert(IST)
    filings["session"] = filings["session"].astype(str).str[:10]
    bars = read("intraday.db", "select symbol, day, ts, open, close from angel_bars where day >= ?", (filings["session"].min(),))
    bars["t"] = pd.to_datetime(bars["ts"]).dt.tz_localize("UTC").dt.tz_convert(IST)
    return follow_filings(filings, bars, datetime.now(ZoneInfo(IST)).date().isoformat())


def report_kinds(followed: pd.DataFrame) -> None:
    print("\nS3 filings by kind: first Angel One bar after we saw it, to the close, against the median stock (no costs taken):")
    if followed.empty or "ret_0" not in followed:
        print("  none yet")
        return
    moved = followed.dropna(subset=["ret_0"])
    moved = moved.assign(excess=moved["ret_0"] - moved["mkt_0"])
    for label, part in (
        ("seen, not evidence", moved[moved["day"] < EVIDENCE_FROM]),
        ("evidence", moved[moved["day"] >= EVIDENCE_FROM]),
    ):
        print(f" {label}:")
        if part.empty:
            print("  none yet")
        for (kind, direction), group in part.groupby(["event_type", "direction"]):
            if len(group) >= S3_MIN:
                print(line(f"{kind} {direction}", stats(group, "excess")))


def report_results_hold(followed: pd.DataFrame) -> None:
    cost = (
        india_delivery_fees("BUY", LIVE_POSITION) + india_delivery_fees("SELL", LIVE_POSITION)
    ) / LIVE_POSITION * 100 + 2 * SLIPPAGE * 100
    print(
        f"\nS4 material good results, bought and held (net of {cost:.2f}% delivery costs on Rs {LIVE_POSITION:,.0f}; excess is over the median stock):"
    )
    if followed.empty:
        print("  none yet")
        return
    good = followed[(followed["event_type"] == "results") & (followed["direction"] == "up") & followed["material"]]
    for hold in HOLDS:
        col = f"ret_{hold}"
        if col not in good:
            print(f"  {hold} sessions: none matured yet")
            continue
        held = good.dropna(subset=[col])
        held = held.assign(net=held[col] - cost, excess=held[col] - held[f"mkt_{hold}"] - cost)
        for label, part in (("seen", held[held["day"] < EVIDENCE_FROM]), ("evidence", held[held["day"] >= EVIDENCE_FROM])):
            print(line(f"{hold} sessions, {label}", stats(part)))
            print(line("   excess", stats(part, "excess")))
    judged = f"ret_{JUDGED_HOLD}"
    if judged in good:
        held = good[good["day"] >= EVIDENCE_FROM].dropna(subset=[judged])
        held = held.assign(net=held[judged] - cost, excess=held[judged] - held[f"mkt_{JUDGED_HOLD}"] - cost)
        print(" verdict:", results_verdict(stats(held), stats(held, "excess")))
    else:
        print(" verdict: RUNNING (0 results)")


def report_feed(signals: pd.DataFrame) -> None:
    print("\nFeed check: entries whose signal is gone on the final bars, before 10:05 (the step that read Yahoo until the fix):")
    if signals.empty:
        return
    known = signals[signals["final_signal"].notna()]
    early = known[known["bar_time"].dt.strftime("%H:%M") < "10:05"]
    for label, part in (
        ("before the fix", early[early["day"] < FEED_FIX_FROM]),
        ("after the fix", early[early["day"] >= FEED_FIX_FROM]),
    ):
        taken = part[part["outcome"] == "ENTERED"]
        if len(part):
            share = (part["final_signal"] == 0).mean()
            gone = (taken["final_signal"] == 0).mean() if len(taken) else float("nan")
            print(f"  {label:<15} all signals {share:.0%} of {len(part)} | entries {gone:.0%} of {len(taken)}")
        else:
            print(f"  {label:<15} none yet")
    print("  target: the entries' share at or under the 10% limit of docs/trading/forward-tests.md")


def report_unrecorded() -> None:
    print("\nSwing positions opened with no order on record (by hand; the core sleeve is left out):")
    trades = read("trading.db", "select symbol, opened_at, closed_at, net_pnl, entry_price, qty from paper_trades")
    held = read("trading.db", "select symbol, opened_at from paper_positions")
    orders = read("trading.db", "select symbol, created_at from orders where side = 'BUY' and status = 'filled'")
    ordered = list(zip(orders["symbol"], orders["created_at"].str[:10]))
    closed = list(zip(trades["symbol"], trades["opened_at"].str[:10]))
    manual = set(unrecorded_entries(closed, ordered))
    picked = trades[[key in manual for key in closed]]
    for r in picked.itertuples():
        print(f"  {r.symbol:<12} {r.opened_at[:10]} to {r.closed_at[:10]}  {r.net_pnl:+.0f}")
    still = [s for s in unrecorded_entries(list(zip(held["symbol"], held["opened_at"].str[:10])), ordered) if "BEES" not in s[0]]
    for symbol, day in still:
        print(f"  {symbol:<12} {day} still open")
    print(f"  {len(picked)} closed by hand-opened, net {picked['net_pnl'].sum():+.0f}; {len(still)} open")


def main() -> None:
    signals = scored_signals()
    report_news_breakouts(signals)
    report_ai()
    followed = filings_followed()
    report_kinds(followed)
    report_results_hold(followed)
    report_feed(signals)
    report_unrecorded()


if __name__ == "__main__":
    main()
