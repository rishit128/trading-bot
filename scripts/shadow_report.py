"""The record-only tests: news-backed breakouts (S1), the swing AI against the rule (S2), filings (S3 to S6), results-day gaps (S7), the 09:30
feed fix, and swing positions
opened outside the normal order path. Read only. Evidence counts from EVIDENCE_FROM; what came before was seen while the tests
were designed, so it is shown apart. Fixed in advance in src/research/shadow.py and STRATEGY.md.

    python scripts/shadow_report.py            # everything, on screen
    python scripts/shadow_report.py --summary  # the short version, on screen
    python scripts/shadow_report.py --notify   # the short version to Telegram; the intraday bot runs this after each close"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config  # noqa: E402, F401  (loads .env: the Telegram keys)
from src.engine.costs import SLIPPAGE, india_delivery_fees  # noqa: E402
from src.monitoring.telegram import Notifier  # noqa: E402
from src.news.watcher import yahoo_market_cap_crore  # noqa: E402
from src.ops.shadow_watch import SESSION_FINISHED, load_state, new_verdicts, save_state  # noqa: E402
from src.research.shadow import (  # noqa: E402
    DRIFT_FROM,
    DRIFT_HOLD,
    FILING_RULES,
    HOLDS,
    JUDGED_HOLD,
    RESULTS_GAP_FROM,
    Stats,
    ai_cells,
    ai_verdict,
    catalyst_verdict,
    drift_candidates,
    drift_trades,
    drift_verdict,
    filing_trades,
    filing_verdict,
    follow_filings,
    profit_rise_crore,
    results_gap_trades,
    results_gap_verdict,
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
CAPS_FILE = ROOT / "research_cache" / "drift_caps.json"
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


def tag_news(signals: pd.DataFrame) -> pd.DataFrame:
    filings = read("news.db", "select symbol, session, published from news_filings where material = 1 and direction = 'up'")
    filings["published"] = pd.to_datetime(filings["published"], utc=True)
    filings["session"] = filings["session"].astype(str).str[:10]
    return signals.assign(news=with_catalyst(signals, filings))


def report_news_breakouts(signals: pd.DataFrame) -> None:
    print(
        f"\nS1 breakouts in stocks with material good news (orb-v4 and {INTRADAY_VERSION}, net of costs, taken or skipped, not trade-to-trade):"
    )
    if signals.empty:
        print("  no signals")
        return
    signals = tag_news(signals)
    for label, part in (
        ("seen, not evidence", signals[signals["day"] < EVIDENCE_FROM]),
        ("evidence", signals[signals["day"] >= EVIDENCE_FROM]),
    ):
        print(f" {label}:")
        print(line("with news", stats(part[part["news"]])))
        print(line("the rest", stats(part[~part["news"]])))
    evidence = signals[signals["day"] >= EVIDENCE_FROM]
    print(" verdict:", catalyst_verdict(stats(evidence[evidence["news"]]), stats(evidence[~evidence["news"]])))


def ai_decisions() -> pd.DataFrame:
    decisions = read(
        "trading.db",
        "select symbol, bar_date, rule_action, final_action, candidate_source from decisions where rule_action is not null order by id",
    )
    outcomes = read(
        "trading.db", "select symbol, bar_date, gross_return from decision_outcomes where horizon = ?", (SWING_HORIZON,)
    )
    return ai_cells(decisions, outcomes)


def report_ai() -> None:
    print(f"\nS2 the swing AI against the rule (new entries, {SWING_HORIZON}-session gross return):")
    cells = ai_decisions()
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


def filings_and_bars() -> tuple:
    filings = read(
        "news.db",
        "select symbol, session, created_at, event_type, direction, material, numbers_json from news_filings "
        "where label_status = 'LABELLED'",
    )
    filings["seen"] = pd.to_datetime(filings["created_at"], utc=True).dt.tz_convert(IST)
    filings["session"] = filings["session"].astype(str).str[:10]
    bars = read(
        "intraday.db",
        "select symbol, day, ts, open, high, low, close from angel_bars where day >= ?",
        (filings["session"].min(),),
    )
    bars["t"] = pd.to_datetime(bars["ts"]).dt.tz_localize("UTC").dt.tz_convert(IST)
    return filings, bars


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


def delivery_cost() -> float:
    fees = india_delivery_fees("BUY", LIVE_POSITION) + india_delivery_fees("SELL", LIVE_POSITION)
    return fees / LIVE_POSITION * 100 + 2 * SLIPPAGE * 100


def results_hold_verdict(good: pd.DataFrame, cost: float) -> str:
    judged = f"ret_{JUDGED_HOLD}"
    if judged not in good:
        return "RUNNING (0 results)"
    held = good[good["day"] >= EVIDENCE_FROM].dropna(subset=[judged])
    held = held.assign(net=held[judged] - cost, excess=held[judged] - held[f"mkt_{JUDGED_HOLD}"] - cost)
    return results_verdict(stats(held), stats(held, "excess"))


def good_results(followed: pd.DataFrame) -> pd.DataFrame:
    return followed[(followed["event_type"] == "results") & (followed["direction"] == "up") & followed["material"]]


def report_results_hold(followed: pd.DataFrame) -> None:
    cost = delivery_cost()
    print(
        f"\nS4 material good results, bought and held (net of {cost:.2f}% delivery costs on Rs {LIVE_POSITION:,.0f}; excess is over the median stock):"
    )
    if followed.empty:
        print("  none yet")
        return
    good = good_results(followed)
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
    print(" verdict:", results_hold_verdict(good, cost))


def report_filing_trades(trades: pd.DataFrame) -> None:
    print("\nS5 same-day trades on material filings (net of 0.21% intraday costs; excess is net, beyond the median stock):")
    for rule in FILING_RULES:
        part = trades[trades["rule"] == rule] if not trades.empty else trades
        print(f" {rule}:")
        if part.empty:
            print("  none yet")
        seen, evidence = (part, part) if part.empty else (part[part["day"] < EVIDENCE_FROM], part[part["day"] >= EVIDENCE_FROM])
        print(line("seen, not evidence", stats(seen)))
        print(line("   excess", stats(seen, "excess")))
        print(line("evidence", stats(evidence)))
        print(line("   excess", stats(evidence, "excess")))
        print("  verdict:", filing_verdict(stats(evidence), stats(evidence, "excess"), stats(evidence, "net_2x")))


def drift_caps(filings: pd.DataFrame) -> dict:
    """Market values for S6, read once near the filing and kept: asked again later, they would already hold what the stock did next."""
    known = json.loads(CAPS_FILE.read_text()) if CAPS_FILE.exists() else {}
    wanted = filings[(filings["event_type"] == "results") & (filings["session"] >= DRIFT_FROM)]
    for symbol, session, numbers in zip(wanted["symbol"], wanted["session"], wanted["numbers_json"]):
        key = f"{symbol}|{session}"
        rise = profit_rise_crore(numbers)
        if key not in known and rise is not None and rise > 0:
            cap = yahoo_market_cap_crore(symbol)
            if cap:
                known[key] = cap
    CAPS_FILE.write_text(json.dumps(known, indent=0))
    return known


def profit_drift(filings: pd.DataFrame, bars: pd.DataFrame, today: str) -> tuple:
    candidates = drift_candidates(filings, drift_caps(filings))
    return drift_trades(candidates, bars[["symbol", "day", "t", "open", "close"]], today, delivery_cost())


def report_profit_drift(trades: pd.DataFrame, running: int) -> None:
    print(
        f"\nS6 large profit rises held {DRIFT_HOLD} sessions (net of {delivery_cost():.2f}% delivery costs; excess is over the median stock):"
    )
    print(line("finished", stats(trades)))
    print(line("   excess", stats(trades, "excess")))
    print(" verdict:", drift_verdict(stats(trades), stats(trades, "excess"), running))


def report_results_gaps(gaps: pd.DataFrame) -> None:
    print("\nS7 the gap-and-go entry rule on the session after results (net of 0.21% intraday costs):")
    if gaps.empty:
        print("  none yet")
    seen = gaps[gaps["day"] < RESULTS_GAP_FROM] if not gaps.empty else gaps
    evidence = gaps[gaps["day"] >= RESULTS_GAP_FROM] if not gaps.empty else gaps
    print(line("seen, not evidence", stats(seen)))
    print(line("evidence", stats(evidence)))
    print(" verdict:", results_gap_verdict(stats(evidence), stats(evidence, "net_2x")))


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


def session_cutoff(now: datetime) -> str:
    """The first day that does not count as finished: today's session counts once its final bars are in."""
    return (now.date() + timedelta(days=1) if now.time() >= SESSION_FINISHED else now.date()).isoformat()


def verdicts(
    signals: pd.DataFrame, cells: pd.DataFrame, followed: pd.DataFrame, trades: pd.DataFrame, drift: tuple, gaps: pd.DataFrame
) -> dict:
    """Every judged test's standing, from evidence alone."""
    out = {}
    if signals.empty:
        out["S1 breakouts on news"] = "RUNNING (no signals yet)"
    else:
        evidence = tag_news(signals[signals["day"] >= EVIDENCE_FROM]) if (signals["day"] >= EVIDENCE_FROM).any() else None
        out["S1 breakouts on news"] = (
            catalyst_verdict(stats(evidence[evidence["news"]]), stats(evidence[~evidence["news"]]))
            if evidence is not None
            else "RUNNING (0 signals)"
        )
    out["S2 AI against the rule"] = (
        ai_verdict(cells[cells["day"] >= EVIDENCE_FROM]) if not cells.empty else "RUNNING (0 disagreements)"
    )
    out["S4 hold good results"] = (
        results_hold_verdict(good_results(followed), delivery_cost()) if not followed.empty else "RUNNING (0 results)"
    )
    for rule in FILING_RULES:
        part = trades[trades["rule"] == rule] if not trades.empty else trades
        evidence = part[part["day"] >= EVIDENCE_FROM] if not part.empty else part
        out[f"S5 {rule}"] = filing_verdict(stats(evidence), stats(evidence, "excess"), stats(evidence, "net_2x"))
    out["S6 profit drift"] = drift_verdict(stats(drift[0]), stats(drift[0], "excess"), drift[1])
    evidence = gaps[gaps["day"] >= RESULTS_GAP_FROM] if not gaps.empty else gaps
    out["S7 results-day gap-and-go"] = results_gap_verdict(stats(evidence), stats(evidence, "net_2x"))
    return out


def feed_line(signals: pd.DataFrame) -> str:
    """The 09:30 feed fix's share, against the 10% limit."""
    if signals.empty:
        return "09:30 feed fix: no entries yet"
    known = signals[signals["final_signal"].notna() & (signals["day"] >= FEED_FIX_FROM) & (signals["outcome"] == "ENTERED")]
    early = known[known["bar_time"].dt.strftime("%H:%M") < "10:05"]
    if early.empty:
        return "09:30 feed fix: no entries yet"
    return f"09:30 feed fix: {(early['final_signal'] == 0).mean():.0%} of {len(early)} early entries vanished (limit 10%)"


def summary(day: str, standing: dict, trades: pd.DataFrame, feed: str) -> str:
    lines = [f"Test tracker, {day} (calculated only: no trades, no money)"]
    for name, verdict in standing.items():
        lines.append(f"{name}: {verdict}")
        rule = name[3:]
        part = trades[(trades["rule"] == rule) & (trades["day"] >= EVIDENCE_FROM)] if not trades.empty else trades
        if len(part):
            s = stats(part)
            lines.append(f"   so far net {s.mean:+.2f}% a trade, won {s.win:.0%}")
    lines.append(feed)
    return "\n".join(lines)


def tracker(notify_owner: bool) -> None:
    now = datetime.now(ZoneInfo(IST))
    cutoff = session_cutoff(now)
    signals, cells = scored_signals(), ai_decisions()
    filings, bars = filings_and_bars()
    followed, trades = follow_filings(filings, bars, cutoff), filing_trades(filings, bars, cutoff)
    day = now.date().isoformat()
    gaps = results_gap_trades(filings, bars, cutoff)
    standing = verdicts(signals, cells, followed, trades, profit_drift(filings, bars, cutoff), gaps)
    text = summary(day, standing, trades, feed_line(signals))
    print(text)
    if not notify_owner:
        return
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise SystemExit("Telegram is not configured")
    notifier = Notifier(token, chat)
    before = load_state().get("verdicts", {})
    if not notifier.send(text):
        raise SystemExit("Telegram did not take the message")
    for verdict in new_verdicts(before, standing):
        notifier.send(
            f"Test verdict reached.\n{verdict}\nNothing changes in any account until you agree to it. Ask Claude to look at it."
        )
    save_state({"sent_day": day, "verdicts": standing, "summary": text})


def main() -> None:
    if "--notify" in sys.argv or "--summary" in sys.argv:
        tracker("--notify" in sys.argv)
        return
    signals = scored_signals()
    report_news_breakouts(signals)
    report_ai()
    filings, bars = filings_and_bars()
    today = session_cutoff(datetime.now(ZoneInfo(IST)))
    followed = follow_filings(filings, bars, today)
    report_kinds(followed)
    report_results_hold(followed)
    report_filing_trades(filing_trades(filings, bars, today))
    report_profit_drift(*profit_drift(filings, bars, today))
    report_results_gaps(results_gap_trades(filings, bars, today))
    report_feed(signals)
    report_unrecorded()


if __name__ == "__main__":
    main()
