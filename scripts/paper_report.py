"""Full paper-trading history, both accounts: every closed trade and open position, and realized P&L by day, by week, and
in total, for the swing account, the intraday account, and the two combined.

    python scripts/paper_report.py                         # the live databases (DATABASE_URL / INTRADAY_DATABASE_URL)
    python scripts/paper_report.py --no-trades              # P&L tables only, skip the trade-by-trade listing
    python scripts/paper_report.py --database-url sqlite:///path/to/trading.db --intraday-database-url sqlite:///path/to/intraday.db"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import intraday_db_url, make_paper_broker  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.india import IST  # noqa: E402
from src.engine.paper_report import group_by_day, group_by_week, merge_periods, totals  # noqa: E402
from src.intraday.strategy import intraday_fees  # noqa: E402


def period_table(title: str, rows: list) -> None:
    print(f"\n{title}\n{'period':<12}{'trades':>7}{'win':>6}{'net P&L':>12}{'fees':>9}")
    for r in rows:
        win = f"{r.win_rate:.0%}" if r.win_rate is not None else "n/a"
        print(f"{r.label:<12}{r.trades:>7}{win:>6}{r.net_pnl:>+12,.0f}{r.fees:>9,.0f}")


def trade_table(label: str, trades: list) -> None:
    print(f"\n{label}: every closed trade, oldest first\n"
          f"{'opened':<11}{'closed':<11}{'symbol':<12}{'qty':>5}{'buy':>10}{'sell':>10}{'net P&L':>10}{'why':>8}")
    for t in sorted(trades, key=lambda t: t["closed_at"]):
        print(f"{t['opened_at'].date()!s:<11}{t['closed_at'].date()!s:<11}{t['symbol']:<12}{t['qty']:>5}"
              f"{t['entry_price']:>10,.2f}{t['exit_price']:>10,.2f}{t['net_pnl']:>+10,.0f}{t['reason']:>8}")


def account_report(label: str, trades: list, holdings: list, equity: float, start: float, show_trades: bool) -> None:
    by_day, by_week, total = group_by_day(trades, IST), group_by_week(trades, IST), totals(trades, IST)
    win_rate = f"{total.win_rate:.0%}" if total.win_rate is not None else "n/a"
    print(f"\n{'=' * 60}\n{label}: {len(holdings)} open, {total.trades} closed, "
          f"started Rs {start:,.0f} -> now Rs {equity:,.0f} ({equity / start - 1:+.2%})\n"
          f"realized P&L Rs {total.net_pnl:+,.0f}, fees Rs {total.fees:,.0f}, win rate {win_rate}")
    period_table("by day", by_day)
    period_table("by week", by_week)
    if show_trades and trades:
        trade_table(label, trades)
    if holdings:
        print(f"\n{label}: open positions\n{'symbol':<12}{'bought':<11}{'qty':>5}{'buy':>10}{'now':>10}{'P&L':>10}")
        for h in sorted(holdings, key=lambda h: h["pnl"]):
            print(f"{h['symbol']:<12}{h['opened_at'].date()!s:<11}{h['qty']:>5}{h['avg_price']:>10,.2f}{h['price']:>10,.2f}{h['pnl']:>+10,.0f}")


def main():
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--database-url", default=settings.database_url, help="the swing account's database")
    ap.add_argument("--intraday-database-url", default=intraday_db_url(), help="the intraday account's database")
    ap.add_argument("--no-trades", action="store_true", help="skip the trade-by-trade listing, just the P&L tables")
    args = ap.parse_args()

    swing = make_paper_broker(settings, args.database_url)
    intraday = make_paper_broker(settings, args.intraday_database_url, fees=intraday_fees)
    swing_summary, intraday_summary = swing.summary(), intraday.summary()
    swing_trades, intraday_trades = swing.trade_history(), intraday.trade_history()

    account_report("SWING", swing_trades, swing.holdings(), swing_summary["equity"], swing_summary["initial_cash"], not args.no_trades)
    account_report("INTRADAY", intraday_trades, intraday.holdings(), intraday_summary["equity"], intraday_summary["initial_cash"],
                   not args.no_trades)

    combined_start = swing_summary["initial_cash"] + intraday_summary["initial_cash"]
    combined_equity = swing_summary["equity"] + intraday_summary["equity"]
    print(f"\n{'=' * 60}\nCOMBINED (swing + intraday): started Rs {combined_start:,.0f} -> now Rs {combined_equity:,.0f} "
          f"({combined_equity / combined_start - 1:+.2%})")
    period_table("by day", merge_periods(group_by_day(swing_trades, IST), group_by_day(intraday_trades, IST)))
    period_table("by week", merge_periods(group_by_week(swing_trades, IST), group_by_week(intraday_trades, IST)))


if __name__ == "__main__":
    main()
