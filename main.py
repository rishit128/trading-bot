"""Command-line entry point: parses the arguments and runs one command (the wiring lives in src/app/bootstrap.py)."""

import argparse
import os
import sys

from src.app.bootstrap import (
    load_settings_or_exit,
    preflight,
    run_portfolio_agent,
    run_trading_bot,
    show_intraday_reports,
)
from src.app.intraday_cli import run_intraday
from src.app.reports import print_candidates, print_graphs, print_positions, print_report
from src.logging_setup import configure_logging

# Read-only reports, in the order they are tried: if more than one flag is given, the first match wins.
REPORT_COMMANDS = (
    ("graph", print_graphs),
    ("screen", print_candidates),
    ("positions", print_positions),
    ("report", print_report),
)


def build_parser() -> argparse.ArgumentParser:
    """Every command-line option, grouped the way the README lists them."""
    parser = argparse.ArgumentParser(description="AI trading bot (Indian NSE stocks, paper trading).")
    run = parser.add_argument_group("run the swing bot")
    run.add_argument("--loop", type=int, metavar="MINUTES", help="repeat every N minutes while the market is open (N >= 1)")
    run.add_argument(
        "--paper-orders",
        "--live",
        dest="live",
        action="store_true",
        help="place PAPER orders in the simulator (default: dry run, decisions only). Nothing here ever trades real money; "
        "--live is the older name for the same flag",
    )
    run.add_argument(
        "--holdout",
        action="store_true",
        help="after each cycle, mark the independent holdout reserve account (see docs/holdout.md)",
    )
    intraday = parser.add_argument_group("intraday engines")
    intraday.add_argument(
        "--intraday",
        action="store_true",
        help="run the intraday engines: breakout, gap-and-go and gap-only, each with its own paper account "
        "(add --paper-orders to place paper orders)",
    )
    intraday.add_argument(
        "--intraday-report", action="store_true", help="positions, trade history and status of the intraday paper accounts"
    )
    reports = parser.add_argument_group("reports and checks (read-only)")
    reports.add_argument("--screen", action="store_true", help="only scan the whole market and print the candidates")
    reports.add_argument("--graph", action="store_true", help="print the LangGraph decision workflow as a Mermaid diagram")
    reports.add_argument(
        "--check", action="store_true", help="test keys and connectivity (OpenRouter, broker, data, Telegram) and exit"
    )
    reports.add_argument("--report", action="store_true", help="print the paper-trading account summary (India simulator)")
    reports.add_argument(
        "--positions", action="store_true", help="every open position (P&L, buy date), closed-trade history, overall status"
    )
    portfolio = parser.add_argument_group("your real Integrated account (read-only: never places an order)")
    portfolio.add_argument(
        "--portfolio",
        action="store_true",
        help="read the REAL Integrated account (read-only; OTP on Telegram), analyse it, log out",
    )
    portfolio.add_argument("--show", action="store_true", help="with --portfolio: show the browser window")
    portfolio.add_argument(
        "--otp-from", choices=("telegram", "terminal"), default="telegram", help="with --portfolio: where you give the OTP"
    )
    portfolio.add_argument("--no-ai", action="store_true", help="with --portfolio: skip the AI-written note")
    return parser


def main() -> None:
    """Parse arguments and run the requested command."""
    if hasattr(sys.stdout, "reconfigure"):  # not on a redirected or replaced stream
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    if args.loop is not None and args.loop < 1:
        parser.error("--loop must be at least 1 minute")
    try:
        configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FORMAT", "text"))
    except ValueError as e:
        sys.exit(f"Invalid configuration: {e}")

    if args.portfolio:
        sys.exit(run_portfolio_agent(args.show, args.otp_from, not args.no_ai))
    if args.intraday:
        run_intraday(args.live)
    elif args.intraday_report:
        show_intraday_reports(load_settings_or_exit())
    elif report := next((fn for name, fn in REPORT_COMMANDS if getattr(args, name)), None):
        report(load_settings_or_exit())
    elif args.check:
        sys.exit(1 if preflight(None) else 0)
    else:
        run_trading_bot(args.live, args.holdout, args.loop)


if __name__ == "__main__":
    main()
