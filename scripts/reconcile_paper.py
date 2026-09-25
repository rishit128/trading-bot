"""Reconcile the paper account against its own ledger (orders, positions, trades, cash). Exit nonzero on anything."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import make_session_factory  # noqa: E402
from src.reconciliation import reconcile_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default="sqlite:///trading.db")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    print(reconcile_report(make_session_factory(args.database_url)))


if __name__ == "__main__":
    main()