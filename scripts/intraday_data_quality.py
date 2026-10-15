"""How far each intraday engine's live data differed from the final bars (src/intraday/data_quality.py): signal-bar close
and volume, signals the final bars do not confirm, and how old the prices behind each fill were. No verdict before 10
verified sessions.

    python scripts/intraday_data_quality.py
On the server: sudo docker compose -f docker-compose.aws.yml exec intraday-bot python scripts/intraday_data_quality.py"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import gapgo_db_url, gaponly_db_url, intraday_db_url  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.intraday.data_quality import MIN_SESSIONS, data_quality_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    args = ap.parse_args()
    for label, url in (("INTRADAY", intraday_db_url()), ("GAP-GO", gapgo_db_url()), ("GAP-ONLY", gaponly_db_url())):
        print(data_quality_report(make_session_factory(url), label, args.min_sessions))
        print()


if __name__ == "__main__":
    main()
