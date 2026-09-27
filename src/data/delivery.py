"""NSE daily delivery percentage (share of traded quantity that was actually taken for delivery, not squared off intraday).

Source: NSE's full bhavcopy, one CSV per trading day, saved by a person from a browser into NSE_FILES_DIR/bhavcopy under
the name NSE gives it (sec_bhavdata_full_DDMMYYYY.csv). NSE's terms of use prohibit automated data collection, so the
bot never downloads these itself. Without recent files the delivery filter is simply off for the day (it fails open)."""
import io
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.data.india import nse_files_dir

log = logging.getLogger(__name__)
FILE_NAME = re.compile(r"sec_bhavdata_full_(\d{8})\.csv$", re.IGNORECASE)


def bhavcopy_dir(directory: Optional[Path] = None) -> Path:
    """The folder of hand-downloaded bhavcopy files (NSE_FILES_DIR/bhavcopy unless given)."""
    return directory if directory is not None else nse_files_dir() / "bhavcopy"


def parse_bhavcopy(text: str) -> pd.Series:
    """Symbol -> delivery percent for the EQ series of one day's file."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = pd.Index([c.strip() for c in df.columns])
    df = df[df["SERIES"].str.strip() == "EQ"]
    return pd.to_numeric(df.set_index("SYMBOL")["DELIV_PER"], errors="coerce")


def load_delivery(years: float = 3, directory: Optional[Path] = None,
                  today: Callable[[], date] = date.today) -> pd.DataFrame:
    """Dates x symbols delivery percent from the bhavcopy files in the folder dated within the last `years`. Empty when
    there are none; a file that cannot be parsed is skipped with a warning, never guessed."""
    folder = bhavcopy_dir(directory)
    start, end = today() - timedelta(days=int(years * 365.25)), today()
    rows = {}
    if folder.exists():
        for path in sorted(folder.iterdir()):
            m = FILE_NAME.search(path.name)
            if not m:
                continue
            day = datetime.strptime(m.group(1), "%d%m%Y").date()
            if not start <= day <= end:
                continue
            try:
                rows[pd.Timestamp(day)] = parse_bhavcopy(path.read_text(encoding="utf-8-sig"))
            except Exception as e:
                log.warning("bhavcopy %s could not be read (%s: %s); skipped", path.name, type(e).__name__, e)
    return pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame()
