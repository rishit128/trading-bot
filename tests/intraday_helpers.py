"""Shared fixtures for the intraday tests: a synthetic session of 5-minute bars and a known breakout."""
from datetime import datetime, timedelta

import pandas as pd

from src.data.india import IST

DAY = datetime(2026, 9, 21, tzinfo=IST)


def make_bars(closes, highs=None, lows=None, volumes=None, start=(9, 15)):
    """5-minute bars from 09:15 IST; open = previous close."""
    idx = pd.DatetimeIndex([DAY.replace(hour=start[0], minute=start[1]) + timedelta(minutes=5 * i) for i in range(len(closes))])
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame({"Open": opens, "High": highs or [max(o, c) + 0.1 for o, c in zip(opens, closes)],
                         "Low": lows or [min(o, c) - 0.1 for o, c in zip(opens, closes)], "Close": closes,
                         "Volume": volumes or [1000.0] * len(closes)}, index=idx)


BREAKOUT = [100, 100.4, 100.2, 100.3, 101.5]  # range 99.9..100.5, then a close above it on the 5th bar
VOLS = [1000, 1000, 1000, 1000, 3000]
