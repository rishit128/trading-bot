"""How far the intraday engines' live data differed from the final data (C4 of the data-quality plan).

During the day each engine trades on Yahoo's 5-minute bars as served at that moment; Yahoo revises them afterwards. Every
signal records what the engine saw (`signal_price`, `seen_volume`, `step_at`), and after the close the engine records the
final bar and whether the rule still fires on it (`final_close`, `final_volume`, `final_signal`; IntradayEngine.
_verify_signals). Every fill records the start of the bar its price came from (`entry_price_bar`, `exit_price_bar`).

This report reads those columns. It judges nothing before `MIN_SESSIONS` verified sessions: until then it shows the
numbers so far and says so."""

import statistics
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from sqlalchemy import select

from src.database import IntradaySignalRecord, PaperTradeRecord

MIN_SESSIONS = 10


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _minutes(values: Sequence[float]) -> str:
    return "n/a" if not values else f"median {statistics.median(values):.1f} min, worst {max(values):.1f} min"


def _utc(at: datetime) -> datetime:
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at  # SQLite hands times back naive (stored as UTC)


def data_quality_report(sessions, label: str, min_sessions: int = MIN_SESSIONS) -> str:
    """Plain-text report for one engine's database."""
    with sessions() as s:
        signals = list(s.scalars(select(IntradaySignalRecord).where(IntradaySignalRecord.step_at.isnot(None))))
        trades = list(s.scalars(select(PaperTradeRecord).where(PaperTradeRecord.entry_price_bar.isnot(None))))
    verified = [r for r in signals if r.final_close is not None]
    days = sorted({r.day for r in verified})
    lines: List[str] = [f"[{label}] live data vs the final bars"]
    lines.append(
        f"  signals recorded with what was seen: {len(signals)}; verified after the close: {len(verified)} on {len(days)} sessions"
    )
    if len(days) < min_sessions:
        lines.append(f"  too few sessions to judge ({len(days)} of {min_sessions}); the numbers so far:")
    closes = [abs(r.final_close / r.signal_price - 1) for r in verified if r.signal_price]
    lines.append(
        f"  signal-bar close, seen vs final: median gap {_pct(_median(closes))}, worst {_pct(max(closes) if closes else None)}"
    )
    volumes = [r.seen_volume / r.final_volume for r in verified if r.seen_volume is not None and r.final_volume]
    lines.append(
        f"  signal-bar volume, seen / final: median {_median(volumes) or 0:.2f}x"
        + (f", {sum(v < 0.9 for v in volumes)} of {len(volumes)} under 0.9x" if volumes else " (none)")
    )
    unconfirmed = [r for r in verified if r.final_signal is False]
    entered = [r for r in verified if r.outcome == "ENTERED"]
    lines.append(
        f"  signals the final bars do not confirm at the same bar: {len(unconfirmed)} of {len(verified)}"
        f" ({sum(r.outcome == 'ENTERED' for r in unconfirmed)} of {len(entered)} entered trades)"
    )
    entry_age = [(_utc(t.opened_at) - _utc(t.entry_price_bar)).total_seconds() / 60 for t in trades]
    exit_age = [(_utc(t.closed_at) - _utc(t.exit_price_bar)).total_seconds() / 60 for t in trades if t.exit_price_bar]
    lines.append(
        f"  entry price taken from a bar that started before the fill by: {_minutes(entry_age)} ({len(entry_age)} fills)"
    )
    lines.append(
        f"  exit price from a bar that started before the exit was booked by: {_minutes(exit_age)} ({len(exit_age)} exits)"
    )
    return "\n".join(lines)
