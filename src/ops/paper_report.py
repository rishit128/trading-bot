"""Combined profit/loss report across both paper accounts (swing and intraday): every closed trade, and realized P&L
grouped by day and by ISO week, for each account and combined.

Pure functions on what `Broker.trade_history()` already returns (dicts with `closed_at`, `net_pnl`, `fees`, ...), so this
adds no new recording and cannot show anything the two accounts did not already log."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from src.data.india import IST


@dataclass(frozen=True)
class PeriodPnl:
    """Realized P&L over one calendar day or ISO week."""
    label: str          # "2026-09-29" for a day, "2026-W40" for a week
    trades: int
    wins: int
    net_pnl: float
    fees: float

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades if self.trades else None


def _closed_date(trade: dict) -> date:
    """The IST calendar date a trade closed on (trade_history's `closed_at` is UTC)."""
    return trade["closed_at"].astimezone(IST).date()


def group_by_day(trades: List[dict]) -> List[PeriodPnl]:
    """One row per calendar day that had a closed trade, oldest first."""
    return _group(trades, lambda d: d.isoformat())


def group_by_week(trades: List[dict]) -> List[PeriodPnl]:
    """One row per ISO week (YYYY-Www) that had a closed trade, oldest first."""
    return _group(trades, lambda d: f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}")


def _group(trades: List[dict], key) -> List[PeriodPnl]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in trades:
        buckets[key(_closed_date(t))].append(t)
    return [PeriodPnl(label, len(rows), sum(1 for r in rows if r["net_pnl"] > 0), sum(r["net_pnl"] for r in rows),
                      sum(r["fees"] for r in rows)) for label, rows in sorted(buckets.items())]


def totals(trades: List[dict]) -> PeriodPnl:
    """All closed trades as one period, labelled "total"."""
    rows = group_by_day(trades)
    return PeriodPnl("total", sum(r.trades for r in rows), sum(r.wins for r in rows),
                     sum(r.net_pnl for r in rows), sum(r.fees for r in rows))


def merge_periods(a: List[PeriodPnl], b: List[PeriodPnl]) -> List[PeriodPnl]:
    """Two accounts' period rows combined into one set, same labels summed."""
    by_label: Dict[str, PeriodPnl] = {r.label: r for r in a}
    for r in b:
        if r.label in by_label:
            o = by_label[r.label]
            by_label[r.label] = PeriodPnl(r.label, o.trades + r.trades, o.wins + r.wins, o.net_pnl + r.net_pnl, o.fees + r.fees)
        else:
            by_label[r.label] = r
    return [by_label[label] for label in sorted(by_label)]
