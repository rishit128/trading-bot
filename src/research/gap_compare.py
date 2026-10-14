"""Gap-and-go against its gap-only control, from the two paper accounts' closed trades and recorded signals.

The question: does waiting for the breakout add anything over buying every 2% gap at 09:30? Gap-and-go is credited with a
timing edge only if it wins BOTH per trade and date by date (paired over the dates both traded): per trade alone can
look better simply by sitting out bad days. Everything is net of fees and slippage, as the paper broker charged them."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.data.india import IST
from src.research.intraday_hypotheses import T_BAR, cluster_t

MIN_SHARED_DATES = 30  # fixed in advance: a handful of trades says nothing about a difference of a fraction of a percent


@dataclass(frozen=True)
class AccountStats:
    trades: int
    net_per_trade: float  # mean of each trade's net P&L over the value it bought
    win_rate: float
    net_pnl: float
    fees: float
    max_drawdown: float  # of the closed-trade equity curve, from its running peak (0 or negative)
    mean_daily_return: float  # over the days the account traded
    utilisation: float  # mean share of the start-of-day equity bought on a trading day
    days_traded: int


def trades_frame(trades: List[dict]) -> pd.DataFrame:
    """Closed trades (PaperBroker.trade_history rows) with the IST trade date and each trade's net return."""
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "value", "net_pnl", "fees", "ret"])
    df["date"] = pd.to_datetime(df["opened_at"], utc=True).dt.tz_convert(IST).dt.date
    df["value"] = df["entry_price"] * df["qty"]
    df["ret"] = df["net_pnl"] / df["value"]
    return df.sort_values("closed_at")


def account_stats(trades: pd.DataFrame, initial_cash: float) -> Optional[AccountStats]:
    if trades.empty:
        return None
    daily = trades.groupby("date").agg(pnl=("net_pnl", "sum"), bought=("value", "sum"))
    equity_end = initial_cash + daily["pnl"].cumsum()
    equity_start = equity_end.shift(1).fillna(initial_cash)
    curve = pd.concat([pd.Series([initial_cash]), equity_end.reset_index(drop=True)])
    return AccountStats(
        trades=len(trades),
        net_per_trade=float(trades["ret"].mean()),
        win_rate=float((trades["net_pnl"] > 0).mean()),
        net_pnl=float(trades["net_pnl"].sum()),
        fees=float(trades["fees"].sum()),
        max_drawdown=float((curve / curve.cummax() - 1).min()),
        mean_daily_return=float((daily["pnl"] / equity_start).mean()),
        utilisation=float((daily["bought"] / equity_start).mean()),
        days_traded=len(daily),
    )


@dataclass(frozen=True)
class Comparison:
    gapgo: Optional[AccountStats]
    gaponly: Optional[AccountStats]
    shared_dates: int
    paired_diff: float  # mean over shared dates of (gap-and-go - gap-only) per-trade net return
    paired_t: float
    same_stock_days: int  # stocks both bought on the same day
    same_stock_gapgo: float
    same_stock_gaponly: float
    capped: Dict[str, int]  # signals each account had to skip because its position limit was full

    @property
    def verdict(self) -> str:
        if self.gapgo is None or self.gaponly is None:
            return "not enough trades yet to compare"
        if self.shared_dates < MIN_SHARED_DATES:
            return (
                f"too early to judge: {self.shared_dates} of the {MIN_SHARED_DATES} dates on which both accounts "
                "traded that a verdict needs (the numbers above are anecdotes until then)"
            )
        per_trade = self.gapgo.net_per_trade > self.gaponly.net_per_trade
        by_date = self.paired_diff > 0 and self.paired_t >= T_BAR
        if per_trade and by_date:
            return "gap-and-go wins per trade AND date by date: the breakout timing adds something"
        if per_trade:
            return (
                "gap-and-go wins per trade only, not reliably date by date: not yet evidence that the breakout "
                "timing adds anything"
            )
        return "gap-and-go does not beat buying every gap: no evidence for the breakout timing"


def compare(
    gapgo_trades: List[dict],
    gaponly_trades: List[dict],
    gapgo_cash: float,
    gaponly_cash: float,
    capped: Optional[Dict[str, int]] = None,
) -> Comparison:
    a, b = trades_frame(gapgo_trades), trades_frame(gaponly_trades)
    pa, pb = (
        a.groupby("date")["ret"].mean() if len(a) else pd.Series(dtype=float),
        b.groupby("date")["ret"].mean() if len(b) else pd.Series(dtype=float),
    )
    shared = pa.index.intersection(pb.index)
    diff = pa[shared] - pb[shared]
    # the same stock bought by both accounts on the same day (one trade per stock per day in each)
    pairs = a.merge(b, on=["symbol", "date"], suffixes=("_go", "_only")) if len(a) and len(b) else pd.DataFrame()
    return Comparison(
        gapgo=account_stats(a, gapgo_cash),
        gaponly=account_stats(b, gaponly_cash),
        shared_dates=len(shared),
        paired_diff=float(diff.mean()) if len(diff) else float("nan"),
        paired_t=cluster_t(diff) if len(diff) > 1 else float("nan"),
        same_stock_days=len(pairs),
        same_stock_gapgo=float(pairs["ret_go"].mean()) if len(pairs) else float("nan"),
        same_stock_gaponly=float(pairs["ret_only"].mean()) if len(pairs) else float("nan"),
        capped=capped or {},
    )


def format_comparison(c: Comparison, cur: str = "Rs ") -> str:
    rows = []
    for name, s in (("gap-and-go", c.gapgo), ("gap-only", c.gaponly)):
        if s is None:
            rows.append(f"{name:<11} no closed trades yet")
            continue
        rows.append(
            f"{name:<11} {s.trades:>5} trades on {s.days_traded:>3} days | net/trade {s.net_per_trade:+.3%} | win "
            f"{s.win_rate:.0%} | net {cur}{s.net_pnl:+,.0f} | fees {cur}{s.fees:,.0f} | daily {s.mean_daily_return:+.3%} | "
            f"max drawdown {s.max_drawdown:.1%} | invested {s.utilisation:.0%} of equity on trading days"
        )
    if c.shared_dates > 1:
        rows.append(
            f"date by date ({c.shared_dates} shared dates): gap-and-go minus gap-only {c.paired_diff:+.3%} a trade, "
            f"paired t {c.paired_t:+.1f} (needs >= {T_BAR})"
        )
    else:
        rows.append("date by date: fewer than 2 dates on which both traded yet")
    if c.same_stock_days:
        rows.append(
            f"the same stock on the same day ({c.same_stock_days}): gap-and-go {c.same_stock_gapgo:+.3%}, "
            f"gap-only {c.same_stock_gaponly:+.3%}"
        )
    if c.capped:
        rows.append("skipped because the position limit was full: " + ", ".join(f"{k} {v}" for k, v in c.capped.items()))
    rows.append("VERDICT: " + c.verdict)
    return "\n".join(rows)


def capped_signals(signals: pd.DataFrame) -> int:
    """How many recorded signals an account skipped for a full position limit (see IntradayEngine.record_capped)."""
    if signals.empty or "reason" not in signals:
        return 0
    return int(signals["reason"].fillna("").str.startswith("position limit reached").sum())
