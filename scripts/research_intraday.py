"""Can the intraday opening-range breakout be fixed? A handful of PRE-DECLARED variants on Yahoo's last ~59 days of
5-minute bars (cached in research_cache/ so every run sees identical data).

PRE-DECLARED before any result was seen. Each variant changes ONE idea; six are tried, so a lucky pass is plausible.
A variant is a CANDIDATE only if all hold: >= 100 trades; net P&L > 0 in BOTH the first and the second half of the
days; total net expectancy per trade > +0.10% of the trade value; and it is still > 0 with slippage doubled.
A candidate is a lead for forward paper trading, never proof: 59 days is one regime and today's Nifty 100 members only.

    python scripts/research_intraday.py [--refresh]"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_intraday import download_5m, simulate  # noqa: E402
from src.data.india import fetch_index_symbols  # noqa: E402
from src.intraday import strategy as st  # noqa: E402

CACHE = Path("research_cache/intraday_5m_59d.pkl")
NOTIONAL = 50_000.0  # a fixed trade value so variants compare on expectancy, not on compounding


def load(refresh):
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    data = download_5m(fetch_index_symbols(100), 59)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(data))
    return data


def index_days(data):
    """Per date: the fraction of stocks whose first-bar open is above the previous session's close (a breadth/gap proxy for
    the market's mood that day, known at 09:15)."""
    gaps = {}
    for df in data.values():
        days = list(df.groupby(df.index.date))
        for (d0, g0), (d1, g1) in zip(days, days[1:]):
            gaps.setdefault(d1, []).append(float(g1["Open"].iloc[0]) > float(g0["Close"].iloc[-1]))
    return {d: float(np.mean(v)) for d, v in gaps.items()}


VARIANTS = {
    "baseline (live rule)": {},
    "early only: entries until 11:00": {"until": "11:00"},
    "1R target instead of 2R": {"target_r": 1.0},
    "narrow range only (<=1.5%)": {"max_range": 0.015},
    "strong volume only (>=3x)": {"volume_mult": 3.0},
    "market up: >60% of stocks gap up": {"breadth": 0.6},
    "no breakout above +2% from open": {"max_extension": 0.02},
}


def run(data, breadth, slippage, until=None, target_r=None, max_range=None, volume_mult=None, breadth_min=None, max_extension=None):
    saved = (st.ENTRY_UNTIL, st.TARGET_R, st.MAX_RANGE, st.VOLUME_MULT)
    if until:
        st.ENTRY_UNTIL = pd.Timestamp(until).time()
    st.TARGET_R = target_r or st.TARGET_R
    st.MAX_RANGE = max_range or st.MAX_RANGE
    st.VOLUME_MULT = volume_mult or st.VOLUME_MULT
    rows = []
    try:
        for sym, df in data.items():
            for day, g in df.groupby(df.index.date):
                if len(g) < 60 or (breadth_min and breadth.get(day, 0) < breadth_min):
                    continue
                s = st.find_setup(sym, g)
                if not s or (max_extension and s.signal_price > float(g["Open"].iloc[0]) * (1 + max_extension)):
                    continue
                entry, exit_, why = simulate(g, s, slippage)
                qty = int(NOTIONAL / entry)
                fees = st.intraday_fees("BUY", qty * entry) + st.intraday_fees("SELL", qty * exit_)
                rows.append({"day": day, "why": why, "pnl": qty * (exit_ - entry) - fees, "ret": (qty * (exit_ - entry) - fees) / (qty * entry)})
    finally:
        st.ENTRY_UNTIL, st.TARGET_R, st.MAX_RANGE, st.VOLUME_MULT = saved
    return pd.DataFrame(rows)


def score(t):
    days = sorted(t["day"].unique())
    half = set(days[:len(days) // 2])
    first, second = t[t["day"].isin(half)].pnl.sum(), t[~t["day"].isin(half)].pnl.sum()
    return len(t), t.ret.mean(), first, second, (t.pnl > 0).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    data = load(args.refresh)
    breadth = index_days(data)
    print(f"{len(data)} stocks, {min(breadth)}..{max(breadth)}; fixed trade value Rs {NOTIONAL:,.0f}; slippage 0.05%/side")
    print(f"{'variant':<36}{'trades':>7}{'win':>6}{'expect':>9}{'1st half':>10}{'2nd half':>10}{'2x slip':>10}")
    for name, kw in VARIANTS.items():
        kw = dict(kw)
        kw["breadth_min"] = kw.pop("breadth", None)
        t = run(data, breadth, 0.0005, **{("until" if k == "until" else k): v for k, v in kw.items()})
        if t.empty:
            print(f"{name:<36} no trades"); continue
        n, exp, a, b, win = score(t)
        doubled = run(data, breadth, 0.001, **kw).ret.mean()
        ok = n >= 100 and a > 0 and b > 0 and exp > 0.001 and doubled > 0
        print(f"{name:<36}{n:>7}{win:>6.0%}{exp:>+9.2%}{a:>+10,.0f}{b:>+10,.0f}{doubled:>+10.2%}{'  <-- CANDIDATE' if ok else ''}")


if __name__ == "__main__":
    main()
