"""Diagnostic only: re-simulate cached LLM signals under different stop/target/hold settings (no LLM calls).
Do NOT pick the best cell and adopt it -- that is overfitting to one year of data."""
import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_backtest import load_bars, read_cache  # noqa: E402
from src.backtest import START_EQUITY, curve_metrics, simulate, trade_metrics  # noqa: E402
from src.config import load_settings  # noqa: E402


def main():
    cache = read_cache()
    symbols = sorted({k.split("|")[0] for k in cache})
    bars = load_bars(symbols)
    signals = {s: {} for s in symbols}
    for key, (action, conf) in cache.items():
        s, d = key.split("|")
        signals[s][pd.Timestamp(d)] = (action, conf)
    first = min(d for s in signals.values() for d in s)
    window = {s: df.loc[first:] for s, df in bars.items()}
    base = load_settings().risk

    print(f"{'stop':>5} {'target':>6} {'hold':>4} | {'return':>7} {'maxDD':>7} {'trades':>6} {'win':>4}")
    for stop, target, hold in [(0.02, 0.05, 10), (0.03, 0.05, 10), (0.05, 0.05, 10), (0.05, 0.10, 20),
                               (0.08, 0.10, 20), (0.02, 0.05, 20), (0.10, 0.20, 40)]:
        limits = dataclasses.replace(base, stop_loss_pct=stop, take_profit_pct=target)
        r = simulate(window, signals, limits, max_hold_days=hold)
        m, t = curve_metrics(r.equity, START_EQUITY), trade_metrics(r.trades)
        print(f"{stop:5.0%} {target:6.0%} {hold:4d} | {m['total_return']:+7.2%} {m['max_drawdown']:7.2%} {t['trades']:6d} {t.get('win_rate', 0):4.0%}")


if __name__ == "__main__":
    main()
