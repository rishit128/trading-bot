"""Walk-forward backtest of the real technical agent + risk engine vs. baselines. Resumable via backtest_cache.json.

Limits: technical agent only (no point-in-time news), symbol name hidden from the model but price levels are not,
so a model that memorised these tickers' history could still leak. Treat results as a sanity check, not proof."""
import argparse
import dataclasses
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.agents import TechnicalAgent  # noqa: E402
from src.agents.base import AgentContext  # noqa: E402
from src.backtest import START_EQUITY, buy_and_hold_curve, curve_metrics, rule_signals, simulate, trade_metrics  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.indicators import Snapshot, build_snapshot  # noqa: E402
from src.llm import SIGNAL_SCHEMA, LLMClient  # noqa: E402

CACHE = Path("backtest_cache.json")


def prompt_fingerprint() -> str:
    """Cached signals are only valid for the exact prompt + schema that produced them."""
    sample = Snapshot("STOCK", 100.0, 98.0, 95.0, 60.0, 1000)
    blob = TechnicalAgent.build_prompt(sample) + json.dumps(SIGNAL_SCHEMA, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def read_cache(path: Path = CACHE) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if "signals" not in data:  # legacy flat file, written before fingerprints existed (same prompt)
        return data
    if data.get("prompt_hash") != prompt_fingerprint():
        print("Agent prompt changed since the cache was written; discarding stale cached signals.", flush=True)
        return {}
    return data["signals"]


def write_cache(cache: dict, path: Path = CACHE) -> None:
    path.write_text(json.dumps({"prompt_hash": prompt_fingerprint(), "signals": cache}))


def load_bars(symbols):
    bars = {}
    for s in symbols:
        df = yf.download(s, period="3y", interval="1d", progress=False, auto_adjust=True)
        df.columns = df.columns.get_level_values(0)
        bars[s] = df.dropna(subset=["Open", "High", "Low", "Close"])
    return bars


def llm_signals(bars, dates, workers):
    cache = read_cache()
    wanted = set(dates)
    agent = TechnicalAgent(LLMClient(load_settings().models))
    lock, done, failures = threading.Lock(), [0], [0]
    jobs = [(s, d) for s in bars for d in dates if f"{s}|{d.date()}" not in cache]
    print(f"{len(cache)} cached, {len(jobs)} to compute", flush=True)

    def work(job):
        sym, d = job
        snap = dataclasses.replace(build_snapshot(sym, bars[sym].loc[:d]), symbol="STOCK")
        sig = agent.analyze(AgentContext("STOCK", snap))
        with lock:
            done[0] += 1
            if sig.degraded:
                failures[0] += 1
            else:
                cache[f"{sym}|{d.date()}"] = [sig.action, sig.confidence]
            if done[0] % 20 == 0:
                write_cache(cache)
                print(f"  {done[0]}/{len(jobs)} done, {failures[0]} failed", flush=True)

    try:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(work, jobs))
    finally:
        write_cache(cache)
    if failures[0]:
        print(f"WARNING: {failures[0]} signals failed and were left out; rerun to retry them.", flush=True)

    out = {s: {} for s in bars}
    for key, (action, conf) in cache.items():
        s, d = key.split("|")
        if s in out and pd.Timestamp(d) in wanted:
            out[s][pd.Timestamp(d)] = (action, conf)
    return out


def report(name, curve, result=None):
    m = curve_metrics(curve, START_EQUITY)
    line = f"{name:22s} return={m['total_return']:+7.2%}  maxDD={m['max_drawdown']:7.2%}  sharpe={m['sharpe']:5.2f}"
    if result is not None:
        t = trade_metrics(result.trades)
        if t["trades"]:
            line += f"  trades={t['trades']} win={t['win_rate']:.0%} avg={t['avg_trade_return']:+.2%} exits={t['exits']}"
        else:
            line += "  trades=0"
        line += f" open_at_end={result.open_at_end}"
    print(line)
    return {**m, **(trade_metrics(result.trades) if result else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT,GOOGL,AMZN,TSLA")
    ap.add_argument("--lookback", type=int, default=250, help="trading days to backtest")
    ap.add_argument("--step", type=int, default=3, help="decide every N trading days")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    settings = load_settings()
    bars = load_bars([s.strip().upper() for s in args.symbols.split(",")])
    calendar = sorted(set.intersection(*[set(df.index) for df in bars.values()]))
    dates = calendar[-args.lookback::args.step]
    print(f"{len(bars)} symbols, {len(dates)} decision dates {dates[0].date()} .. {dates[-1].date()}", flush=True)

    signals = llm_signals(bars, dates, args.workers)
    print("\nLLM signal mix:", {a: sum(1 for s in signals.values() for v in s.values() if v[0] == a) for a in ("BUY", "SELL", "HOLD")})

    window = {s: df.loc[dates[0]:] for s, df in bars.items()}
    llm = simulate(window, signals, settings.risk)
    rule = simulate(window, rule_signals(bars, dates), settings.risk)
    hold = buy_and_hold_curve(window)

    print("\n=== RESULTS (same risk engine, stops, sizing for LLM and rule) ===")
    out = {
        "llm_technical": report("LLM technical agent", llm.equity, llm),
        "rule_baseline": report("Trend rule (no LLM)", rule.equity, rule),
        "buy_and_hold": report("Buy & hold equal-weight", hold),
    }
    Path("backtest_report.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
