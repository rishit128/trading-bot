"""Backtests the production AI (src.agents.technical.TechnicalAgent, real OpenRouter calls, the production prompt minus ATR/ADX, which need High/Low) on
Indian stock history, point-in-time, and compares it with the mechanical rule it sits on top of in production.

This directly answers the open gap in STRATEGY.md: the AI has run live since 2026-09-21, but "running" is not
"validated" -- every prior backtest used the plain rule as a stand-in for the AI.

Design (read before trusting the numbers):
  * Monthly rebalance, not daily. Daily would need ~20x more LLM calls for the same period; monthly still matches the
    bot's holding horizon (the trend exit typically holds weeks to months). This is a real scope compromise, not free.
  * Each month: replicate the live scanner's filter (price > MA50 > MA200, RSI(14) < 70, liquid) and rank by 12-1
    momentum (today's live ranking) to get a candidate list, computed from data available strictly up to that month's
    close (no lookahead) via the same build_snapshot() production code path.
  * The REAL TechnicalAgent + LLMClient (configured OpenRouter models) scores each candidate BUY/SELL/HOLD + confidence,
    from the exact production prompt. Responses are cached to disk (keyed by the prompt text) so a re-run or a crash
    does not repeat paid/rate-limited calls.
  * Two portfolios, same universe and costs: "AI-filtered" holds the top N candidates the AI said BUY with
    confidence >= MIN_CONFIDENCE; "mechanical" holds the top N candidates by momentum with no AI involved (N and
    MIN_CONFIDENCE come from live settings). The gap between them is what the AI is worth over the rule alone.
  * Today's Nifty 100 membership only (survivorship bias applies, same caveat as every other backtest in this repo).
  * Single run, not pre-registered like scripts/research_signals.py: there is no honest way to pre-declare pass/fail
    criteria for a script that costs real API calls to iterate on. Read the comparison, not a verdict."""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.technical import TechnicalAgent  # noqa: E402
from src.agents.base import AgentContext  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.indicators import Snapshot, build_snapshot, compute_rsi  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.research import engine  # noqa: E402
from src.data.price_history import load_universe  # noqa: E402
from src.research.signals import liquid_mask, month_end_dates  # noqa: E402

CACHE = Path("research_cache/ai_backtest_signals.json")


def point_in_time_snapshot(symbol: str, close: pd.Series, volume: pd.Series, as_of: pd.Timestamp) -> Snapshot:
    """Exactly src.data.indicators.build_snapshot on data up to (and including) `as_of`, so the prompt carries the same
    indicators as production (MA, RSI, MACD, Bollinger, momentum, volume trend, average volume). Only ATR and ADX stay
    n/a: the cached research frames hold Close and Volume, not High/Low."""
    c = close.loc[:as_of].dropna()
    bars = pd.DataFrame({"Close": c, "Volume": volume.loc[c.index].fillna(0.0)})
    return build_snapshot(symbol, bars)


class DiskCachedAgent:
    """Wraps the real TechnicalAgent so identical (symbol, date, indicators) prompts are never asked twice, across runs."""

    def __init__(self, agent: TechnicalAgent, cache_path: Path = CACHE):
        self.agent, self.path = agent, cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.calls = 0

    def signal(self, snapshot: Snapshot):
        key = hashlib.sha256(self.agent.build_prompt(snapshot).encode()).hexdigest()
        if key not in self.cache:
            # TechnicalAgent.analyze() never raises: it already catches LLMUnavailable and returns a degraded
            # fail-safe HOLD AgentSignal, so there is nothing further to catch here.
            sig = self.agent.analyze(AgentContext(snapshot.symbol, snapshot))
            self.cache[key] = {"action": sig.action, "confidence": sig.confidence, "degraded": sig.degraded}
            self.calls += 1
            if self.calls % 10 == 0:
                self._save()
        return self.cache[key]

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache))


def monthly_candidates(close, volume, cfg_top_n, lookback_days_needed=253):
    """Per month-end date, the live-style candidate list (trend filter + liquidity, ranked by 12-1 momentum).
    RSI is checked lazily per candidate (only for stocks that already pass the cheaper filters) since a full
    rolling RSI over every column and date is expensive and most columns never reach this filter."""
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    ok = liquid_mask(close, volume, 1e8) & (close > ma50) & (ma50 > ma200)
    mom = close.shift(21) / close.shift(lookback_days_needed) - 1
    out = {}
    for d in month_end_dates(close.index):
        elig = ok.loc[d] & mom.loc[d].notna()
        scores = mom.loc[d].where(elig).dropna().sort_values(ascending=False)
        candidates = []
        for sym in scores.index:
            r = compute_rsi(close[sym].loc[:d].dropna().tail(300))
            if r < 70:
                candidates.append(sym)
            if len(candidates) >= cfg_top_n:
                break
        out[d] = candidates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--index", type=int, default=100)
    ap.add_argument("--cost", type=float, default=0.0012)
    ap.add_argument("--cot-only", action="store_true",
                    help="the five-step reasoning only; skip the history/market/reflection refinement phases "
                         "that make every candidate cost extra real LLM calls and are not disk-cached")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = load_settings()
    top_n = settings.risk.max_open_positions
    min_conf = settings.risk.min_confidence
    print(f"live settings used: max_open_positions={top_n} min_confidence={min_conf} models={settings.models}")

    # Load extra history before the test window so the 253-day momentum lookback and 200-day MA have data on day 1
    # of the window; only month-ends inside the actual test window are then scored.
    close, volume, nifty = load_universe(args.index, args.years + 2)
    window_start = close.index[-1] - pd.DateOffset(years=args.years)
    candidates_by_date = monthly_candidates(close, volume, top_n * 3)  # scan a wider pool than N so the AI has room to reject some
    candidates_by_date = {d: v for d, v in candidates_by_date.items() if d >= window_start}
    llm = LLMClient.from_settings(settings, cache_ttl_seconds=0)
    phase = ("CoT + history + market + reflection" if not args.cot_only else "CoT only (no refinement phases)")
    agent = DiskCachedAgent(TechnicalAgent(llm, use_learning=not args.cot_only, use_context=not args.cot_only,
                                           use_reflect=not args.cot_only))
    print(f"agent: {phase}")

    ai_weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    mech_weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    total_candidates = sum(len(v) for v in candidates_by_date.values())
    done, buy_count, hold_or_sell_count = 0, 0, 0
    for d, symbols in candidates_by_date.items():
        buys = []
        for sym in symbols:
            done += 1
            snap = point_in_time_snapshot(sym, close[sym], volume[sym], d)
            sig = agent.signal(snap)
            if sig["action"] == "BUY" and sig["confidence"] >= min_conf:
                buys.append((sym, sig["confidence"]))
                buy_count += 1
            else:
                hold_or_sell_count += 1
        buys.sort(key=lambda x: -x[1])
        chosen = [s for s, _ in buys[:top_n]]
        if chosen:
            ai_weights.loc[d, chosen] = 1.0 / len(chosen)
        mech_chosen = symbols[:top_n]
        if mech_chosen:
            mech_weights.loc[d, mech_chosen] = 1.0 / len(mech_chosen)
        print(f"  {d.date()}: {len(symbols)} candidates -> AI approved {len(buys)}, bought (capped at {top_n}) {len(chosen)}, "
              f"mechanical bought {len(mech_chosen)} ({done}/{total_candidates} candidates scored, {agent.calls} real calls so far)",
              flush=True)
    agent._save()

    def hold_between(w):
        w = w.copy()
        w.loc[~w.index.isin(month_end_dates(close.index))] = float("nan")
        return w.ffill().fillna(0.0)

    ai_bt = engine.portfolio_returns(close, hold_between(ai_weights), args.cost)
    mech_bt = engine.portfolio_returns(close, hold_between(mech_weights), args.cost)
    basket = engine.portfolio_returns(close, hold_between(liquid_mask(close, volume, 1e8).astype(float)
                                                          .div(liquid_mask(close, volume, 1e8).sum(axis=1), axis=0)), args.cost)

    print(f"\nreal AI calls made this run: {agent.calls} (cached thereafter at {CACHE})")
    print(f"{'':32s} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>7} {'turn/y':>6} | vs mechanical | vs basket")
    for name, res in (("AI-filtered (BUY, conf>=min)", ai_bt), ("mechanical (top-N momentum, no AI)", mech_bt),
                      ("BENCH equal-weight liquid basket", basket)):
        s = engine.summarize(res)
        vs_mech = engine.cagr(res["net"]) - engine.cagr(mech_bt["net"])
        vs_basket = engine.cagr(res["net"]) - engine.cagr(basket["net"])
        print(f"{name:32s} {s['cagr']:7.1%} {s['sharpe']:6.2f} {s['max_drawdown']:7.1%} {s['turnover_per_year']:6.1f} "
              f"| {vs_mech:+13.1%} | {vs_basket:+8.1%}")

    approval_rate = buy_count / max(1, buy_count + hold_or_sell_count)
    print(f"\nAI approved {approval_rate:.0%} of the {total_candidates} candidate-months it was shown as BUY with "
          f"confidence >= {min_conf:.2f} ({buy_count} approvals, {hold_or_sell_count} HOLD/SELL/low-confidence)")
    print("Read this as: does the AI's judgment on top of the rule help, hurt, or do nothing? It is not a green light")
    print("for real money either way -- see STRATEGY.md's other caveats (survivorship bias, single short sample).")


if __name__ == "__main__":
    main()
