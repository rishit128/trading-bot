"""Performance benchmark for the LLM-upgrade hot paths (review doc 5.2).

Measures the parts that do not need a live network so it is repeatable offline:
  * prompt construction for all four phases (time + bytes) - the per-stock constant cost
  * the phase-2 history lookup against the live database (SQL query + snapshot rebuild)
  * the reflection monotonic/humility enforcement (client-side arithmetic on the chain)
A live measurement of the actual model round-trips needs an API key and real calls; the script
says so rather than pretending. Run repeatedly to compare optimisations.

    python scripts/benchmark_performance.py --iterations 1000"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.technical import TechnicalAgent, ReflectionChain, ReflectionStage  # noqa: E402
from src.agents.history import PatternStats, history_stats  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.indicators import Snapshot  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.llm import AgentSignal  # noqa: E402

SNAP = Snapshot("RELIANCE", 2500.0, 2400.0, 2300.0, 62.0, 1_200_000, avg_volume=900_000, momentum_6m=0.08)
MKT = None
try:
    from src.data.market_context import MarketContext  # noqa: E402
    MKT = MarketContext(index_price=25000.0, index_ma200=24200.0, index_above_ma200=True,
                        index_rsi=68.0, index_ytd=0.12, vix=14.5, vix_percentile=0.42,
                        vs_index_6m=0.03, beta_6m=1.1)
except Exception:
    pass

BASE = AgentSignal(action="BUY", confidence=0.75, reasoning="Clear uptrend, volume confirms, not overbought.",
              details={"base_confidence": 0.75, "confluence_score": 8, "edge_confidence": 0.6,
                       "risks": ["a", "b", "c"]})
STATS = PatternStats(sample_size=12, win_rate=0.55, avg_win_pct=8.0, avg_loss_pct=-4.0, profit_factor=2.0,
                     best_holding_days=30, worst_holding_days=3, confidence_in_pattern=0.4)


def build_prompts():
    prompts = []
    for _ in range(10):
        prompts.append(TechnicalAgent.build_prompt(SNAP))
        prompts.append(TechnicalAgent.build_learning_prompt(BASE, "RELIANCE", STATS, "P>MA50+MA50>MA200|RSI60"))
        prompts.append(TechnicalAgent.build_context_prompt(BASE, "RELIANCE", MKT))
        prompts.append(TechnicalAgent.build_reflection_prompt(
            BASE, "RELIANCE", "Base reasoning: x\nMarket: bull, VIX 14."))
    return prompts


def make_stage(c):
    return ReflectionStage(conviction=c, reason="r")


def reflect_chain():
    return ReflectionChain(action="BUY", confidence=0.5,
                           step2_reflection=make_stage(0.70), step3_fundamental=make_stage(0.70),
                           step4_macro=make_stage(0.65), step5_integration=make_stage(0.60),
                           step6_risk=make_stage(0.60),
                           biggest_risk="a", what_proves_us_wrong="b", bias_check=["confirmation"])


def bench(label, fn, n):
    fn()  # warm up (caches, imports)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    print(f"{label:52s} {dt*1e6:9.1f} us/op  over {n} ops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1000)
    args = ap.parse_args()
    n = args.iterations

    print(f"benchmarking deterministic hot paths ({n} iterations each); live LLM round-trips are not measurable here")
    prompts = build_prompts()
    bench("build 4 phase prompts (10x each build)", build_prompts, max(1, n // 10))
    bench("build technical prompt", lambda: TechnicalAgent.build_prompt(SNAP), n)
    bench("build learning prompt", lambda: TechnicalAgent.build_learning_prompt(BASE, "RELIANCE", STATS, "PID"), n)
    bench("build context prompt", lambda: TechnicalAgent.build_context_prompt(BASE, "RELIANCE", MKT), n)
    bench("build reflection prompt", lambda: TechnicalAgent.build_reflection_prompt(BASE, "RELIANCE", "s"), n)
    chain = reflect_chain()
    base = AgentSignal(action="BUY", confidence=0.75, reasoning="r")
    agent = TechnicalAgent(llm=None)  # _apply_reflection is instance-only; none of the LLM path is touched here
    bench("reflect chain monotonic+humility", lambda: agent._apply_reflection(base, chain), n)
    avg_prompt_len = sum(len(p) for p in prompts) / len(prompts)
    print(f"\naverage prompt bytes: {avg_prompt_len:.0f} (the per-stock constant cost is the data, not the template)")

    settings = load_settings()
    db_path = Path(settings.database_url.replace("sqlite:///", "", 1))
    if db_path.exists():
        sessions = make_session_factory(settings.database_url)
        t0 = time.perf_counter()
        st = history_stats(sessions, SNAP)
        dt = (time.perf_counter() - t0) * 1000
        result = "stats found" if st else "None (no similar closed trades yet)"
        print(f"\nhistory lookup on live DB: {dt:6.1f} ms -> {result}")
    else:
        print("\nno live database found; skipping the history-lookup benchmark.")

    print("\nnote: real end-to-end latency = above + model round-trip (network + inference). Measure that only with a")
    print("live OPENROUTER_API_KEY, e.g. scripts/test_openrouter_api.py; free-tier latency varies a lot by model.")


if __name__ == "__main__":
    main()
