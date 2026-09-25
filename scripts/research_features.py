"""Do simple indicator buckets predict 20-session net returns, out of sample? Report only: nothing is adopted or tuned.
The pre-declared pass criteria are in src/research/feature_study.py.

    python scripts/research_features.py [--split 2021-12-31] [--horizon 20]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.price_history import load_ohlc_universe  # noqa: E402
from src.research.feature_study import T_BAR, build_observations, study  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="2021-12-31", help="last day of the train period")
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()
    obs = build_observations(load_ohlc_universe(500, 10), args.horizon)
    print(f"{len(obs):,} observations, {obs['symbol'].nunique()} stocks, {obs['date'].min().date()} .. {obs['date'].max().date()}; "
          f"train <= {args.split}; excess = net return minus the same-date universe average; bar |t| >= {T_BAR}")
    print(f"{'feature':<13}{'bucket':<7}{'range':<22}{'train n':>8}{'mean':>8}{'t':>6}   {'test n':>7}{'mean':>8}{'t':>6}")
    results = study(obs, args.split)
    for r in results:
        rng = f"{r.low:9.2f}..{r.high:<9.2f}".replace("inf", "∞")
        print(f"{r.feature:<13}{r.bucket:<7}{rng:<22}{r.train_n:>8}{r.train_mean:>+8.2%}{r.train_t:>6.1f}   {r.test_n:>7}{r.test_mean:>+8.2%}{r.test_t:>6.1f}"
              f"{'  <-- CANDIDATE' if r.candidate else ''}")
    found = [r for r in results if r.candidate]
    print(f"\ncandidates passing both periods: {len(found)} of {len(results)}"
          + (" -> leads only; test them properly before any change" if found else " -> no bucket predicts out of sample; the memory stays a guard, not an edge"))


if __name__ == "__main__":
    main()
