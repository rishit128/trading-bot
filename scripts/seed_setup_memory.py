"""Seed the decision memory with setups rebuilt from 10 years of NSE price history, and check whether it predicts anything.

    python scripts/seed_setup_memory.py build            # rebuild weekly scanner picks + 60-session outcomes -> JSON lines
    python scripts/seed_setup_memory.py check            # walk-forward test of the memory on them (the gate, see below)
    python scripts/seed_setup_memory.py import FILE      # load them into DATABASE_URL (the running bot's database)

`build` reads the cached all-NSE history (research_cache/ohlc_nse_10y.pkl; downloaded on first use). The pre-declared
gate and the known biases are in src/research/seed_memory.py. The live bot only uses these rows with LEARNING_SEED=true."""
import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.wiring import screen_config  # noqa: E402
from src.config import Settings, load_settings  # noqa: E402
from src.data.price_history import load_ohlc_nse  # noqa: E402
from src.database import make_session_factory  # noqa: E402
from src.research.seed_memory import Observation, gate, rebuild, store, walk_forward  # noqa: E402

DEFAULT_FILE = Path("research_cache/setup_seed_h60.jsonl")


def read(path: Path):
    return [Observation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--years", type=int, default=10)
    b.add_argument("--horizon", type=int, default=60)
    b.add_argument("--out", type=Path, default=DEFAULT_FILE)
    c = sub.add_parser("check")
    c.add_argument("file", type=Path, nargs="?", default=DEFAULT_FILE)
    c.add_argument("--test-years", type=float, default=4.0, help="the most recent years are the out-of-sample window")
    c.add_argument("--min-samples", type=int, default=30)
    i = sub.add_parser("import")
    i.add_argument("file", type=Path, nargs="?", default=DEFAULT_FILE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.command == "build":
        settings = Settings()  # the scanner's code defaults, so the result does not depend on a local .env
        observations = rebuild(load_ohlc_nse(args.years), screen_config(settings), args.horizon,
                               settings.risk.stop_loss_pct, progress=print)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(o.to_json() for o in observations) + "\n", encoding="utf-8")
        days = sorted({o.bar_date for o in observations})
        print(f"{len(observations)} setups over {len(days)} weeks ({days[0]} .. {days[-1]}) -> {args.out}")
    elif args.command == "check":
        observations = read(args.file)
        last = max(o.bar_date for o in observations)
        test_from = str((pd.Timestamp(last) - pd.DateOffset(years=args.test_years)).date())
        horizon = observations[0].horizon
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            sessions = make_session_factory(f"sqlite:///{Path(tmp) / 'seed.db'}")
            store(sessions, observations)
            groups = walk_forward(sessions, observations, horizon, args.min_samples, test_from)
        print(f"walk-forward test window {test_from} .. {last}, horizon {horizon} sessions, net of round-trip costs:")
        passed, lines = gate(groups)
        print("\n".join(lines))
        sys.exit(0 if passed else 1)
    else:
        settings = load_settings()
        count = store(make_session_factory(settings.database_url), read(args.file))
        print(f"{count} setups stored in {settings.database_url}; the bot uses them only with LEARNING_SEED=true")


if __name__ == "__main__":
    main()
