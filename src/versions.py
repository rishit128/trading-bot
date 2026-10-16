"""Version stamps recorded with every decision so behaviour changes can be attributed without guessing.

Bump FEATURE_VERSION when the indicator inputs to the prompt change, PROMPT_VERSION when the technical prompt text
changes, SCHEMA_VERSION when the model's output schema changes, and STRATEGY_VERSION when the decision or execution
rules change. The decisions table stores these per-decision (columns added by the automatic migration), so a report
can answer "why did behaviour change on that date" by comparing stamps instead of guessing."""

FEATURE_VERSION = "indicators-v4"  # v3 + 12-1 month momentum (the scanner's ranking signal) in the snapshot
PROMPT_VERSION = "technical-v12"  # v11 + the market pulse (a surging sector) and its own baseline, when the stock has one
SCHEMA_VERSION = "signal-v1"  # the llm.SIGNAL_SCHEMA output contract, frozen for replay compatibility
# swing-v1: mechanical BUY filter + trend exit + 15% protective stop, sized and gated by the risk engine, paper broker.
# swing-v2: the same, with the stocks of open broker Swing/Momentum/Stellar Pick buy calls added to the candidates.
# swing-v3: stocks of surging sectors (the market pulse) are also candidates, bought while above MA50 and sold on a
#           close below MA50 (the other stocks keep the MA200 exit).
STRATEGY_VERSION = "swing-v3"

# The intraday engine (a separate paper account). Recorded with every signal so results can be split by system version.
#   orb-v1: sizing borrowed from the swing settings (5% of equity), stop/target replay dropped each bar's tail, no record
#           of skipped signals, universe chosen silently (Nifty 100 file, else Yahoo top-100).
#   orb-v2: risk-based sizing with a price cap (INTRADAY_*), completed-bar replay, every signal and skip recorded, one
#           shared universe selector, opening-range integrity guard, stale positions closed at the open, and each
#           finished session's bars archived (no change to what trades: the vectorised rule is pinned to the old one).
#   orb-v3: the stocks of today's open broker Intraday buy calls are added to the scan, and a stock the broker has an
#           open SELL call on is not bought (the breakout rule itself is unchanged).
INTRADAY_VERSION = "orb-v3"
# The gap-and-go paper strategy (its own account): gap >= 2% then a close above the opening range by 10:30, stop at
# the range low, held to the square-off.
GAPGO_VERSION = "gapgo-v1"
# The gap-only paper control (its own account): the same gap, bought on the 09:30 close with no breakout.
GAPONLY_VERSION = "gaponly-v1"

VERSIONS = {
    "feature": FEATURE_VERSION,
    "prompt": PROMPT_VERSION,
    "schema": SCHEMA_VERSION,
    "strategy": STRATEGY_VERSION,
}
