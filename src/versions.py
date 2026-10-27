"""Version stamps recorded with every decision so behaviour changes can be attributed without guessing.

Bump FEATURE_VERSION when the indicator inputs to the prompt change, PROMPT_VERSION when the technical prompt text
changes, SCHEMA_VERSION when the model's output schema changes, and STRATEGY_VERSION when the decision or execution
rules change. The decisions table stores these per-decision (columns added by the automatic migration), so a report
can answer "why did behaviour change on that date" by comparing stamps instead of guessing."""

FEATURE_VERSION = "indicators-v4"  # v3 + 12-1 month momentum (the scanner's ranking signal) in the snapshot
PROMPT_VERSION = "technical-v13"  # v12 + Brent crude and its monthly change in the market line
SCHEMA_VERSION = "signal-v1"  # the llm.SIGNAL_SCHEMA output contract, frozen for replay compatibility
# swing-v1: mechanical BUY filter + trend exit + 15% protective stop, sized and gated by the risk engine, paper broker.
# swing-v2: the same, with the stocks of open broker Swing/Momentum/Stellar Pick buy calls added to the candidates.
# swing-v3: stocks of surging sectors (the market pulse) are also candidates, bought while above MA50 and sold on a
#           close below MA50 (the other stocks keep the MA200 exit).
# swing-v4 (2026-10-26): no new buy within 3 days of an announced results meeting, none into an F&O short build-up
#           (both refuse entries through the entry context), and the core index sleeve (CORE_SLEEVE).
# swing-v5 (2026-10-27): the 24-hour re-buy cooldown also covers a stop-out (it only looked at our own buy orders, so a stock
#           stopped out after more than a day in the position could be bought straight back, against STRATEGY.md).
STRATEGY_VERSION = "swing-v5"

# The intraday engine (a separate paper account). Recorded with every signal so results can be split by system version.
#   orb-v1: sizing borrowed from the swing settings (5% of equity), stop/target replay dropped each bar's tail, no record
#           of skipped signals, universe chosen silently (Nifty 100 file, else Yahoo top-100).
#   orb-v2: risk-based sizing with a price cap (INTRADAY_*), completed-bar replay, every signal and skip recorded, one
#           shared universe selector, opening-range integrity guard, stale positions closed at the open, and each
#           finished session's bars archived (no change to what trades: the vectorised rule is pinned to the old one).
#   orb-v3: the stocks of today's open broker Intraday buy calls are added to the scan, and a stock the broker has an
#           open SELL call on is not bought (the breakout rule itself is unchanged).
#   orb-v4: the universe is the explicit 500-stock INTRADAY_UNIVERSE (was Yahoo's top 100, since 2026-10-19), and Angel One's
#           entry checks refuse a buy within 1% of the upper circuit (since 2026-10-23) or into an F&O short build-up
#           (since 2026-10-26). Bumped 2026-10-26: those changes had gone live under orb-v3.
INTRADAY_VERSION = "orb-v4"
# The gap-and-go paper strategy (its own account): gap >= 2% then a close above the opening range by 10:30, stop at
# the range low, held to the square-off.
# gapgo-v2: v1 plus the 500-stock universe (live since 2026-10-19) and the Angel One entry checks (2026-10-23, 10-26) that
# orb-v4 also carries (bumped 2026-10-26).
GAPGO_VERSION = "gapgo-v2"
# The gap-only paper control (its own account): the same gap, bought on the 09:30 close with no breakout.
# gaponly-v2: v1 plus the universe and Angel One entry checks of orb-v4 (bumped 2026-10-26).
GAPONLY_VERSION = "gaponly-v2"
# The news-catalyst paper strategy (its own account): a material exchange filing, read by the AI, traded long by fixed
# rules (src.news.strategy).
NEWS_VERSION = "news-v3"  # v2: gap floor 0% (was +2%), live order-book veto; v3: rule A until 12:00, filings after 14:30 roll over, no entry into an F&O short build-up

VERSIONS = {
    "feature": FEATURE_VERSION,
    "prompt": PROMPT_VERSION,
    "schema": SCHEMA_VERSION,
    "strategy": STRATEGY_VERSION,
}
