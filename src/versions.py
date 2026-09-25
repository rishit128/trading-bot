"""Version stamps recorded with every decision so behaviour changes can be attributed without guessing.

Bump FEATURE_VERSION when the indicator inputs to the prompt change, PROMPT_VERSION when the technical prompt text
changes, SCHEMA_VERSION when the model's output schema changes, and STRATEGY_VERSION when the decision or execution
rules change. The decisions table stores these per-decision (columns added by the automatic migration), so a report
can answer "why did behaviour change on that date" by comparing stamps instead of guessing."""

FEATURE_VERSION = "indicators-v3"  # Phase-5 enrichment: MACD, Bollinger (levels+position), ATR, ADX, 14d momentum, volume trend
PROMPT_VERSION = "technical-v8"  # 5-step strict-JSON chain-of-thought + STEP 6 mechanical-baseline rule check + falsification
SCHEMA_VERSION = "signal-v1"  # the llm.SIGNAL_SCHEMA output contract, frozen for replay compatibility
STRATEGY_VERSION = "swing-v1"  # mechanical BUY filter + trend exit + 8% stop, sized and gated by the risk engine, paper broker

VERSIONS = {
    "feature": FEATURE_VERSION,
    "prompt": PROMPT_VERSION,
    "schema": SCHEMA_VERSION,
    "strategy": STRATEGY_VERSION,
}