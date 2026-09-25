"""The strategy's deterministic rules, each defined exactly once.

The bot's entry filter ("price above MA50, MA50 above MA200, RSI below 70") and its trend exit ("close below the 200-day
average") used to be written out separately in the live scanner, the AI's mechanical baseline, the backtest, the replay
tool and the research signals. When one copy changed the others silently disagreed, and a backtest could test a different
strategy than the one running. Every consumer now imports from here.

Scalar functions take floats (the live path, one stock at a time); the `_mask` functions take pandas Series or numpy
arrays (the research path, whole tables at once). tests/test_rules.py proves the two forms agree."""

RSI_OVERBOUGHT = 70.0  # RSI(14) at or above this is "overbought": no new entry


# ---- scalar rules ---------------------------------------------------------------------------------------------------
def above_ma50(price: float, ma50: float) -> bool:
    """Price is above its 50-day average."""
    return price > ma50


def ma_stack_bullish(ma50: float, ma200: float) -> bool:
    """The 50-day average is above the 200-day average (a long-term uptrend structure)."""
    return ma50 > ma200


def uptrend(price: float, ma50: float, ma200: float) -> bool:
    """Price > MA50 > MA200."""
    return above_ma50(price, ma50) and ma_stack_bullish(ma50, ma200)


def overbought(rsi: float) -> bool:
    """RSI(14) is at or above the overbought line."""
    return rsi >= RSI_OVERBOUGHT


def entry_filter(price: float, ma50: float, ma200: float, rsi: float) -> bool:
    """The mechanical entry filter: an uptrend that is not overbought."""
    return uptrend(price, ma50, ma200) and not overbought(rsi)


def trend_broken(price: float, ma200: float) -> bool:
    """The last completed close is below the 200-day average: the long-term trend has failed (the exit rule)."""
    return price < ma200


# ---- the same rules over whole tables ------------------------------------------------------------------------------
def uptrend_mask(close, ma50, ma200):
    """`uptrend` for pandas/numpy inputs (elementwise)."""
    return (close > ma50) & (ma50 > ma200)


def overbought_mask(rsi):
    """`overbought` for pandas/numpy inputs."""
    return rsi >= RSI_OVERBOUGHT


def entry_filter_mask(close, ma50, ma200, rsi):
    """`entry_filter` for pandas/numpy inputs."""
    return uptrend_mask(close, ma50, ma200) & ~overbought_mask(rsi)


def trend_broken_mask(close, ma200):
    """`trend_broken` for pandas/numpy inputs."""
    return close < ma200
