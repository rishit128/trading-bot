"""The core index sleeve of the swing account (2026-10-26, at the owner's request).

A share of the account sits in index ETFs that the stock-picking logic never sees: no AI analysis, no trend exit, no
stop that matters. The ten-year attribution found the swing strategy's return to be mostly broad-market exposure with
no significant stock-selection alpha (STRATEGY.md, 2026-10-26); a 25-40% core cut the worst drawdown by 5 to 10 points
(Sharpe about equal, return lower) in the blend test. The sleeve only BUYS toward its target; it never sells.

CORE_SLEEVE="NIFTYBEES:0.15,MID150BEES:0.15" gives each ETF its share of equity."""

import math
from typing import Dict, List, Mapping, Tuple

TOLERANCE = 0.25  # buy only when the holding is more than this far below its target (no dribbling of tiny buys)
RESERVE_PCT = 0.03  # of equity, kept as cash
MIN_BUY_VALUE = 1500.0  # a smaller buy is not worth its fixed charges


def parse(spec: str) -> Dict[str, float]:
    """{symbol: share of equity} from "SYM:0.15,SYM2:0.15"; empty text means no sleeve. Raises on a bad entry or a total over 60%."""
    weights: Dict[str, float] = {}
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        symbol, _, share = part.partition(":")
        weight = float(share)
        if not symbol.strip() or not 0 < weight <= 0.5:
            raise ValueError(f"CORE_SLEEVE entry {part!r}: expected SYMBOL:share with a share in (0, 0.5]")
        weights[symbol.strip().upper()] = weight
    if sum(weights.values()) > 0.6:
        raise ValueError(f"CORE_SLEEVE shares total {sum(weights.values()):.0%}; the stock-picking sleeve needs at least 40%")
    return weights


def buys(weights: Mapping[str, float], held_value: Mapping[str, float], prices: Mapping[str, float], equity: float, cash: float) -> List[Tuple[str, int]]:
    """(symbol, shares) to buy now: each ETF more than TOLERANCE under its target, in the order given, from the cash above
    the reserve. Whole shares only; skips an ETF with no price."""
    spendable = cash - RESERVE_PCT * equity
    out: List[Tuple[str, int]] = []
    for symbol, weight in weights.items():
        price, target = prices.get(symbol), weight * equity
        have = held_value.get(symbol, 0.0)
        if not price or have >= (1 - TOLERANCE) * target:
            continue
        budget = min(target - have, spendable)
        qty = math.floor(budget / price)
        if qty >= 1 and qty * price >= MIN_BUY_VALUE:
            out.append((symbol, qty))
            spendable -= qty * price
    return out
