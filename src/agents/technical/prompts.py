"""Everything the agent says to the model: the exact prompt text for each call, and the mechanical baseline the prompt
quotes. Pure functions of their inputs (no model, no I/O), so a stored decision can rebuild the very prompt that was
sent (`scripts/replay_decision.py`) and a wording change is a one-file diff. Changing any text here means bumping
`PROMPT_VERSION` in src/versions.py."""

from typing import Optional

from src.data.house_calls import HouseCall, prompt_line
from src.data.market_pulse import HOT_MAX_RSI, HotStock, pulse_line
from src.engine.enums import Action
from src.engine.rules import RSI_OVERBOUGHT, above_ma50, entry_filter, ma_stack_bullish, overbought, rule_verdict


def _surging(s, pulse) -> bool:
    """A stock of a surging sector (market pulse) that is still above its 50-day average and not extreme."""
    return pulse is not None and above_ma50(s.price, s.ma50) and s.rsi < HOT_MAX_RSI


def mechanical_action(s, pulse=None) -> Action:
    """The deterministic long-only filter that produced the candidate: BUY only when price is above MA50, MAs are
    stacked (MA50 > MA200) and RSI(14) is below 70; otherwise HOLD. A stock of a surging sector (market pulse) is also
    a BUY while it stays above MA50 with RSI(14) below the pulse's own line. This is the baseline the AI agrees with
    or overrides, and the yardstick the weekly report scores the AI against."""
    if _surging(s, pulse):
        return Action.BUY
    return rule_verdict(s.price, s.ma50, s.ma200, s.rsi)


def mechanical_baseline(s, pulse=None) -> str:
    """Human-readable form of the mechanical verdict, rendered into the prompt."""
    above, stacked = above_ma50(s.price, s.ma50), ma_stack_bullish(s.ma50, s.ma200)
    if not entry_filter(s.price, s.ma50, s.ma200, s.rsi) and _surging(s, pulse):
        return (
            f"BUY (market-pulse rule: price above MA50, RSI(14) below {HOT_MAX_RSI:.0f}, and the stock's sector is "
            "surging. The long-term filter (MA50 above MA200, RSI below 70) does not apply to these stocks; they are "
            "sold on a close below MA50)"
        )
    if entry_filter(s.price, s.ma50, s.ma200, s.rsi):
        return (
            f"BUY (price above MA50, MA50 above MA200, RSI(14) below {RSI_OVERBOUGHT:.0f} - the technical filter "
            "this candidate passed)"
        )
    divergences = []
    if not above:
        divergences.append(f"price {s.price:.2f} is below MA50 {s.ma50:.2f}")
    if not stacked:
        divergences.append(f"MA50 {s.ma50:.2f} is below MA200 {s.ma200:.2f}")
    if overbought(s.rsi):
        divergences.append(f"RSI(14) {s.rsi:.1f} is at or above {RSI_OVERBOUGHT:.0f}")
    return "HOLD (" + "; ".join(divergences) + " - the mechanical filter did NOT select this candidate)"


def _pct(value: Optional[float], signed: bool = True) -> str:
    return "n/a" if value is None else (f"{value:+.1%}" if signed else f"{value:.1%}")


def company_text(c) -> str:
    """The company's fundamentals as the prompt shows them (`c` is a Fundamentals, or its stored dict), or a plain
    statement that there are none."""
    if c is None:
        return "no company data available - judge without it and say so"
    get = c.get if isinstance(c, dict) else lambda k: getattr(c, k)
    crore = 1e7
    cap, pe, debt = get("market_cap"), get("pe"), get("debt_to_equity")
    lines = [
        f"{get('sector') or 'sector n/a'} ({get('industry') or 'industry n/a'})",
        f"market cap Rs {cap / crore:,.0f} crore" if cap else "market cap n/a",
        f"P/E {pe:.1f}" if pe is not None else "P/E n/a (losses or no earnings)",
        f"debt/equity {debt:.2f}" if debt is not None else "debt/equity n/a",
        f"profit margin {_pct(get('profit_margin'), signed=False)}",
        f"latest quarter vs a year earlier: sales {_pct(get('revenue_growth'))}, profit {_pct(get('earnings_growth'))}",
        f"promoters/insiders hold {_pct(get('promoter_holding'), signed=False)}",
    ]
    quarters = []
    for q in get("quarters") or ():
        q = q if isinstance(q, dict) else q.__dict__
        rev = f"{q['revenue'] / crore:,.1f}" if q.get("revenue") is not None else "n/a"
        net = f"{q['net_income'] / crore:,.1f}" if q.get("net_income") is not None else "n/a"
        quarters.append(f"{q['period_end'][:7]} {rev} / {net}")
    text = " | ".join(lines)
    if quarters:
        text += "\nRecent quarters, newest first (sales / net profit, Rs crore): " + "; ".join(quarters)
    return text


def market_text(m) -> str:
    """The market situation as the prompt shows it (`m` is a MarketContext, or its stored dict)."""
    if m is None:
        return "no market data available - judge without it and say so"
    get = m.get if isinstance(m, dict) else lambda k: getattr(m, k)
    above = get("index_above_ma200")
    trend = "n/a" if above is None else ("above" if above else "BELOW")
    vix = "n/a"
    if get("vix") is not None:
        vix = f"{get('vix'):.1f}" + (
            f" ({get('vix_percentile'):.0%} of the past year's range)" if get("vix_percentile") is not None else ""
        )
    rsi = f"{get('index_rsi'):.0f}" if get("index_rsi") is not None else "n/a"
    return (
        (
            f"Nifty 50 {get('index_price'):,.0f}, {trend} its 200-day average {get('index_ma200'):,.0f}; index RSI(14) {rsi}; "
            f"6-month index return {_pct(get('index_mom_6m'))}; India VIX {vix}; this stock vs the index over 6 months "
            f"{_pct(get('vs_index_6m'))}"
        )
        if get("index_price") is not None
        else "no market data available"
    )


def house_section(h) -> str:
    """The broker research call as the prompt shows it (`h` is a HouseCall, or its stored dict), or nothing at all when
    there is none: a stock without a call gets exactly the prompt it got before research calls were wired in."""
    if h is None:
        return ""
    call = h if isinstance(h, HouseCall) else HouseCall.from_dict(h)
    return (
        f"HOUSE RESEARCH: {prompt_line(call)}. This is a second opinion from the broker's analysts, whose track record "
        "is not proven: it may raise or lower your confidence, but it is not on its own a reason to "
        "buy a stock the MECHANICAL BASELINE did not select. A house SELL call, or a call that hit its stop-loss, is a "
        "risk to name in step5_risks.\n"
    )


def pulse_section(p) -> str:
    """The market pulse as the prompt shows it (`p` is a HotStock, or its stored dict), or nothing at all when there is
    none: a stock outside a surging sector gets exactly the prompt it got before the pulse was wired in."""
    if p is None:
        return ""
    return (
        f"MARKET PULSE: {pulse_line(p if isinstance(p, HotStock) else HotStock.from_dict(p))}. The strategy "
        "also buys stocks riding a sector-wide surge that has not yet lifted them above their 200-day average. Judge "
        "from the COMPANY data whether this stock's own results support the move; a surge that the results do not "
        "support is a risk to name in step5_risks.\n"
    )


def cot_prompt(s, market=None, company=None, house=None, pulse=None) -> str:
    """The exact prompt sent to the model (also used to replay and fingerprint decisions)."""
    above, stacked = above_ma50(s.price, s.ma50), ma_stack_bullish(s.ma50, s.ma200)
    avg = s.avg_volume
    ratio = f"{s.volume / avg:.2f}x" if avg else "n/a"
    macd_hist = f"{s.macd_histogram:+.5f} (positive=bullish)" if s.macd_histogram is not None else "n/a"
    momentum = f"{s.momentum:+.1%} over 14 sessions" if s.momentum is not None else "n/a"
    bb = f"{s.bb_position:+.0%} of the Bollinger range" if s.bb_position is not None else "n/a"
    vol_trend = f"{s.volume_trend:+.1%} vs the prior 20 days" if s.volume_trend is not None else "n/a"
    atr = f"{s.atr / s.price:.1%} of price" if s.atr is not None and s.price else "n/a"
    adx = f"{s.adx:.0f} of 100" if s.adx is not None else "n/a"
    strength = (
        f"Its 12-month momentum (return from 12 months ago to 1 month ago) is {s.momentum_12_1:+.0%}: the scanner "
        "ranks NSE stocks by this, the one signal that beat holding the whole market in this strategy's 9-year "
        "research. Strong stocks often look stretched on a chart; that is expected, not a flaw.\n"
        if s.momentum_12_1 is not None
        else ""
    )
    return (
        f"You are the analyst in a momentum strategy for Indian stocks. {s.symbol} is a candidate the scanner selected "
        "(or a stock already held). The scanner has already confirmed the price trend; your job is to judge what a "
        "price chart cannot show: the company's recent results and the market situation. Answer in the requested JSON.\n"
        f"{strength}"
        f"PRICE: {s.price:.2f} | MA50: {s.ma50:.2f} | MA200: {s.ma200:.2f} | RSI(14): {s.rsi:.1f} | "
        f"Volume: {s.volume:,} | 20-day avg volume: {avg if avg else 'n/a'} | volume ratio: {ratio} | 14-day momentum "
        f"{momentum} | MACD histogram {macd_hist} | price at {bb} | volume trend {vol_trend} | ATR {atr} | ADX {adx}\n"
        f"MECHANICAL BASELINE (the deterministic filter): {mechanical_baseline(s, pulse)}\n"
        f"COMPANY: {company_text(company)}\n"
        f"MARKET: {market_text(market)}\n"
        f"{house_section(house)}"
        f"{pulse_section(pulse)}"
        "STEP 1 (trend): is price above MA50 ("
        + ("yes" if above else "no")
        + ") and MA50 above MA200 ("
        + ("yes" if stacked else "no")
        + ")? Fresh breakout or extended run? step1_trend: one sentence.\n"
        "STEP 2 (overbought): RSI, MACD and the Bollinger position in one sentence. step2_overbought.\n"
        "STEP 3 (volume): is the move confirmed by volume? step3_volume: one sentence.\n"
        "COMPANY CHECK: company_check: one sentence quoting the numbers: are sales and profit growing or shrinking over "
        "the recent quarters, and is anything alarming (losses, a sharp fall in sales or profit, debt far above equity, "
        "very low promoter holding)?\n"
        "MARKET CHECK: market_check: one sentence: is the market supportive or stressed (Nifty vs its 200-day average, "
        "how high the VIX is)?\n"
        "STEP 4 (confluence): step4_confluence: an integer 1-10 for how well trend, company and market agree.\n"
        "STEP 5 (risks): step5_risks: at least three concrete risks, most severe first.\n"
        "STEP 6 (rule check): rule_alignment: 'agree' if your action matches the MECHANICAL BASELINE, 'deviate' if you "
        "override it. falsification: the single most concrete, checkable thing that would prove this call wrong.\n"
        "DECISION: follow the MECHANICAL BASELINE unless the COMPANY data gives a concrete reason against it that you "
        "can name with its number (for example net losses, sales or profit falling sharply year on year, debt far above "
        "equity). These are NOT reasons: an extended-looking chart, below-average volume, a short pause in momentum "
        "(the strategy's tested edge comes from buying exactly these strong stocks), or a weak market on its own (in "
        "the 9-year test, skipping buys whenever the Nifty was below its 200-day average halved the return). Use the "
        "MARKET data for your risks and confidence, not to veto. For a stock already held, SELL only for a concrete "
        "company reason or a clearly broken trend. confidence (0-1) is your probability the call is right; "
        "edge_confidence (0-1) how repeatable the setup is. final_reasoning: 2-3 sentences naming the deciding facts."
    )


def simple_prompt(s) -> str:
    """The compact one-sentence prompt: the fallback when a weak model cannot satisfy the full schema."""
    return (
        f"You are a swing-trading analyst. Assess {s.symbol} using only this data.\n"
        f"Price: {s.price:.2f}\n50-day MA: {s.ma50:.2f}\n200-day MA: {s.ma200:.2f}\n"
        f"RSI(14): {s.rsi:.1f}\nLatest daily volume: {s.volume:,}\n\n"
        "BUY only for a clear uptrend that is not overbought; SELL only if the trend has clearly broken; "
        "otherwise HOLD. Confidence is your probability estimate that the call is right (0-1). "
        "Give one sentence of reasoning."
    )


def reflection_prompt(base, symbol: str, summary: str) -> str:
    """Reflection prompt: a six-stage self-critique of the call (review doc 4). Step 1 (the base) is seeded client-side
    from the call entering this phase; the model produces steps 2-6, each a conviction 0-1 no higher than the last."""
    return (
        f"You are a careful analyst doing a final check before a trade. The proposed call for {symbol} is {base.action} "
        f"at confidence {base.confidence:.2f}.\n"
        f"Step 1 (technical base) conviction starts at {base.confidence:.2f}. Go through the remaining steps and "
        "for each give a conviction 0-1 that is equal to or below the previous step's, plus a one-sentence reason.\n"
        f"{summary}\n"
        "Step 2 (reflection): what did the first pass overlook? step2_reflection.\n"
        "Step 3 (fundamental): challenge the call against the company data above; if there is none, keep the conviction "
        "unchanged and say so. step3_fundamental.\n"
        "Step 4 (macro): challenge the call against the market situation above. step4_macro.\n"
        "Step 5 (integration): weigh steps 2-4 together - does the call survive? step5_integration.\n"
        "Step 6 (risk): restate at the surviving conviction with the single biggest risk. step6_risk.\n"
        "Then give: biggest_risk, what_proves_us_wrong (a falsifiable condition for exiting at once), bias_check (the "
        "cognitive biases most likely at play). action: the final BUY/SELL/HOLD - change it only for a concrete company "
        "reason with its number, not for a stretched-looking chart or a weak market on its own. confidence: your "
        "proposed final probability 0-1."
    )
