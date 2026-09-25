"""Everything the technical agent says to the model: the exact prompt text for each call, and the mechanical baseline the
prompt quotes. Pure functions of their inputs (no model, no I/O), so a stored decision can rebuild the very prompt that
was sent (`scripts/replay_decision.py`) and a wording change is a one-file diff. Changing any text here means bumping
`PROMPT_VERSION` in src/versions.py."""
from src.agents.history import PatternStats
from src.engine.enums import Action
from src.engine.rules import above_ma50, entry_filter, ma_stack_bullish, overbought
from src.engine.agent_signal import AgentSignal


def mechanical_action(s) -> Action:
    """The deterministic long-only filter that produced the candidate: BUY only when price is above MA50, MAs are
    stacked (MA50 > MA200) and RSI(14) is below 70; otherwise HOLD. This is the mechanical baseline the AI is asked
    to agree with or deliberately override, and the same source of truth the analytics scripts use to measure how
    often the AI rubber-stamps the filter."""
    return Action.BUY if entry_filter(s.price, s.ma50, s.ma200, s.rsi) else Action.HOLD


def mechanical_baseline(s) -> str:
    """Human-readable form of the mechanical verdict, rendered into the CoT prompt and reused by the metrics dashboards."""
    above, stacked = above_ma50(s.price, s.ma50), ma_stack_bullish(s.ma50, s.ma200)
    if entry_filter(s.price, s.ma50, s.ma200, s.rsi):
        return ("BUY (price above MA50, MA50 above MA200, RSI(14) below 70 - the technical filter this candidate "
                "passed)")
    divergences = []
    if not above:
        divergences.append(f"price {s.price:.2f} is below MA50 {s.ma50:.2f}")
    if not stacked:
        divergences.append(f"MA50 {s.ma50:.2f} is below MA200 {s.ma200:.2f}")
    if overbought(s.rsi):
        divergences.append(f"RSI(14) {s.rsi:.1f} is at or above 70")
    return "HOLD (" + "; ".join(divergences) + " - the mechanical filter did NOT select this candidate)"


def cot_prompt(s) -> str:
    """The exact chain-of-thought prompt sent to the model (also used to replay and fingerprint decisions).

    Each step carries 2-3 sub-questions (review doc 1.2), and step 5 asks for at least three risks ranked most
    severe first."""
    above, stacked = above_ma50(s.price, s.ma50), ma_stack_bullish(s.ma50, s.ma200)
    avg = s.avg_volume
    ratio = f"{s.volume / avg:.2f}x" if avg else "n/a"
    macd_hist = f"{s.macd_histogram:+.5f} (positive=bullish)" if s.macd_histogram is not None else "n/a"
    momentum = f"{s.momentum:+.1%} over 14 sessions" if s.momentum is not None else "n/a"
    bb_status = ("stretched at the upper band (overbought)"
                 if s.bb_position is not None and s.bb_position > 0.8
                 else "pressed at the lower band (oversold)"
                 if s.bb_position is not None and s.bb_position < 0.2
                 else "mid-band, not stretched")
    bb = (f"{s.bb_position:+.0%} of the Bollinger range ({bb_status})"
          if s.bb_position is not None else "n/a")
    vol_trend = f"{s.volume_trend:+.1%} vs the prior 20 days" if s.volume_trend is not None else "n/a"
    atr = f"{s.atr / s.price:.1%} of price" if s.atr is not None and s.price else "n/a"
    adx_label = ("strong" if s.adx is not None and s.adx > 40
                 else "weak" if s.adx is not None and s.adx < 25
                 else "moderate" if s.adx is not None else "n/a")
    adx = f"{s.adx:.0f} of 100 ({adx_label})" if s.adx is not None else "n/a"
    return (
        f"You are a cautious swing-trading technical analyst. Analyze {s.symbol} in five fixed steps and answer "
        "in the requested JSON. Reason step by step; answer the sub-questions of each step in your head before "
        "writing the step's one sentence.\n"
        f"Price: {s.price:.2f} | MA50: {s.ma50:.2f} | MA200: {s.ma200:.2f} | RSI(14): {s.rsi:.1f} | "
        f"Volume: {s.volume:,} | 20-day avg volume: {avg if avg else 'n/a'} | volume ratio: {ratio}\n"
        f"MECHANICAL BASELINE (the deterministic filter that produced this candidate): {mechanical_baseline(s)}\n"
        "STEP 1 (trend) - sub-questions: (a) is price above MA50? " + ("yes" if above else "no") +
        f" [{s.price:.2f} vs {s.ma50:.2f}]; (b) is MA50 above MA200? " + ("yes" if stacked else "no") +
        f" [{s.ma50:.2f} vs {s.ma200:.2f}]; (c) is the move a fresh breakout or an extended run? "
        "step1_trend: one sentence covering (a)-(c).\n"
        "STEP 2 (overbought) - sub-questions: (a) is RSI(14) at or above 70 (overbought)? "
        "(b) is it between 55 and 70 (strong but with room)? (c) has RSI(14) been pinned above 70 for many "
        "sessions (exhaustion)? Short-term momentum and the price-in-band context: "
        f"14-day momentum {momentum}; MACD histogram {macd_hist}; price at {bb}. "
        "step2_overbought: one sentence covering (a)-(c) plus whether MACD and the Bollinger position agree.\n"
        "STEP 3 (volume) - sub-questions: (a) is volume above its 20-day average? (b) is volume expanding on "
        "up-moves and shrinking on pullbacks (healthy) rather than the reverse? (c) does any spike look like "
        f'distribution? Context: volume trend {vol_trend}; volatility ATR {atr}; trend strength {adx} ADX. '
        'step3_volume: one sentence covering (a)-(c), e.g. "Volume 1.3x average; confirms".\n'
        "STEP 4 (confluence) - sub-questions: does (a) trend, (b) momentum and (c) volume all point the same "
        "way? step4_confluence: an integer 1-10 capturing how aligned the three are.\n"
        "STEP 5 (risks) - enumerate the ways this call loses money and rank them so the most severe is first. "
        "step5_risks: a list of at least three concrete risks, most severe first.\n"
        "STEP 6 (rule check) - compare your own analysis against the MECHANICAL BASELINE. rule_alignment: "
        "'agree' if your action matches the baseline, 'deviate' if you are deliberately overriding it. If you "
        "deviate, your final_reasoning must justify exactly why the additional context overrides the filter. "
        "falsification: the single most concrete, checkable thing - a price level, an event, a number - that "
        "would prove this call wrong.\n"
        "DECISION: action is BUY only for a clear uptrend that is not overbought; SELL only if the trend has "
        "clearly broken; otherwise HOLD. confidence (0-1) is your probability the call is right; edge_confidence "
        "(0-1) is how repeatable the pattern is (separate from confidence). final_reasoning: 2-3 sentences tying "
        "the steps together."
    )


def simple_prompt(s) -> str:
    """The original one-sentence prompt, kept for A/B comparison and `LLM_COT=false`."""
    return (
        f"You are a cautious swing-trading technical analyst. Assess {s.symbol} using only this data.\n"
        f"Price: {s.price:.2f}\n50-day MA: {s.ma50:.2f}\n200-day MA: {s.ma200:.2f}\n"
        f"RSI(14): {s.rsi:.1f}\nLatest daily volume: {s.volume:,}\n\n"
        "BUY only for a clear uptrend that is not overbought; SELL only if the trend has clearly broken; "
        "otherwise HOLD. Confidence is your probability estimate that the call is right (0-1). "
        "Give one sentence of reasoning."
    )


def learning_prompt(base: AgentSignal, symbol: str, stats: PatternStats, pattern_id: str) -> str:
    """Decision-memory prompt: does the historical record of similar setups support the base call?"""
    pf = f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "n/a (no losing trades yet)"
    rel = ("reasonably well supported" if stats.confidence_in_pattern >= 0.5 else "too thin to fully trust")
    opening = (f"You are a cautious swing-trading technical analyst. A colleague proposes {symbol} {base.action} at "
               f"confidence {base.confidence:.2f} from the technical setup (pattern {pattern_id}). Before you finalise, ")
    if stats.scope == "market":
        record = (
            "check how setups like this one actually turned out across ALL the stocks this bot has analysed. "
            f"Similar means: {stats.matched_on}. Each outcome is what really happened after entering at the next session's "
            f"open and holding {stats.best_holding_days} sessions, net of an estimated round-trip cost; overlapping "
            "observations of one stock are counted once per week.\n"
            f"Independent observations: {stats.sample_size}\n"
            f"win rate {stats.win_rate:.0%}"
            + (f" (all comparable setups: {stats.baseline_win_rate:.0%}; only the difference from that is an edge)"
               if stats.baseline_win_rate is not None else "")
            + f" | avg win {stats.avg_win_pct:+.1f}% | avg loss {stats.avg_loss_pct:+.1f}% | profit factor {pf} | "
            f"evidence is {rel} (sufficiency {stats.confidence_in_pattern:.0%})\n")
    else:
        record = (
            "check how similar past setups on this same symbol actually played out. Similar means the same trend shape "
            "(price vs MA50, MA50 vs MA200), a price within 5% and an RSI(14) within 5 points.\n"
            f"Closed trades on similar setups in the last year: {stats.sample_size}\n"
            f"win rate {stats.win_rate:.0%} | avg win {stats.avg_win_pct:+.1f}% | avg loss {stats.avg_loss_pct:+.1f}% | "
            f"profit factor {pf} | holding {stats.worst_holding_days}-{stats.best_holding_days} days | "
            f"evidence is {rel} (sufficiency {stats.confidence_in_pattern:.0%})\n")
    return (
        opening + record +
        "DECISION: does the historical record support, contradict, or leave the call unchanged? "
        "action: BUY/SELL/HOLD (the final call, possibly unchanged). confidence: your final probability 0-1. "
        "pattern_reliability: yes if the record clearly supports this pattern, no if it clearly contradicts it, "
        "maybe if it is mixed or thin. reason_for_adjustment: one sentence. what_could_break: one "
        "concrete scenario that would invalidate the pattern."
    )


def context_prompt(base: AgentSignal, symbol: str, market) -> str:
    """Market-context prompt: does the market/sector regime support the call? Missing context renders as 'n/a'."""
    vix = "n/a"
    if market.vix is not None:
        vix = f"{market.vix:.0f}" + (f" ({(market.vix_percentile * 100):.0f}th percentile of the last year)"
                                     if market.vix_percentile is not None else "")
    index = f"at {market.index_price:,.0f}" if market.index_price is not None else "price n/a"
    above = ("above" if market.index_above_ma200 else "below") if market.index_above_ma200 is not None else "? vs"
    ma200 = f"{market.index_ma200:,.0f}" if market.index_ma200 is not None else "n/a"
    rsi = f"{market.index_rsi:.0f}" if market.index_rsi is not None else "n/a"
    rel = f"{market.vs_index_6m:+.1%} vs index over 6 months" if market.vs_index_6m is not None else "n/a (no stock history)"
    ytd = f"{market.index_ytd:+.1%} YTD" if market.index_ytd is not None else "YTD n/a"
    sector = market.sector_trend if market.sector_trend else "n/a (no sector feed)"
    earnings = (f"~{market.earnings_days_until} sessions away" if market.earnings_days_until is not None
                else "n/a (no earnings feed)")
    beta = f"{market.beta_6m:.2f}" if market.beta_6m is not None else "n/a (no beta feed)"
    correlation = (f"{market.correlation_with_index:.2f}" if market.correlation_with_index is not None
                   else "n/a (no correlation feed)")
    return (
        f"You are a cautious swing-trading technical analyst. A colleague proposes {symbol} {base.action} at "
        f"confidence {base.confidence:.2f} from the technical setup. Check the market regime before finalising:\n"
        f"NIFTY 50 {index}, {above} its 200-day average ({ma200}); index RSI(14) {rsi}; {ytd}. "
        f"Regime: {market.regime_label}.\n"
        f"VIX: {vix}. | {symbol} relative strength: {rel}.\n"
        f"Sector trend: {sector} | next earnings: {earnings} | 6-month beta proxy vs index: {beta} | "
        f"correlation with index: {correlation}\n"
        "Answer these four context questions in your reasoning: (1) does the sector trend support this call? "
        "sector_support: yes/neutral/no. (2) is the stock near an earnings event, and does that threaten the "
        "call? earnings_risk: true/false. (3) if the market drops, does the stock's beta make this call riskier? "
        "fold that into your confidence. (4) how diversified is this single-stock call as protection against a "
        "sector-specific shock? diversification_score: 1-10.\n"
        "DECISION: does the regime support, contradict, or leave the call unchanged? action: BUY/SELL/HOLD. "
        "confidence: your final probability 0-1. macro_support: yes/neutral/no. context_reasoning: one sentence. "
        "key_context_risks: 1-2 concrete risks this regime adds to the call."
    )


def reflection_prompt(base: AgentSignal, symbol: str, summary: str) -> str:
    """Reflection prompt: a six-stage self-critique of the assembled call (review doc 4).

    Step 1 (the technical base) is seeded client-side from the call entering this phase; the model produces steps
    2-6. Each stage returns a conviction 0-1 that must not exceed the previous stage's - the client enforces this
    even if the model drifts, and applies the fixed humility discount at the end."""
    return (
        f"You are a cautious swing-trading technical analyst doing a final self-critique. The proposed call for "
        f"{symbol} is {base.action} at confidence {base.confidence:.2f}.\n"
        f"Step 1 (technical base) conviction starts at {base.confidence:.2f}. Go through the remaining steps and "
        "for each give a conviction 0-1 that is equal to or below the previous step's, plus a one-sentence reason.\n"
        f"{summary}\n"
        "Step 2 (reflection): attack the technical case - what did the first pass overlook? "
        "step2_reflection.\n"
        "Step 3 (fundamental): the bot has NO fundamental or earnings feed. If no fundamentals are visible here, "
        "keep your conviction unchanged and say that honestly; only move it if you actually have data. "
        "step3_fundamental.\n"
        "Step 4 (macro): challenge the call against the regime summary above. step4_macro.\n"
        "Step 5 (integration): weigh steps 2-4 together - netting out, does the call survive? step5_integration.\n"
        "Step 6 (risk): restate at the surviving conviction identified with the single biggest risk. "
        "step6_risk.\n"
        "Then give: biggest_risk: the single most likely way this call loses money that the analysis may have "
        "underweighted. what_proves_us_wrong: a falsifiable market condition that would mean the position should "
        "be exited immediately. bias_check: the cognitive biases (e.g. recency, anchoring, confirmation, loss "
        "aversion) most likely at play here, listed. action: the final BUY/SELL/HOLD. confidence: your proposed "
        "final probability 0-1 (the system still applies its own monotonic and humility rules on top)."
    )
