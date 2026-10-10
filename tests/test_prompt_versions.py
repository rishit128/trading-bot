"""The prompt text is versioned (`PROMPT_VERSION`, stored on every decision). Changing the wording without bumping the
version would make two different prompts share a label and break attribution ("why did behaviour change on that date?").
This test fingerprints every prompt and JSON schema; if it fails after you edited src/agents/technical/, bump
PROMPT_VERSION in src/versions.py and record the new fingerprint here in the same commit."""

import hashlib
import json

from src.agents.technical import COT_SCHEMA, REFLECT_SCHEMA, mechanical_action, mechanical_baseline
from src.agents.technical.prompts import cot_prompt, reflection_prompt, simple_prompt
from src.data.fundamentals import Fundamentals, Quarter
from src.data.indicators import Snapshot
from src.data.market_context import MarketContext
from src.engine.agent_signal import AgentSignal
from src.versions import PROMPT_VERSION

FINGERPRINTS = {
    "technical-v10": "2b025ee232b16e3f959e75a3c53a1d44dfce2c9991b626d275d8840b1098d06e"
}  # earlier versions' texts live in git history


def fingerprint() -> str:
    snaps = [
        Snapshot(
            "UP",
            250.0,
            240.0,
            220.0,
            58.0,
            1_200_000,
            avg_volume=900_000,
            momentum=0.03,
            macd=1.2,
            macd_signal=0.9,
            macd_histogram=0.3,
            bb_position=0.85,
            volume_trend=0.1,
            atr=5.0,
            adx=45.0,
            momentum_6m=0.2,
            momentum_12_1=0.9,
        ),
        Snapshot("DOWN", 90.0, 100.0, 110.0, 75.0, 500, adx=10.0, bb_position=0.1),
        Snapshot("BARE", 100.0, 98.0, 95.0, 60.0, 1000),
        Snapshot("OB", 300.0, 280.0, 250.0, 72.5, 1000, avg_volume=800),
    ]
    base = AgentSignal(action="BUY", confidence=0.72, reasoning="uptrend intact")
    market = MarketContext(
        index_price=23400.0,
        index_ma200=24500.0,
        index_above_ma200=False,
        index_rsi=41.0,
        index_ytd=-0.03,
        index_mom_6m=-0.04,
        vix=18.0,
        vix_percentile=0.85,
        vs_index_6m=0.05,
    )
    company = Fundamentals(
        sector="Industrials",
        industry="Machinery",
        market_cap=35e9,
        pe=19.5,
        debt_to_equity=0.18,
        profit_margin=0.145,
        revenue_growth=0.27,
        earnings_growth=-0.12,
        promoter_holding=0.66,
        quarters=(Quarter("2026-06-30", 3.47e9, 5.66e8), Quarter("2026-03-31", 3.39e9, None)),
    )
    out = {"schemas": {n: json.dumps(v, sort_keys=True) for n, v in (("cot", COT_SCHEMA), ("refl", REFLECT_SCHEMA))}}
    out["prompts"] = {
        s.symbol: {
            "cot": cot_prompt(s),
            "simple": simple_prompt(s),
            "baseline": mechanical_baseline(s),
            "action": str(mechanical_action(s)),
        }
        for s in snaps
    }
    out["prompts"]["informed"] = cot_prompt(snaps[0], market, company)
    out["prompts"]["reflection"] = reflection_prompt(base, "UP", "summary line one\nsummary line two")
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


def test_the_prompt_text_matches_the_recorded_version():
    assert PROMPT_VERSION in FINGERPRINTS, f"record the fingerprint of {PROMPT_VERSION} in FINGERPRINTS"
    assert fingerprint() == FINGERPRINTS[PROMPT_VERSION], (
        "a prompt or schema changed but PROMPT_VERSION did not: bump src/versions.py PROMPT_VERSION and add the new "
        f"fingerprint ({fingerprint()}) to FINGERPRINTS"
    )
