"""The prompt text is versioned (`PROMPT_VERSION`, stored on every decision). Changing the wording without bumping the
version would make two different prompts share a label and break attribution ("why did behaviour change on that date?").
This test fingerprints every prompt and JSON schema; if it fails after you edited src/agents/technical/, bump
PROMPT_VERSION in src/versions.py and record the new fingerprint here in the same commit."""

import hashlib
import json
from datetime import date, datetime

from src.agents.technical import COT_SCHEMA, REFLECT_SCHEMA, mechanical_action, mechanical_baseline
from src.agents.technical.prompts import cot_prompt, reflection_prompt, simple_prompt
from src.data.fundamentals import Fundamentals, Quarter
from src.data.house_calls import HouseCall
from src.data.india import IST
from src.data.indicators import Snapshot
from src.data.market_context import MarketContext
from src.data.market_pulse import HotStock
from src.engine.agent_signal import AgentSignal
from src.versions import PROMPT_VERSION

FINGERPRINTS = {
    "technical-v10": "2b025ee232b16e3f959e75a3c53a1d44dfce2c9991b626d275d8840b1098d06e",
    "technical-v11": "be59a29a6f2a4ec570e8043432657ec9a5ec2f38e3b6aa4ef06b8de025d37ab5",
    "technical-v12": "bc29cd47022acf18b728bc5bade85e783a839f3ccb7606ebd3211031ccef70b1",
    "technical-v13": "2d8c5569a7c12f17ee66d215e0df7515905a32b77627169b952329d06321d61d",
}  # earlier versions' texts live in git history


PULSE = HotStock("UP", "Information Technology", 6, 0.11, 0.13, 0.09)
HOUSE = HouseCall(
    symbol="UP",
    kind="Swing Trade",
    side="buy",
    entry_low=248.0,
    entry_high=249.5,
    stop=240.0,
    target=262.0,
    published=datetime(2026, 10, 15, 14, 49, tzinfo=IST),
    valid_to=date(2026, 10, 28),
    is_open=True,
)


def fingerprint(with_house: bool = True, with_pulse: bool = True, with_crude: bool = True) -> str:
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
        crude=85.0 if with_crude else None,
        crude_1m=0.062 if with_crude else None,
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
    if with_house:
        out["prompts"]["house"] = cot_prompt(snaps[0], market, company, HOUSE)
        out["prompts"]["house_stored"] = cot_prompt(snaps[0], market, company, json.loads(HOUSE.to_json()))
    if with_pulse:
        out["prompts"]["pulse"] = cot_prompt(snaps[2], market, company, None, PULSE)
        out["prompts"]["pulse_stored"] = cot_prompt(snaps[2], market, company, None, json.loads(PULSE.to_json()))
        out["prompts"]["pulse_baseline"] = [mechanical_baseline(s, PULSE) for s in snaps]
        out["prompts"]["pulse_action"] = [str(mechanical_action(s, PULSE)) for s in snaps]
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


def test_the_prompt_text_matches_the_recorded_version():
    assert PROMPT_VERSION in FINGERPRINTS, f"record the fingerprint of {PROMPT_VERSION} in FINGERPRINTS"
    assert fingerprint() == FINGERPRINTS[PROMPT_VERSION], (
        "a prompt or schema changed but PROMPT_VERSION did not: bump src/versions.py PROMPT_VERSION and add the new "
        f"fingerprint ({fingerprint()}) to FINGERPRINTS"
    )


def test_a_stock_without_a_house_call_gets_exactly_the_v10_prompt():
    """v11 only ADDS the research line when there is a call: every other prompt is byte-for-byte what v10 sent."""
    assert fingerprint(with_house=False, with_pulse=False, with_crude=False) == FINGERPRINTS["technical-v10"]


def test_a_stock_without_a_pulse_gets_exactly_the_v11_prompt():
    """v12 only ADDS the market-pulse line and baseline when the stock's sector is surging."""
    assert fingerprint(with_pulse=False, with_crude=False) == FINGERPRINTS["technical-v11"]


def test_a_market_without_crude_data_gets_exactly_the_v12_prompt():
    """v13 only ADDS the Brent crude clause to the market line when the price could be read."""
    assert fingerprint(with_crude=False) == FINGERPRINTS["technical-v12"]


def test_a_surging_sector_stock_above_its_50_day_average_is_a_mechanical_buy():
    s = Snapshot("UP", 250.0, 240.0, 260.0, 74.0, 1000)  # MA50 below MA200, RSI 74: the long-term filter says HOLD
    assert str(mechanical_action(s)) != str(mechanical_action(s, PULSE))
    assert "market-pulse rule" in mechanical_baseline(s, PULSE) and "did NOT select" in mechanical_baseline(s)
    broken = Snapshot("UP", 230.0, 240.0, 260.0, 55.0, 1000)  # below MA50: the pulse does not make it a buy
    assert mechanical_action(broken, PULSE) == mechanical_action(broken)
    stretched = Snapshot("UP", 250.0, 240.0, 260.0, 85.0, 1000)  # RSI past the pulse's own line
    assert mechanical_action(stretched, PULSE) == mechanical_action(stretched)


def test_a_stored_pulse_rebuilds_the_same_prompt_on_replay():
    s = Snapshot("UP", 250.0, 240.0, 260.0, 74.0, 1000)
    assert cot_prompt(s, pulse=PULSE) == cot_prompt(s, pulse=json.loads(PULSE.to_json()))
    assert "MARKET PULSE" in cot_prompt(s, pulse=PULSE) and "MARKET PULSE" not in cot_prompt(s)


def test_a_stored_house_call_rebuilds_the_same_prompt_on_replay():
    assert cot_prompt(Snapshot("UP", 250.0, 240.0, 220.0, 58.0, 1000), house=HOUSE) == cot_prompt(
        Snapshot("UP", 250.0, 240.0, 220.0, 58.0, 1000), house=json.loads(HOUSE.to_json())
    )
