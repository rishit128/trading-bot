"""The typed vocabularies: identical to the old strings on the wire, strict in code, and recorded on every decision."""
import json

import pytest
from sqlalchemy import select

from src.agents.technical import AdvancedSignal, COT_SCHEMA
from src.database import DecisionRecord, OrderRecord
from src.engine.enums import ACTION_VALUES, NOT_A_PLACED_BUY, SILENT_STATUSES, Action, DecisionSource, OrderStatus
from src.llm import SIGNAL_SCHEMA, AgentSignal
from src.ops.replay import replay
from tests.characterization import build, observe


def test_members_are_the_old_strings_so_nothing_stored_or_sent_changes():
    assert [a.value for a in Action] == ["BUY", "SELL", "HOLD"] == ACTION_VALUES
    assert Action.BUY == "BUY" and "SELL" == Action.SELL and f"{Action.HOLD}" == "HOLD"
    assert json.dumps({"action": Action.BUY}) == '{"action": "BUY"}'
    assert {s.value for s in OrderStatus} == {"filled", "confirmed", "DRY_RUN", "PAUSED", "MARKET_CLOSED", "SKIPPED_COOLDOWN",
                                              "SKIPPED_OPEN_ORDER", "FAILED"}
    assert {d.value for d in DecisionSource} == {"agents", "trend_exit"}


def test_every_json_schema_sent_to_the_model_offers_exactly_the_three_actions():
    from src.agents.technical import schemas as agents
    schemas = [SIGNAL_SCHEMA, COT_SCHEMA, agents.LEARNING_SCHEMA, agents.CONTEXT_SCHEMA, agents.REFLECT_SCHEMA]
    for schema in schemas:
        assert schema["schema"]["properties"]["action"]["enum"] == ["BUY", "SELL", "HOLD"], schema["name"]


def test_typos_are_rejected_instead_of_becoming_a_different_branch():
    with pytest.raises(ValueError):
        Action("buy")  # the wrong case is not silently accepted
    with pytest.raises(AttributeError):
        Action.BYU  # noqa: B018
    with pytest.raises(ValueError):
        AgentSignal(action="MAYBE", confidence=0.5, reasoning="r")
    assert AgentSignal(action="BUY", confidence=0.5, reasoning="r").action is Action.BUY  # a model's string becomes the enum
    body = dict(action="HOLD", confidence=0.5, edge_confidence=0.5, step1_trend="up trend", step2_overbought="fine",
                step3_volume="ok volume", step4_confluence=5, step5_risks=["gap risk"], final_reasoning="stand aside")
    assert AdvancedSignal(**body).action is Action.HOLD


def test_the_placement_groups_are_derived_from_the_enum_not_retyped():
    assert OrderStatus.SKIPPED_COOLDOWN in NOT_A_PLACED_BUY and OrderStatus.FILLED not in NOT_A_PLACED_BUY
    assert OrderStatus.DRY_RUN in SILENT_STATUSES and OrderStatus.FAILED not in SILENT_STATUSES


def test_orders_and_decisions_round_trip_through_the_database_as_plain_strings(tmp_path):
    pipeline, broker, sessions, messages = build(tmp_path)
    observe(pipeline, broker, sessions, messages)
    with sessions() as s:
        assert s.scalar(select(OrderRecord.status).where(OrderRecord.symbol == "NEWBUY")) == "accepted"  # the broker's word
        statuses = {o.status for o in s.scalars(select(OrderRecord))}
        assert {"SKIPPED_OPEN_ORDER", "SKIPPED_COOLDOWN"} <= statuses  # readable by any SQL client, unchanged
        raw = s.connection().exec_driver_sql("select final_action, decision_source from decisions where symbol='HELD2'").one()
    assert raw == ("SELL", "trend_exit")


def test_every_decision_records_who_made_it(tmp_path):
    pipeline, broker, sessions, messages = build(tmp_path)
    observe(pipeline, broker, sessions, messages)
    with sessions() as s:
        sources = {d.symbol: d.decision_source for d in s.scalars(select(DecisionRecord))}
    assert sources["HELD2"] == "trend_exit"  # the deterministic exit
    assert {v for k, v in sources.items() if k != "HELD2"} == {"agents"}


def _record(reasoning, source):
    return DecisionRecord(id=1, symbol="X", price=90.0, final_action="SELL", final_confidence=1.0, reasoning=reasoning,
                          risk_approved=True, risk_quantity=1, risk_reason="ok", decision_source=source,
                          snapshot_json=json.dumps({"symbol": "X", "price": 90.0, "ma50": 92.0, "ma200": 95.0, "rsi": 55.0, "volume": 1}),
                          signals_json=json.dumps({}))


def test_replay_trusts_the_recorded_source_not_the_wording():
    assert replay(_record("some other wording entirely", "trend_exit")).rule_based is True
    # an AI reasoning that merely STARTS with the old marker is no longer mistaken for the rule:
    assert replay(_record("trend exit: the model said this itself", "agents")).rule_based is False
    # rows from before the column existed still work, by the old text check:
    assert replay(_record("trend exit: last close 90.00 is below the 200-day average 95.00", None)).rule_based is True
    assert replay(_record("technical SELL: weak", None)).rule_based is False


def test_a_database_from_before_decision_source_gains_the_column_and_replay_still_reads_its_rows(tmp_path):
    """The migration adds the nullable column to an existing paper database without touching its rows."""
    import sqlite3
    from src.database import make_session_factory

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE decisions (id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), price FLOAT,
            final_action VARCHAR(8), final_confidence FLOAT, reasoning TEXT, risk_approved BOOLEAN,
            risk_quantity INTEGER, risk_reason TEXT);
        INSERT INTO decisions VALUES (1, '2026-09-22 05:00:00', 'OLD', 90.0, 'SELL', 1.0,
            'trend exit: last close 90.00 is below the 200-day average 95.00', 1, 100, 'closing full OLD position');
    """)
    conn.commit()
    conn.close()
    sessions = make_session_factory(f"sqlite:///{path}")  # opening the database upgrades it
    with sessions() as s:
        row = s.scalars(select(DecisionRecord)).one()
        assert row.decision_source is None and row.symbol == "OLD"  # the old row is intact; the new column is empty
        assert replay(row).rule_based is True  # ... and replay falls back to the wording it always used
