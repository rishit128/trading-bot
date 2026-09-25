"""Golden tests: one full trading cycle over nine deliberately different situations, pinned end to end (results, broker
calls, audit rows, alerts). They describe what the bot does TODAY; a restructuring that changes any line here has
changed behaviour, not just structure, and must say so."""
from pathlib import Path

from tests.characterization import build, observe

LIVE = {
    "results": [
        ("NEWBUY", "BUY", 0.9, True, 50, "accepted", None),
        ("LOWCONF", "BUY", 0.5, False, 0, None, None),
        ("HOLDER", "HOLD", 0.7, False, 0, None, None),
        ("HELD1", "BUY", 0.9, False, 0, None, None),
        ("HELD2", "SELL", 1.0, True, 100, "accepted", None),  # the deterministic trend exit, whatever the agent said
        ("BROKEN", "ERROR", 0.0, None, None, None, "ValueError: BROKEN: no market data returned"),
        ("OPENORD", "BUY", 0.9, True, 50, "SKIPPED_OPEN_ORDER", None),
        ("COOL", "BUY", 0.9, True, 50, "SKIPPED_COOLDOWN", None),
        ("SELLNOTHELD", "SELL", 0.9, False, 0, None, None),
    ],
    "broker_buys": [("NEWBUY", 50, 100.0, 0.15, 1.0)],  # quantity from the risk engine, 15% stop, no practical target
    "broker_sells": [("HELD2", 100)],
    "decisions": [
        ("NEWBUY", "BUY", 0.9, True, 50, "approved 50 shares"),
        ("LOWCONF", "BUY", 0.5, False, 0, "confidence 0.50 below minimum 0.60"),
        ("HOLDER", "HOLD", 0.7, False, 0, "HOLD: nothing to do"),
        ("HELD1", "BUY", 0.9, False, 0, "no room: position_room=-15,000 exposure_room=40,000 cash=60,"),
        ("HELD2", "SELL", 1.0, True, 100, "closing full HELD2 position"),
        ("OPENORD", "BUY", 0.9, True, 50, "approved 50 shares"),
        ("COOL", "BUY", 0.9, True, 50, "approved 50 shares"),
        ("SELLNOTHELD", "SELL", 0.9, False, 0, "no SELLNOTHELD position to sell"),
    ],
    "orders": [
        ("COOL", "BUY", 5, "filled"),  # the earlier purchase that starts COOL's cooldown
        ("NEWBUY", "BUY", 50, "accepted"),
        ("HELD2", "SELL", 100, "accepted"),
        ("OPENORD", "BUY", 50, "SKIPPED_OPEN_ORDER"),
        ("COOL", "BUY", 50, "SKIPPED_COOLDOWN"),
    ],
    "alerts": [
        "BUY 50 NEWBUY @ ~₹100.00: accepted",
        "SELL 100 HELD2 @ ~₹90.00: accepted",
        "ERROR processing BROKEN: ValueError: BROKEN: no market data returned",
        "BUY 50 OPENORD @ ~₹100.00: SKIPPED_OPEN_ORDER",
    ],
}


def test_a_live_cycle_over_nine_situations(tmp_path: Path):
    assert observe(*build(tmp_path)) == LIVE


def test_the_same_cycle_as_a_dry_run_records_everything_and_places_nothing(tmp_path: Path):
    seen = observe(*build(tmp_path, dry_run=True))
    assert seen["broker_buys"] == [] and seen["broker_sells"] == []
    assert seen["orders"] == [("COOL", "BUY", 5, "filled"), ("NEWBUY", "BUY", 50, "DRY_RUN"), ("HELD2", "SELL", 100, "DRY_RUN"),
                              ("OPENORD", "BUY", 50, "DRY_RUN"), ("COOL", "BUY", 50, "SKIPPED_COOLDOWN")]
    assert seen["alerts"] == ["ERROR processing BROKEN: ValueError: BROKEN: no market data returned"]
    assert seen["decisions"] == LIVE["decisions"] and seen["results"][0][5] == "DRY_RUN"  # same decisions, different execution


def test_a_drawdown_halt_blocks_every_buy_but_never_an_exit(tmp_path: Path):
    pipeline, broker, sessions, messages = build(tmp_path)
    pipeline._record_equity(130_000.0)  # equity is 100,000: 23% below the recorded peak
    seen = observe(pipeline, broker, sessions, messages)
    assert seen["broker_buys"] == [] and seen["broker_sells"] == [("HELD2", 100)]
    approved = [r[:2] for r in seen["results"] if r[3]]
    assert approved == [("HELD2", "SELL")]
    assert {d[5] for d in seen["decisions"]} >= {"drawdown 23.08% reached limit 20.00%"}
    assert seen["alerts"][0] == "RISK HALT: drawdown 23.08% reached limit 20.00%. New buys are blocked."


def test_the_cycle_is_deterministic_across_runs(tmp_path: Path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first, second = observe(*build(tmp_path / "a")), observe(*build(tmp_path / "b"))
    assert first == second  # ordering and outcomes do not depend on thread timing
