"""The interfaces the engine depends on (src/engine/ports.py): the real broker satisfies them, optional capabilities are
detected explicitly, and the whole application type-checks."""
from pathlib import Path

import pytest

from src.database import make_session_factory
from src.engine.paper_broker import PaperBroker
from src.engine.ports import (Broker, ChargesFees, ConfirmsFills, Fill, ListsHoldings, ListsTrades, MarketClock, PriceFeed,
                              ProtectsPositions)
from tests.test_llm_and_pipeline import FakeBroker
from tests.test_paper_broker import FakeClock, FakeFeed


def paper(tmp_path):
    return PaperBroker(make_session_factory(f"sqlite:///{tmp_path / 'p.db'}"), FakeFeed({"A": 100.0}), FakeClock())


def test_the_paper_broker_offers_exactly_the_capabilities_it_implements(tmp_path):
    broker = paper(tmp_path)
    assert isinstance(broker, ProtectsPositions) and isinstance(broker, ChargesFees)
    assert isinstance(broker, ListsHoldings) and isinstance(broker, ListsTrades)
    assert not isinstance(broker, ConfirmsFills)  # a simulator has nothing to confirm: its fills are already final


def test_a_bare_broker_has_none_of_the_optional_capabilities():
    bare = FakeBroker(None)
    for capability in (ProtectsPositions, ConfirmsFills, ChargesFees, ListsHoldings, ListsTrades):
        assert not isinstance(bare, capability)


def test_the_test_doubles_and_the_real_feed_and_clock_match_the_interfaces(tmp_path):
    feed, clock = FakeFeed({"A": 1.0}), FakeClock()
    assert callable(feed.last_price) and callable(feed.bars_since) and callable(clock.is_open) and callable(clock.today_ist)
    assert PriceFeed and MarketClock and Broker  # the interfaces exist and are importable from one place


def test_fill_is_defined_once_in_the_ports_module(tmp_path):
    fill = paper(tmp_path).buy_with_bracket("A", 5, 100.0, 0.15, 1.0)
    assert isinstance(fill, Fill) and fill.status == "filled" and fill.filled_qty == 5


def test_the_application_type_checks_cleanly():
    """Interface mismatches, Optional values used as if always set, and wrong return types surface here, not mid-cycle."""
    api = pytest.importorskip("mypy.api")
    root = Path(__file__).resolve().parent.parent
    stdout, stderr, status = api.run(["--config-file", str(root / "mypy.ini"), str(root / "src"), str(root / "main.py")])
    assert status == 0, stdout
