"""/sell SYMBOL: the owner sells a whole swing holding (paper) after a preview, and the bot does not buy it back for 30 days."""

from datetime import datetime, timedelta, timezone

from src.config import RiskLimits, Settings
from src.engine.paper_broker import MANUAL_COOLDOWN_DAYS
from src.monitoring.telegram import HELP, handle_command
from tests.test_llm_and_pipeline import make_pipeline
from tests.test_paper_broker import make

SETTINGS = Settings(dry_run=True, risk=RiskLimits(), currency="Rs ", core_sleeve="NIFTYBEES:0.15")


def holding(tmp_path):
    broker, feed, clock, now, _ = make(tmp_path, prices={"AAA": 100.0, "NIFTYBEES": 250.0})
    broker.buy_with_bracket("AAA", 10, 100.0, 0.15, 1.0)
    broker.buy_with_bracket("NIFTYBEES", 4, 250.0, 0.9, 10.0)
    feed.prices["AAA"] = 130.0
    return broker, clock, now


def test_sell_shows_a_preview_first_and_sells_only_on_confirm(tmp_path):
    broker, *_ = holding(tmp_path)
    preview = handle_command("/sell aaa", None, broker, SETTINGS)
    assert "10 shares bought at Rs 100.00, now Rs 130.00 (+30.0%" in preview and "/sell AAA again" in preview
    assert any(h["symbol"] == "AAA" for h in broker.holdings())  # nothing sold yet
    done = handle_command("/sell AAA confirm", None, broker, SETTINGS)
    assert "Sold 10 AAA at Rs 130.00, Rs +300 after fees" in done
    assert not any(h["symbol"] == "AAA" for h in broker.holdings())
    assert broker.trade_history()[0]["reason"] == "MANUAL"


def test_sell_refuses_when_closed_not_held_a_core_etf_or_no_symbol(tmp_path):
    broker, clock, _ = holding(tmp_path)
    assert "Usage: /sell SYMBOL" in handle_command("/sell", None, broker, SETTINGS)
    assert "not held" in handle_command("/sell XYZ confirm", None, broker, SETTINGS)
    assert "core index ETF" in handle_command("/sell NIFTYBEES confirm", None, broker, SETTINGS)
    clock.open = False
    assert "market is closed" in handle_command("/sell AAA confirm", None, broker, SETTINGS)
    assert any(h["symbol"] == "AAA" for h in broker.holdings())
    assert "/sell SYMBOL" in HELP


def test_a_stock_sold_by_hand_is_not_bought_back_for_thirty_days(tmp_path):
    broker, _, now = holding(tmp_path)
    now[0] = datetime.now(timezone.utc) - timedelta(days=MANUAL_COOLDOWN_DAYS - 1)
    handle_command("/sell AAA confirm", None, broker, SETTINGS)
    pipe, *_ = make_pipeline(tmp_path, broker=broker)
    assert pipe._in_cooldown("AAA") and not pipe._in_cooldown("BBB")


def test_after_thirty_days_it_may_be_bought_again(tmp_path):
    broker, _, now = holding(tmp_path)
    now[0] = datetime.now(timezone.utc) - timedelta(days=MANUAL_COOLDOWN_DAYS + 1)
    handle_command("/sell AAA confirm", None, broker, SETTINGS)
    pipe, *_ = make_pipeline(tmp_path, broker=broker)
    assert not pipe._in_cooldown("AAA")


def test_sending_the_same_sell_twice_confirms_it(tmp_path, monkeypatch):
    import src.monitoring.telegram as tg

    broker, *_ = holding(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(tg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tg, "_sell_previews", {})
    assert "/sell AAA again" in handle_command("/sell aaa", None, broker, SETTINGS)
    clock[0] += tg.CONFIRM_WINDOW_S + 1  # too late: just a fresh preview
    assert "again within 5 minutes" in handle_command("/sell AAA", None, broker, SETTINGS)
    assert any(h["symbol"] == "AAA" for h in broker.holdings())
    clock[0] += 60
    assert "Sold 10 AAA" in handle_command("/sell AAA", None, broker, SETTINGS)
    assert not any(h["symbol"] == "AAA" for h in broker.holdings())
