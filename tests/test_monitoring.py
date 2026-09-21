import json
import logging

import httpx
import pytest

from src.config import RiskLimits, Settings
from src.control import Control
from src.database import make_session_factory
from src.engine.risk_engine import Portfolio
from src.monitoring.telegram import HELP, CommandListener, Html, Notifier, handle_command
from tests.test_llm_and_pipeline import FakeBroker, make_pipeline

SETTINGS = Settings(dry_run=True, risk=RiskLimits(), market="india", currency="₹")


@pytest.fixture
def control(tmp_path):
    return Control(make_session_factory(f"sqlite:///{tmp_path / 'c.db'}"))


def test_pause_flag_defaults_off_and_persists(control):
    assert control.is_paused() is False
    control.set_paused(True)
    assert control.is_paused() is True
    control.set_paused(False)
    assert control.is_paused() is False


def test_commands_pause_resume_status_positions(control):
    broker = FakeBroker(Portfolio(90_000.0, 100_500.0, {"AAPL": 10_500.0}, {"AAPL": 70}, 100_000.0))
    assert "PAUSED" in handle_command("/pause", control, broker, SETTINGS) and control.is_paused()
    status = handle_command("/status", control, broker, SETTINGS)
    assert "PAUSED" in status and "₹100,500.00" in status and "+0.50% today" in status and "DRY RUN" in status
    assert "RESUMED" in handle_command("/resume", control, broker, SETTINGS) and not control.is_paused()
    assert "AAPL: 70 sh" in handle_command("/positions", control, broker, SETTINGS)
    assert handle_command("/pause@my_bot", control, broker, SETTINGS).startswith("Trading PAUSED")
    assert handle_command("hello", control, broker, SETTINGS) == HELP
    assert handle_command("", control, broker, SETTINGS) == HELP


def telegram_transport(updates, sent):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(200, json={"ok": True, "result": updates})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})
    return httpx.MockTransport(handler)


def test_listener_answers_authorised_chat_and_ignores_others():
    sent, replies = [], []
    updates = [
        {"update_id": 5, "message": {"chat": {"id": 111}, "text": "/pause"}},
        {"update_id": 6, "message": {"chat": {"id": 999}, "text": "/resume"}},
    ]
    client = httpx.Client(transport=telegram_transport(updates, sent))
    listener = CommandListener("tok", "111", lambda text: replies.append(text) or "ok", client=client)
    assert listener.poll_once() == 1
    assert replies == ["/pause"]  # the message from chat 999 never reached the handler
    assert [m["chat_id"] for m in sent] == ["111"] and listener.offset == 7


def test_listener_survives_handler_exception():
    sent = []
    updates = [{"update_id": 1, "message": {"chat": {"id": 1}, "text": "/status"}}]
    client = httpx.Client(transport=telegram_transport(updates, sent))

    def boom(_):
        raise RuntimeError("broker down")

    assert CommandListener("t", "1", boom, client=client).poll_once() == 1
    assert "failed" in sent[0]["text"]


def test_notifier_sends_and_never_raises_on_failure():
    sent = []
    ok = Notifier("tok", "42", httpx.Client(transport=telegram_transport([], sent)))
    assert ok.send("hi") and sent == [{"chat_id": "42", "text": "hi"}]
    bad = Notifier("tok", "42", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert bad.send("hi") is False


def test_bot_token_is_not_logged(caplog):
    caplog.set_level(logging.INFO)
    sent = []
    Notifier("SECRET-TOKEN-123", "1", httpx.Client(transport=telegram_transport([], sent))).send("x")
    assert "SECRET-TOKEN-123" not in caplog.text


def test_paused_pipeline_logs_decision_but_places_no_order(tmp_path):
    pipe, broker, sessions = make_pipeline(tmp_path, dry_run=False)
    pipe.control = Control(sessions)
    pipe.control.set_paused(True)
    [r] = pipe.run_once()
    assert r.order_status == "PAUSED" and broker.buys == []
    pipe.control.set_paused(False)
    [r] = pipe.run_once()
    assert r.order_status == "accepted" and len(broker.buys) == 1


def test_pipeline_alerts_on_order_failure_error_and_risk_halt(tmp_path):
    messages = []
    broker = FakeBroker(Portfolio(97_000.0, 97_000.0, {}, {}, 100_000.0))  # -3% day -> daily loss halt
    pipe, _, _ = make_pipeline(tmp_path, dry_run=False, broker=broker)
    pipe._notify = messages.append
    pipe.run_once()
    pipe.run_once()
    assert sum("RISK HALT" in m for m in messages) == 1  # deduplicated across cycles

    messages.clear()
    ok_broker = FakeBroker(Portfolio(100_000.0, 100_000.0, {}, {}, 100_000.0))
    ok_broker.buy_with_bracket = lambda *a: (_ for _ in ()).throw(RuntimeError("rejected"))
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    pipe2, _, _ = make_pipeline(second_dir, dry_run=False, broker=ok_broker)
    pipe2._notify = messages.append
    pipe2.run_once()
    assert any("FAILED" in m and "rejected" in m for m in messages)


def test_notification_failure_does_not_break_trading(tmp_path):
    pipe, broker, _ = make_pipeline(tmp_path, dry_run=False)
    pipe._notify = lambda m: (_ for _ in ()).throw(RuntimeError("telegram down"))
    [r] = pipe.run_once()
    assert r.order_status == "accepted" and len(broker.buys) == 1


def test_status_and_positions_use_rupees_for_the_india_market(control):
    broker = FakeBroker(Portfolio(900_000.0, 1_005_000.0, {"RELIANCE": 105_000.0}, {"RELIANCE": 85}, 1_000_000.0))
    india = Settings(dry_run=True, risk=RiskLimits(), market="india", currency="₹")
    assert "₹1,005,000.00" in handle_command("/status", control, broker, india)
    assert "RELIANCE: 85 sh, ₹105,000" in handle_command("/positions", control, broker, india)


def test_positions_command_shows_buy_date_price_and_profit_when_the_broker_has_holdings(control):
    from datetime import datetime, timezone

    class Detailed(FakeBroker):
        def holdings(self):
            return [{"symbol": "RELIANCE", "qty": 10, "avg_price": 100.0, "price": 110.0, "pnl": 100.0, "pnl_pct": 0.1, "stop": 92.0,
                     "opened_at": datetime(2026, 9, 21, tzinfo=timezone.utc)}]

    text = handle_command("/positions", control, Detailed(Portfolio(1, 1)), SETTINGS)
    assert isinstance(text, Html) and "<b>RELIANCE</b>  🟢 +10.0% (+₹100)" in text
    assert "10 sh · bought 21 Sep at ₹100.00" in text and "now ₹110.00" in text and "Total unrealised</b>  🟢" in text


def test_history_command_and_html_replies_are_sent_with_parse_mode(control):
    from datetime import datetime, timezone

    class Detailed(FakeBroker):
        def trade_history(self):
            return [{"symbol": "A<B", "qty": 5, "entry_price": 100.0, "exit_price": 90.0, "reason": "STOP", "fees": 12.0,
                     "net_pnl": -62.0, "opened_at": datetime(2026, 9, 21, tzinfo=timezone.utc),
                     "closed_at": datetime(2026, 9, 25, tzinfo=timezone.utc)}]

    text = handle_command("/history", control, Detailed(Portfolio(1, 1)), SETTINGS)
    assert "A&lt;B" in text and "🔴 -10.0% (-₹62)" in text and "21 Sep → 25 Sep" in text and "Realised net</b>  -₹62" in text
    sent = []
    Notifier("tok", "42", httpx.Client(transport=telegram_transport([], sent))).send(text)
    assert sent[0]["parse_mode"] == "HTML"


def test_intraday_commands_show_the_separate_account(control):
    from datetime import datetime, timezone

    class Intra(FakeBroker):
        def holdings(self):
            return [{"symbol": "ORB1", "qty": 10, "avg_price": 100.0, "price": 101.0, "pnl": 10.0, "pnl_pct": 0.01,
                     "stop": 99.0, "opened_at": datetime(2026, 9, 21, tzinfo=timezone.utc)}]

        def trade_history(self):
            return []

    intra = Intra(Portfolio(900_000.0, 1_000_500.0, {"ORB1": 1010.0}, {"ORB1": 10}, 1_000_000.0))
    text = handle_command("/intraday", control, FakeBroker(Portfolio(1, 1)), SETTINGS, intra)
    assert "INTRADAY account" in text and "₹1,000,500.00" in text and "+0.05% today" in text and "<b>ORB1</b>" in text
    assert handle_command("/intraday_history", control, FakeBroker(Portfolio(1, 1)), SETTINGS, intra) == "No closed trades yet."
    assert handle_command("/intraday", control, FakeBroker(Portfolio(1, 1)), SETTINGS) == HELP  # no intraday account wired
