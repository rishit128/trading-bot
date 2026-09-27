"""The real-portfolio agent's building blocks without a browser: the network guard, the OTP exchange (shared Telegram
bot), the credential vault, holdings extraction/analysis, the graph's failure path and the /portfolio command."""
import json

import httpx
import keyring
import pytest
from keyring.backend import KeyringBackend

from src.portfolio import cli, vault
from src.portfolio.runs import RunStore
from src.portfolio.guard import verdict
from src.portfolio.telegram_otp import OtpCancelled, OtpInbox, OtpTimeout, TelegramError, TelegramOtp

HOST = "webapp.integrated.investments"
URL = f"https://{HOST}/StandardService/api/v1/Common/Claim"
TOKEN = "123456:" + "A" * 35


# -- guard -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method,url,headers", [
    ("POST", URL, {"App-Action-Name": "PlaceOrder"}),
    ("POST", URL, {"app-process-name": "OrderEntry", "app-action-name": "Submit"}),
    ("POST", URL, {"App-Action-Name": "SellHolding"}),
    ("POST", URL, {"App-Action-Name": "PayoutRequest"}),
    ("POST", URL, {"App-Action-Name": "PledgeRequest"}),
    ("POST", URL, {"App-Action-Name": "MfRedeem"}),
    ("POST", f"https://{HOST}/ODINService/api/v1/Order/Place", {}),
    ("POST", f"https://{HOST}/CIAMS/api/v1/UserManagement/managecredentials", {}),
    ("POST", f"https://{HOST}/CIAMS/api/v1/UserManagement/changempin", {}),
    ("GET", f"https://{HOST}/api/placeorder?qty=1", {}),
    ("POST", "https://payments.example.com/charge", {}),
    ("POST", "https://evilintegrated.investments/x", {}),  # a look-alike domain is third-party
])
def test_the_guard_blocks_anything_that_could_trade_or_move_money(method, url, headers):
    assert verdict(method, url, headers, HOST) is not None


@pytest.mark.parametrize("method,url,headers", [
    ("POST", URL, {"App-Action-Name": "Holdings", "App-Process-Name": "Portfolio"}),
    ("POST", f"https://{HOST}/ODINService/api/v1/Funds/FundsViewDetailClaim", {}),
    ("POST", f"https://{HOST}/CIAMS/api/v1/SSAuth/login", {}),
    ("POST", f"https://{HOST}/CIAMS/api/v1/SSAuth/logout", {"App-Action-Name": "Logout"}),
    ("POST", f"https://{HOST}/CIAMS/api/v1/TwoFA/coidverify", {}),
    ("GET", f"https://{HOST}/nv1/Scripts/OrderWindow.js", {}),  # a script file is not an action
    ("GET", "https://cdnjs.cloudflare.com/ajax/libs/jquery.min.js", {}),
])
def test_the_guard_lets_reads_login_and_logout_through(method, url, headers):
    assert verdict(method, url, headers, HOST) is None


# -- Telegram OTP ------------------------------------------------------------------------------------------------
class FakeTelegram:
    """Telegram's Bot API in memory: a queue of updates plus a log of calls."""

    def __init__(self, updates=(), chat="42"):
        self.queue = list(updates)
        self.calls, self.chat, self.later = [], chat, []

    def msg(self, uid, text, chat=None):
        return {"update_id": uid, "message": {"message_id": 1000 + uid, "chat": {"id": int(chat or self.chat)}, "text": text}}

    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        self.calls.append((method, body))
        if method == "getUpdates":
            offset = body.get("offset", 0)
            if offset == -1:
                return httpx.Response(200, json={"ok": True, "result": self.queue[-1:]})
            self.queue = [u for u in self.queue if u["update_id"] >= offset]
            result = list(self.queue)
            if not result and self.later:
                self.queue.append(self.later.pop(0))
            return httpx.Response(200, json={"ok": True, "result": result})
        return httpx.Response(200, json={"ok": True, "result": True})

    def client(self):
        return TelegramOtp(TOKEN, self.chat, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_the_otp_is_taken_only_from_the_authorised_chat_after_the_request_and_then_deleted():
    tg = FakeTelegram(updates=[])
    tg.queue = [tg.msg(1, "1111")]  # an old message: must never count
    otp = tg.client()
    otp.begin()
    tg.later = [tg.msg(2, "5555", chat="999"), tg.msg(3, "hello"), tg.msg(4, " 4321 ")]
    assert otp.wait_for_otp(timeout_s=60) == "4321"
    methods = [m for m, _ in tg.calls]
    assert ("deleteMessage", {"chat_id": "42", "message_id": 1004}) in tg.calls
    assert methods.count("sendMessage") == 2  # the prompt, and the "not a 4-digit OTP" hint for "hello"
    assert methods[-2:] == ["getUpdates", "deleteMessage"]
    assert otp.offset == 5  # everything up to the OTP acknowledged: it cannot be fetched again


def test_cancel_and_timeout():
    tg = FakeTelegram()
    tg.later = [tg.msg(1, "cancel")]
    with pytest.raises(OtpCancelled):
        tg.client().wait_for_otp(timeout_s=60)

    now = [0.0]
    quiet = FakeTelegram()
    otp = TelegramOtp(TOKEN, "42", client=httpx.Client(transport=httpx.MockTransport(quiet.handler)),
                      clock=lambda: now.__setitem__(0, now[0] + 30) or now[0])
    with pytest.raises(OtpTimeout):
        otp.wait_for_otp(timeout_s=100)


def test_telegram_errors_never_contain_the_bot_token():
    def down(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    def rejected(request):
        return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

    for handler in (down, rejected):
        otp = TelegramOtp(TOKEN, "42", client=httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(TelegramError) as info:
            otp.send("x")
        chained = info.value.__cause__ or (None if info.value.__suppress_context__ else info.value.__context__)
        assert TOKEN not in str(info.value) and chained is None


def test_a_running_trading_bot_is_detected_instead_of_fighting_it_for_messages():
    conflict = httpx.MockTransport(lambda r: httpx.Response(409, json={"ok": False, "description": "Conflict"}))
    with pytest.raises(TelegramError, match="send /portfolio"):
        TelegramOtp(TOKEN, "42", client=httpx.Client(transport=conflict)).begin()


def test_the_inbox_takes_the_otp_from_the_shared_bot_listener():
    import threading

    inbox, prompts, got = OtpInbox(), [], []
    assert inbox.offer("1234") is not None  # no login waiting: still taken (and deleted), never passed on
    assert inbox.offer("/status") is None and inbox.offer("cancel") is None  # normal commands untouched
    waiter = threading.Thread(target=lambda: got.append(inbox.wait(prompts.append, timeout_s=5)))
    waiter.start()
    while not prompts:
        pass
    assert inbox.offer("hello") is None and inbox.offer(" 4321 ") == "OTP received, logging in."
    waiter.join()
    assert got == ["4321"] and "4-digit OTP" in prompts[0] and inbox.offer("5555").startswith("No portfolio login")

    def cancel_soon():
        while not inbox.offer("cancel"):
            pass

    threading.Thread(target=cancel_soon).start()
    with pytest.raises(OtpCancelled):
        inbox.wait(lambda _: None, timeout_s=5)
    with pytest.raises(OtpTimeout):
        inbox.wait(lambda _: None, timeout_s=0.05)


def test_the_trading_bot_listener_hands_the_otp_over_and_deletes_it():
    from src.monitoring.telegram import CommandListener

    calls, replies = [], []
    updates = [{"update_id": 1, "message": {"message_id": 77, "chat": {"id": 42}, "text": "4321"}},
               {"update_id": 2, "message": {"message_id": 78, "chat": {"id": 42}, "text": "/status"}}]

    def handler(request):
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(200, json={"ok": True, "result": updates})
        calls.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    listener = CommandListener(TOKEN, "42", lambda text: replies.append(text) or "status",
                               client=httpx.Client(transport=httpx.MockTransport(handler)),
                               intercept=lambda text: "OTP received" if text == "4321" else None)
    assert listener.poll_once() == 2
    assert replies == ["/status"]  # the OTP never reached the command handler
    assert ("deleteMessage", {"chat_id": "42", "message_id": 77}) in calls
    assert [b["text"] for m, b in calls if m == "sendMessage"] == ["OTP received", "status"]


# -- vault -------------------------------------------------------------------------------------------------------
class MemoryKeyring(KeyringBackend):
    priority = 1  # type: ignore[assignment]

    def __init__(self):
        super().__init__()
        self.data = {}

    def get_password(self, service, username):
        return self.data.get((service, username))

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.data:
            import keyring.errors
            raise keyring.errors.PasswordDeleteError(username)
        del self.data[(service, username)]


GOOD = vault.Credentials("9876543210", "123456", "")


@pytest.fixture
def memory_keyring():
    previous = keyring.get_keyring()
    store = MemoryKeyring()
    keyring.set_keyring(store)
    yield store
    keyring.set_keyring(previous)


def test_credentials_round_trip_through_the_os_store_and_never_print(memory_keyring):
    with pytest.raises(vault.MissingCredentials):
        vault.load()
    vault.save(GOOD)
    assert vault.load() == GOOD and "9876543210" not in repr(GOOD) and "123456" not in repr(vault.load())
    vault.forget()
    assert memory_keyring.data == {}


def test_invalid_credentials_are_not_stored_and_not_echoed(memory_keyring):
    bad = vault.Credentials("55555", "12ab", "x1")
    with pytest.raises(ValueError) as info:
        vault.save(bad)
    assert memory_keyring.data == {} and "12ab" not in str(info.value) and "55555" not in str(info.value)
    assert len(vault.problems(vars(bad))) == 3


def test_a_plaintext_or_missing_credential_store_is_refused():
    from keyring.backends import fail

    previous = keyring.get_keyring()
    keyring.set_keyring(fail.Keyring())
    try:
        with pytest.raises(vault.InsecureStore):
            vault.save(GOOD)
    finally:
        keyring.set_keyring(previous)


def test_an_unknown_backend_name_is_rejected(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_SECRET_BACKEND", "sqlite")
    with pytest.raises(ValueError, match="PORTFOLIO_SECRET_BACKEND"):
        vault.load()


# -- vault: AWS Secrets Manager backend (the cloud deployment; no Windows Credential Manager there) ----------------
class FakeSecretsManager:
    """A minimal stand-in for boto3's Secrets Manager client: one named secret, holding a JSON string."""

    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

    def __init__(self):
        self.secrets: dict = {}
        self.calls: list = []

    def get_secret_value(self, SecretId):
        self.calls.append(("get", SecretId))
        if SecretId not in self.secrets:
            raise self.exceptions.ResourceNotFoundException(SecretId)
        return {"SecretString": self.secrets[SecretId]}

    def put_secret_value(self, SecretId, SecretString):
        self.calls.append(("put", SecretId))
        if SecretId not in self.secrets:
            raise self.exceptions.ResourceNotFoundException(SecretId)
        self.secrets[SecretId] = SecretString

    def create_secret(self, Name, SecretString, Description=""):
        self.calls.append(("create", Name))
        self.secrets[Name] = SecretString

    def delete_secret(self, SecretId, ForceDeleteWithoutRecovery=True):
        self.calls.append(("delete", SecretId))
        if SecretId not in self.secrets:
            raise self.exceptions.ResourceNotFoundException(SecretId)
        del self.secrets[SecretId]


@pytest.fixture
def aws_backend(monkeypatch):
    """PORTFOLIO_SECRET_BACKEND=aws, with boto3.client(...) swapped for an in-memory fake (no network, no real AWS)."""
    fake = FakeSecretsManager()
    monkeypatch.setenv("PORTFOLIO_SECRET_BACKEND", "aws")
    monkeypatch.setattr("boto3.client", lambda service, region_name=None: fake)
    return fake


def test_aws_backend_round_trips_credentials_as_one_json_secret(aws_backend):
    with pytest.raises(vault.MissingCredentials):
        vault.load()
    vault.save(GOOD)
    assert vault.load() == GOOD
    body = aws_backend.secrets[vault.SERVICE]
    assert "9876543210" in body and "123456" in body  # it is the one JSON secret, not per-field like keyring
    assert ("create", vault.SERVICE) in aws_backend.calls  # first save creates the secret
    vault.forget()
    with pytest.raises(vault.MissingCredentials):
        vault.load()


def test_aws_backend_second_save_updates_the_existing_secret_instead_of_recreating_it(aws_backend):
    vault.save(GOOD)
    changed = vault.Credentials("9876543211", "654321", "")
    vault.save(changed)
    assert vault.load() == changed
    assert ("put", vault.SERVICE) in aws_backend.calls and aws_backend.calls.count(("create", vault.SERVICE)) == 1


def test_aws_backend_invalid_credentials_are_never_sent_to_aws(aws_backend):
    with pytest.raises(ValueError):
        vault.save(vault.Credentials("55555", "12ab", "x1"))
    assert aws_backend.calls == []


def test_aws_backend_forgetting_an_unset_secret_does_not_raise(aws_backend):
    vault.forget()  # nothing was ever saved; must not blow up
    assert ("delete", vault.SERVICE) in aws_backend.calls


def test_aws_backend_uses_a_custom_secret_name_when_given(aws_backend, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_SECRET_NAME", "custom/name")
    vault.save(GOOD)
    assert "custom/name" in aws_backend.secrets and vault.SERVICE not in aws_backend.secrets


# -- CLI -----------------------------------------------------------------------------------------------------------
def test_check_masks_everything_and_setup_clears_an_mpin_block(tmp_path, memory_keyring, capsys, monkeypatch):
    store = RunStore(tmp_path)
    vault.save(GOOD)
    store.block_mpin()
    assert cli.main(["check"], store) == 0
    out = capsys.readouterr().out
    assert "******3210" in out and "9876543210" not in out and "123456" not in out and "BLOCKED" in out
    answers = iter(["9876543210", "654321"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: next(answers))
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert cli.main(["setup"], store) == 0 and vault.load().mpin == "654321" and store.mpin_blocked() is None


# -- holdings ----------------------------------------------------------------------------------------------------
def test_numbers_in_indian_formats():
    from src.portfolio.holdings import number

    assert number("₹1,23,456.50") == 123456.5 and number("(1,200)") == -1200 and number("-3.5%") == -3.5
    assert number("") is None and number("N/A") is None and number(12) == 12.0


def test_holdings_are_found_by_their_headings_and_totals_are_skipped():
    from src.portfolio.holdings import from_json, from_tables

    table = [["Scrip Name", "Net Qty", "Avg. Price", "LTP", "Buy Value", "Current Value"],
             ["NBCC", "1,000", "95.00", "112.40", "95,000", "1,12,400"],
             ["IRFC", "500", "150", "138.10", "75,000", "69,050"],
             ["Total", "", "", "", "1,70,000", "1,81,450"]]
    found = from_tables([[["Date", "Amount"], ["1 Sep", "500"]], table])
    assert [(h.symbol, h.value, h.invested) for h in found] == [("NBCC", 112400.0, 95000.0), ("IRFC", 69050.0, 75000.0)]
    body = {"data": {"holdings": [{"tradingsymbol": "TCS", "quantity": 2, "average_price": 3000, "ltp": 3500}]}}
    assert [(h.symbol, h.value, h.invested) for h in from_json(body)] == [("TCS", 7000.0, 6000.0)]
    assert from_tables([[["Date", "Amount"], ["1 Sep", "500"]]]) == []


def test_analysis_flags_concentration_and_deep_losses_and_the_ai_sees_no_amounts():
    from src.portfolio.holdings import Holding, analyse, anonymised

    holdings = [Holding("BIG", 100, 100, 100, 50_000, 40_000), Holding("LOSER", 10, 100, 60, 600, 1_000)] + \
        [Holding(f"S{i}", 1, 1, 1, 5_000, 5_000) for i in range(6)]
    a = analyse(holdings)
    assert a["holdings"] == 8 and a["positions"][0]["symbol"] == "BIG" and a["total_value"] == 80_600
    assert any("BIG is 62%" in f for f in a["flags"]) and any("LOSER is 40% below its average cost" in f for f in a["flags"])
    assert any("five largest" in f for f in a["flags"])
    shared = json.dumps(anonymised(a))
    assert "BIG" in shared and "50000" not in shared and "80600" not in shared and "qty" not in shared


def test_the_graph_reports_a_failed_login_without_analysing_anything():
    from src.portfolio.graph import build_portfolio_graph

    sent = []
    graph = build_portfolio_graph(lambda: {"error": "login stopped at otp: portal said: Invalid OTP", "blocked": 0,
                                           "logged_out": True}, sent.append,
                                  explain=lambda d: pytest.fail("no AI call on a failed run"))
    state = graph.invoke({})
    assert "analysis" not in state and "Invalid OTP" in sent[0] and "Logged out safely" in sent[0]


def test_the_portfolio_command_runs_one_background_run_at_a_time():
    import threading

    from src.app.portfolio_agent import PortfolioCommand

    release, sent, started = threading.Event(), [], []

    def runner(get_otp, notify, explain):
        started.append(1)
        release.wait(5)
        return {}

    command = PortfolioCommand(sent.append, runner=runner)
    assert "started" in command() and "already in progress" in command()
    release.set()
    command._thread.join(5)
    assert started == [1] and "started" in command()


def test_dashboard_totals_and_indian_digit_grouping():
    from src.portfolio.graph import format_report
    from src.portfolio.holdings import inr, totals

    text = ("My portfolio\nTotal Portfolio Value\n\u20b9 12,34,567.80\n12.00 %\nTotal Invested\n\u20b9 11,00,000.00\n"
            "Gain/Loss\n\u20b9 1,34,567.80\n100%\nEquity\nAvailable Margin\n\u20b9 250.00\nAdd funds")
    t = totals(text)
    assert t == {"value": 1234567.8, "invested": 1100000.0, "gain": 134567.8, "margin": 250.0}
    assert totals("nothing here") == {}
    assert (inr(1234567.8), inr(999), inr(-134567.8), inr(0)) == ("\u20b912,34,568", "\u20b9999", "-\u20b91,34,568", "\u20b90")
    report = format_report({"totals": t, "pages": {"dashboard": {"reached": True}}, "blocked": 2, "logged_out": True})
    assert "\u20b912,34,568" in report and "(+12.2%)" in report and "no per-stock table" in report


def test_a_days_gain_column_is_not_taken_for_the_total_profit():
    from src.portfolio.holdings import from_rows

    # only the generic word "gain" can match here, so the day column must be skipped for the total one
    [h] = from_rows(["Stock", "Qty", "LTP", "Day's Gain", "Total Gain"], [["TCS", "2", "3500", "50", "1000"]])
    assert h.invested == 6000.0 and round(h.pnl_pct, 4) == round(7000 / 6000 - 1, 4)


def test_the_report_warns_about_saturday_mock_prices(tmp_path):
    from src.portfolio.graph import extract, format_report

    (tmp_path / "dashboard.txt").write_text("Total Portfolio Value\n\u20b9 100.00\nSATURDAY MOCK TRADING SESSION NOTICE",
                                            encoding="utf-8")
    state = extract({"run_dir": str(tmp_path)})
    assert state["warnings"] and "mock-session prices" in format_report({**state, "pages": {}, "logged_out": True})
    (tmp_path / "dashboard.txt").write_text("Total Portfolio Value\n\u20b9 100.00", encoding="utf-8")
    assert extract({"run_dir": str(tmp_path)})["warnings"] == []


def _analyzer_body():
    """The Portfolio Analyzer response in the portal's real layout (JSON inside a string), with made-up holdings plus
    the 'ZZZ' sector copies the portal adds, which must be ignored."""
    def row(isin, name, qty, rate, value, sector, kind="Stocks"):
        return {"tsh_isin": isin, "tsh_isin_desc": name, "tsh_bal": qty, "Rate": rate, "tsh_valuation": value,
                "sector_name": sector, "tsh_acct_desc": kind, "Asset_Type": "EQUITY"}
    rows = [row("INE000A01011", "Alpha Ltd", "10", 500.0, "5,000.00", "IT"),
            row("INE000B01012", "Beta Ltd", "30", 100.0, "3,000.00", "BANKS"),
            row("INE000C01013", "Gamma Ltd", "4", 500.0, "2,000.00", "IT"),
            row("ZZZ", "ZZZ", "1", 0.0, "7,000.00", "IT", kind="XSTL")]
    return json.dumps([{"LiveStatus": "N", "PayData": rows, "ClientDetails": {"Name": "not read"}}])


def test_the_portfolio_analyzer_statement_is_read_field_by_field():
    from src.portfolio.holdings import analyse, anonymised, from_portfolio_analyzer

    found = from_portfolio_analyzer(_analyzer_body())
    assert [(h.symbol, h.qty, h.value, h.sector) for h in found] == [
        ("Alpha Ltd", 10.0, 5000.0, "IT"), ("Beta Ltd", 30.0, 3000.0, "BANKS"), ("Gamma Ltd", 4.0, 2000.0, "IT")]
    a = analyse(found)
    assert a["sectors"] == [{"sector": "IT", "weight": 0.7}, {"sector": "BANKS", "weight": 0.3}]
    assert any("IT is 70%" in f for f in a["flags"]) and any("Alpha Ltd is 50%" in f for f in a["flags"])
    shared = json.dumps(anonymised(a))
    assert "IT" in shared and "5000" not in shared and "not read" not in shared
    assert from_portfolio_analyzer("not json") == [] and from_portfolio_analyzer({"other": 1}) == []


def test_extract_prefers_the_statement_and_rejects_market_lists_that_do_not_add_up(tmp_path):
    from src.portfolio.graph import extract

    api = tmp_path / "api"
    api.mkdir()
    market = [{"Symbol": f"S{i}", "Qty": 1000, "LTP": 999} for i in range(50)]  # "52-week highs": not the account
    (api / "dashboard_18_D1_Portfolio_52Week.json").write_text(json.dumps({"body": json.dumps(market)}), encoding="utf-8")
    (api / "dashboard_19_Other.json").write_text(json.dumps({"body": market}), encoding="utf-8")
    (tmp_path / "dashboard.txt").write_text("Total Portfolio Value\n\u20b9 10,000.00", encoding="utf-8")
    assert extract({"run_dir": str(tmp_path)})["holdings"] == []  # adds up to 5 crore, not 10,000: rejected

    (api / "analyzer_00_1_Statements_PortfolioAnalyzer.json").write_text(json.dumps({"body": _analyzer_body()}),
                                                                         encoding="utf-8")
    state = extract({"run_dir": str(tmp_path)})
    assert state["source"].startswith("analyzer_00") and len(state["holdings"]) == 3 and not state["warnings"]

    (tmp_path / "dashboard.txt").write_text("Total Portfolio Value\n\u20b9 99,000.00", encoding="utf-8")
    state = extract({"run_dir": str(tmp_path)})
    assert state["holdings"] == [] and "not the portal" in state["warnings"][0]


def test_long_reports_are_split_for_telegram_without_losing_a_line():
    from src.portfolio.graph import split_message

    text = "\n".join(f"{i}. holding number {i} with a long enough description" for i in range(300))
    parts = split_message(text, limit=1000)
    assert len(parts) > 1 and all(len(x) <= 1000 + 20 for x in parts) and parts[0].startswith(f"<i>(1/{len(parts)})</i>")
    body = "\n".join(x.split("\n", 1)[1] for x in parts)
    assert body == text  # every line arrives, in order
    assert split_message("short") == ["short"]


def test_the_report_lists_every_holding_and_errors_say_what_to_do():
    from src.portfolio.graph import format_report
    from src.portfolio.holdings import Holding, analyse

    holdings = [Holding(f"Stock {i}", 10, 100.0, 100.0, 1000.0 + i, None, "IT - SOFTWARE", cap="Large Cap") for i in range(25)]
    state = {"analysis": analyse(holdings), "totals": {}, "pages": {"analyzer": {"reached": True, "data_calls": [1],
             "tabs": {"Dividend": {"status": "read", "rows": 3, "data_calls": 1}}}}, "logged_out": True,
             "blocked_list": ["ForceRemoveChild: x", "ForceRemoveChild: x"], "seconds": 70}
    report = format_report(state)
    assert all(f". Stock {i}" in report for i in range(25)) and "Large Cap · IT - Software" in report
    assert "Portfolio analyser (holdings): read + tabs: Dividend" in report and "(\u00d72)" in report and "took 70s" in report
    for error, todo in [("login stopped at otp: portal said: Invalid OTP", "type the new OTP"),
                        ("no OTP: no OTP within 180s", "within 3 minutes"),
                        ("login stopped at mpin: portal said: Invalid MPIN", "run setup again")]:
        failed = format_report({"error": error, "logged_out": True})
        assert "Why: " + error in failed and todo in failed and "Logged out safely" in failed


# -- Chromium sandbox toggle (for the AWS/Docker deployment) --------------------------------------------------------
def test_no_sandbox_defaults_off_and_is_only_true_on_the_literal_string(monkeypatch):
    from src.portfolio.session import _no_sandbox

    monkeypatch.delenv("PORTFOLIO_NO_SANDBOX", raising=False)
    assert _no_sandbox() is False
    for off in ("false", "False", "0", "no", ""):
        monkeypatch.setenv("PORTFOLIO_NO_SANDBOX", off)
        assert _no_sandbox() is False
    monkeypatch.setenv("PORTFOLIO_NO_SANDBOX", "true")
    assert _no_sandbox() is True
    monkeypatch.setenv("PORTFOLIO_NO_SANDBOX", "TRUE")
    assert _no_sandbox() is True
