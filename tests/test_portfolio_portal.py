"""The login/read/logout flow in a real Chromium against a local fake of the Integrated portal that reproduces its page
flow, element IDs, error box and redirects (mapped from the portal's public login scripts). Includes a trap: the fake
dashboard tries to place an order, which the guard must stop before it reaches the server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from src.portfolio.session import LoginFailed, PortalRun, read_pages  # noqa: E402
from src.portfolio.vault import Credentials  # noqa: E402

MOBILE, OTP, MPIN = "9876543210", "4321", "135790"
CREDS = Credentials(MOBILE, MPIN, "")
NOTE = ("Two holdings: NBCC is the larger one and shows a gain on its cost, while IRFC is smaller and sits "
        "below its cost, so the result depends mostly on NBCC.")

ALERT_JS = """<script>function alertBox(m){document.body.insertAdjacentHTML('beforeend',
 '<div class="mask-overlay"><div class="alert-box"><p>'+m+'</p></div></div>')}
async function call(path, headers, body){const r = await fetch(path,{method:'POST',headers:headers||{},body:JSON.stringify(body||{})});
 return r.json()}</script>"""
LOGOUT_JS = """<script>function Logoutfunc(){call('/CIAMS/api/v1/SSAuth/logout',{'App-Action-Name':'Logout'})
 .then(()=>{sessionStorage.clear();location.href='../login/login.html'})}</script>"""
READ_JS = """<script>call('/StandardService/api/v1/Common/Claim',{'App-Action-Name':'Holdings','App-Process-Name':'Portfolio'})
 .then(j=>{document.getElementById('h').innerHTML='<tr><th>Scrip</th><th>Net Qty</th><th>Avg. Price</th><th>LTP</th></tr>'+
  j.rows.map(r=>'<tr><td>'+[r.Symbol,r.Qty,r.AvgPrice,r.LTP].join('</td><td>')+'</td></tr>').join('')});
 call('/StandardService/api/v1/Common/Claim',{'App-Action-Name':'PlaceOrder'},{symbol:'NBCC',qty:1000}).catch(()=>{});
 fetch('/SMDService/api/v1/SMD/MarketData',{method:'POST',headers:{'App-Action-Name':'LiveFeed'}}).catch(()=>{});</script>"""


def pages(s: dict) -> dict:
    customers = json.dumps(s["customers"])
    return {
        "/nv1/login/login.html": f"""{ALERT_JS}<input type=radio id=MobileNoRadioBtn name=RadioCheck>
<input type=radio id=CustRadioBtn name=RadioCheck checked><input id=MobileNoTxt maxlength=10>
<input type=button id=Submitbtn value=Submit disabled>
<script>MobileNoTxt.oninput=()=>Submitbtn.disabled=MobileNoTxt.value.length<10;
Submitbtn.onclick=async()=>{{if(!MobileNoRadioBtn.checked){{alertBox('customer-id login is not tested');return}}
 const j=await call('/CIAMS/api/v1/TwoFA/init',{{}},{{m:MobileNoTxt.value}});
 if(j.ok){{sessionStorage.setItem('LoginWith','L1');location.href='OtpVerify.html?V1.0'}}else alertBox(j.description)}}</script>""",
        "/nv1/login/OtpVerify.html": f"""{ALERT_JS}<input id=OTPTxt maxlength=4><input type=button id=Submitbtn value=Verify disabled>
<script>OTPTxt.addEventListener('keyup',()=>{{Submitbtn.disabled={"true" if s["otp_button_stuck"] else "OTPTxt.value.length<4"}}});
Submitbtn.onclick=async()=>{{const j=await call('/CIAMS/api/v1/TwoFA/coidverify',{{}},{{otp:OTPTxt.value}});
 if(j.ok){{ {"" if s["otp_hangs"] else "location.href='CustomerIDSelection.html?V=1.0'"} }}else alertBox(j.description)}}</script>""",
        "/nv1/login/CustomerIDSelection.html": f"""{ALERT_JS}<div id=d></div><input type=button id=Submitbtn value=Submit>
<script>const ids={customers};
function go(r){{sessionStorage.setItem('CustomerId',r.value);
 if(r.dataset.mpinsts!='Y'){{alert('Your MPin has Expired. Please Reset your Mpin.');location.href='ResetMpin.html';return}}
 location.href='MpinPassword.html'}}
d.innerHTML=ids.map(c=>'<input type=radio name=CustomerId data-mpinsts="'+c[1]+'" value="'+c[0]+'">').join('');
Submitbtn.onclick=()=>go(document.querySelector("input[name='CustomerId']:checked"));
if(ids.length==1){{d.style.display='none';document.querySelector("input[name='CustomerId']").checked=true;Submitbtn.click()}}</script>""",
        "/nv1/login/MpinPassword.html": f"""{ALERT_JS}{''.join(f'<input id=MpinTxt{i} class=InputBox maxlength=1>' for i in range(1, 7))}
<input type=button id=Submitbtn value=Login disabled>
<script>document.querySelectorAll('.InputBox').forEach(b=>b.addEventListener('keyup',()=>{{
 Submitbtn.disabled=[...document.querySelectorAll('.InputBox')].some(x=>!x.value)}}));
Submitbtn.onclick=async()=>{{let v='';for(let i=1;i<=6;i++)v+=document.getElementById('MpinTxt'+i).value;
 const j=await call('/CIAMS/api/v1/SSAuth/login',{{}},{{mpin:v,customer:sessionStorage.getItem('CustomerId')}});
 if(j.ok)location.href='../Dashboard/Dashboardv1.html';else alertBox(j.description)}}</script>""",
        "/nv1/login/ResetMpin.html": "reset your MPIN",
        "/nv1/dashboard/dashboardv1.html": f"{ALERT_JS}{LOGOUT_JS}<h1>Dashboard</h1><table id=h></table>{READ_JS}",
        "/nv1/portfoliotracker/livesummarydetailsv1.html": f"{ALERT_JS}{LOGOUT_JS}<h1>Holdings</h1><table id=h></table>{READ_JS}",
        "/nv1/demat/portfolioanalyzer.html": f"""{ALERT_JS}{LOGOUT_JS}<p>Analyzer: sector split</p>
<span id=t1>Dividend</span> <span id=t2>Sell All</span> <div id=out></div>
<script>t1.onclick=()=>call('/StandardService/api/v1/Common/Claim',{{'App-Action-Name':'Dividend'}})
 .then(j=>{{out.innerHTML='<table><tr><th>Scrip</th><th>Amount</th></tr><tr><td>NBCC</td><td>'+j.amount+'</td></tr></table>'}});
t2.onclick=()=>call('/StandardService/api/v1/Common/Claim',{{'App-Action-Name':'SellAll'}});</script>""",
        "/nv1/dashboard/myaccount.html": f"{ALERT_JS}<p>My account (this page has no logout function)</p>",
    }


class FakePortal:
    def __init__(self, **scenario):
        self.scenario = {"customers": [["700001", "Y"]], "otp_hangs": False, "otp_button_stuck": False, **scenario}
        self.requests = []  # (method, path, action header, body)
        self.closing = threading.Event()
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body, kind="text/html"):
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                page = {k.lower(): v for k, v in pages(portal.scenario).items()}.get(self.path.split("?")[0].lower())
                self._reply(200, "<html><body>" + page + "</body></html>") if page else self._reply(404, "not found")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                path = self.path.split("?")[0]
                portal.requests.append(("POST", path, self.headers.get("App-Action-Name", ""), body))
                if self.headers.get("App-Action-Name") == "LiveFeed":  # a price stream that never ends
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ticks":[')
                    self.wfile.flush()
                    portal.closing.wait(120)
                    return
                self._reply(200, json.dumps(portal.api(path, self.headers.get("App-Action-Name", ""), body)),
                            "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}/nv1/"

    def api(self, path, action, body):
        wrong = {"ok": False, "description": "Invalid details"}
        if path.endswith("TwoFA/init"):
            return {"ok": True} if body.get("m") == MOBILE else {"ok": False, "description": "Mobile number not registered"}
        if path.endswith("TwoFA/coidverify"):
            return {"ok": True} if body.get("otp") == OTP else {"ok": False, "description": "Invalid OTP"}
        if path.endswith("SSAuth/login"):
            return {"ok": True} if body.get("mpin") == MPIN else {"ok": False, "description": "Invalid MPIN"}
        if action == "Dividend":
            return {"ok": True, "amount": "250.00"}
        if action == "Holdings":
            return {"ok": True, "rows": [{"Symbol": "NBCC", "Qty": 1000, "AvgPrice": 95.0, "LTP": 112.4},
                                         {"Symbol": "IRFC", "Qty": 500, "AvgPrice": 150.0, "LTP": 138.1}]}
        return {"ok": True} if path.endswith("logout") else wrong

    def calls(self, suffix):
        return [r for r in self.requests if r[1].endswith(suffix)]

    def close(self):
        self.closing.set()
        self.server.shutdown()


@pytest.fixture
def portal():
    servers = []

    def make(**scenario):
        servers.append(FakePortal(**scenario))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def run(fake, tmp_path, creds=CREDS, otp=OTP, timeout=10):
    asked = []

    def get_otp():
        asked.append(1)
        return otp

    r = PortalRun(creds, get_otp, tmp_path / "profile", tmp_path / "evidence", base=fake.base, step_timeout_s=timeout)
    return r, asked


def all_saved_text(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in folder.rglob("*") if p.suffix in (".txt", ".json"))


def test_logs_in_reads_every_page_blocks_the_order_and_logs_out(portal, tmp_path, caplog):
    fake = portal()
    r, asked = run(fake, tmp_path)
    with r as page:
        summary = read_pages(page, r.base, tmp_path / "out", settle_ms=3000)
    assert asked == [1] and r.logged_out is True
    assert all(e["reached"] for e in summary.values())
    assert "NBCC" in (tmp_path / "out" / "portfolio.txt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out" / "portfolio.tables.json").read_text(encoding="utf-8"))[0][1] == ["NBCC", "1000", "95", "112.4"]
    holdings = [c for e in summary.values() for c in e["data_calls"] if c["action"].startswith("Holdings")]
    assert holdings and all((tmp_path / "out" / "api" / c["file"]).exists() for c in holdings)
    # the trap: every load of the dashboard/holdings pages tried to place an order; the server never saw one
    assert not [q for q in fake.requests if q[2] == "PlaceOrder"]
    assert len(r.blocked) >= 2 and {b.action for b in r.blocked} == {"PlaceOrder"}
    assert len(fake.calls("SSAuth/logout")) == 1
    saved = all_saved_text(tmp_path / "out") + caplog.text
    assert MOBILE not in saved and MPIN not in saved and OTP not in saved
    # a second run reuses the same browser profile (the portal's device ID survives), and cookies were cleared
    r2, _ = run(fake, tmp_path)
    with r2 as page:
        assert not page.context.cookies()


def test_an_unregistered_mobile_stops_before_any_otp_and_leaves_no_session(portal, tmp_path):
    fake = portal()
    r, asked = run(fake, tmp_path, creds=Credentials("9000000000", MPIN, ""))
    with pytest.raises(LoginFailed, match="mobile: portal said: Mobile number not registered"):
        with r:
            pass
    assert asked == [] and r.logged_out is None and not fake.calls("logout")
    assert (tmp_path / "evidence" / "failure.png").exists()


def test_a_wrong_otp_is_tried_once_and_the_half_session_is_logged_out(portal, tmp_path):
    fake = portal()
    r, _ = run(fake, tmp_path, otp="1111")
    with pytest.raises(LoginFailed, match="otp: portal said: Invalid OTP"):
        with r:
            pass
    assert len(fake.calls("coidverify")) == 1 and r.logged_out is None and not fake.calls("logout")


def test_a_wrong_mpin_is_tried_once(portal, tmp_path):
    fake = portal()
    r, _ = run(fake, tmp_path, creds=Credentials(MOBILE, "000000", ""))
    with pytest.raises(LoginFailed) as info:
        with r:
            pass
    assert info.value.stage == "mpin" and len(fake.calls("SSAuth/login")) == 1


def test_several_customer_ids_need_the_saved_one(portal, tmp_path):
    fake = portal(customers=[["700001", "Y"], ["700002", "Y"]])
    r, _ = run(fake, tmp_path)
    with pytest.raises(LoginFailed, match="2 customer IDs"):
        with r:
            pass
    assert not fake.calls("SSAuth/login")

    r, _ = run(fake, tmp_path, creds=Credentials(MOBILE, MPIN, "700002"))
    with r:
        pass
    assert fake.calls("SSAuth/login")[-1][3]["customer"] == "700002" and r.logged_out is True

    r, _ = run(fake, tmp_path, creds=Credentials(MOBILE, MPIN, "799999"))
    with pytest.raises(LoginFailed, match="not one of those linked"):
        with r:
            pass


def test_an_expired_mpin_popup_stops_the_run(portal, tmp_path):
    fake = portal(customers=[["700001", "N"]])
    r, _ = run(fake, tmp_path)
    with pytest.raises(LoginFailed, match="MPin has Expired"):
        with r:
            pass
    assert not fake.calls("SSAuth/login")


def test_buttons_enabled_only_by_key_presses_work_and_a_stuck_one_stops_cleanly(portal, tmp_path):
    """The real portal enables Submit from key events (a bulk fill left it disabled on 2026-09-26). Typing key by key
    works (all other tests); a button that never enables stops the run without sending anything."""
    fake = portal(otp_button_stuck=True)
    r, _ = run(fake, tmp_path)
    with pytest.raises(LoginFailed) as info:
        with r:
            pass
    # nothing was submitted, so no session exists and no logout is attempted (or falsely reported as failed)
    assert info.value.stage == "otp entry" and not fake.calls("coidverify") and r.logged_out is None


def test_a_portal_that_stops_responding_times_out_instead_of_hanging(portal, tmp_path):
    fake = portal(otp_hangs=True)
    r, _ = run(fake, tmp_path, timeout=3)
    with pytest.raises(LoginFailed, match="timed out"):
        with r:
            pass
    assert r.logged_out is None  # the login never completed: nothing to log out of; the browser is closed



def test_the_langgraph_agent_end_to_end_with_its_mpin_lockout(portal, tmp_path):
    """fetch (fake portal) -> extract -> analyse -> explain -> report, then a wrong MPIN blocks the next run."""
    from src.portfolio.graph import build_portfolio_graph
    from src.portfolio.runs import RunStore, fetch_portfolio

    fake, store, sent, seen_by_ai = portal(), RunStore(tmp_path / "data"), [], []

    def agent(creds):
        return build_portfolio_graph(lambda: fetch_portfolio(creds, lambda: OTP, store, base=fake.base, step_timeout_s=10,
                                                             progress=sent.append),
                                     sent.append, explain=lambda data: seen_by_ai.append(data) or NOTE)

    state = agent(CREDS).invoke({})
    a = state["analysis"]
    assert [p["symbol"] for p in a["positions"]] == ["NBCC", "IRFC"] and a["total_value"] == 1000 * 112.4 + 500 * 138.1
    assert state["logged_out"] is True and state["blocked"] >= 1 and any(NOTE in s for s in sent)
    # step-by-step updates on Telegram: login, each page (with what the guard blocked), logout, then the report
    steps = "\n".join(sent)
    assert "Logged in (read-only)" in steps and "1/4 Dashboard (totals)</b>: read" in steps and "2/4 Capital gains" in steps
    assert "blocked 1 write request(s) the page tried: PlaceOrder" in steps and "Logged out of Integrated" in steps
    order = ["Opening the Integrated login page", "Mobile number accepted", "OTP accepted", "Logged in (read-only)",
             "1/4 Dashboard", "Logged out of Integrated", "PORTFOLIO SUMMARY"]
    positions = [next(i for i, s in enumerate(sent) if step in s) for step in order]
    assert positions == sorted(positions)  # every step reaches Telegram, in the order it happened
    assert [i for i, s in enumerate(sent) if "Logged in" in s][0] < [i for i, s in enumerate(sent) if "Logged out" in s][0]
    assert (Path(state["run_dir"]) / "report.json").exists() and not store.lock_file.exists()
    ai_text = json.dumps(seen_by_ai)
    assert "NBCC" in ai_text and "1000" not in ai_text and "112.4" not in ai_text  # symbols and %, never qty/prices

    logins = len(fake.calls("SSAuth/login"))
    state = agent(Credentials(MOBILE, "000000", "")).invoke({})
    assert "login stopped at mpin" in state["error"] and store.mpin_blocked()
    state = agent(CREDS).invoke({})
    assert "another wrong MPIN could lock" in state["error"] and len(fake.calls("SSAuth/login")) == logins + 1
    assert all(MOBILE not in s and MPIN not in s for s in sent)


def test_read_only_tabs_are_clicked_by_exact_label_and_action_labels_are_refused(portal, tmp_path):
    fake = portal()
    r, _ = run(fake, tmp_path)
    tabs = {"analyzer": ("Dividend", "Sell All", "Missing Tab")}
    with r as page:
        summary = read_pages(page, r.base, tmp_path / "out", settle_ms=2000, tabs=tabs)
    got = summary["analyzer"]["tabs"]
    assert got["Dividend"]["status"] == "read" and got["Missing Tab"]["status"] == "not found"
    assert got["Sell All"]["status"].startswith("refused")  # never clicked: its request never happened at all
    assert not [q for q in fake.requests if q[2] == "SellAll"] and fake.calls("SSAuth/logout")
    saved = json.loads((tmp_path / "out" / "analyzer__dividend.tables.json").read_text(encoding="utf-8"))
    assert saved == [[["Scrip", "Amount"], ["NBCC", "250.00"]]]


def test_a_navigation_to_the_expected_page_is_not_mistaken_for_an_unexpected_page():
    """2026-09-26 real run: CustomerIDSelection -> MpinPassword happened while the page checks were throwing (the page
    was navigating), and the login stopped with 'unexpected page /nv1/login/MpinPassword.html'."""
    from playwright.sync_api import Error as PlaywrightError

    class Navigating:
        def __init__(self):
            self.urls = iter(["https://x/nv1/login/CustomerIDSelection.html", "https://x/nv1/login/MpinPassword.html"])
            self.url = next(self.urls)

        def locator(self, _selector):
            self.url = next(self.urls, self.url)  # the navigation lands while the page is being inspected...
            raise PlaywrightError("Execution context was destroyed")  # ...which makes the inspection throw

        def wait_for_timeout(self, _ms):
            pass

    run = PortalRun(CREDS, lambda: OTP, Path("."), Path("."), step_timeout_s=2)
    run.page = Navigating()
    assert run._expect(("mpin", "dashboard"), "customer") == "mpin"


def test_the_no_sandbox_flag_still_logs_in_and_out_cleanly(portal, tmp_path, monkeypatch):
    """PORTFOLIO_NO_SANDBOX=true (the AWS/Docker deployment) must not change the login flow itself."""
    monkeypatch.setenv("PORTFOLIO_NO_SANDBOX", "true")
    fake = portal()
    r, _ = run(fake, tmp_path)
    with r:
        pass
    assert r.logged_out is True
