"""The portal's statements, read field by field, and the report built from them. Every record here is made up but uses
the real layouts (JSON inside a string, the portal's field names, dd/mm/yyyy dates)."""
import json
from datetime import date

from src.portfolio.holdings import Holding, analyse
from src.portfolio.report import dividend_view, income_section, sections, strip_tags
from src.portfolio.statements import current_holdings, dividends, load, lots, name_key, realised, trades

CLIENT = {"clientName": "NOT READ", "Address1": "NOT READ", "ClientId": "NOT READ"}


def wrapped(obj) -> str:
    return json.dumps(obj)  # the portal sends the JSON as a string


def present():
    def row(name, qty, avg, price, cost, value, realised_pl, bought, sold):
        return {"companyName": name, "portfolioPosition": str(qty), "averageCostPrice": str(avg),
                "currentMarketPrice": str(price), "valueAtCost": str(cost), "valueAtMarketPrice": str(value),
                "realizedProfitLoss": str(realised_pl), "BuyQty": str(bought), "SellQty": str(sold),
                "lastPriceDate": "25-09-2026", "weightagePercentage": "0"}
    return wrapped({"portfolioAnalysis": {"clientInfo": CLIENT, "holdings": [
        row("Alpha Ltd", 100, 50.0, 150.0, 5000.0, 15000.0, 1200.5, 150, 50),
        row("Beta Ltd", 10, 400.0, 200.0, 4000.0, 2000.0, 0, 10, 0),
        row("Demerged Co Ltd", 20, 0.01, 50.0, 0.2, 1000.0, 0, 20, 0)],   # cost of a few paise: a demerger
        "totals": {"grandTotal": {"valueAtCost": "9000.20", "valueAtMarketPrice": "18000.00",
                                  "realizedProfitLoss": "1200.50", "unrealizedProfitLoss": "8999.80", "ROIper": 99.99}}}})


def test_current_holdings_with_cost_history_and_totals():
    held, totals = current_holdings(present())
    alpha = held[0]
    assert (alpha.symbol, alpha.qty, alpha.avg, alpha.ltp, alpha.invested, alpha.value) == \
        ("Alpha Ltd", 100, 50.0, 150.0, 5000.0, 15000.0)
    assert (alpha.realized, alpha.bought_qty, alpha.sold_qty, alpha.price_date) == (1200.5, 150, 50, "25-09-2026")
    assert round(alpha.pnl_pct, 4) == 2.0 and round(held[1].pnl_pct, 4) == -0.5
    assert held[2].cost_recorded is False and held[2].pnl_pct is None  # no absurd +499,900% for demerged shares
    assert totals == {"cost": 9000.2, "value": 18000.0, "realised": 1200.5, "unrealised": 8999.8, "roi_pct": 99.99}
    assert current_holdings("not json") == ([], {}) and current_holdings({"other": 1}) == ([], {})


def test_lots_dividends_sales_and_trades():
    turn = wrapped({"HoldingsTurningLongTerm": {"clientInfo": CLIENT, "transactions": [
        {"CompanyName": "Beta Ltd", "Details": [
            {"DateOfTransaction": "17/03/2026", "Quantity": "10", "Rate": "433.85", "RemainingDays": "172"}]}]}})
    assert lots(turn)[0] == lots(turn)[0].__class__("Beta Ltd", "17/03/2026", 10.0, 433.85, 172)

    div = wrapped({"ClientDetails": CLIENT, "DividendISINDetails": [
        {"ISIN": {"No": "INE000A01011", "Description": "Alpha Ltd"}, "RecordDate": "03/06/2026",
         "EligibleQuantity": "400", "DividendPerShare": "2.50", "GrossDividendAmount": "1,000",
         "TDS@10Percentage": "0", "NetDividentAmount": "1,000.00"},
        {"ISIN": {"Description": "Alpha Ltd"}, "RecordDate": "03/06/2024", "EligibleQuantity": "400",
         "DividendPerShare": "1", "GrossDividendAmount": "400", "TDS@10Percentage": "40", "NetDividentAmount": "360"},
        {"ISIN": {"Description": "Nothing"}, "RecordDate": "", "GrossDividendAmount": "0"}]})
    got = dividends(div)
    assert [(d.company, d.net, d.tds) for d in got] == [("Alpha Ltd", 1000.0, 0.0), ("Alpha Ltd", 360.0, 40.0)]

    sold = wrapped({"portfolioGainLossStatement": {"financialYear": "FY 2026", "clientInfo": CLIENT, "transactions": [
        {"company": "Gamma Ltd", "SoldQuantity": 75, "BoughtValue": "26,179", "SoldValue": "25,148", "GainOrLoss": "-1,031"}]}})
    fy, sales = realised(sold)
    assert fy == "FY 2026" and sales[0].gain == -1031.0

    txn = wrapped({"FundPortfolio": {"clientInfo": CLIENT, "fundHoldings": [
        {"fundName": "Gamma Ltd", "transactions": [{"transactionDate": "29/08/2026", "rate": "335.30", "creditQty": 0.0,
                                                    "debitQty": 75.0, "description": "To  Merger"}]},
        {"fundName": "Delta Ltd", "transactions": [{"transactionDate": "03/09/2026", "rate": "343.45", "creditQty": 64.0,
                                                    "debitQty": 0.0, "description": "By Scheme Of Arrangement"}]},
        {"fundName": "Beta Ltd", "transactions": [{"transactionDate": "17/08/2026", "rate": "89.85", "creditQty": 50.0,
                                                   "debitQty": 0.0, "description": "By thru NSE,T1-NORMAL"}]}]}})
    t = trades(txn)
    assert [(x.company, x.side, x.qty, x.corporate_action) for x in t] == [
        ("Gamma Ltd", "Sell", 75.0, True), ("Delta Ltd", "Buy", 64.0, True), ("Beta Ltd", "Buy", 50.0, False)]
    assert name_key("KOTAK MAHINDRA BANK LIMITED") == name_key("Kotak Mahindra Bank Ltd.")


def test_load_joins_the_statements_and_the_report_shows_every_detail(tmp_path):
    api = tmp_path / "api"
    api.mkdir()

    def save(name, body):
        (api / name).write_text(json.dumps({"body": body}), encoding="utf-8")

    save("portfolio__current_holdings_00_1_Portfolio_Present.json", present())
    save("analyzer__analyser_01_1_Statements_EquityPortfolioAnalyzerDtl.json", wrapped([
        {"isindesc": "ALPHA LIMITED", "sectorname": "IT - SOFTWARE", "Market_cap": "Large Cap", "valuation": "15000"},
        {"isindesc": "Beta Ltd", "sectorname": "BANKS", "Market_cap": "Small Cap", "valuation": "2000"}]))
    save("analyzer__dividend_00_1_Statements_Dividend.json", wrapped({"DividendISINDetails": [
        {"ISIN": {"Description": "Alpha Ltd"}, "RecordDate": "03/06/2026", "EligibleQuantity": "100",
         "DividendPerShare": "2.50", "GrossDividendAmount": "250", "TDS@10Percentage": "0", "NetDividentAmount": "250"}]}))
    save("portfolio__realised_gain_loss_00_1_Portfolio_RealizedGainLoss.json", wrapped({"portfolioGainLossStatement": {
        "financialYear": "FY 2026", "transactions": [{"company": "Gamma Ltd", "SoldQuantity": 75, "BoughtValue": "100",
                                                      "SoldValue": "90", "GainOrLoss": "-10"}]}}))
    save("portfolio__transaction_history_00_1_Portfolio_StatementOfTransactionV1.json", wrapped({"FundPortfolio": {
        "fundHoldings": [{"fundName": "Gamma Ltd", "transactions": [
            {"transactionDate": "29/08/2026", "rate": "1", "creditQty": 0, "debitQty": 75, "description": "To Merger"}]}]}}))
    st = load(tmp_path)
    assert [(h.symbol, h.sector, h.cap) for h in st.holdings[:2]] == [("Alpha Ltd", "IT - SOFTWARE", "Large Cap"),
                                                                     ("Beta Ltd", "BANKS", "Small Cap")]
    a = analyse(st.holdings)
    state = {"analysis": a, "statements": st, "totals": {"margin": 497.0}, "logged_out": True,
             "explanation": "Two gainers."}
    parts = sections(state, today=date(2026, 9, 26))
    summary, holdings, analysis, income, note, run = parts
    assert_balanced_tags(parts)
    assert "Value: <b>₹18,000</b>" in summary and "+₹8,999" not in summary  # 8,999.80 rounds to 9,000
    assert "P&amp;L: <b>+₹9,000 (+100.0%)</b>" in summary and "Booked P&amp;L (past sales): +₹1,200" in summary
    assert "Dividends (last 12 months): ₹250 net · 1 payouts" in summary and "Cash / margin: ₹497" in summary
    assert "Prices as of 25-09-2026" in summary and "Best: <b>Alpha Ltd</b> +200%" in summary
    assert "<b>1. Alpha Ltd</b>" in holdings and "Large Cap · IT - Software" in holdings
    assert "100 sh · avg ₹50.00 · now ₹150.00" in holdings and "₹5,000 → ₹15,000 · 83.3% of portfolio" in holdings
    assert "bought 150, sold 50 · booked +₹1,200 · dividends ₹250 (12 m)" in holdings
    assert "cost not recorded (e.g. received through a demerger or bonus)" in holdings
    assert "In profit: 1 · In loss: 1 · Cost not recorded: 1" in analysis and "Beta Ltd is 50% below" in analysis
    assert "(merger/scheme conversion, not a market sale)" in income and "🔄 Gamma Ltd: 75 sh transferred out" in income
    assert note.endswith("Two gainers.") and run.startswith("<b>⚙️ RUN DETAILS</b>")
    assert "👉 <b>Next:</b> nothing needed" in run  # every successful run ends by saying what to do next
    plain = "\n".join(parts)
    assert "NOT READ" not in plain and "<b>" not in strip_tags(plain) and "&amp;" not in strip_tags(plain)


def assert_balanced_tags(parts) -> None:
    """Every opening tag in a Telegram message has a matching close, and (b, i, code) are the only tags used — a
    malformed tag would make Telegram refuse the whole message."""
    import re

    allowed = {"b", "i", "code"}
    for part in parts:
        stack = []
        for closing, name in re.findall(r"<(/?)([a-zA-Z]+)[^>]*>", part):
            assert name in allowed, f"unexpected tag <{name}> in {part!r}"
            if closing:
                assert stack and stack.pop() == name, f"unbalanced </{name}> in {part!r}"
            else:
                stack.append(name)
        assert not stack, f"unclosed tag(s) {stack} in {part!r}"


def test_only_dividends_from_the_last_12_months_are_counted():
    from src.portfolio.statements import Dividend, Statements

    st = Statements(dividends=[Dividend("A", "01/10/2025", 1, 1, 100, 10, 90),     # inside the window
                               Dividend("A", "20/09/2025", 1, 1, 50, 0, 50),      # older than 12 months
                               Dividend("B", "", 1, 1, 70, 0, 70)])               # no record date: not counted
    view = dividend_view(st, date(2026, 9, 26))
    assert (view["count"], view["net"], view["tds"]) == (1, 90, 10) and view["by_stock"] == {"a": 90}
    assert "DIVIDENDS</b>, last 12 months: ₹90 net · 1 payouts (TDS ₹10)" in income_section(st, view)


GOOD_NOTE = ("The portfolio holds 32 stocks across 19 sectors, with large caps close to half of it. Oil India and "
             "Power Finance drive most of the gains, while the IT names are the weakest.")


def test_a_bad_ai_note_is_retried_once_and_then_dropped_never_shown():
    from src.app.portfolio_agent import make_explainer
    from src.portfolio.graph import make_explain, usable_note

    assert usable_note(GOOD_NOTE)
    for bad in ("Invalid input: expected JSON with portfolio data.", GOOD_NOTE + " You should sell TCS.",
                GOOD_NOTE + " Return data is missing for one stock.", "Too short.", GOOD_NOTE + " Shall I go on?",
                "The JSON input appears to have been provided twice. Please confirm which version to use, or provide one."):
        assert not usable_note(bad)

    class FakeLLM:
        def __init__(self, answers):
            self.answers, self.prompts = list(answers), []

        def structured_call(self, prompt, model, schema, max_tokens=None):
            self.prompts.append(prompt)
            return model(note=self.answers.pop(0))

    llm = FakeLLM(["Invalid input: expected JSON.", GOOD_NOTE])
    assert make_explainer(llm)({"stocks": 1}) == GOOD_NOTE
    assert len(llm.prompts) == 2 and llm.prompts[0] != llm.prompts[1]  # reworded, so the cache cannot answer it

    analysis = analyse([])
    state = {"analysis": analysis}
    assert make_explain(lambda d: "Invalid input.")(state) == {"explanation": ""}
    assert make_explain(lambda d: GOOD_NOTE)(state) == {"explanation": GOOD_NOTE}


def test_a_company_name_with_an_ampersand_never_breaks_the_telegram_formatting():
    """Real example: 'Balmer Lawrie & Company Ltd'. An unescaped '&' in HTML mode risks Telegram rejecting the whole
    message; strip_tags() must also decode it back to plain '&' for the file saved to disk."""
    from src.portfolio.statements import Dividend, Statements

    name = "Balmer Lawrie & Company Ltd"
    h = [Holding(name, 10, 100.0, 150.0, 1500.0, 1000.0, "Trading & Logistics", cap="Small Cap")]
    st = Statements(holdings=h, dividends=[Dividend(name, "01/09/2026", 10, 5.0, 50.0, 0.0, 50.0)])
    state = {"analysis": analyse(h), "statements": st, "totals": {}, "logged_out": True}
    parts = sections(state, today=date(2026, 9, 26))
    assert_balanced_tags(parts)
    joined = "\n".join(parts)
    assert "Balmer Lawrie &amp; Company Ltd" in joined and name not in joined  # raw '&' never sent unescaped
    plain = strip_tags(joined)
    assert name in plain and "&amp;" not in plain  # the saved file reads with a plain '&', not the HTML entity
