"""The portfolio agent as a LangGraph graph.

    fetch -> extract -> analyse -> explain -> report
      |         |                              ^
      +---------+------ (failed / nothing found) +

* fetch    log in (OTP on Telegram), save the allow-listed read pages and tabs, log out  (browser; read-only)
* extract  the portal's statements: holdings with cost and P&L, lots, dividends, sales, transactions  (deterministic)
* analyse  weights, returns, concentration, sectors, market caps, flags                              (deterministic)
* explain  a short plain-English note from a free AI model, given ONLY symbols, sectors and percentages (optional)
* report   save report.json / report.txt with the run and send the report to Telegram, section by section

The AI never sees quantities, amounts, names or account IDs, and cannot act: it only writes the note. Every number in
the report comes from the deterministic nodes."""
import json
import logging
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.portfolio.holdings import (Holding, analyse, anonymised, from_json, from_portfolio_analyzer, from_tables, inr,
                                    plausible, totals)
from src.portfolio.report import sections, split_message, strip_tags
from src.portfolio.session import MARKET_DATA
from src.portfolio.statements import Statements, load, name_key

log = logging.getLogger(__name__)
MOCK_NOTICE = re.compile(r"mock trading session", re.I)  # Saturday exchange test sessions show simulated prices
MOCK_WARNING = "⚠ The portal is showing mock-session prices today (exchange test session): values are not real."
# A free model's note is used only if it reads like a description of this portfolio: long enough, not an error or a
# complaint about the input, and not advice to trade (it is told not to, and this checks it did not).
NOTE_MIN_CHARS = 120
NOTE_REJECT = re.compile(r"invalid|\bjson\b|no data|not provided|missing|cannot determine|unable to|as an ai|"
                         r"please (confirm|provide|clarify)|\?\s*$|"
                         r"\b(you should|i recommend|we recommend|consider (buying|selling)|buy more|sell (it|them|now))\b",
                         re.I)
PAGE_ORDER = ("analyzer", "portfolio", "dashboard", "account")  # where a holdings table is most likely shown
ANALYZER = re.compile(r"PortfolioAnalyzer", re.I)       # the saved data response holding the demat statement

__all__ = ["build_portfolio_graph", "extract", "analyse_node", "format_report", "split_message"]


class PortfolioState(TypedDict, total=False):
    run_dir: str
    pages: dict
    blocked: int
    blocked_list: List[str]
    seconds: float
    logged_out: Optional[bool]
    error: str
    holdings: List[Holding]
    source: str
    statements: Statements
    totals: dict
    warnings: List[str]
    analysis: dict
    explanation: str
    report: str


def extract(state: PortfolioState) -> PortfolioState:
    """The portal's statements, and the holdings from the best source that adds up to the portal's own total:
    1. Portfolio/Present (cost, P&L, history per stock), 2. the Portfolio Analyzer statement (no cost),
    3. any table that looks like holdings. The dashboard's printed totals are the cross-check."""
    run_dir = Path(state["run_dir"])
    dashboard = run_dir / "dashboard.txt"
    text = dashboard.read_text(encoding="utf-8") if dashboard.exists() else ""
    found_totals = totals(text)
    warnings = [MOCK_WARNING] if MOCK_NOTICE.search(text) else []
    st = load(run_dir)
    result: PortfolioState = {"statements": st, "totals": found_totals, "warnings": warnings}
    portal_value = found_totals.get("value")
    if st.holdings and plausible(st.holdings, portal_value):
        return {**result, "holdings": st.holdings, "source": "Portfolio/Present"}
    responses = {f.name: _body(f) for f in sorted((run_dir / "api").glob("*.json"))}
    for name, body in responses.items():
        if ANALYZER.search(name) and (found := from_portfolio_analyzer(body)):
            if plausible(found, portal_value):
                return {**result, "holdings": [_with_class(h, st) for h in found], "source": name}
            warnings.append(f"⚠ The holdings statement adds up to {inr(sum(h.value for h in found))}, not the portal's "
                            f"{inr(found_totals['value'])}; not used.")
    best: List[Holding] = []
    source = ""
    for page in PAGE_ORDER:
        candidates = [(from_json(body), name) for name, body in responses.items()
                      if name.startswith(f"{page}_") and not MARKET_DATA.search(name)]
        tables = run_dir / f"{page}.tables.json"
        if tables.exists():
            candidates.append((from_tables(json.loads(tables.read_text(encoding="utf-8"))), tables.name))
        for found, name in candidates:
            if len(found) > len(best) and plausible(found, portal_value):
                best, source = found, name
    return {**result, "holdings": best, "source": source}


def _with_class(h: Holding, st: Statements) -> Holding:
    cls = st.classes.get(name_key(h.symbol))
    return replace(h, sector=h.sector or cls[0], cap=h.cap or cls[1]) if cls else h


def _body(file: Path):
    try:
        return json.loads(file.read_text(encoding="utf-8")).get("body")
    except (OSError, ValueError, AttributeError):
        return None


def analyse_node(state: PortfolioState) -> PortfolioState:
    return {"analysis": analyse(state["holdings"])}


def usable_note(text: str) -> bool:
    """True if the AI's note reads like a description (see NOTE_REJECT); a bad note is dropped, never shown."""
    text = (text or "").strip()
    return len(text) >= NOTE_MIN_CHARS and not NOTE_REJECT.search(text)


def make_explain(explain_fn: Optional[Callable[[dict], str]]):
    def explain(state: PortfolioState) -> PortfolioState:
        if explain_fn is None:
            return {"explanation": ""}
        try:
            note = explain_fn(anonymised(state["analysis"])).strip()
        except Exception as e:  # the note is optional; the numbers stand without it
            log.warning("portfolio explanation failed: %s", type(e).__name__)
            return {"explanation": ""}
        if not usable_note(note):
            log.warning("portfolio explanation rejected (%d chars)", len(note))
            return {"explanation": ""}
        return {"explanation": note}
    return explain


def format_report(state: PortfolioState) -> str:
    """The whole report as one plain-text string (what report.txt holds and what tests match against). Telegram gets
    the same content, formatted (bold headers, escaped text), section by section — see `sections()`."""
    return strip_tags("\n\n".join(sections(dict(state))))


def make_report(notify: Callable[[str], object]):
    def report(state: PortfolioState) -> PortfolioState:
        parts = sections(dict(state))
        text = strip_tags("\n\n".join(parts))
        if state.get("run_dir") and Path(state["run_dir"]).exists():
            out = Path(state["run_dir"])
            (out / "report.txt").write_text(text, encoding="utf-8")
            st = state.get("statements")
            record = {"totals": state.get("totals"), "warnings": state.get("warnings"), "source": state.get("source"),
                      **(state.get("analysis") or {}), "explanation": state.get("explanation", ""),
                      "statements": asdict(st) if st is not None else None}
            (out / "report.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
        for section in parts:
            for message in split_message(section):
                notify(message)
        return {"report": text}
    return report


def build_portfolio_graph(fetch: Callable[[], dict], notify: Callable[[str], object],
                          explain: Optional[Callable[[dict], str]] = None):
    """Compile the graph. `fetch` is the portal visit (runs.fetch_portfolio bound to credentials and an OTP source)."""
    g = StateGraph(PortfolioState)
    g.add_node("fetch", lambda state: fetch())
    g.add_node("extract", extract)
    g.add_node("analyse", analyse_node)
    g.add_node("explain", make_explain(explain))
    g.add_node("report", make_report(notify))
    g.add_edge(START, "fetch")
    g.add_conditional_edges("fetch", lambda s: "report" if s.get("error") or not s.get("run_dir") else "extract",
                            ["extract", "report"])
    g.add_conditional_edges("extract", lambda s: "analyse" if s.get("holdings") else "report", ["analyse", "report"])
    g.add_edge("analyse", "explain")
    g.add_edge("explain", "report")
    g.add_edge("report", END)
    return g.compile()
