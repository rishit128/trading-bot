"""The trading workflow as three LangGraph graphs, built from whatever agents are registered on the pipeline.

    cycle graph     open_cycle -> select_universe --(map: one Send per stock)--> analyze_symbol x N --> execute_cycle
    analysis graph  fetch_data -> agent_<name> (ALL agents in parallel) -> collect          (one stock)
    decision graph  decide -> [risk approved?] -> execute                                     (one stock)

* Adding an agent means writing a class with `name`/`role`/`analyze(ctx)` and registering it: the analysis graph grows
  a parallel node automatically and `combine_signals` folds it in by role. No graph code changes.
* Stocks are analysed concurrently (the slow, LLM-bound part) but decided and executed one at a time in a fixed order,
  so risk limits, position counts and cash are always evaluated against fresh state and results are deterministic.
* Risk limits and order sizing stay in plain deterministic code (`RiskEngine`); the graphs only orchestrate."""
import logging
import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.base import LEAD, AgentContext
from src.data.indicators import Snapshot
from src.engine.risk_engine import Portfolio, RiskDecision
from src.engine.strategy import Decision, combine_signals
from src.llm import Signal
from src.results import SymbolResult

log = logging.getLogger(__name__)


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer letting parallel agent nodes each write their own key into one shared dict."""
    return {**(left or {}), **(right or {})}


class AnalysisState(TypedDict, total=False):
    """State of the per-stock analysis graph."""
    symbol: str
    snapshot: Snapshot
    signals: Annotated[Dict[str, Optional[Signal]], merge_dicts]


class DecisionState(TypedDict, total=False):
    """State of the per-stock decision graph."""
    symbol: str
    portfolio: Portfolio
    snapshot: Snapshot
    signals: Dict[str, Optional[Signal]]
    decision: Decision
    risk: RiskDecision
    price: float
    decision_id: int
    order_status: Optional[str]


class SymbolInput(TypedDict):
    """Input for one parallel analysis task."""
    symbol: str


class CycleState(TypedDict, total=False):
    """State of the whole-cycle graph."""
    portfolio: Portfolio
    symbols: List[str]
    analyses: Annotated[List[dict], operator.add]
    results: List[SymbolResult]


def agent_node_name(name: str) -> str:
    """Graph node name for an agent."""
    return f"agent_{name}"


def build_analysis_graph(p):
    """p is the TradingPipeline. Agents are looked up by name at call time, so an implementation can be swapped."""
    if not p.agents:
        raise ValueError("at least one agent is required")

    def fetch_data(state: AnalysisState):
        """Fetch the stock's indicator snapshot."""
        return {"snapshot": p.snapshot_fn(state["symbol"])}

    def make_agent_node(name: str):
        """Build the graph node that runs one agent."""
        def node(state: AnalysisState):
            """Run one agent for the stock."""
            symbol = state["symbol"]
            ctx = AgentContext(symbol, state["snapshot"], headlines=lambda: p.headlines_fn(symbol))
            return {"signals": {name: p.agents[name].analyze(ctx)}}

        return node

    def collect(state: AnalysisState):
        """Join point after all agents finish."""
        return {}

    graph = StateGraph(AnalysisState)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("collect", collect)
    graph.add_edge(START, "fetch_data")
    for name in p.agents:
        node = agent_node_name(name)
        graph.add_node(node, make_agent_node(name))
        graph.add_edge("fetch_data", node)
    graph.add_edge([agent_node_name(n) for n in p.agents], "collect")
    graph.add_edge("collect", END)
    return graph.compile()


def build_decision_graph(p):
    """Per-stock graph: combine signals, apply the risk engine, log, and place the order if approved."""
    def decide(state: DecisionState):
        """Combine signals (or apply the trend exit), size with the risk engine, and record the decision."""
        symbol, snap, signals = state["symbol"], state["snapshot"], state["signals"]
        held = state["portfolio"].position_qty.get(symbol, 0)
        if p.settings.trend_exit and held > 0 and snap.price < snap.ma200:
            # Deterministic exit (like the risk engine, not an AI opinion): a broken long-term trend closes the position.
            decision = Decision("SELL", 1.0, f"trend exit: last close {snap.price:,.2f} is below the 200-day average {snap.ma200:,.2f}")
        else:
            decision = combine_signals(signals, {n: a.role for n, a in p.agents.items()})
        price = p._quote(symbol, snap.price) if decision.action != "HOLD" else snap.price
        risk = p.risk.evaluate(decision.action, decision.confidence, symbol, price, state["portfolio"])
        decision_id = p._log_decision(symbol, snap, signals, decision, risk, price)
        return {"decision": decision, "risk": risk, "price": price, "decision_id": decision_id}

    def execute(state: DecisionState):
        """Place the approved order."""
        d = state["decision"]
        return {"order_status": p._place(state["decision_id"], state["symbol"], d.action, state["risk"].quantity, state["price"])}

    def route(state: DecisionState) -> str:
        """Send approved trades to execute and everything else to the end."""
        return "execute" if state["risk"].approved else "end"

    graph = StateGraph(DecisionState)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, {"execute": "execute", "end": END})
    graph.add_edge("execute", END)
    return graph.compile()


def build_cycle_graph(p, analysis_graph, decision_graph):
    """Whole-cycle graph: open the account, choose stocks, analyse them in parallel, then decide each in order."""
    def open_cycle(state: CycleState):
        """Read the account, record equity, and check halts and stop protection."""
        portfolio = p.broker.portfolio()
        p._record_equity(portfolio.equity)
        portfolio = p._with_peak(portfolio)
        p._check_halt(portfolio)
        p._reconcile_protection()
        p._cycle_llm = []
        return {"portfolio": portfolio, "analyses": [], "results": []}

    def select_universe(state: CycleState):
        """Choose which stocks to analyse this cycle."""
        return {"symbols": list(p._symbols(state["portfolio"]))}

    def scatter(state: CycleState):
        """Fan out one parallel analysis task per stock."""
        symbols = state["symbols"]
        return [Send("analyze_symbol", {"symbol": s}) for s in symbols] if symbols else "execute_cycle"

    def analyze_symbol(payload: SymbolInput):
        """Analyse one stock with all agents; a failure is captured, not raised."""
        symbol = payload["symbol"]
        try:
            # The cycle's max_concurrency (stocks at once) is inherited by nested graphs; override it so every agent
            # for this stock still runs in parallel regardless of how many stocks are in flight.
            out = analysis_graph.invoke({"symbol": symbol}, config={"max_concurrency": len(p.agents)})
            return {"analyses": [{"symbol": symbol, "snapshot": out["snapshot"], "signals": out["signals"], "error": None}]}
        except Exception as e:  # one stock failing must not abort the others
            log.exception("%s analysis failed", symbol)
            return {"analyses": [{"symbol": symbol, "error": f"{type(e).__name__}: {e}"}]}

    def execute_cycle(state: CycleState):
        """Decide and execute each analysed stock in universe order against fresh account state."""
        order = {s: i for i, s in enumerate(state["symbols"])}
        roles = {n: a.role for n, a in p.agents.items()}
        portfolio, results = state["portfolio"], []
        for a in sorted(state["analyses"], key=lambda a: order[a["symbol"]]):
            symbol = a["symbol"]
            if a["error"]:
                results.append(SymbolResult(symbol, "ERROR", 0.0, None, None, a["error"]))
                p.notify(f"ERROR processing {symbol}: {a['error']}")
                continue
            try:
                out = decision_graph.invoke({"symbol": symbol, "portfolio": portfolio, "snapshot": a["snapshot"],
                                             "signals": a["signals"]})
            except Exception as e:
                log.exception("%s decision failed", symbol)
                results.append(SymbolResult(symbol, "ERROR", 0.0, None, None, f"{type(e).__name__}: {e}"))
                p.notify(f"ERROR processing {symbol}: {type(e).__name__}: {e}")
                continue
            decision, risk, status = out["decision"], out["risk"], out.get("order_status")
            said = {n: (f"{a['signals'][n].action}({a['signals'][n].confidence:.2f})" if a["signals"].get(n) else "none")
                    for n in p.agents}  # registry order, not the order parallel nodes happened to finish
            log.info("%s %s conf=%.2f risk=%s order=%s agents=%s", symbol, decision.action, decision.confidence,
                     risk.reason, status, " ".join(f"{n}:{v}" for n, v in said.items()),
                     extra={"symbol": symbol, "action": decision.action, "confidence": round(decision.confidence, 3),
                            "risk_approved": risk.approved, "order_status": status, "agents": said})
            p._cycle_llm.append(any(s is not None and s.degraded for n, s in a["signals"].items() if roles.get(n) == LEAD))
            results.append(SymbolResult(symbol, decision.action, decision.confidence, risk, status))
            if status and not p.settings.dry_run:
                portfolio = p._with_peak(p.broker.portfolio())
        p._track_llm()
        return {"results": results}

    graph = StateGraph(CycleState)
    graph.add_node("open_cycle", open_cycle)
    graph.add_node("select_universe", select_universe)
    graph.add_node("analyze_symbol", analyze_symbol)
    graph.add_node("execute_cycle", execute_cycle)
    graph.add_edge(START, "open_cycle")
    graph.add_edge("open_cycle", "select_universe")
    graph.add_conditional_edges("select_universe", scatter, ["analyze_symbol", "execute_cycle"])
    graph.add_edge("analyze_symbol", "execute_cycle")
    graph.add_edge("execute_cycle", END)
    return graph.compile()
