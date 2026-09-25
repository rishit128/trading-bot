"""The trading workflow as three LangGraph graphs, built from whatever agents are registered.

    cycle graph     open_cycle -> select_universe --(map: one Send per stock)--> analyze_symbol x N --> execute_cycle
    analysis graph  fetch_data -> agent_<name> (ALL agents in parallel) -> collect          (one stock)
    decision graph  decide -> [risk approved?] -> execute                                     (one stock)

* Adding an agent means writing a class with `name`/`role`/`analyze(ctx)` and registering it: the analysis graph grows
  a parallel node automatically and `combine_signals` folds it in by role. No graph code changes.
* Stocks are analysed concurrently (the slow, LLM-bound part) but decided and executed one at a time in a fixed order,
  so risk limits, position counts and cash are always evaluated against fresh state and results are deterministic.
* Risk limits and order sizing stay in plain deterministic code (`RiskEngine`); the graphs only orchestrate.
* The graphs know the application only through `CycleServices` (below): a handful of public methods. Nothing here reaches
  into the pipeline's internals, so a node can be tested with a small fake and the pipeline can change freely."""
import logging
import operator
from typing import Annotated, Dict, List, Mapping, Optional, Protocol, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from src.agents.base import LEAD, Agent, AgentContext
from src.config import Settings
from src.data.indicators import Snapshot
from src.data.retry import is_transient_data_error
from src.engine.risk_engine import Portfolio
from src.engine.agent_signal import AgentSignal
from src.results import SymbolResult, Verdict

log = logging.getLogger(__name__)


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer letting parallel agent nodes each write their own key into one shared dict."""
    return {**(left or {}), **(right or {})}


class AnalysisState(TypedDict, total=False):
    """State of the per-stock analysis graph."""
    symbol: str
    snapshot: Snapshot
    signals: Annotated[Dict[str, Optional[AgentSignal]], merge_dicts]


class DecisionState(TypedDict, total=False):
    """State of the per-stock decision graph."""
    symbol: str
    portfolio: Portfolio
    snapshot: Snapshot
    signals: Dict[str, Optional[AgentSignal]]
    verdict: Verdict
    order_status: Optional[str]
    universe_size: int


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


class CycleServices(Protocol):
    """What the graphs need from the application, and nothing more. `TradingPipeline` is the production implementation; a
    test can pass a small fake. Every member is public: the graphs never touch anything with a leading underscore."""
    @property
    def agents(self) -> Mapping[str, Agent]: ...
    @property
    def settings(self) -> Settings: ...

    def snapshot(self, symbol: str) -> Snapshot: ...
    def agent_context(self, symbol: str, snapshot: Snapshot) -> AgentContext: ...
    def open_cycle(self) -> Portfolio: ...
    def select_symbols(self, portfolio: Portfolio) -> Sequence[str]: ...
    def decide(self, symbol: str, portfolio: Portfolio, snapshot: Snapshot, signals: Mapping[str, Optional[AgentSignal]],
               universe_size: Optional[int] = None) -> Verdict: ...
    def place(self, symbol: str, verdict: Verdict) -> Optional[str]: ...
    def refresh_portfolio(self) -> Portfolio: ...
    def note_lead_health(self, degraded: bool) -> None: ...
    def close_cycle(self) -> None: ...
    def notify(self, text: str) -> None: ...


# A snapshot download that fails on a transient error (dropped connection, rate limit) gets one more try after a moment,
# instead of costing that stock a whole 30-minute cycle. A definite answer about the data (stale, missing, too short) is
# never retried: see `is_transient_data_error`.
SNAPSHOT_RETRY = RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0, retry_on=is_transient_data_error)


def build_analysis_graph(services: CycleServices, snapshot_retry: RetryPolicy = SNAPSHOT_RETRY):
    """Per-stock graph: fetch the snapshot (retrying transient failures), then run ALL agents in parallel. Agents are
    looked up by name at call time, so an implementation can be swapped."""
    if not services.agents:
        raise ValueError("at least one agent is required")

    def fetch_data(state: AnalysisState):
        """Fetch the stock's indicator snapshot."""
        return {"snapshot": services.snapshot(state["symbol"])}

    def make_agent_node(name: str):
        """Build the graph node that runs one agent."""
        def node(state: AnalysisState):
            """Run one agent for the stock."""
            ctx = services.agent_context(state["symbol"], state["snapshot"])
            return {"signals": {name: services.agents[name].analyze(ctx)}}

        return node

    def collect(state: AnalysisState):
        """Join point after all agents finish."""
        return {}

    graph = StateGraph(AnalysisState)
    graph.add_node("fetch_data", fetch_data, retry_policy=snapshot_retry)
    graph.add_node("collect", collect)
    graph.add_edge(START, "fetch_data")
    for name in services.agents:
        node = agent_node_name(name)
        graph.add_node(node, make_agent_node(name))
        graph.add_edge("fetch_data", node)
    graph.add_edge([agent_node_name(n) for n in services.agents], "collect")
    graph.add_edge("collect", END)
    return graph.compile()


def build_decision_graph(services: CycleServices):
    """Per-stock graph: decide (combine, risk-size, record), then place the order if the risk engine approved it."""
    def decide(state: DecisionState):
        """Decide and record one stock's call."""
        return {"verdict": services.decide(state["symbol"], state["portfolio"], state["snapshot"], state["signals"],
                                           state.get("universe_size"))}

    def execute(state: DecisionState):
        """Place the approved order."""
        return {"order_status": services.place(state["symbol"], state["verdict"])}

    def route(state: DecisionState) -> str:
        """Send approved trades to execute and everything else to the end."""
        return "execute" if state["verdict"].risk.approved else "end"

    graph = StateGraph(DecisionState)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, {"execute": "execute", "end": END})
    graph.add_edge("execute", END)
    return graph.compile()


def build_cycle_graph(services: CycleServices, analysis_graph, decision_graph):
    """Whole-cycle graph: open the account, choose stocks, analyse them in parallel, then decide each in order."""
    def open_cycle(state: CycleState):
        """Read the account and run the start-of-cycle safeguards."""
        return {"portfolio": services.open_cycle(), "analyses": [], "results": []}

    def select_universe(state: CycleState):
        """Choose which stocks to analyse this cycle."""
        return {"symbols": list(services.select_symbols(state["portfolio"]))}

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
            out = analysis_graph.invoke({"symbol": symbol}, config={"max_concurrency": len(services.agents)})
            return {"analyses": [{"symbol": symbol, "snapshot": out["snapshot"], "signals": out["signals"], "error": None}]}
        except Exception as e:  # one stock failing must not abort the others
            log.exception("%s analysis failed", symbol)
            return {"analyses": [{"symbol": symbol, "error": f"{type(e).__name__}: {e}"}]}

    def execute_cycle(state: CycleState):
        """Decide and execute each analysed stock in universe order against fresh account state."""
        order = {s: i for i, s in enumerate(state["symbols"])}
        roles = {n: a.role for n, a in services.agents.items()}
        portfolio, results = state["portfolio"], []
        for a in sorted(state["analyses"], key=lambda a: order[a["symbol"]]):
            symbol = a["symbol"]
            if a["error"]:
                results.append(SymbolResult(symbol, "ERROR", 0.0, None, None, a["error"]))
                services.notify(f"ERROR processing {symbol}: {a['error']}")
                continue
            try:
                out = decision_graph.invoke({"symbol": symbol, "portfolio": portfolio, "snapshot": a["snapshot"],
                                             "signals": a["signals"], "universe_size": len(state["symbols"])})
            except Exception as e:
                log.exception("%s decision failed", symbol)
                results.append(SymbolResult(symbol, "ERROR", 0.0, None, None, f"{type(e).__name__}: {e}"))
                services.notify(f"ERROR processing {symbol}: {type(e).__name__}: {e}")
                continue
            decision, risk, status = out["verdict"].decision, out["verdict"].risk, out.get("order_status")
            said = {n: (f"{a['signals'][n].action}({a['signals'][n].confidence:.2f})" if a["signals"].get(n) else "none")
                    for n in services.agents}  # registry order, not the order parallel nodes happened to finish
            log.info("%s %s conf=%.2f risk=%s order=%s agents=%s", symbol, decision.action, decision.confidence,
                     risk.reason, status, " ".join(f"{n}:{v}" for n, v in said.items()),
                     extra={"symbol": symbol, "action": decision.action, "confidence": round(decision.confidence, 3),
                            "risk_approved": risk.approved, "order_status": status, "agents": said})
            services.note_lead_health(any(s is not None and s.degraded for n, s in a["signals"].items()
                                          if roles.get(n) == LEAD))
            results.append(SymbolResult(symbol, decision.action, decision.confidence, risk, status))
            if status and not services.settings.dry_run:
                portfolio = services.refresh_portfolio()
        services.close_cycle()
        return {"results": results}

    graph = StateGraph(CycleState)
    graph.add_node("open_cycle", open_cycle)
    graph.add_node("select_universe", select_universe)
    graph.add_node("analyze_symbol", analyze_symbol)  # type: ignore[arg-type]  # a Send payload is not the graph state type
    graph.add_node("execute_cycle", execute_cycle)
    graph.add_edge(START, "open_cycle")
    graph.add_edge("open_cycle", "select_universe")
    graph.add_conditional_edges("select_universe", scatter, ["analyze_symbol", "execute_cycle"])
    graph.add_edge("analyze_symbol", "execute_cycle")
    graph.add_edge("execute_cycle", END)
    return graph.compile()
