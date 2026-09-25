"""Trading pipeline: owns the agents, broker and audit log, and runs one cycle through the LangGraph workflow."""
import dataclasses
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from sqlalchemy import func, select

from src.agents.base import Agent, AgentContext
from src.config import Settings
from src.data.indicators import Snapshot
from src.data.market_context import MarketContext
from src.database import DecisionRecord, EquityRecord, OrderRecord
from src.engine.enums import NOT_A_PLACED_BUY, SILENT_STATUSES, Action, OrderStatus
from src.engine.ports import Broker, ChargesFees, ConfirmsFills, ProtectsPositions
from src.engine.risk_engine import Portfolio, RiskDecision, RiskEngine, drawdown_pause
from src.engine.strategy import Decision, combine_signals, trend_exit_decision
from src.versions import VERSIONS
from src.engine.agent_signal import AgentSignal, SignalDetails
from src.results import SymbolResult, Verdict
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph

log = logging.getLogger(__name__)

__all__ = ["TradingPipeline", "SymbolResult", "NOT_A_PLACED_BUY"]  # the last is re-exported for callers of the old name



def details_columns(details: Optional[SignalDetails]) -> dict:
    """The lead agent's audit trail as the denormalised `decisions` columns (queryable without parsing the JSON blob).
    Everything is None when the agent recorded no details or a section did not run."""
    if details is None:
        return {}
    learning, context, reflection, chain = details.learning, details.context, details.reflection, details.reasoning_chain
    return dict(
        # chain-of-thought extras
        confluence_score=details.confluence_score, edge_confidence=details.edge_confidence,
        risks_json=json.dumps(details.risks) if details.risks is not None else None,
        reasoning_chain_json=json.dumps(chain.model_dump()) if chain is not None else None,
        # refinement phases: the untouched base confidence, the historical record, the regime and the review columns
        base_confidence=details.base_confidence,
        context_regime=context.regime if context else None,
        historical_win_rate=learning.win_rate if learning else None,
        historical_sample_size=learning.sample_size if learning else None,
        pattern_id=details.pattern_id, adjusted_signal_confidence=details.adjusted_signal_confidence,
        adjustment_reason=details.adjustment_reason,
        macro_support=context.macro_support if context else None,
        sector_support=context.sector_support if context else None,
        earnings_risk=context.earnings_risk if context else None,
        diversification_score=context.diversification_score if context else None,
        market_context_json=json.dumps(details.market_context_json) if details.market_context_json is not None else None,
        biggest_risk=reflection.biggest_risk if reflection else None,
        what_proves_us_wrong=reflection.what_proves_us_wrong if reflection else None,
        bias_check=json.dumps(reflection.bias_check) if reflection else None,
        conviction_adjustments=json.dumps([c.model_dump() for c in reflection.conviction_adjustments]) if reflection else None,
        # accountability: who answered and whether the call agreed with the mechanical filter
        raw_model=details.raw_model, rule_alignment=details.rule_alignment, falsification=details.falsification)


class TradingPipeline:
    """Runs the analyse, decide, execute cycle and holds the safeguards around it (cooldown, halts, alerts, protection sweep)."""
    def __init__(
        self,
        settings: Settings,
        agents: Sequence[Agent],
        broker: Broker,
        session_factory,
        snapshot_fn: Callable[[str], Snapshot],
        headlines_fn: Callable[[str], Sequence[str]],
        control=None,
        notify: Optional[Callable[[str], object]] = None,
        universe_fn: Optional[Callable[[Portfolio], Sequence[str]]] = None,
        quote_fn: Optional[Callable[[str], float]] = None,
        market_context_fn: Optional[Callable[[Snapshot], Optional[MarketContext]]] = None,
        broker_label: str = "",
    ):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"agent names must be unique: {names}")
        self.settings = settings
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.broker = broker
        self.sessions = session_factory
        self.snapshot_fn = snapshot_fn
        self.headlines_fn = headlines_fn
        self.broker_label = broker_label  # a human name for the broker, shown in the start-up banner
        self.risk = RiskEngine(settings.risk, fees=broker.fees if isinstance(broker, ChargesFees) else None)
        self.control = control
        self.universe_fn = universe_fn
        self.quote_fn = quote_fn
        self.market_context_fn = market_context_fn
        self._notify = notify
        self._halt_kind: Optional[str] = None
        self._llm_down = False
        self._cycle_llm: List[bool] = []
        self._naked_alerted: set = set()
        self.protect_grace_seconds = 3.0  # right after a fill the stop legs may still be activating
        self._sleep = time.sleep
        self.rebuild_graphs()

    def rebuild_graphs(self) -> None:
        """Call after adding or removing agents so the analysis graph gains/loses the matching parallel nodes."""
        self.analysis_graph = build_analysis_graph(self)
        self.decision_graph = build_decision_graph(self)
        self.cycle_graph = build_cycle_graph(self, self.analysis_graph, self.decision_graph)

    # ---- the CycleServices interface: the only way the workflow graphs touch the application (see workflow.py) --------
    def snapshot(self, symbol: str) -> Snapshot:
        """The stock's indicator snapshot as of the last completed session."""
        return self.snapshot_fn(symbol)

    def agent_context(self, symbol: str, snapshot: Snapshot) -> AgentContext:
        """What an agent is given for one stock. Headlines and market context are lazy: only agents that ask pay for them."""
        return AgentContext(symbol, snapshot, headlines=lambda: self.headlines_fn(symbol),
                            market=lambda: self.market_context_fn(snapshot) if self.market_context_fn else None)

    def open_cycle(self) -> Portfolio:
        """Start a cycle: read the account, record its equity, apply the drawdown pause, check halts and stop protection."""
        portfolio = self.broker.portfolio()
        self._record_equity(portfolio.equity)
        portfolio = self._apply_drawdown_pause(self._with_peak(portfolio))
        self._check_halt(portfolio)
        self._reconcile_protection()
        self._cycle_llm = []
        return portfolio

    def select_symbols(self, portfolio: Portfolio) -> List[str]:
        """Which stocks this cycle analyses (held positions always, so exits keep working)."""
        return list(self._symbols(portfolio))

    def decide(self, symbol: str, portfolio: Portfolio, snapshot: Snapshot, signals: Mapping[str, Optional[AgentSignal]],
               universe_size: Optional[int] = None) -> Verdict:
        """Turn the agents' signals into one recorded decision: the deterministic trend exit if it applies, else the agents
        combined by role; then size it with the risk engine and write the audit row."""
        held = portfolio.position_qty.get(symbol, 0)
        decision = (trend_exit_decision(snapshot, held, self.settings.trend_exit)
                    or combine_signals(signals, {n: a.role for n, a in self.agents.items()}))
        price = self._quote(symbol, snapshot.price) if decision.action != Action.HOLD else snapshot.price
        risk = self.risk.evaluate(decision.action, decision.confidence, symbol, price, portfolio)
        decision_id = self._log_decision(symbol, snapshot, signals, decision, risk, price, universe_size=universe_size)
        return Verdict(decision, risk, price, decision_id)

    def place(self, symbol: str, verdict: Verdict) -> str:
        """Carry out an approved verdict (or record why it was not sent) and return the order status."""
        return self._place(verdict.decision_id, symbol, verdict.decision.action, verdict.risk.quantity, verdict.price)

    def refresh_portfolio(self) -> Portfolio:
        """The account as it is now, after orders, with the peak-equity baseline applied."""
        return self._with_peak(self.broker.portfolio())

    def note_lead_health(self, degraded: bool) -> None:
        """Record whether a stock's lead agent fell back to its fail-safe (used to detect an AI outage)."""
        self._cycle_llm.append(degraded)

    def close_cycle(self) -> None:
        """End a cycle: log AI health and alert on an outage or its recovery."""
        self._track_llm()

    def notify(self, text: str) -> None:
        """Send an alert if Telegram is configured; never raises."""
        if self._notify:
            try:
                self._notify(text)
            except Exception:
                log.exception("notification failed")

    def run_once(self) -> List[SymbolResult]:
        """Run one full cycle and return one result per stock."""
        state = self.cycle_graph.invoke({"symbols": []}, config={"max_concurrency": self.settings.analysis_workers})
        return state["results"]

    def _symbols(self, portfolio: Portfolio) -> Sequence[str]:
        """Held positions are always analysed (so exits work) even if the market scan fails."""
        if self.universe_fn is None:
            return self.settings.watchlist
        try:
            return list(self.universe_fn(portfolio))
        except Exception as e:
            log.exception("universe scan failed")
            self.notify(f"UNIVERSE SCAN FAILED, analysing held positions only: {type(e).__name__}: {e}")
            return sorted(portfolio.positions)

    def _quote(self, symbol: str, fallback: float) -> float:
        """Live price for sizing and bracket levels: the analysis runs on completed bars, the order on today's price."""
        if self.quote_fn is None:
            return fallback
        try:
            return float(self.quote_fn(symbol))
        except Exception as e:
            log.warning("no live quote for %s, using last completed close: %s", symbol, e)
            return fallback

    def _reconcile_protection(self) -> None:
        """Every long position must have a working stop. Alerts when one does not and, for live orders, places one.

        Covers a SELL that fails after its stop/target legs were cancelled, or any stop lost for another reason.
        A second look after a short pause avoids acting on legs that are still activating right after a fill."""
        if not isinstance(self.broker, ProtectsPositions):
            return
        broker = self.broker
        acting = self.settings.auto_protect and not self.settings.dry_run
        try:
            naked = broker.unprotected_positions()
            if naked and acting:
                self._sleep(self.protect_grace_seconds)
                naked = broker.unprotected_positions()
        except Exception as e:
            log.exception("protection check failed")
            self.notify(f"PROTECTION CHECK FAILED: {type(e).__name__}: {e}")
            return
        for symbol, qty, avg_entry in naked:
            if symbol not in self._naked_alerted:
                self.notify(f"UNPROTECTED POSITION: {qty} {symbol} has no working stop order.")
            if acting:
                stop = avg_entry * (1 - self.settings.risk.stop_loss_pct)
                try:
                    broker.protect(symbol, qty, stop)
                    self.notify(f"Protective stop placed: {symbol} {qty} sh, stop {self.settings.currency}{stop:,.2f}.")
                except Exception as e:
                    log.exception("could not protect %s", symbol)
                    self.notify(f"COULD NOT PROTECT {symbol}: {type(e).__name__}: {e}. Close it manually.")
        self._naked_alerted = {s for s, _, _ in naked}

    def _record_equity(self, equity: float) -> None:
        with self.sessions() as s:
            s.add(EquityRecord(equity=equity))
            s.commit()

    def _with_peak(self, p: Portfolio) -> Portfolio:
        since = self.control.peak_since() if self.control else None  # set by /rebase after a drawdown halt
        query = select(func.max(EquityRecord.equity))
        if since is not None:
            query = query.where(EquityRecord.created_at >= since)
        with self.sessions() as s:
            recorded = s.scalar(query)
        peak = max(recorded or 0.0, p.equity)
        return Portfolio(p.cash, p.equity, p.positions, p.position_qty, p.start_of_day_equity, peak)

    def _apply_drawdown_pause(self, p: Portfolio) -> Portfolio:
        """End a drawdown halt by itself after `drawdown_pause_days`: rebase the peak (what /rebase does by hand) and
        re-read the portfolio against it. The clock only runs while the halt is continuously in force."""
        if self.control is None or p.peak_equity is None:
            return p
        lim, now = self.settings.risk, datetime.now(timezone.utc)
        since = self.control.drawdown_since()
        peak, new_since = drawdown_pause(p.peak_equity, p.equity, since, now, lim.max_drawdown_pct, lim.drawdown_pause_days)
        if new_since != since:
            self.control.set_drawdown_since(new_since)
        if peak != p.peak_equity:
            self.control.rebase_peak(now)
            self.notify(f"Drawdown pause over after {lim.drawdown_pause_days:g} days: the peak is rebased to equity "
                        f"{self.settings.currency}{p.equity:,.0f} and buying resumes.")
            return self._with_peak(p)
        return p

    def _log_decision(self, symbol: str, snap: Snapshot, signals: Mapping[str, Optional[AgentSignal]],
                      decision: Decision, risk: RiskDecision, price: float,
                      universe_size: Optional[int] = None) -> int:
        tech, sent = signals.get("technical"), signals.get("sentiment")
        with self.sessions() as s:
            record = DecisionRecord(
                symbol=symbol,
                price=price,
                technical_action=tech.action if tech else None,
                technical_confidence=tech.confidence if tech else None,
                sentiment_action=sent.action if sent else None,
                sentiment_confidence=sent.confidence if sent else None,
                final_action=decision.action,
                final_confidence=decision.confidence,
                reasoning=decision.reasoning,
                risk_approved=risk.approved,
                risk_quantity=risk.quantity,
                risk_reason=risk.reason,
                signals_json=json.dumps({n: sig.to_json_dict() if sig else None for n, sig in signals.items()}),
                snapshot_json=json.dumps(dataclasses.asdict(snap)),
                **details_columns(tech.details if tech is not None else None),
                prompt_version=VERSIONS["prompt"],
                feature_version=VERSIONS["feature"],
                strategy_version=VERSIONS["strategy"],
                decision_source=decision.source,
                bar_date=snap.bar_date,
                universe_size=universe_size,
            )
            s.add(record)
            s.commit()
            return record.id

    def _check_halt(self, portfolio: Portfolio) -> None:
        halt = self.risk.halt_reason(portfolio)
        kind = halt[0] if halt else None
        if kind != self._halt_kind:
            self.notify(f"RISK HALT: {halt[1]}. New buys are blocked." if halt else "Risk halt cleared: buys allowed again.")
            self._halt_kind = kind

    def _log_llm_health(self) -> None:
        """One INFO line per cycle on how the AI models behaved (calls that worked, failures by kind, benched models)."""
        for agent in self.agents.values():
            health = getattr(getattr(agent, "llm", None), "health_line", None)
            line = health() if callable(health) else None
            if line:
                log.info(line)

    def _track_llm(self) -> None:
        """Alert only when every analysis in a cycle failed; one flaky call is not an outage."""
        self._log_llm_health()
        if not self._cycle_llm:
            return
        down = all(self._cycle_llm)
        if down and not self._llm_down:
            self.notify("LLM UNAVAILABLE: every configured model failed; the bot is holding (no new trades) until it recovers.")
        elif not down and self._llm_down:
            self.notify("LLM recovered: trading signals are working again.")
        self._llm_down = down

    def _in_cooldown(self, symbol: str) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.settings.rebuy_cooldown_hours)
        with self.sessions() as s:
            recent = s.scalar(
                select(OrderRecord.id)
                .where(OrderRecord.symbol == symbol, OrderRecord.side == Action.BUY,
                       OrderRecord.status.not_in(NOT_A_PLACED_BUY), OrderRecord.created_at >= cutoff)
                .limit(1)
            )
        return recent is not None

    def _place(self, decision_id: int, symbol: str, action: Action, qty: int, price: float) -> str:
        filled_qty: Optional[int] = None
        fill_price: Optional[float] = None
        status: str
        broker_id: Optional[str]
        error: Optional[str]
        if self.control and self.control.is_paused():
            status, broker_id, error = OrderStatus.PAUSED, None, None
        elif action == Action.BUY and self._in_cooldown(symbol):
            status, broker_id, error = OrderStatus.SKIPPED_COOLDOWN, None, None
        elif self.settings.dry_run:
            status, broker_id, error = OrderStatus.DRY_RUN, None, None
        elif not self.broker.is_market_open():
            status, broker_id, error = OrderStatus.MARKET_CLOSED, None, None
        else:
            status, broker_id, error, filled_qty, fill_price = self._submit(symbol, action, qty, price)
        with self.sessions() as s:
            s.add(
                OrderRecord(
                    decision_id=decision_id,
                    symbol=symbol,
                    side=action,
                    quantity=qty,
                    status=status,
                    broker_order_id=broker_id,
                    error=error,
                    filled_qty=filled_qty,
                    fill_price=fill_price,
                )
            )
            s.commit()
        if status not in SILENT_STATUSES:
            detail = f" ({error})" if error else ""
            self.notify(f"{action} {qty} {symbol} @ ~{self.settings.currency}{price:,.2f}: {status}{detail}")
        if filled_qty is not None and filled_qty != qty:
            self.notify(f"PARTIAL/UNFILLED {action} {symbol}: ordered {qty}, broker filled {filled_qty} ({status}). Check the position.")
        return status

    def _submit(self, symbol: str, action: str, qty: int, price: float):
        try:
            if action == Action.BUY:
                if self.broker.has_open_order(symbol):
                    return OrderStatus.SKIPPED_OPEN_ORDER, None, None, None, None
                fill = self.broker.buy_with_bracket(
                    symbol, qty, price, self.settings.risk.stop_loss_pct, self.settings.risk.take_profit_pct
                )
            else:
                fill = self.broker.sell(symbol, qty)
            if isinstance(self.broker, ConfirmsFills):
                fill = self.broker.confirm(fill, qty)  # ask the broker what really happened instead of trusting "accepted"
            return fill.status, fill.broker_order_id, None, fill.filled_qty, fill.avg_price
        except Exception as e:
            log.exception("order submission failed for %s", symbol)
            if action == Action.SELL:
                self._reconcile_protection()  # a failed sell may have cancelled the stop: repair it right away
            return OrderStatus.FAILED, None, f"{type(e).__name__}: {e}", None, None
