"""Trading pipeline: owns the agents, broker and audit log, and runs one cycle through the LangGraph workflow."""
import dataclasses
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from sqlalchemy import func, select

from src.agents.base import Agent
from src.config import Settings
from src.data.indicators import Snapshot
from src.database import DecisionRecord, EquityRecord, OrderRecord
from src.engine.risk_engine import Portfolio, RiskDecision, RiskEngine, drawdown_pause
from src.engine.strategy import Decision
from src.versions import VERSIONS
from src.llm import Signal
from src.results import SymbolResult
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph

log = logging.getLogger(__name__)

__all__ = ["TradingPipeline", "SymbolResult", "NOT_A_PLACED_BUY"]

# Statuses that mean "we actually went for this buy"; failures and skips must not start a cooldown.
NOT_A_PLACED_BUY = ("SKIPPED_OPEN_ORDER", "SKIPPED_COOLDOWN", "FAILED", "PAUSED", "MARKET_CLOSED")


class TradingPipeline:
    """Runs the analyse, decide, execute cycle and holds the safeguards around it (cooldown, halts, alerts, protection sweep)."""
    def __init__(
        self,
        settings: Settings,
        agents: Sequence[Agent],
        broker,
        session_factory,
        snapshot_fn: Callable[[str], Snapshot],
        headlines_fn: Callable[[str], Sequence[str]],
        control=None,
        notify: Optional[Callable[[str], object]] = None,
        universe_fn: Optional[Callable[[Portfolio], Sequence[str]]] = None,
        quote_fn: Optional[Callable[[str], float]] = None,
        market_context_fn: Optional[Callable[[Snapshot], Optional[object]]] = None,
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
        self.risk = RiskEngine(settings.risk, fees=getattr(broker, "fees", None))
        self.control = control
        self.universe_fn = universe_fn
        self.quote_fn = quote_fn
        self.market_context_fn = market_context_fn
        self._notify = notify
        self._halt_kind = None
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
        check = getattr(self.broker, "unprotected_positions", None)
        if check is None:
            return
        acting = self.settings.auto_protect and not self.settings.dry_run
        try:
            naked = check()
            if naked and acting:
                self._sleep(self.protect_grace_seconds)
                naked = check()
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
                    self.broker.protect(symbol, qty, stop)
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

    def _log_decision(self, symbol: str, snap: Snapshot, signals: Mapping[str, Optional[Signal]],
                      decision: Decision, risk: RiskDecision, price: float,
                      universe_size: Optional[int] = None) -> int:
        tech, sent = signals.get("technical"), signals.get("sentiment")
        details = tech.details if tech is not None else None
        learning = (details or {}).get("learning") or {}
        context = (details or {}).get("context") or {}
        reflection = (details or {}).get("reflection") or {}
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
                signals_json=json.dumps({n: sig.model_dump() if sig else None for n, sig in signals.items()}),
                snapshot_json=json.dumps(dataclasses.asdict(snap)),
                # Phase 1 chain-of-thought extras, taken from the lead agent's stored details (None before Phase 1).
                confluence_score=details.get("confluence_score") if details else None,
                edge_confidence=details.get("edge_confidence") if details else None,
                risks_json=json.dumps(details["risks"]) if details and details.get("risks") is not None else None,
                reasoning_chain_json=json.dumps(details["reasoning_chain"])
                if details and details.get("reasoning_chain") is not None else None,
                # Phases 2-4 extras: the untouched base confidence, the regime label (None on normal markets),
                # the historical pattern record when the learning phase consulted one, and the review-doc columns.
                base_confidence=details.get("base_confidence") if details else None,
                context_regime=context.get("regime") if context else None,
                historical_win_rate=learning.get("win_rate") if learning else None,
                historical_sample_size=learning.get("sample_size") if learning else None,
                pattern_id=details.get("pattern_id") if details else None,
                adjusted_signal_confidence=details.get("adjusted_signal_confidence") if details else None,
                adjustment_reason=details.get("adjustment_reason") if details else None,
                macro_support=context.get("macro_support") if context else None,
                sector_support=context.get("sector_support") if context else None,
                earnings_risk=context.get("earnings_risk") if context else None,
                diversification_score=context.get("diversification_score") if context else None,
                market_context_json=json.dumps(details["market_context_json"])
                if details and details.get("market_context_json") is not None else None,
                biggest_risk=reflection.get("biggest_risk") if reflection else None,
                what_proves_us_wrong=reflection.get("what_proves_us_wrong") if reflection else None,
                bias_check=json.dumps(reflection["bias_check"])
                if reflection and reflection.get("bias_check") is not None else None,
                conviction_adjustments=json.dumps(reflection["conviction_adjustments"])
                if reflection and reflection.get("conviction_adjustments") is not None else None,
                raw_model=details.get("raw_model") if details else None,
                rule_alignment=details.get("rule_alignment") if details else None,
                falsification=details.get("falsification") if details else None,
                prompt_version=VERSIONS["prompt"],
                feature_version=VERSIONS["feature"],
                strategy_version=VERSIONS["strategy"],
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
                .where(OrderRecord.symbol == symbol, OrderRecord.side == "BUY",
                       OrderRecord.status.not_in(NOT_A_PLACED_BUY), OrderRecord.created_at >= cutoff)
                .limit(1)
            )
        return recent is not None

    def _place(self, decision_id: int, symbol: str, action: str, qty: int, price: float) -> str:
        filled_qty = fill_price = None
        if self.control and self.control.is_paused():
            status, broker_id, error = "PAUSED", None, None
        elif action == "BUY" and self._in_cooldown(symbol):
            status, broker_id, error = "SKIPPED_COOLDOWN", None, None
        elif self.settings.dry_run:
            status, broker_id, error = "DRY_RUN", None, None
        elif not self.broker.is_market_open():
            status, broker_id, error = "MARKET_CLOSED", None, None
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
        if status not in ("DRY_RUN", "PAUSED", "MARKET_CLOSED", "SKIPPED_COOLDOWN"):
            detail = f" ({error})" if error else ""
            self.notify(f"{action} {qty} {symbol} @ ~{self.settings.currency}{price:,.2f}: {status}{detail}")
        if filled_qty is not None and filled_qty != qty:
            self.notify(f"PARTIAL/UNFILLED {action} {symbol}: ordered {qty}, broker filled {filled_qty} ({status}). Check the position.")
        return status

    def _submit(self, symbol: str, action: str, qty: int, price: float):
        try:
            if action == "BUY":
                if self.broker.has_open_order(symbol):
                    return "SKIPPED_OPEN_ORDER", None, None, None, None
                fill = self.broker.buy_with_bracket(
                    symbol, qty, price, self.settings.risk.stop_loss_pct, self.settings.risk.take_profit_pct
                )
            else:
                fill = self.broker.sell(symbol, qty)
            confirm = getattr(self.broker, "confirm", None)
            if confirm is not None:
                fill = confirm(fill, qty)  # ask the broker what really happened instead of trusting "accepted"
            return fill.status, fill.broker_order_id, None, fill.filled_qty, fill.avg_price
        except Exception as e:
            log.exception("order submission failed for %s", symbol)
            if action == "SELL":
                self._reconcile_protection()  # a failed sell may have cancelled the stop: repair it right away
            return "FAILED", None, f"{type(e).__name__}: {e}", None, None
