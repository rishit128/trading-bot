"""Trading pipeline: owns the agents, broker and audit log, and runs one cycle through the LangGraph workflow."""

import dataclasses
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set

from sqlalchemy import func, select

from src.agents.base import Agent, AgentContext
from src.config import Settings
from src.data.angel_quotes import EntryContext
from src.data.fundamentals import Fundamentals
from src.data.house_calls import HouseCall, call_for, open_sells, recently_ended, swing_buys
from src.data.indicators import Snapshot
from src.data.market_context import MarketContext
from src.data.market_pulse import MAX_OPEN as MAX_PULSE_OPEN
from src.data.market_pulse import HotStock, MarketPulse
from src.data.market_pulse import summary as pulse_summary
from src.data.sector_watch import OIL_GAS
from src.database import DecisionRecord, EquityRecord, OrderRecord
from src.engine import core_sleeve
from src.engine.agent_signal import AgentSignal, SignalDetails
from src.engine.enums import NOT_A_PLACED_BUY, SILENT_STATUSES, Action, OrderStatus
from src.engine.ports import Broker, ChargesFees, ListsTrades
from src.engine.risk_engine import Portfolio, RiskDecision, RiskEngine, drawdown_pause
from src.engine.rules import rule_verdict
from src.engine.strategy import Decision, combine_signals, trend_exit_decision
from src.results import SymbolResult, Verdict
from src.versions import VERSIONS
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph

log = logging.getLogger(__name__)

# The core sleeve is bought through the bracket-order API with a stop and a target that can never trigger: -90% and +1000%.
CORE_STOP_PCT, CORE_TARGET_PCT = 0.90, 10.0
STOP_LOOKBACK = 100  # the newest closed trades searched for a recent stop-out (the cooldown is hours, far fewer trades)

__all__ = ["TradingPipeline", "SymbolResult", "NOT_A_PLACED_BUY"]  # the last is re-exported for callers of the old name


def details_columns(details: Optional[SignalDetails]) -> dict:
    """The lead agent's audit trail as the denormalised `decisions` columns (queryable without parsing the JSON blob).
    Everything is None when the agent recorded no details or a section did not run. Columns from removed phases
    (decision memory, the separate market-context call) stay in the table for older rows and are left empty."""
    if details is None:
        return {}
    reflection, chain = details.reflection, details.reasoning_chain
    return dict(
        # analysis extras
        confluence_score=details.confluence_score,
        edge_confidence=details.edge_confidence,
        risks_json=json.dumps(details.risks) if details.risks is not None else None,
        reasoning_chain_json=json.dumps(chain.model_dump()) if chain is not None else None,
        market_context_json=json.dumps(details.market_context_json) if details.market_context_json is not None else None,
        # the review: the untouched base confidence, what the critic did and why
        base_confidence=details.base_confidence,
        adjusted_signal_confidence=details.adjusted_signal_confidence,
        adjustment_reason=details.adjustment_reason,
        biggest_risk=reflection.biggest_risk if reflection else None,
        what_proves_us_wrong=reflection.what_proves_us_wrong if reflection else None,
        bias_check=json.dumps(reflection.bias_check) if reflection else None,
        conviction_adjustments=json.dumps([c.model_dump() for c in reflection.conviction_adjustments]) if reflection else None,
        # accountability: who answered and whether the call agreed with the mechanical filter
        raw_model=details.raw_model,
        rule_alignment=details.rule_alignment,
        falsification=details.falsification,
    )


class TradingPipeline:
    """Runs the analyse, decide, execute cycle and holds the safeguards around it (cooldown, halts, alerts)."""

    def __init__(
        self,
        settings: Settings,
        agents: Sequence[Agent],
        broker: Broker,
        session_factory,
        snapshot_fn: Callable[[str], Snapshot],
        control=None,
        notify: Optional[Callable[[str], object]] = None,
        universe_fn: Optional[Callable[[Portfolio], Sequence[str]]] = None,
        quote_fn: Optional[Callable[[str], float]] = None,
        market_context_fn: Optional[Callable[[Snapshot], Optional[MarketContext]]] = None,
        fundamentals_fn: Optional[Callable[[str], Optional[Fundamentals]]] = None,
        broker_label: str = "",
        house_calls_fn: Optional[Callable[[], List[HouseCall]]] = None,
        screen_verdict_fn: Optional[Callable[[str, float], str]] = None,
        pulse: Optional[MarketPulse] = None,
        context_fn: Optional[Callable[[str, float], EntryContext]] = None,
    ):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"agent names must be unique: {names}")
        self.settings = settings
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.broker = broker
        self.core = core_sleeve.parse(settings.core_sleeve)  # index ETFs held outside the stock logic (src.engine.core_sleeve)
        self.sessions = session_factory
        self.snapshot_fn = snapshot_fn
        self.broker_label = broker_label  # a human name for the broker, shown in the start-up banner
        self.risk = RiskEngine(settings.risk, fees=broker.fees if isinstance(broker, ChargesFees) else None)
        self.control = control
        self.universe_fn = universe_fn
        self.quote_fn = quote_fn
        self.market_context_fn = market_context_fn
        self.fundamentals_fn = fundamentals_fn
        self.house_calls_fn = house_calls_fn  # the broker's research calls (src.data.house_calls); None = not wired
        self.screen_verdict_fn = screen_verdict_fn  # the screener's verdict on one stock (recorded for research-only ones)
        self.pulse = pulse  # today's surging sectors (src.data.market_pulse); None = not wired
        self.context_fn = context_fn  # Angel One's circuit limits, spread and F&O data for a BUY; None = not wired
        self._hot: Dict[str, HotStock] = {}  # the day's surging stocks by symbol, read once per cycle
        self._announced: Optional[tuple] = None  # the (day, sectors) last told to the user
        self._pulse_bought: Optional[Set[str]] = None  # per cycle: stocks whose latest filled BUY came from the pulse
        self._calls: List[HouseCall] = []  # read once per cycle, so every stock in a cycle sees the same snapshot
        self._sources: Dict[str, str] = {}  # why each stock of this cycle is analysed (recorded with its decision)
        self._screened: Dict[str, str] = {}  # the screener's verdict on this cycle's research-only stocks
        self._warned: set = set()  # house warnings already sent about held stocks
        self._notify = notify
        self._halt_kind: Optional[str] = None
        self._llm_down = False
        self._cycle_llm: List[bool] = []
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

    def remembered_signals(self, symbol: str, snapshot: Snapshot) -> Optional[Dict[str, Optional[AgentSignal]]]:
        """The agents' answers already recorded for this stock on this bar, under the same prompt and features, or None.

        A completed bar does not change during the session, so neither should the call. Asking again (after a restart, or
        when the re-downloaded data differs in the 12th decimal) let a noisy model flip HOLD -> BUY and buy on the flip.
        A fail-safe HOLD from an AI outage is never reused: that stock is asked again next cycle."""
        if snapshot.bar_date is None:
            return None
        with self.sessions() as s:
            raw = s.scalar(
                select(DecisionRecord.signals_json)
                .where(
                    DecisionRecord.symbol == symbol,
                    DecisionRecord.bar_date == snapshot.bar_date,
                    DecisionRecord.prompt_version == VERSIONS["prompt"],
                    DecisionRecord.feature_version == VERSIONS["feature"],
                    DecisionRecord.signals_json.isnot(None),
                    self._house_matches(symbol),
                    self._pulse_matches(symbol),
                )
                .order_by(DecisionRecord.id.desc())
                .limit(1)
            )
        if not raw:
            return None
        try:
            stored = json.loads(raw)
            signals = {name: AgentSignal(**stored[name]) for name in self.agents}
        except (ValueError, TypeError, KeyError):  # an older row shape, or an agent added since: ask afresh
            return None
        if any(sig.degraded for sig in signals.values()):
            return None
        return dict(signals)

    def agent_context(self, symbol: str, snapshot: Snapshot) -> AgentContext:
        """What an agent is given for one stock. Market and company data are lazy: only an agent that asks pays for them."""
        return AgentContext(
            symbol,
            snapshot,
            market=lambda: self.market_context_fn(snapshot) if self.market_context_fn else None,
            company=lambda: self.fundamentals_fn(symbol) if self.fundamentals_fn else None,
            house=lambda: self.house_call(symbol),
            pulse=lambda: self._hot.get(symbol),
        )

    def house_call(self, symbol: str) -> Optional[HouseCall]:
        """The broker research call the AI is shown for this stock this cycle (see house_calls.call_for), or None."""
        return call_for(self._calls, symbol, datetime.now(timezone.utc)) if self._calls else None

    def _house_matches(self, symbol: str):
        """A remembered answer is reused only if it was given with the same research call (or with none): a call
        published mid-session makes the stock be asked again, once."""
        call = self.house_call(symbol)
        return DecisionRecord.house_call_json.is_(None) if call is None else DecisionRecord.house_call_json == call.to_json()

    def _pulse_matches(self, symbol: str):
        """Likewise for the market pulse: an answer given without it is not reused once the stock's sector is surging."""
        hot = self._hot.get(symbol)
        return DecisionRecord.pulse_json.is_(None) if hot is None else DecisionRecord.pulse_json == hot.to_json()

    def open_cycle(self) -> Portfolio:
        """Start a cycle: read the account, record its equity, apply the drawdown pause and check halts."""
        portfolio = self.broker.portfolio()
        self._record_equity(portfolio.equity)
        portfolio = self._apply_drawdown_pause(self._with_peak(portfolio))
        self._check_halt(portfolio)
        self._cycle_llm = []
        self._calls = self._read_house_calls()
        self._read_pulse()
        self._warn_on_held(portfolio)
        self._top_up_core(portfolio)
        return portfolio

    def _top_up_core(self, portfolio: Portfolio) -> None:
        """Buy the index ETFs of the core sleeve toward their share of equity (see src.engine.core_sleeve). Not while paused,
        in a dry run or with the market closed; a failure is logged and never stops the cycle."""
        weights = self.core
        if not weights or self.settings.dry_run or (self.control and self.control.is_paused()) or not self.broker.is_market_open():
            return
        try:
            prices = {s: self._quote(s, 0.0) for s in weights}
            for symbol, qty in core_sleeve.buys(weights, portfolio.positions, prices, portfolio.equity, portfolio.cash):
                fill = self.broker.buy_with_bracket(symbol, qty, prices[symbol], CORE_STOP_PCT, CORE_TARGET_PCT)
                self.notify(f"CORE sleeve: bought {qty} {symbol} @ ~{self.settings.currency}{fill.avg_price or prices[symbol]:,.2f}")
        except Exception:
            log.exception("core sleeve top-up failed; the cycle goes on without it")

    def _read_pulse(self) -> None:
        """Read today's surging stocks (computed once a day from the scan) and tell the user when the surging sectors
        change. Any failure means no pulse this cycle: the bot then runs as it did before the pulse existed."""
        self._pulse_bought = None
        self._hot = {}
        if self.pulse is None or not self.settings.market_pulse:
            return
        try:
            hot = self.pulse.hot()
        except Exception:
            log.exception("could not read the market pulse; this cycle runs without it")
            return
        self._hot = {h.symbol: h for h in hot}
        key = (datetime.now(timezone.utc).date(), tuple(sorted({h.sector for h in hot})))
        if hot and key != self._announced:
            self._announced = key
            self.notify("🔥 Market pulse: sectors surging today (liquid stocks up 8%+ in a month):\n" + pulse_summary(hot))

    def _entered_via_pulse(self, symbol: str) -> bool:
        """Whether a position in `symbol` was bought as a market-pulse candidate (its latest filled BUY came from a
        decision whose candidate_source is "pulse"): such a position is sold on a close below its 50-day average. One
        query a cycle answers for every stock."""
        if self._pulse_bought is None:
            with self.sessions() as s:
                rows = s.execute(
                    select(OrderRecord.symbol, DecisionRecord.candidate_source)
                    .join(DecisionRecord, OrderRecord.decision_id == DecisionRecord.id)
                    .where(OrderRecord.side == Action.BUY, OrderRecord.status == OrderStatus.FILLED)
                    .order_by(OrderRecord.id)
                )
                latest = dict(rows.all())  # ordered by id: the last BUY of each stock wins
            self._pulse_bought = {symbol for symbol, source in latest.items() if source == "pulse"}
        return symbol in self._pulse_bought

    def _read_house_calls(self) -> List[HouseCall]:
        if self.house_calls_fn is None:
            return []
        try:
            return list(self.house_calls_fn())
        except Exception:
            log.exception("could not read the house research calls; this cycle runs without them")
            return []

    def _warn_on_held(self, portfolio: Portfolio) -> None:
        """Tell the user, once per call, when the house turns against a stock the bot holds (an open SELL call, or its
        call on the stock hit its stop-loss). The bot's own stop is unchanged; the stock is analysed every cycle anyway,
        now with that call in its prompt."""
        if not self._calls:
            return
        now, sells = datetime.now(timezone.utc), open_sells(self._calls)
        for symbol in sorted(portfolio.positions):
            call = sells.get(symbol) or recently_ended(self._calls, symbol, now)
            if call is None or (symbol, call.published, call.is_open) in self._warned:
                continue
            self._warned.add((symbol, call.published, call.is_open))
            what = "an open SELL call" if call.is_open else f"its {call.kind} call ended: {call.exit_comment}"
            self.notify(f"HOUSE WARNING on held {symbol}: the broker's research desk has {what}. The AI sees this in this "
                        "cycle's analysis; the bot's own stop is unchanged.")  # fmt: skip

    def select_symbols(self, portfolio: Portfolio) -> List[str]:
        """Which stocks this cycle analyses (held positions always, so exits keep working)."""
        return list(self._symbols(portfolio))

    def decide(
        self,
        symbol: str,
        portfolio: Portfolio,
        snapshot: Snapshot,
        signals: Mapping[str, Optional[AgentSignal]],
        universe_size: Optional[int] = None,
    ) -> Verdict:
        """Turn the agents' signals into one recorded decision: the deterministic trend exit if it applies, else the agents
        combined by role; then size it with the risk engine and write the audit row."""
        held = portfolio.position_qty.get(symbol, 0)
        short = held > 0 and self._entered_via_pulse(symbol)
        decision = trend_exit_decision(snapshot, held, self.settings.trend_exit, short) or combine_signals(
            signals, {n: a.role for n, a in self.agents.items()}
        )
        if (  # RESEARCH_RULE_GATE: a stock only a broker research call listed needs the rule's BUY too
            self.settings.research_rule_gate
            and decision.action == Action.BUY
            and self._sources.get(symbol) == "research"
            and rule_verdict(snapshot.price, snapshot.ma50, snapshot.ma200, snapshot.rsi) != Action.BUY
        ):
            decision = dataclasses.replace(
                decision, action=Action.HOLD, reasoning=f"research-call gate: rule says HOLD. AI said: {decision.reasoning}"
            )
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
        """Held positions are always analysed (so exits work) even if the market scan fails. The stocks of open broker
        Swing/Momentum/Stellar Pick buy calls are added after the scan's own list. They skip the screener's filters but
        go through the same AI and risk engine; what the screener and the mechanical rule would have said is recorded with
        each decision (`screen_verdict`, `rule_action`), not acted on."""
        scanned = self._scanned(portfolio)
        research = swing_buys(self._calls)
        self._sources = {}
        found = "screener" if self.universe_fn is not None else "watchlist"
        for s in scanned:
            self._sources[s] = "held" if s in portfolio.positions else (f"{found}+research" if s in research else found)
            if s in OIL_GAS and self._sources[s] == found:
                self._sources[s] = f"{found}+oilgas"
        for s in research:
            self._sources.setdefault(s, "research")
        self._screened = {s: v for s in research if self._sources[s] == "research" and (v := self._screen(s, portfolio))}
        for h in self._pulse_picks(portfolio):
            self._sources.setdefault(h.symbol, "pulse")
        return [s for s in self._sources if s not in self.core]  # the core sleeve's ETFs are never analysed, so never sold

    def _pulse_picks(self, portfolio: Portfolio) -> List[HotStock]:
        """Stocks of surging sectors that the screener did not pick: they skip the screener's long-term filters and go
        through the same AI, risk engine and stop. At most MAX_PULSE_OPEN such positions are open at once."""
        if self.pulse is None or not self._hot:  # not wired, switched off, failed to read, or nothing surging
            return []
        open_now = sum(1 for s in portfolio.positions if self._entered_via_pulse(s))
        budget = portfolio.equity * self.settings.risk.min_position_pct  # one share must fit the smallest position
        return self.pulse.picks(budget, set(portfolio.positions) | set(self._sources), MAX_PULSE_OPEN - open_now)

    def _screen(self, symbol: str, portfolio: Portfolio) -> Optional[str]:
        """The screener's verdict on a research-only stock, or None (no screener, or it failed: this is a record only)."""
        if self.screen_verdict_fn is None:
            return None
        try:
            return self.screen_verdict_fn(symbol, portfolio.equity)
        except Exception as e:
            log.warning("no screen verdict for %s: %s: %s", symbol, type(e).__name__, e)
            return None

    def _scanned(self, portfolio: Portfolio) -> List[str]:
        if self.universe_fn is None:
            return list(self.settings.watchlist)
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
            self.notify(
                f"Drawdown pause over after {lim.drawdown_pause_days:g} days: the peak is rebased to equity "
                f"{self.settings.currency}{p.equity:,.0f} and buying resumes."
            )
            return self._with_peak(p)
        return p

    def _log_decision(
        self,
        symbol: str,
        snap: Snapshot,
        signals: Mapping[str, Optional[AgentSignal]],
        decision: Decision,
        risk: RiskDecision,
        price: float,
        universe_size: Optional[int] = None,
    ) -> int:
        tech = signals.get("technical")
        with self.sessions() as s:
            record = DecisionRecord(
                symbol=symbol,
                price=price,
                technical_action=tech.action if tech else None,
                technical_confidence=tech.confidence if tech else None,
                final_action=decision.action,
                final_confidence=decision.confidence,
                reasoning=decision.reasoning,
                risk_approved=risk.approved,
                risk_quantity=risk.quantity,
                risk_reason=risk.reason,
                signals_json=json.dumps({n: sig.to_json_dict() if sig else None for n, sig in signals.items()}),
                snapshot_json=json.dumps(dataclasses.asdict(snap)),
                rule_action=rule_verdict(snap.price, snap.ma50, snap.ma200, snap.rsi),
                **details_columns(tech.details if tech is not None else None),
                prompt_version=VERSIONS["prompt"],
                feature_version=VERSIONS["feature"],
                strategy_version=VERSIONS["strategy"],
                decision_source=decision.source,
                bar_date=snap.bar_date,
                universe_size=universe_size,
                candidate_source=self._sources.get(symbol),
                screen_verdict=self._screened.get(symbol),
                house_call_json=call.to_json() if (call := self.house_call(symbol)) is not None else None,
                pulse_json=hot.to_json() if (hot := self._hot.get(symbol)) is not None else None,
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
        """True when `symbol` was bought, or stopped out, within the cooldown (STRATEGY.md: a stop-out is not re-bought for 24
        hours). The broker closes a stop on its own, so a stop-out is read from its trade history, not from our orders."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.settings.rebuy_cooldown_hours)
        if isinstance(self.broker, ListsTrades):
            stopped = (t for t in self.broker.trade_history(limit=STOP_LOOKBACK) if t["closed_at"] >= cutoff)
            if any(t["symbol"] == symbol and t["reason"] == "STOP" for t in stopped):
                return True
        with self.sessions() as s:
            recent = s.scalar(
                select(OrderRecord.id)
                .where(
                    OrderRecord.symbol == symbol,
                    OrderRecord.side == Action.BUY,
                    OrderRecord.status.not_in(NOT_A_PLACED_BUY),
                    OrderRecord.created_at >= cutoff,
                )
                .limit(1)
            )
        return recent is not None

    def _place(self, decision_id: int, symbol: str, action: Action, qty: int, price: float) -> str:
        filled_qty: Optional[int] = None
        fill_price: Optional[float] = None
        status: str
        broker_id: Optional[str]
        error: Optional[str]
        context = EntryContext()
        if self.control and self.control.is_paused():
            status, broker_id, error = OrderStatus.PAUSED, None, None
        elif action == Action.BUY and self._in_cooldown(symbol):
            status, broker_id, error = OrderStatus.SKIPPED_COOLDOWN, None, None
        elif self.settings.dry_run:
            status, broker_id, error = OrderStatus.DRY_RUN, None, None
        elif not self.broker.is_market_open():
            status, broker_id, error = OrderStatus.MARKET_CLOSED, None, None
        else:
            if action == Action.BUY:
                context = self._buy_context(symbol, price)
            if context.skip:
                status, broker_id, error = OrderStatus.SKIPPED_CONTEXT, None, context.skip
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
                    context_json=json.dumps(context.note) if context.note else None,
                )
            )
            s.commit()
        if status not in SILENT_STATUSES:
            detail = f" ({error})" if error else ""
            self.notify(f"{action} {qty} {symbol} @ ~{self.settings.currency}{price:,.2f}: {status}{detail}")
        if filled_qty is not None and filled_qty != qty:
            self.notify(
                f"PARTIAL/UNFILLED {action} {symbol}: ordered {qty}, broker filled {filled_qty} ({status}). Check the position."
            )
        return status

    def _buy_context(self, symbol: str, price: float) -> EntryContext:
        """Angel One's checks for a BUY; with none wired, or an error from it, the order goes ahead without them."""
        if self.context_fn is None:
            return EntryContext()
        try:
            return self.context_fn(symbol, price)
        except Exception as e:
            log.warning("Angel One quote failed for %s (%s: %s); ordering without it", symbol, type(e).__name__, e)
            return EntryContext()

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
            return fill.status, fill.broker_order_id, None, fill.filled_qty, fill.avg_price
        except Exception as e:
            log.exception("order submission failed for %s", symbol)
            return OrderStatus.FAILED, None, f"{type(e).__name__}: {e}", None, None
