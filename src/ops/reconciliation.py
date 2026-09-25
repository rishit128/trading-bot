"""Paper-level broker reconciliation: the database ledger must tie back to the paper account line by line.

Every decision that was approved must have an order record, every filled order must have landed in a position or a
closed trade, and the cash the account reports must be rebuildable from the recorded fills. The script
scripts/reconcile_paper.py prints the report and exits nonzero on any finding; tests pin the catch-the-drift case."""
from typing import Dict, List, Optional

from sqlalchemy import select

from src.database import DecisionRecord, OrderRecord, PaperAccountRecord, PaperPositionRecord, PaperTradeRecord
from src.engine.enums import Action, OrderStatus
from src.engine.costs import india_delivery_fees

TOLERANCE = 0.01  # rupees; fees and STT round into sub-paisa noise


def _filled_qty(order) -> int:
    """Quantity the broker actually filled, from the order's own fill fields."""
    if order.filled_qty is not None:
        return int(order.filled_qty)
    return int(order.quantity) if order.status in (OrderStatus.FILLED, OrderStatus.CONFIRMED) else 0


def reconcile_paper(sessions, fees=None, cash: Optional[float] = None) -> dict:
    """Reconcile the paper account, orders, positions and trades in `sessions`. Returns {ok, findings, ...}; `cash`
    overrides the recorded cash when called from a test that already changed it."""
    fees = fees or india_delivery_fees
    with sessions() as s:
        acct = s.get(PaperAccountRecord, 1)
        positions = s.scalars(select(PaperPositionRecord)).all()
        trades = s.scalars(select(PaperTradeRecord)).all()
        orders = s.scalars(select(OrderRecord)).all()
        decisions = s.scalars(select(DecisionRecord)).all()
    findings: List[str] = []

    if acct is None:
        findings.append("no paper account row (id=1)")
        return {"ok": False, "findings": findings}

    # ---- 1. cash: rebuild what the account should hold from the fills, position by position -------------------
    spent = sum(t.qty * t.entry_price + t.fees for t in trades) + sum(
        p.qty * p.avg_price + fees(Action.BUY, p.qty * p.avg_price) for p in positions)
    received = sum(t.qty * t.exit_price for t in trades)
    expected = acct.initial_cash - spent + received
    recorded = cash if cash is not None else acct.cash
    if abs(expected - recorded) > TOLERANCE:
        findings.append(f"cash does not tie: rebuilt {expected:,.2f} vs recorded {recorded:,.2f}")

    # ---- 2. orders: every approved trade decision must have produced an order, and every filled order a fill ---- 
    buy_orders: Dict[str, int] = {}
    sell_orders: Dict[str, int] = {}
    for o in orders:
        n = _filled_qty(o)
        if not n or o.side not in (Action.BUY, Action.SELL):
            continue
        if o.side == Action.BUY:
            buy_orders[o.symbol] = buy_orders.get(o.symbol, 0) + n
        else:
            sell_orders[o.symbol] = sell_orders.get(o.symbol, 0) + n
    buy_fills: Dict[str, int] = {}
    sell_fills: Dict[str, int] = {}
    for t in trades:
        buy_fills[t.symbol] = buy_fills.get(t.symbol, 0) + t.qty
        sell_fills[t.symbol] = sell_fills.get(t.symbol, 0) + t.qty
    for p in positions:
        buy_fills[p.symbol] = buy_fills.get(p.symbol, 0) + p.qty
    for sym, n in (list(sell_orders.items())):
        if sell_fills.get(sym, 0) != n:
            findings.append(f"{sym}: filled SELL orders total {n} but trades only show {sell_fills.get(sym, 0)}")
    for sym, n in (list(buy_orders.items())):
        if buy_fills.get(sym, 0) != n:
            findings.append(f"{sym}: filled BUY orders total {n} but positions+trades only show {buy_fills.get(sym, 0)}")

    traded = sum(1 for d in decisions if d.risk_approved and d.final_action in (Action.BUY, Action.SELL))
    ordered_ids = {o.decision_id for o in orders if o.status not in ("rejected", "cancelled")}
    ordered = sum(1 for d in decisions if d.id in ordered_ids)
    if ordered != traded:
        findings.append(f"decision/order mismatch: {traded} approved trade decisions but {ordered} order records")

    # ---- 3. positions sanity -----------------------------------------------------------------------------------
    for p in positions:
        if p.qty <= 0:
            findings.append(f"{p.symbol}: position quantity {p.qty} is not positive")
        if p.avg_price <= 0 or not (p.stop < p.avg_price < p.target):
            findings.append(f"{p.symbol}: bracket out of order (stop {p.stop}, entry {p.avg_price}, target {p.target})")

    return {"ok": not findings, "findings": findings, "expected_cash": expected, "cash": recorded,
            "positions": len(positions), "closed_trades": len(trades),
            "fees_paid": sum(t.fees for t in trades)}


def reconcile_report(sessions) -> str:
    """A human-readable report for scripts/reconcile_paper.py; raises SystemExit(1) with it when anything fails."""
    r = reconcile_paper(sessions)
    lines = [
        f"paper reconciliation: {'PASS' if r['ok'] else 'FAIL'}",
        f"  rebuilt cash {r['expected_cash']:,.2f} vs recorded {r['cash']:,.2f}",
        f"  open positions {r['positions']}, closed trades {r['closed_trades']}, fees paid {r['fees_paid']:,.2f}",
    ]
    for f in r["findings"]:
        lines.append(f"  FINDING: {f}")
    report = "\n".join(lines)
    if not r["ok"]:
        raise SystemExit(report)
    return report