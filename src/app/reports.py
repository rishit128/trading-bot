"""Read-only command-line reports: scanner candidates, workflow graphs, account summary, positions and trade history."""
from types import SimpleNamespace
from typing import Optional

from src.app.kit import build_india_screener, make_paper_broker
from src.config import Settings
from src.data.india import IST
from src.workflow import build_analysis_graph, build_cycle_graph, build_decision_graph


def print_graphs(settings: Settings) -> None:
    """Print the three LangGraph workflows as Mermaid diagrams."""
    names = ["technical"]
    stub = SimpleNamespace(agents={n: None for n in names})
    analysis, decision = build_analysis_graph(stub), build_decision_graph(stub)
    for title, graph in (("CYCLE: scan -> analyse all stocks in parallel -> decide each in order", build_cycle_graph(stub, analysis, decision)),
                         (f"ANALYSIS (per stock): agents {names} run in parallel", analysis),
                         ("DECISION (per stock): combine -> risk -> execute", decision)):
        print(f"%% {title}")
        print(graph.get_graph().draw_mermaid())


def print_candidates(settings: Settings) -> None:
    """Print today's scanner candidates."""
    screener = build_india_screener(settings)
    cur = settings.currency
    for c in screener.candidates():
        print(f"{c.symbol:12s} price={cur}{c.price:10,.2f} 12-1m={(c.momentum_12_1 or 0):+.0%} 3m={c.momentum_63d:+.1%} vol/day={c.daily_volatility:.1%} "
              f"rsi={c.rsi:4.1f} traded/day={cur}{c.avg_traded_value / 1e6:9,.1f}M")


def print_report(settings: Settings, database_url: Optional[str] = None) -> None:
    """Print a paper account's summary (default: the swing account; pass a database URL for another account)."""
    s = make_paper_broker(settings, database_url).summary()
    cur = settings.currency
    win = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "n/a"
    print(f"Paper account ({settings.market.upper()}): started {cur}{s['initial_cash']:,.0f} -> equity {cur}{s['equity']:,.0f} "
          f"({s['return_pct']:+.2%})\ncash {cur}{s['cash']:,.0f} | open positions {s['open_positions']} | closed trades "
          f"{s['closed_trades']} (win rate {win}) | realized net P&L {cur}{s['realized_net_pnl']:,.0f} | fees paid "
          f"{cur}{s['fees_paid']:,.0f}\nexits: {s['exits']}")


def print_positions(settings: Settings, database_url: Optional[str] = None) -> None:
    """Print every open position (profit/loss, buy date), the closed-trade history and the overall account status."""
    broker = make_paper_broker(settings, database_url)
    cur = settings.currency
    day = lambda dt: dt.astimezone(IST).strftime("%d-%b-%y")  # noqa: E731
    held = broker.holdings()
    print(f"OPEN POSITIONS ({len(held)}), best to worst")
    print(f"{'symbol':12s} {'bought':9s} {'qty':>5s} {'buy':>9s} {'now':>9s} {'P&L':>10s} {'P&L%':>7s} {'stop':>9s}")
    for h in held:
        print(f"{h['symbol']:12s} {day(h['opened_at']):9s} {h['qty']:5d} {h['avg_price']:9,.2f} {h['price']:9,.2f} "
              f"{h['pnl']:+10,.0f} {h['pnl_pct']:+7.1%} {h['stop']:9,.2f}")
    if held:
        winners = sum(1 for h in held if h["pnl"] > 0)
        print(f"{'':12s} {winners} in profit, {len(held) - winners} in loss, unrealised P&L {cur}{sum(h['pnl'] for h in held):+,.0f}")
    trades = broker.trade_history()
    print(f"\nCLOSED TRADES ({len(trades)}), newest first")
    if trades:
        print(f"{'symbol':12s} {'bought':9s} {'sold':9s} {'qty':>5s} {'buy':>9s} {'sell':>9s} {'net P&L':>10s} {'why':6s}")
    for t in trades:
        print(f"{t['symbol']:12s} {day(t['opened_at']):9s} {day(t['closed_at']):9s} {t['qty']:5d} {t['entry_price']:9,.2f} "
              f"{t['exit_price']:9,.2f} {t['net_pnl']:+10,.0f} {t['reason']:6s}")
    print("\nOVERALL")
    print_report(settings, database_url)
