"""The Telegram report: readable, formatted messages built section by section.

    1 summary      value, cost, unrealised and booked P&L, dividends, cash, market-cap mix, best and worst
    2 holdings     every stock as its own card: shares, average buy price, price now, cost -> value, P&L, weight, history
    3 analysis     gainers, weakest, concentration, sectors, market caps, flags
    4 income/tax   dividends (last 12 months), sales booked this year, short-term lots and when they turn long-term,
                   recent transactions
    5 AI note      (explains only)
    6 the run      pages and tabs read, what the guard stopped, logout, time taken, where the files are, what to do next

Every number is read from the portal or is plain arithmetic on those numbers. Sections use light Telegram HTML (bold
headers, bold key figures) for readability; every piece of text that came from the portal, the portal's data, or the AI
(company names, sector names, error text, the AI note) is escaped with `esc()` first, so a stray "&", "<" or ">" in a
company name (e.g. "Balmer Lawrie & Company Ltd") can never break the formatting or the message. Bold tags are always
opened and closed on the same line, so splitting a long section across several Telegram messages (`split_message`)
can never cut a tag in half.

`sections()` returns the HTML version, for Telegram. `format_report()` (in graph.py) strips the tags for the plain-text
copy saved to disk."""
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from html import escape as esc, unescape
from typing import Dict, List, Optional

from src.portfolio.holdings import inr, nice
from src.portfolio.session import PAGE_NAMES
from src.portfolio.statements import Statements, name_key, parse_date

LONG_TERM_SOON_DAYS = 60
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
COST_NOTE = ("ℹ️ Returns use the cost the portal records. After a demerger or merger (e.g. Reliance → Jio Financial, "
             "Tata Motors' split) that cost may be divided between the companies differently from your records.")
TAX_NOTE = ("ℹ️ Listed shares: gains on shares held up to 1 year are short-term (taxed 20%), after 1 year long-term "
            "(12.5% above ₹1.25 lakh a year). Rates from 23 Jul 2024; confirm with your CA.")
FLAGS_NOTE = "ℹ️ Flags are informational only. This agent never recommends or places a trade — the decision is yours."
ADVICE = (  # (words in the error, what the user should do next)
    ("MPIN step failed on", "Run <code>python -m src.portfolio setup</code> on the PC with the correct MPIN — runs "
                            "stay blocked until then."),
    ("login stopped at mpin", "The portal did not accept the MPIN. Check it, then run setup again on the PC. Runs "
                              "are blocked until then so a wrong MPIN is never tried twice."),
    ("Invalid OTP", "Send /portfolio again and type the new OTP carefully (the old one is used up)."),
    ("no OTP", "Send /portfolio again and reply with the 4-digit OTP within 3 minutes."),
    ("entry", "The portal did not accept the typed entry (its page may have changed). Tell Claude; nothing was sent."),
    ("customer IDs are linked", "Run setup on the PC and enter the customer ID to use."),
    ("timed out", "The portal was slow or down. Try /portfolio again later."),
    ("another portfolio run", "A run is already going — wait for its report."),
    ("unexpected", "Something unexpected happened; details are in the bot window on the PC. Tell Claude."),
)


def strip_tags(text: str) -> str:
    """The plain-text rendering of an HTML-tagged section (tags removed, entities like '&amp;' decoded back to '&'),
    for the copy saved to disk and for tests that check plain substrings."""
    return unescape(_TAG.sub("", text))


def signed(amount: float) -> str:
    return ("+" if amount >= 0 else "") + inr(amount)


def marker(amount: float) -> str:
    return "🟢" if amount > 0 else "🔴" if amount < 0 else "⚪"


def advice(error: str) -> str:
    return next((todo for words, todo in ADVICE if words in error), "Try again later; if it repeats, tell Claude.")


def heading(text: str) -> str:
    return f"<b>{text}</b>"


def sections(state: dict, today: Optional[date] = None) -> List[str]:
    """The report as a list of HTML-formatted messages, in the order they are sent."""
    today = today or date.today()
    logout = logout_line(state.get("logged_out"))
    if state.get("error"):
        lines = [heading("❌ PORTFOLIO RUN STOPPED"), "", f"<b>Why:</b> {esc(state['error'])}",
                 f"👉 <b>What to do next:</b> {advice(state['error'])}", "", *run_footer(state, logout, next_line=False)]
        return ["\n".join(lines)]
    a, st = state.get("analysis"), state.get("statements") or Statements()
    divs = dividend_view(st, today)
    out = [summary(state, a, st, divs)]
    if a:
        out += [holdings_section(a, divs["by_stock"]), analysis_section(a)]
        income = income_section(st, divs)
        if income:
            out.append(income)
        if state.get("explanation"):
            out.append(heading("🤖 AI NOTE") + "  <i>(explains only — never advises a trade)</i>\n\n"
                       + esc(state["explanation"]))
    out.append("\n".join(run_footer(state, logout)))
    return out


def logout_line(logged_out: Optional[bool]) -> str:
    return {True: "✅ Logged out safely.",
            False: "⚠️ <b>LOGOUT NOT CONFIRMED</b> — please log out on the Integrated portal yourself.",
            None: "🔒 No portal session was left open."}[logged_out]


def summary(state: dict, a: Optional[dict], st: Statements, divs: dict) -> str:
    t, dash = st.totals, state.get("totals") or {}
    value, cost = t.get("value", dash.get("value")), t.get("cost", dash.get("invested"))
    lines = [heading("📊 PORTFOLIO SUMMARY") + "  <i>(read-only)</i>"]
    dated = next((h.price_date for h in st.holdings if h.price_date), "")
    if dated:
        lines.append(f"🕒 Prices as of {esc(dated)}")
    lines.append("")
    if value is not None:
        lines.append(f"💰 Value: <b>{inr(value)}</b>")
    if cost:
        lines.append(f"💵 Invested (cost): {inr(cost)}")
    unreal = t.get("unrealised", dash.get("gain"))
    if unreal is not None:
        pct = f" ({unreal / cost:+.1%})" if cost else ""
        lines.append(f"📈 Unrealised P&amp;L: <b>{signed(unreal)}{pct}</b>")
    if t.get("realised") is not None:
        lines.append(f"✅ Booked P&amp;L (past sales): {signed(t['realised'])}")
    if divs["count"]:
        lines.append(f"🎁 Dividends (last 12 months): {inr(divs['net'])} net · {divs['count']} payouts")
    if "margin" in dash:
        lines.append(f"🏦 Cash / margin: {inr(dash['margin'])}")
    if a:
        lines.append("")
        caps = " · ".join(f"{esc(c['cap'])} {c['weight']:.0%}" for c in a.get("caps", []))
        lines.append(f"📦 {a['holdings']} stocks" + (f" · {caps}" if caps else ""))
        ranked = [p for p in a["positions"] if p["pnl_pct"] is not None]
        if ranked:
            best, worst = max(ranked, key=lambda p: p["pnl_pct"]), min(ranked, key=lambda p: p["pnl_pct"])
            lines.append(f"🏆 Best: <b>{esc(best['symbol'])}</b> {best['pnl_pct']:+.0%}")
            lines.append(f"📉 Weakest: <b>{esc(worst['symbol'])}</b> {worst['pnl_pct']:+.0%}")
        lines.append(f"⚠️ Flags: <b>{len(a['flags'])}</b> (see Analysis message below)" if a["flags"]
                     else "✅ No risk flags")
    if state.get("warnings"):
        lines.append("")
        lines += [esc(w) for w in state["warnings"]]
    if not a:
        lines += ["", "📦 <b>Holdings:</b> no per-stock table I can read yet (the pages are saved on the PC)."]
    return "\n".join(lines)


def holdings_section(a: dict, dividends_by_stock: Dict[str, float]) -> str:
    lines = [heading("📦 HOLDINGS") + f" — {a['holdings']} stocks, largest first"]
    for n, p in enumerate(a["positions"], start=1):
        lines.append("")
        pnl = None if p["pnl_pct"] is None else p["value"] - p["invested"]
        tag = f"  {marker(pnl)} {p['pnl_pct']:+.1%} ({signed(pnl)})" if pnl is not None else ""
        lines.append(f"<b>{n}. {esc(p['symbol'])}</b>{tag}")
        tags = " · ".join(x for x in (p.get("cap"), nice(p["sector"]) if p.get("sector") else "") if x)
        if tags:
            lines.append(f"   {esc(tags)}")
        price = [f"{p['qty']:,.0f} sh" if p.get("qty") else "",
                 f"avg ₹{p['avg']:,.2f}" if p.get("avg") else "", f"now ₹{p['ltp']:,.2f}" if p.get("ltp") else ""]
        if any(price):
            lines.append("   " + " · ".join(x for x in price if x))
        if p.get("invested") and p.get("cost_recorded", True):
            lines.append(f"   {inr(p['invested'])} → {inr(p['value'])} · {p['weight']:.1%} of portfolio")
        elif p.get("invested") is not None:
            lines.append(f"   value {inr(p['value'])} · {p['weight']:.1%} of portfolio")
            lines.append("   ⚠️ cost not recorded (e.g. received through a demerger or bonus)")
        else:
            lines.append(f"   value {inr(p['value'])} · {p['weight']:.1%} of portfolio")
        history = []
        if p.get("bought_qty"):
            history.append(f"bought {p['bought_qty']:,.0f}" + (f", sold {p['sold_qty']:,.0f}" if p.get("sold_qty") else ""))
        if p.get("realized"):
            history.append(f"booked {signed(p['realized'])}")
        if dividends_by_stock.get(name_key(p["symbol"])):
            history.append(f"dividends {inr(dividends_by_stock[name_key(p['symbol'])])} (12 m)")
        if history:
            lines.append("   📜 " + " · ".join(history))
    return "\n".join(lines)


def analysis_section(a: dict) -> str:
    ranked = sorted((p for p in a["positions"] if p["pnl_pct"] is not None), key=lambda p: p["pnl_pct"], reverse=True)
    lines = [heading("🔎 ANALYSIS"), ""]
    if ranked:
        lines.append("🏆 <b>Top gainers</b> (return on cost)")
        lines.append("   " + ", ".join(f"{esc(p['symbol'])} {p['pnl_pct']:+.0%}" for p in ranked[:5]))
        lines.append("📉 <b>Weakest</b>")
        lines.append("   " + ", ".join(f"{esc(p['symbol'])} {p['pnl_pct']:+.0%}" for p in ranked[::-1][:5]))
        losers = [p for p in ranked if p["pnl_pct"] < 0]
        unknown = len(a["positions"]) - len(ranked)
        lines.append(f"   In profit: {len(ranked) - len(losers)} · In loss: {len(losers)}"
                     + (f" · Cost not recorded: {unknown}" if unknown else ""))
        lines.append("")
    lines.append("🧭 <b>Concentration</b>")
    lines.append(f"   Largest holding: {a['largest_weight']:.1%} · Top 5 holdings: {a['top5_weight']:.1%}")
    lines.append(f"   Spread: like {a['effective_holdings']} equal holdings")
    if a.get("caps"):
        lines += ["", "📐 <b>Market cap mix</b>", "   " + " · ".join(f"{esc(c['cap'])} {c['weight']:.1%}"
                                                                    for c in a["caps"])]
    if a.get("sectors"):
        lines += ["", f"🏭 <b>Sectors</b> ({len(a['sectors'])})",
                  "   " + ", ".join(f"{esc(nice(s['sector']))} {s['weight']:.1%}" for s in a["sectors"])]
    lines.append("")
    if a["flags"]:
        lines.append(f"⚠️ <b>FLAGS</b> ({len(a['flags'])})")
        lines += [f"   • {esc(f)}" for f in a["flags"]]
    else:
        lines.append("✅ <b>No flags</b> — no stock above 20%, no sector above 30%, top five below 60%, "
                     "none 25%+ below cost")
    lines += ["", FLAGS_NOTE, COST_NOTE]
    return "\n".join(lines)


def dividend_view(st: Statements, today: date) -> dict:
    """Dividends with a record date in the last 12 months: total, count, TDS, per stock, newest first."""
    since = today - timedelta(days=365)
    recent = [d for d in st.dividends if (when := parse_date(d.record_date)) and since <= when <= today]
    by_stock: Dict[str, float] = defaultdict(float)
    for d in recent:
        by_stock[name_key(d.company)] += d.net
    recent.sort(key=lambda d: parse_date(d.record_date) or today, reverse=True)
    return {"recent": recent, "count": len(recent), "net": sum(d.net for d in recent),
            "tds": sum(d.tds for d in recent), "by_stock": dict(by_stock)}


def income_section(st: Statements, divs: dict) -> str:
    lines: List[str] = [heading("💼 INCOME &amp; TAX")]
    if divs["count"]:
        lines += ["", f"🎁 <b>DIVIDENDS</b>, last 12 months: {inr(divs['net'])} net · {divs['count']} payouts"
                     + (f" (TDS {inr(divs['tds'])})" if divs["tds"] else "")]
        lines += [f"   • {esc(d.record_date)} {esc(d.company)}: {d.qty:,.0f} × ₹{d.per_share:,.2f} = {inr(d.net)}"
                  for d in divs["recent"]]
    if st.sales:
        moved = {name_key(t.company) for t in st.trades if t.corporate_action}
        total = sum(s.gain for s in st.sales)
        lines += ["", f"🧾 <b>SOLD THIS FINANCIAL YEAR</b> ({esc(st.financial_year)}): {len(st.sales)} sale(s), "
                     f"{signed(total)}"]
        lines += [f"   • {esc(s.company)}: {s.qty:,.0f} sh · bought {inr(s.bought_value)} → sold {inr(s.sold_value)} "
                  f"· {signed(s.gain)}"
                  + (" <i>(merger/scheme conversion, not a market sale)</i>" if name_key(s.company) in moved else "")
                  for s in st.sales]
    if st.lots:
        lines += ["", "⏳ <b>SHORT-TERM LOTS</b> (bought within a year)"]
        for lot in sorted(st.lots, key=lambda x: (x.days_to_long_term is None, x.days_to_long_term or 0)):
            when = (f"long-term in {lot.days_to_long_term} days" if lot.days_to_long_term is not None else "")
            soon = " ⏰" if lot.days_to_long_term is not None and lot.days_to_long_term <= LONG_TERM_SOON_DAYS else ""
            lines.append(f"   • {esc(lot.company)}: {lot.qty:,.0f} sh bought {esc(lot.bought_on)} @ ₹{lot.rate:,.2f}"
                         + (f" → {when}{soon}" if when else ""))
        lines += ["", TAX_NOTE]
    if st.trades:
        lines += ["", "🔁 <b>RECENT TRANSACTIONS</b>"]
        for t in sorted(st.trades, key=lambda t: parse_date(t.on) or date.min, reverse=True):
            if t.corporate_action:
                way = "received" if t.side == "Buy" else "transferred out"
                lines.append(f"   • {esc(t.on)} 🔄 {esc(t.company)}: {t.qty:,.0f} sh {way} "
                             f"<i>({esc(t.note)}; corporate action)</i>")
            else:
                lines.append(f"   • {esc(t.on)} {t.side} {esc(t.company)}: {t.qty:,.0f} @ ₹{t.rate:,.2f}")
    return "\n".join(lines) if len(lines) > 1 else ""


def run_footer(state: dict, logout: str, next_line: bool = True) -> List[str]:
    """What was read (pages, tabs), what the guard blocked, the logout, time taken, where the files are, and — always
    — what (if anything) the user needs to do next."""
    out = [heading("⚙️ RUN DETAILS")]
    pages = state.get("pages") or {}
    if pages:
        out += ["", "📄 <b>Pages read</b>"]
        for name, p in pages.items():
            title = esc(PAGE_NAMES.get(name, name))
            if not p.get("reached"):
                out.append(f"   • {title}: could not open")
                continue
            tabs = p.get("tabs") or {}
            read = [esc(label) for label, tab in tabs.items() if tab.get("status") == "read"]
            missed = [f"{esc(label)} ({esc(str(tab.get('status')))})" for label, tab in tabs.items()
                      if tab.get("status") != "read"]
            line = f"   • {title}: read" + (f" + tabs: {', '.join(read)}" if read else "")
            out.append(line + (f"; not read: {', '.join(missed)}" if missed else ""))
    blocked = state.get("blocked_list") or []
    out.append("")
    if blocked:
        out.append(f"🛡️ <b>Guard</b> stopped {len(blocked)} request(s) the portal pages fired on their own "
                   "(nothing you need):")
        out += [f"   • {esc(b)}" + (f" (×{n})" if n > 1 else "") for b, n in Counter(blocked).items()]
    else:
        out.append("🛡️ <b>Guard:</b> nothing needed blocking.")
    out.append(logout)
    extra = [f"took {state['seconds']:.0f}s" if state.get("seconds") else "",
             f"files: <code>{esc(state['run_dir'])}</code>" if state.get("run_dir") else ""]
    if any(extra):
        out.append("⏱️ " + " · ".join(x for x in extra if x))
    if next_line:
        out += ["", next_action(state)]
    return out


def next_action(state: dict) -> str:
    """The one line every successful run ends with: what, if anything, the user needs to do now."""
    if state.get("logged_out") is False:
        return "👉 <b>Next:</b> log out on the Integrated portal yourself, then you're done."
    return "👉 <b>Next:</b> nothing needed. Send /portfolio again anytime for a fresh read (e.g. after you buy or sell)."


def split_message(text: str, limit: int = 3800) -> List[str]:
    """Telegram cuts messages at 4096 characters: split on line breaks into parts that fit, numbered (1/3), ...
    Splits only ever fall on a line break, and every line's tags open and close on that same line, so a split can
    never leave a Telegram HTML tag half-open in one message."""
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single overlong line: hard-split it
            parts.append((current + "\n" if current else "") + line[:limit])
            current, line = "", line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts if len(parts) == 1 else [f"<i>({i}/{len(parts)})</i>\n{p}" for i, p in enumerate(parts, start=1)]
