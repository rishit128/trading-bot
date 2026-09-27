"""Deterministic reading and analysis of the holdings the portal showed. No AI here: numbers come from the portal.

The primary source is the Portfolio Analyzer's holdings statement (`from_portfolio_analyzer`, a known layout). As a
fallback, `from_tables` / `from_json` look for a holdings table by its column headings; anything found that way must add
up to the portal's own total value (`plausible`), or it is rejected rather than reported."""
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

COLUMNS = {  # field -> heading words, most specific first
    "symbol": ("symbol", "scrip", "stock", "security", "instrument", "company", "name"),
    "qty": ("net qty", "holding qty", "quantity", "qty", "shares", "units"),
    "avg": ("avg", "average", "buy price", "cost price", "purchase price"),
    "ltp": ("ltp", "cmp", "last price", "current price", "market price", "close"),
    "invested": ("invested", "investment", "buy value", "cost value"),  # before "value", which would take "Buy Value"
    "value": ("current value", "market value", "present value", "cur. val", "value"),
    "pnl": ("unrealised", "unrealized", "p&l", "pnl", "p/l", "gain", "profit"),
}
NOT_TOTAL_PNL = re.compile(r"day", re.I)  # "Day's Gain" / "Today's P&L" is not the holding's total profit/loss
SKIP_ROWS = re.compile(r"^(total|grand total|net|summary)\b", re.I)
CONCENTRATED = 0.20      # one holding above 20% of the portfolio
SECTOR_HEAVY = 0.30      # one sector above 30%
TOP5_HEAVY = 0.60        # the five largest above 60%
DEEP_LOSS = -0.25        # a holding 25% or more below its average cost
MIN_COST_SHARE = 0.005   # a recorded cost under 0.5% of the value is treated as "no cost recorded" (demerger/bonus)


@dataclass(frozen=True)
class Holding:
    symbol: str
    qty: Optional[float]
    avg: Optional[float]
    ltp: Optional[float]
    value: float
    invested: Optional[float]
    sector: str = ""
    cap: str = ""                          # Large / Mid / Small Cap, as the portal classes it
    realized: Optional[float] = None       # profit already booked on this stock's past sales
    bought_qty: Optional[float] = None     # all shares ever bought (history)
    sold_qty: Optional[float] = None       # all shares ever sold (history)
    price_date: str = ""                   # the date of `ltp`

    @property
    def cost_recorded(self) -> bool:
        """False when the portal holds (almost) no cost for the shares, as for shares received through a demerger or
        bonus (Jio Financial from Reliance shows a cost of a few paise): a return on that cost is meaningless."""
        return self.invested is not None and self.invested > 0 and self.invested >= MIN_COST_SHARE * self.value

    @property
    def pnl_pct(self) -> Optional[float]:
        return self.value / self.invested - 1 if self.cost_recorded and self.invested else None


def number(text) -> Optional[float]:
    """'₹1,23,456.50' -> 123456.5; '(1,200)' and '-1,200' -> -1200; blanks and words -> None."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    s = str(text or "").strip()
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^0-9.\-]", "", s)
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    value = float(s)
    return -abs(value) if negative else value


def _column_map(headings: Sequence[str]) -> Dict[str, int]:
    """field -> column index, matching each heading to at most one field (the most specific keyword wins)."""
    found: Dict[str, int] = {}
    lowered = [re.sub(r"\s+", " ", str(h)).strip().lower() for h in headings]
    for field, words in COLUMNS.items():
        for word in words:
            hit = next((i for i, h in enumerate(lowered) if word in h and i not in found.values()
                        and not (field == "pnl" and NOT_TOTAL_PNL.search(h))), None)
            if hit is not None:
                found[field] = hit
                break
    return found


def _holding(get) -> Optional[Holding]:
    symbol = str(get("symbol") or "").strip().split("\n")[0].strip()
    if not symbol or SKIP_ROWS.match(symbol):
        return None
    qty, avg, ltp = number(get("qty")), number(get("avg")), number(get("ltp"))
    value, invested = number(get("value")), number(get("invested"))
    if value is None and qty is not None and ltp is not None:
        value = qty * ltp
    if invested is None and qty is not None and avg is not None:
        invested = qty * avg
    if invested is None and value is not None and (pnl := number(get("pnl"))) is not None:
        invested = value - pnl
    if value is None or value <= 0:
        return None
    return Holding(symbol, qty, avg, ltp, value, invested)


def from_rows(headings: Sequence[str], rows: Iterable[Sequence]) -> List[Holding]:
    cols = _column_map(headings)
    if "symbol" not in cols or not ({"qty", "value"} & cols.keys()):
        return []
    out = []
    for row in rows:
        h = _holding(lambda f: row[cols[f]] if f in cols and cols[f] < len(row) else None)
        if h:
            out.append(h)
    return out


def from_tables(tables: Iterable[List[List[str]]]) -> List[Holding]:
    """The largest holdings table among saved page tables (first row = headings)."""
    best: List[Holding] = []
    for table in tables:
        if len(table) >= 2:
            found = from_rows(table[0], table[1:])
            best = found if len(found) > len(best) else best
    return best


def decode(body):
    """The portal returns JSON *inside a string*; unwrap it (repeatedly) so the records can be read."""
    while isinstance(body, str) and body.lstrip()[:1] in ("{", "["):
        try:
            body = json.loads(body)
        except ValueError:
            break
    return body


ISIN = re.compile(r"IN[A-Z0-9]{10}")


def from_portfolio_analyzer(body) -> List[Holding]:
    """The demat holdings statement behind the portal's Portfolio Analyzer page (Statements/PortfolioAnalyzer):
    `PayData` rows with ISIN, name, quantity, rate, valuation and sector. Rows without a real ISIN (the portal adds
    'ZZZ' copies under another account type) are skipped. It has no purchase cost, so `invested` stays unknown."""
    body = decode(body)
    records = body if isinstance(body, list) else [body]
    out: List[Holding] = []
    for record in records:
        rows = record.get("PayData") if isinstance(record, dict) else None
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not ISIN.fullmatch(str(row.get("tsh_isin", ""))):
                continue
            value = number(row.get("tsh_valuation"))
            if value is None or value <= 0:
                continue
            out.append(Holding(symbol=str(row.get("tsh_isin_desc") or row["tsh_isin"]).strip(), qty=number(row.get("tsh_bal")),
                               avg=None, ltp=number(row.get("Rate")) or None, value=value, invested=None,
                               sector=str(row.get("sector_name") or "").strip()))
    return out


def plausible(holdings: List[Holding], portal_value: Optional[float], tolerance: float = 0.10) -> bool:
    """Holdings found by guessing must add up to the portal's own total (within 10%) when that total is known: a market
    list (e.g. 52-week highs) can look like a holdings table but will not add up to the account's value."""
    if not holdings:
        return False
    if not portal_value:
        return True
    return abs(sum(h.value for h in holdings) / portal_value - 1) <= tolerance


def from_json(body) -> List[Holding]:
    """The largest list of records in a JSON response whose keys look like a holdings table."""
    best: List[Holding] = []
    stack = [decode(body)]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(decode(v) for v in node.values())
        elif isinstance(node, list):
            records = [r for r in node if isinstance(r, dict)]
            if records:
                keys = list(records[0].keys())
                found = from_rows(keys, ([r.get(k) for k in keys] for r in records))
                best = found if len(found) > len(best) else best
            stack.extend(decode(n) for n in node)
    return best


TOTAL_LABELS = {"value": "Total Portfolio Value", "invested": "Total Invested", "gain": "Gain/Loss",
                "margin": "Available Margin"}


def totals(text: str) -> Dict[str, float]:
    """The account totals the dashboard prints as 'label' then '₹ amount' (e.g. Total Portfolio Value / ₹ 19,06,827.92).
    Only labels actually found are returned."""
    out: Dict[str, float] = {}
    for key, label in TOTAL_LABELS.items():
        match = re.search(re.escape(label) + r"\s*\n\s*(-?)\s*₹\s*(-?[\d,]+(?:\.\d+)?)", text)
        if match and (value := number(match.group(1) + match.group(2))) is not None:
            out[key] = value
    return out


ACRONYMS = {"it", "fmcg", "nbfc", "psu", "lpg", "ites", "bpo", "etf"}


def nice(name: str) -> str:
    """'IT - SOFTWARE' -> 'IT - Software'; 'PAINTS/VARNISH' -> 'Paints/Varnish'."""
    return re.sub(r"[A-Za-z]+", lambda m: m.group().upper() if m.group().lower() in ACRONYMS
                  else m.group().capitalize(), name)


def inr(amount: float) -> str:
    """Indian digit grouping: 1906827.92 -> '₹19,06,828'; negatives as '-₹…'."""
    whole = str(int(round(abs(amount))))
    head, tail = whole[:-3], whole[-3:]
    groups = [head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1]
    return ("-" if amount < 0 else "") + "₹" + ",".join(groups + [tail] if head else [tail])


def analyse(holdings: List[Holding]) -> dict:
    """Concentration and loss figures, and plain flags. Weights are shares of the holdings' current value."""
    total = sum(h.value for h in holdings)
    ranked = sorted(holdings, key=lambda h: h.value, reverse=True)
    weights = [h.value / total for h in ranked]
    invested = [h.invested for h in holdings if h.invested]
    flags = []
    for h, w in zip(ranked, weights):
        if w > CONCENTRATED:
            flags.append(f"{h.symbol} is {w:.0%} of the portfolio (above {CONCENTRATED:.0%})")
    if sum(weights[:5]) > TOP5_HEAVY and len(ranked) > 5:
        flags.append(f"the five largest holdings are {sum(weights[:5]):.0%} of the portfolio")
    for h in ranked:
        if h.pnl_pct is not None and h.pnl_pct <= DEEP_LOSS:
            flags.append(f"{h.symbol} is {-h.pnl_pct:.0%} below its average cost")
    sectors: Dict[str, float] = {}
    for h in ranked:
        if h.sector:
            sectors[h.sector] = sectors.get(h.sector, 0.0) + h.value / total
    sector_weights = sorted(sectors.items(), key=lambda kv: kv[1], reverse=True)
    for name, w in sector_weights:
        if w > SECTOR_HEAVY:
            flags.append(f"{nice(name)} is {w:.0%} of the portfolio (one sector above {SECTOR_HEAVY:.0%})")
    caps: Dict[str, float] = {}
    for h in ranked:
        if h.cap:
            caps[h.cap] = caps.get(h.cap, 0.0) + h.value / total
    return {
        "total_value": round(total, 2),
        "invested": round(sum(invested), 2) if len(invested) == len(holdings) else None,
        "holdings": len(holdings),
        "effective_holdings": round(1 / sum(w * w for w in weights), 1) if weights else 0,
        "largest_weight": round(weights[0], 4) if weights else 0,
        "top5_weight": round(sum(weights[:5]), 4),
        "sectors": [{"sector": name, "weight": round(w, 4)} for name, w in sector_weights],
        "caps": [{"cap": name, "weight": round(w, 4)} for name, w in sorted(caps.items(), key=lambda kv: -kv[1])],
        "positions": [{**asdict(h), "weight": round(w, 4), "cost_recorded": h.cost_recorded,
                       "pnl_pct": None if h.pnl_pct is None else round(h.pnl_pct, 4)} for h, w in zip(ranked, weights)],
        "flags": flags,
    }


def anonymised(analysis: dict) -> dict:
    """What an outside AI may see: stock names, sectors, market-cap class, weights and returns in percent. No
    quantities, prices, amounts, personal names or account IDs. Missing returns are left out, not sent as null."""
    positions = []
    for p in analysis["positions"]:
        item = {"stock": p["symbol"], "weight_pct": round(p["weight"] * 100, 1)}
        if p["pnl_pct"] is not None:
            item["return_on_cost_pct"] = round(p["pnl_pct"] * 100, 1)
        item.update({k: v for k, v in (("sector", p.get("sector")), ("market_cap", p.get("cap"))) if v})
        positions.append(item)
    return {"stocks": analysis["holdings"], "spread_like_equal_holdings": analysis["effective_holdings"],
            "positions": positions,
            "sectors": [{"sector": s["sector"], "weight_pct": round(s["weight"] * 100, 1)} for s in analysis["sectors"]],
            "market_caps": [{"cap": c["cap"], "weight_pct": round(c["weight"] * 100, 1)} for c in analysis.get("caps", [])],
            "flags": analysis["flags"]}
