"""Readers for the portal's statements, read field by field (their layouts are known from the real account).

    Portfolio/Present              current holdings: avg buy price, cost, price, value, realised/unrealised P&L
    Portfolio/HoldingsTurnings     short-term purchase lots: date, qty, rate, days until they turn long-term
    Statements/Dividend            dividends: record date, eligible qty, per share, gross, TDS, net
    Portfolio/RealizedGainLoss     shares sold this financial year: bought value, sold value, gain
    Portfolio/StatementOfTransactionV1   recent buys/sells with date, qty and rate
    Statements/EquityPortfolioAnalyzer*  sector and market-cap class per stock

The portal sends most of these as JSON inside a string; every reader unwraps it (`decode`) and skips rows it cannot read
rather than guessing. Personal details (name, address, client/DP IDs) in these responses are never read."""
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.portfolio.holdings import Holding, decode, number


@dataclass(frozen=True)
class Lot:
    company: str
    bought_on: str          # dd/mm/yyyy as the portal prints it
    qty: float
    rate: float
    days_to_long_term: Optional[int]


@dataclass(frozen=True)
class Dividend:
    company: str
    record_date: str        # dd/mm/yyyy
    qty: float
    per_share: float
    gross: float
    tds: float
    net: float


@dataclass(frozen=True)
class Sale:
    company: str
    qty: float
    bought_value: float
    sold_value: float
    gain: float


CORPORATE_ACTION = re.compile(r"scheme|merger|demerger|amalgamation|bonus|split|rights|buy ?back", re.I)


@dataclass(frozen=True)
class Trade:
    company: str
    on: str                 # dd/mm/yyyy
    side: str               # Buy / Sell
    qty: float
    rate: float
    note: str

    @property
    def corporate_action(self) -> bool:
        """A merger/demerger/bonus/split credit or debit: shares moved by the company, not bought or sold on the market."""
        return bool(CORPORATE_ACTION.search(self.note))


@dataclass
class Statements:
    holdings: List[Holding] = field(default_factory=list)
    totals: Dict[str, float] = field(default_factory=dict)   # cost, value, realised, unrealised, roi_pct
    lots: List[Lot] = field(default_factory=list)
    dividends: List[Dividend] = field(default_factory=list)
    financial_year: str = ""
    sales: List[Sale] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    classes: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # name key -> (sector, market-cap class)


def name_key(name: str) -> str:
    """'KOTAK MAHINDRA BANK LIMITED' and 'Kotak Mahindra Bank Ltd' -> 'kotakmahindrabank' (for joining statements)."""
    s = re.sub(r"\b(limited|ltd|the)\b\.?", "", str(name or "").lower())
    return re.sub(r"[^a-z0-9]", "", s)


def _num(value, default: float = 0.0) -> float:
    n = number(value)
    return default if n is None else n


def current_holdings(body) -> Tuple[List[Holding], Dict[str, float]]:
    """Portfolio/Present: one row per stock with average cost, value and P&L, and the account totals."""
    root = decode(body)
    pa = root.get("portfolioAnalysis") if isinstance(root, dict) else None
    if not isinstance(pa, dict):
        return [], {}
    out = []
    for row in pa.get("holdings") or []:
        if not isinstance(row, dict):
            continue
        value, qty = number(row.get("valueAtMarketPrice")), number(row.get("portfolioPosition"))
        if not value or value <= 0 or not qty:
            continue
        out.append(Holding(
            symbol=str(row.get("companyName") or "").strip(), qty=qty, avg=number(row.get("averageCostPrice")),
            ltp=number(row.get("currentMarketPrice")), value=value, invested=number(row.get("valueAtCost")),
            realized=number(row.get("realizedProfitLoss")), bought_qty=number(row.get("BuyQty")),
            sold_qty=number(row.get("SellQty")), price_date=str(row.get("lastPriceDate") or "")))
    grand = ((pa.get("totals") or {}).get("grandTotal")) or {}
    totals = {k: v for k, v in {"cost": number(grand.get("valueAtCost")), "value": number(grand.get("valueAtMarketPrice")),
                                "realised": number(grand.get("realizedProfitLoss")),
                                "unrealised": number(grand.get("unrealizedProfitLoss")),
                                "roi_pct": number(grand.get("ROIper"))}.items() if v is not None}
    return out, totals


def lots(body) -> List[Lot]:
    """Portfolio/HoldingsTurnings: the short-term purchase lots and the days left until each turns long-term."""
    root = decode(body)
    rows = ((root or {}).get("HoldingsTurningLongTerm") or {}).get("transactions") if isinstance(root, dict) else None
    out = []
    for company in rows or []:
        for d in company.get("Details") or []:
            qty, rate = number(d.get("Quantity")), number(d.get("Rate"))
            if qty and rate:
                days = number(d.get("RemainingDays"))
                out.append(Lot(str(company.get("CompanyName") or "").strip(), str(d.get("DateOfTransaction") or ""),
                               qty, rate, None if days is None else int(days)))
    return out


def dividends(body) -> List[Dividend]:
    """Statements/Dividend: one row per dividend credited (record date, shares, per share, gross, TDS, net)."""
    root = decode(body)
    out = []
    for row in (root.get("DividendISINDetails") if isinstance(root, dict) else None) or []:
        gross = number(row.get("GrossDividendAmount"))
        if not gross:
            continue
        isin = row.get("ISIN") or {}
        tds = next((_num(v) for k, v in row.items() if k.upper().startswith("TDS")), 0.0)
        out.append(Dividend(str(isin.get("Description") or "").strip(), str(row.get("RecordDate") or ""),
                            _num(row.get("EligibleQuantity")), _num(row.get("DividendPerShare")), gross, tds,
                            _num(row.get("NetDividentAmount"), gross - tds)))
    return out


def realised(body) -> Tuple[str, List[Sale]]:
    """Portfolio/RealizedGainLoss: the financial year and each sale in it."""
    root = decode(body)
    st = (root.get("portfolioGainLossStatement") if isinstance(root, dict) else None) or {}
    sales = [Sale(str(t.get("company") or "").strip(), _num(t.get("SoldQuantity")), _num(t.get("BoughtValue")),
                  _num(t.get("SoldValue")), _num(t.get("GainOrLoss"))) for t in st.get("transactions") or []]
    return str(st.get("financialYear") or ""), sales


def trades(body) -> List[Trade]:
    """Portfolio/StatementOfTransactionV1: recent buys and sells (the portal's default period)."""
    root = decode(body)
    fp = (root.get("FundPortfolio") if isinstance(root, dict) else None) or {}
    out = []
    for holding in fp.get("fundHoldings") or []:
        for t in holding.get("transactions") or []:
            credit, debit = _num(t.get("creditQty")), _num(t.get("debitQty"))
            if credit or debit:
                out.append(Trade(str(holding.get("fundName") or "").strip(), str(t.get("transactionDate") or ""),
                                 "Buy" if credit else "Sell", credit or debit, _num(t.get("rate")),
                                 re.sub(r"\s+", " ", str(t.get("description") or "")).strip()))
    return out


def classes(body) -> Dict[str, Tuple[str, str]]:
    """Statements/EquityPortfolioAnalyzer*: sector and market-cap class (Large/Mid/Small Cap) per stock name."""
    root = decode(body)
    rows = root.get("Fulldetails") if isinstance(root, dict) else root
    out = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("isindesc") and number(row.get("valuation")):
            out[name_key(row["isindesc"])] = (str(row.get("sectorname") or "").strip(),
                                              str(row.get("Market_cap") or "").strip())
    return out


READERS = {  # file-name pattern of the saved data response -> what to do with it
    "Portfolio_Present": "holdings", "HoldingsTurnings": "lots", "Statements_Dividend": "dividends",
    "RealizedGainLoss": "realised", "StatementOfTransaction": "trades", "EquityPortfolioAnalyzer": "classes",
}


def load(run_dir: Path) -> Statements:
    """Every statement found among the run's saved data responses; missing ones stay empty."""
    s = Statements()
    for file in sorted((Path(run_dir) / "api").glob("*.json")):
        kind = next((k for pattern, k in READERS.items() if pattern in file.name), None)
        if kind is None:
            continue
        try:
            body = json.loads(file.read_text(encoding="utf-8")).get("body")
        except (OSError, ValueError, AttributeError):
            continue
        if kind == "holdings" and not s.holdings:
            s.holdings, s.totals = current_holdings(body)
        elif kind == "lots":
            s.lots = s.lots or lots(body)
        elif kind == "dividends":
            s.dividends = s.dividends or dividends(body)
        elif kind == "realised":
            s.financial_year, s.sales = realised(body) if not s.sales else (s.financial_year, s.sales)
        elif kind == "trades":
            s.trades = s.trades or trades(body)
        elif kind == "classes":
            s.classes = {**classes(body), **s.classes}
    if s.holdings and s.classes:  # add sector and market-cap class to each holding
        s.holdings = [_classed(h, s.classes.get(name_key(h.symbol))) for h in s.holdings]
    return s


def _classed(h: Holding, cls: Optional[Tuple[str, str]]) -> Holding:
    if not cls:
        return h
    return replace(h, sector=h.sector or cls[0], cap=h.cap or cls[1])


def parse_date(text: str) -> Optional[date]:
    try:
        return datetime.strptime(text.strip(), "%d/%m/%Y").date()
    except (ValueError, AttributeError):
        return None
