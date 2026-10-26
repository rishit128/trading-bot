"""What a filing says, as the AI reads it, and whether it is material. The AI reads; it never decides a trade: it returns an
event type, a direction, a surprise rating and the key numbers taken from the filing, and the fixed tests below (from the
News-Catalyst v1 plan) decide whether the filing counts. A filing the AI cannot read is "unclear" and never counts.

Routine filings (trading-window closures, newspaper copies, call transcripts...) are dropped by their subject line before
any AI call: they are most of the feed and cost a call each for nothing."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Callable, Literal, Optional, get_args

from pydantic import BaseModel

from src.data.filings import Filing
from src.data.india import IST
from src.news.strategy import B_UNTIL

EventType = Literal[
    "results",
    "order_win",
    "guidance",
    "merger_deal",
    "approval",
    "fund_raise",
    "promoter_buy",
    "promoter_sell",
    "management_change",
    "routine",
]
EVENT_TYPES = list(get_args(EventType))
RESULTS_MIN_CHANGE = 20.0  # % change in profit on a year earlier
ORDER_MIN_SHARE = 0.02  # an order worth at least this share of the company's market value

# Lower-case fragments of a subject line that mark a routine filing. Checked once against a day of the BSE feed
# (2026-10-21); add one only after seeing it, and never one that a results, order or deal filing could contain.
ROUTINE_SUBJECTS = (
    "trading window",
    "newspaper",
    "news paper",
    "loss of share",
    "duplicate share",
    "74(5)",
    "transcript",
    "audio recording",
    "analyst",
    "investor meet",
    "investors meet",
    "con. call",
    "conference call",
    "earnings call",
    "compliance certificate",
    "compliance report on corporate governance",
    "record date",
    "esop",
    "esos",
    "shareholding pattern",
    "investor complaints",
    "postal ballot",
    "book closure",
    "secretarial",
    "intimation of board meeting",
    "that the meeting of the board of directors of the company is scheduled",  # BSE's wording for a meeting notice
    "prior intimation",
    "change in registrar",
    "kyc",
    "regulation 31(4)",
    "regulation 7(3)",
    "debenture trustee",
    "non-convertible debentures",
    "commercial paper",
    "interest payment",
    "nav as on",
)


def routine_subject(subject: str) -> Optional[str]:
    """The routine marker this subject contains, or None."""
    s = subject.lower()
    return next((m for m in ROUTINE_SUBJECTS if m in s), None)


class FilingLabel(BaseModel):
    event_type: EventType
    direction: Literal["up", "down", "none"]
    surprise: Literal["large", "moderate", "none", "unclear"]
    profit_quarter: Optional[float] = None  # results: net profit of the latest quarter, as printed (any unit)
    profit_year_ago: Optional[float] = None  # results: net profit of the SAME quarter a year earlier, same unit
    order_value_crore: Optional[float] = None  # order win: the order's value in Rs crore
    key_numbers: str = ""  # the figures as written in the filing
    reason: str = ""
    raw_model: Optional[str] = None  # set by the LLM client: which model answered


LABEL_SCHEMA = {
    "name": "filing_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": EVENT_TYPES},
            "direction": {"type": "string", "enum": ["up", "down", "none"]},
            "surprise": {"type": "string", "enum": ["large", "moderate", "none", "unclear"]},
            "profit_quarter": {"type": ["number", "null"]},
            "profit_year_ago": {"type": ["number", "null"]},
            "order_value_crore": {"type": ["number", "null"]},
            "key_numbers": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "event_type",
            "direction",
            "surprise",
            "profit_quarter",
            "profit_year_ago",
            "order_value_crore",
            "key_numbers",
            "reason",
        ],
        "additionalProperties": False,
    },
}

PROMPT = """You read Indian stock-exchange filings for a trading system. You do NOT decide trades; you only classify.

Company: {company} (NSE: {symbol})
Filed: {when} IST
Exchange subject line: {subject}

Filing text (the opening and the lines around its figures, extracted from the PDF; may be empty or garbled):
<<<
{text}
>>>

Return JSON with:
- event_type: one of {types}. Use "routine" for anything procedural (meeting notices, certificates, transcripts, routine
  disclosures) or when the filing has no news for the share price.
- direction: the likely effect on the share price today: "up", "down" or "none". For results it is the sign of the
  year-on-year profit change ("up" if profit grew, "down" if it fell); for an order win, "up".
- surprise: how big the news is, by these fixed measures (you have no analyst estimates, so do not guess at them):
  results: "large" if net profit changed 40% or more on the same quarter a year earlier (either way), "moderate" for
  20% to 40%, "none" under 20%; an order win: "large" if the order is a big part of the company's business as the
  filing describes it, else "moderate"; anything else: your judgement of how much it changes the company's value.
  Use "unclear" whenever you cannot read the numbers you would need from the filing text itself (empty or garbled text,
  numbers missing). Never use outside knowledge for the numbers.
- profit_quarter and profit_year_ago: for results only, the NET PROFIT (profit for the period / profit after tax;
  consolidated if given, else standalone) of the latest quarter and of the SAME quarter one year earlier, copied from the
  results table in the same unit; a loss is negative. Indian results tables usually have three quarter columns: the
  latest quarter, the PREVIOUS quarter, and the same quarter LAST YEAR (e.g. 30.09.2026 | 30.06.2026 | 30.09.2025), then
  the two half-years and the last full year; but some companies put the year-ago quarter SECOND, so always read the
  column dates and take profit_year_ago from the column dated one year before the latest. Both figures must come from the
  SAME table (both consolidated or both standalone). If the dates cannot be read or that table has no year-ago figure,
  both are null; for anything but results, both are null.
- order_value_crore: for an order win only, the order's value in Rs crore as stated in the filing; else null.
- key_numbers: the figures you used with their column dates, copied as written in the filing (e.g. "PAT 30.09.2026: 412.5
  vs 30.09.2025: 301.2 (Rs cr)"). Empty if none.
- reason: one line."""


def ask_label(llm, filing: Filing, text: str) -> FilingLabel:
    """The AI's label for one filing (raises when no model answers)."""
    prompt = PROMPT.format(
        company=filing.company,
        symbol=filing.symbol,
        when=filing.published.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
        subject=filing.subject[:500],
        text=text.strip() or "(no text could be extracted)",
        types=", ".join(EVENT_TYPES),
    )
    return llm.structured_call(prompt, FilingLabel, LABEL_SCHEMA, max_tokens=1200)


def profit_change_pct(label: FilingLabel) -> Optional[float]:
    """The year-on-year change in quarterly profit, computed here from the two figures the AI copied. A loss turning into a
    profit is +100, a profit turning into a loss -100; otherwise the change against the size of the year-ago figure."""
    now, ago = label.profit_quarter, label.profit_year_ago
    if now is None or ago is None or (now == 0 and ago == 0):
        return None
    if ago <= 0 < now:
        return 100.0
    if now <= 0 < ago:
        return -100.0
    return (now - ago) / abs(ago) * 100


@dataclass(frozen=True)
class Materiality:
    material: bool
    reason: str


def materiality(label: FilingLabel, market_cap_crore: Callable[[], Optional[float]]) -> Materiality:
    """The plan's tests. Results: profit at least 20% up or down on a year earlier, with the direction agreeing and a surprise
    rated large or moderate. Order win: at least 2% of the company's market value (whatever the surprise rating: the order's
    size is the test). Anything else: a large surprise.
    `market_cap_crore` is only asked for an order win (it costs a request)."""
    if label.event_type == "routine" or (label.surprise == "unclear" and label.event_type != "order_win"):
        return Materiality(False, f"{label.event_type}, surprise {label.surprise}")
    if label.event_type == "results":
        change = profit_change_pct(label)
        if change is None:
            return Materiality(False, "results without a profit change read from the filing")
        agrees = (change > 0 and label.direction == "up") or (change < 0 and label.direction == "down")
        if abs(change) < RESULTS_MIN_CHANGE:
            return Materiality(False, f"profit {change:+.0f}% on a year earlier (under {RESULTS_MIN_CHANGE:.0f}%)")
        if not agrees or label.surprise not in ("large", "moderate"):
            return Materiality(False, f"profit {change:+.0f}% but direction {label.direction}, surprise {label.surprise}")
        return Materiality(True, f"profit {change:+.0f}% on a year earlier, surprise {label.surprise}")
    if label.event_type == "order_win":
        if not label.order_value_crore:
            return Materiality(False, "order win without an order value read from the filing")
        cap = market_cap_crore()
        if not cap:
            return Materiality(False, "order win, but the market value is unknown")
        share = label.order_value_crore / cap
        if share < ORDER_MIN_SHARE:
            return Materiality(False, f"order {share:.1%} of market value (under {ORDER_MIN_SHARE:.0%})")
        return Materiality(True, f"order Rs {label.order_value_crore:,.0f} cr = {share:.1%} of market value")
    if label.surprise == "large":
        return Materiality(True, f"{label.event_type}, surprise large")
    return Materiality(False, f"{label.event_type}, surprise {label.surprise} (needs large)")


def label_fields(label: FilingLabel, verdict: Materiality) -> dict:
    """The NewsFilingRecord columns for a label."""
    numbers = {
        "profit_quarter": label.profit_quarter,
        "profit_year_ago": label.profit_year_ago,
        "profit_change_pct": profit_change_pct(label),
        "order_value_crore": label.order_value_crore,
    }
    return dict(
        label_status="LABELLED",
        event_type=label.event_type,
        direction=label.direction,
        surprise=label.surprise,
        numbers_json=json.dumps({**numbers, "as_written": label.key_numbers[:400]}),
        ai_reason=label.reason[:500],
        ai_model=(label.raw_model or "")[:80] or None,
        material=verdict.material,
        material_reason=verdict.reason,
    )


def as_utc(when: datetime) -> datetime:
    """A stored time as an aware UTC datetime: SQLite hands back the UTC wall time without its zone, which Python would
    otherwise read as the machine's local (IST) time, five and a half hours off."""
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def first_session(published: datetime, is_session_day: Callable[[datetime], bool]) -> str:
    """The first session (YYYY-MM-DD) a filing can be traded in: a filing after the last time of day rule B takes news
    (14:30) or on a closed day belongs to the next trading day, where rule A treats it as overnight news; one before that to
    the same day if it trades. (Until 2026-10-23 the cut-off was 15:30, which left filings from 14:30 to 15:30 undecided.)"""
    local = published.astimezone(IST)
    day = local.date()
    if local.time() > B_UNTIL:
        day += timedelta(days=1)
    for _ in range(10):
        if is_session_day(datetime.combine(day, dtime(12, 0), IST)):
            break
        day += timedelta(days=1)
    return day.isoformat()
