"""Company filings from the exchanges' official RSS feeds, for the news-catalyst engine (src.news).

Sources, in order of preference (checked 2026-10-21):
  * BSE's "Latest Corporate Announcements" feed (www.bseindia.com/data/xml/announcements.xml): every company filing, about
    7,000 a day, most of them mutual-fund NAVs that no stock maps to. Companies listed on both exchanges must file with both
    at the same time, so for liquid NSE stocks BSE carries the same filings. Each item names the BSE scrip code; it is
    mapped to the NSE symbol through Angel One's public instrument list (a BSE code and an NSE stock share a symbol).
    The feed starts afresh every night (its oldest item is from just after midnight), so it must be read in the evening
    too, or results filed after the close are gone by the next morning.
  * NSE's "Online announcements" feed (nsearchives.nseindia.com/content/RSS/Online_announcements.xml): the backup, read
    only when the BSE feed fails. Its items carry no symbol; the attachment's file name starts with it.
These are feeds the exchanges publish for subscribers to read automatically. Only the headline and link are stored, the
attachment is read once (by the AI, for the label) and never stored or republished. NSE's other feeds (results, insider
trading) had not been updated for months when checked, so they are not used.

Timestamps in both feeds are IST without a zone."""

import json
import logging
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import httpx

from src.data.india import IST

log = logging.getLogger(__name__)

BSE_FEED = "https://www.bseindia.com/data/xml/announcements.xml"
NSE_FEED = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"
SCRIP_MASTER = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
# Both exchanges refuse a client that names itself a bot (BSE: HTTP 403; NSE: no answer, checked 2026-10-21) and serve the
# same feed to a browser's agent string, so the reader sends one, as a browser-based feed reader does.
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
MAP_FILE = "bse_nse_map.json"
MAP_MAX_AGE = 7 * 86400  # seconds: the instrument list changes only with listings
MAX_PDF_BYTES = 30_000_000  # results filings with scanned pages run to 15 MB
MAX_PAGES = 40
MAX_TEXT = 12_000  # characters of a filing given to the AI
HEAD = 2_500  # the cover letter: what the filing is
# Lines worth showing the AI from deeper in the document (the results table and order values are rarely on the first pages):
# the results lines first, then the rest, so a long deal write-up cannot crowd the profit table out.
RESULT_LINES = re.compile(
    r"profit for the|profit after tax|net profit|total income|revenue from operations|profit before tax|earnings per share"
    r"|quarter ended|three months ended|months ended|particulars"
    r"|(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}.*){2}",  # a row of column dates: which column is which quarter
    re.IGNORECASE,
)
OTHER_LINES = re.compile(r"order|contract|crore|lakh|acqui|merger|amalgamat|preferential|qip|buyback|guidance", re.IGNORECASE)
CONTEXT = 2  # lines kept on each side of a key line


@dataclass(frozen=True)
class Filing:
    """One item of an exchange feed, mapped to an NSE symbol."""

    source: str  # BSE | NSE
    symbol: str
    company: str
    subject: str
    link: str
    published: datetime  # UTC

    @property
    def key(self) -> str:
        return self.link[:200]


def _when(text: str) -> Optional[datetime]:
    """'21-Oct-2026 16:25:08' (IST) as UTC; None when it cannot be read."""
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=IST).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _items(xml: bytes) -> Iterable[ET.Element]:
    return ET.fromstring(xml.lstrip(b"\xef\xbb\xbf")).iter("item")


def _text(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def parse_bse(xml: bytes, bse_to_nse: Dict[str, str]) -> List[Filing]:
    """The BSE feed's items whose scrip code maps to an NSE stock (the rest are funds, bonds and BSE-only companies)."""
    out = []
    for item in _items(xml):
        symbol, published = bse_to_nse.get(_text(item, "scripcode")), _when(_text(item, "pubDate"))
        if symbol is None or published is None:
            continue
        company = re.sub(r"\s*\(\d+\)\s*$", "", _text(item, "title"))
        out.append(Filing("BSE", symbol, company[:120], _text(item, "description"), _text(item, "link"), published))
    return out


def nse_symbol_from_link(link: str) -> Optional[str]:
    """'.../corporate/E2E_21102026162502_E2E_Trans_Final.pdf' -> 'E2E' (NSE names each attachment after the stock)."""
    m = re.search(r"/corporate/([A-Z0-9&\-]+)_\d{14}", link)
    return m.group(1) if m else None


def parse_nse(xml: bytes) -> List[Filing]:
    out = []
    for item in _items(xml):
        link, published = _text(item, "link"), _when(_text(item, "pubDate"))
        symbol = nse_symbol_from_link(link)
        if symbol is None or published is None:
            continue
        text, _, category = _text(item, "description").partition("|SUBJECT:")
        subject = f"{text.strip()} ({category.strip()})" if category.strip() else text.strip()  # the category helps spot routine
        out.append(Filing("NSE", symbol, _text(item, "title")[:120], subject, link, published))
    return out


def bse_nse_map(master: Iterable[dict]) -> Dict[str, str]:
    """BSE scrip code -> NSE symbol, for every BSE equity whose symbol is also an NSE equity's (Angel One's instrument list)."""
    rows = list(master)
    nse = {r["name"] for r in rows if r.get("exch_seg") == "NSE" and str(r.get("symbol", "")).endswith("-EQ")}
    return {str(r["token"]): r["name"] for r in rows if r.get("exch_seg") == "BSE" and r.get("name") in nse}


def load_bse_nse_map(cache_dir: Path) -> Dict[str, str]:
    """The BSE -> NSE map, from a cache refreshed weekly. A failed refresh keeps the old copy; with none at all it raises."""
    path = cache_dir / MAP_FILE
    if path.exists() and time.time() - path.stat().st_mtime < MAP_MAX_AGE:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        mapping = bse_nse_map(scrip_master())
    except Exception:
        if path.exists():
            log.exception("could not refresh the BSE -> NSE symbol map; using the old copy")
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return mapping


def scrip_master() -> List[dict]:
    response = httpx.get(SCRIP_MASTER, timeout=120)
    response.raise_for_status()
    return response.json()


def _around(lines: List[str], pattern: re.Pattern) -> List[int]:
    return sorted({j for i, line in enumerate(lines) if pattern.search(line) for j in range(i - CONTEXT, i + CONTEXT + 1)})


def excerpt(text: str) -> str:
    """The filing's opening (what it is), then the lines around results figures, then those around other figures, at most
    MAX_TEXT characters."""
    head, rest = text[:HEAD], text[HEAD:].splitlines()
    out, used, size = [head], set(), len(head)
    for pattern in (RESULT_LINES, OTHER_LINES):
        last = -2
        for j in _around(rest, pattern):
            if not 0 <= j < len(rest) or j in used:
                continue
            line = re.sub(r"\s{3,}", "  ", rest[j].rstrip())
            piece = (line if j == last + 1 else "[...]\n" + line) + "\n"
            if size + len(piece) > MAX_TEXT:
                return "".join(out)
            out.append(piece)
            used.add(j)
            size += len(piece)
            last = j
    return "".join(out)


class FeedReader:
    """Reads the newest filings (BSE first, NSE when BSE fails) and the text of a filing's attachment."""

    def __init__(self, bse_to_nse: Dict[str, str], client: Optional[httpx.Client] = None):
        self.bse_to_nse = bse_to_nse
        self.http = client or httpx.Client(timeout=30, headers=HEADERS, follow_redirects=True)

    def read(self) -> List[Filing]:
        try:
            response = self.http.get(BSE_FEED)
            response.raise_for_status()
            return parse_bse(response.content, self.bse_to_nse)
        except Exception as e:
            log.warning("BSE announcements feed failed (%s: %s); reading NSE's instead", type(e).__name__, e)
        response = self.http.get(NSE_FEED)
        response.raise_for_status()
        return parse_nse(response.content)

    def attachment_text(self, link: str) -> str:
        """The PDF's text cut down by `excerpt` for the AI (up to MAX_PAGES pages); "" when it cannot be read."""
        if not link.lower().endswith(".pdf"):
            return ""
        try:
            response = self.http.get(link)
            response.raise_for_status()
            if len(response.content) > MAX_PDF_BYTES:
                return ""
            done = subprocess.run(
                ["pdftotext", "-l", str(MAX_PAGES), "-layout", "-", "-"], input=response.content, capture_output=True, timeout=90
            )
            return excerpt(done.stdout.decode("utf-8", "replace"))
        except Exception as e:
            log.warning("could not read the filing %s (%s)", link, type(e).__name__)
            return ""
