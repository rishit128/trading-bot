"""`.env.example` is the user's only settings documentation: it must list every setting the code reads, list nothing the
code ignores, and show the defaults that actually apply."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = (ROOT / ".env.example").read_text()
SOURCES = [ROOT / "main.py"] + sorted((ROOT / "src").rglob("*.py"))
SECRETS = {"OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}
COMPUTED_DEFAULTS = {"WATCHLIST"}

READ = re.compile(r"""(?:getenv|environ\.get|_float|_int|_bool)\(\s*["']([A-Z][A-Z0-9_]+)["']|environ\[["']([A-Z][A-Z0-9_]+)["']\]""")
LINE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]+)=([^\s#]*)", re.M)


def read_by_code():
    names = set()
    for path in SOURCES:
        for a, b in READ.findall(path.read_text()):
            names.add(a or b)
    return names


def documented():
    """name -> value shown in the example (commented defaults and the active required lines alike)."""
    return {name: value for _, name, value in LINE.findall(EXAMPLE)}


def test_every_setting_the_code_reads_is_documented():
    missing = read_by_code() - set(documented())
    assert not missing, f"settings read by the code but missing from .env.example: {sorted(missing)}"


def test_nothing_documented_is_ignored_by_the_code():
    stale = set(documented()) - read_by_code()
    assert not stale, f"documented in .env.example but no longer read by the code: {sorted(stale)}"


def test_each_setting_appears_once():
    names = [name for _, name, _ in LINE.findall(EXAMPLE)]
    assert len(names) == len(set(names)), f"duplicated: {sorted({n for n in names if names.count(n) > 1})}"


def test_only_the_required_lines_are_active():
    active = {name for hash_, name, _ in LINE.findall(EXAMPLE) if not hash_}
    assert active == SECRETS | {"OPENROUTER_MODEL", "OPENROUTER_FALLBACK_MODEL"}


def test_documented_defaults_are_the_defaults_in_the_source():
    """Compare each shown default with the literal default written next to its os.getenv/_float call."""
    shown, checked = documented(), 0
    for path in SOURCES:
        text = path.read_text()
        for name, default in re.findall(r"""(?:getenv|environ\.get)\(\s*["']([A-Z0-9_]+)["']\s*,\s*["']([^"']*)["']""", text):
            if name in COMPUTED_DEFAULTS:
                continue  # built from per-market data, not a literal: covered by the end-to-end test below
            assert name in shown and shown[name] == default, f"{name}: source default {default!r}, example {shown.get(name)!r}"
            checked += 1
        for name, default in re.findall(r"""_float\(\s*["']([A-Z0-9_]+)["']\s*,\s*([0-9._]+)\s*\)""", text):
            assert name in shown and float(shown[name]) == float(default.replace("_", "")), \
                f"{name}: source default {default}, example {shown.get(name)}"
            checked += 1
    assert checked >= 30  # the scan really found the defaults (guards against the regex silently matching nothing)


def test_setting_the_documented_values_changes_nothing(monkeypatch):
    """End to end: a bot configured with exactly the example's values behaves like one with no configuration at all."""
    from src.config import load_settings

    values = {n: v for n, v in documented().items() if v and n not in SECRETS}
    for name in read_by_code():
        monkeypatch.delenv(name, raising=False)
    baseline = load_settings()
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert load_settings() == baseline


def test_the_example_explains_the_free_model_advice():
    assert "LLM_REASONING_OFF" in EXAMPLE and "LLM_LEARNING" in EXAMPLE and "best left OFF" in EXAMPLE
