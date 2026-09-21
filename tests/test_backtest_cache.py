import json

from scripts import run_backtest as rb
from src.agents.agents import TechnicalAgent


def test_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    rb.write_cache({"AAPL|2026-01-02": ["BUY", 0.8]}, path)
    assert rb.read_cache(path) == {"AAPL|2026-01-02": ["BUY", 0.8]}


def test_missing_file_is_empty(tmp_path):
    assert rb.read_cache(tmp_path / "none.json") == {}


def test_legacy_flat_cache_is_still_accepted(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"AAPL|2026-01-02": ["HOLD", 0.5]}))
    assert rb.read_cache(path) == {"AAPL|2026-01-02": ["HOLD", 0.5]}


def test_cache_is_discarded_when_the_prompt_changes(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    rb.write_cache({"AAPL|2026-01-02": ["BUY", 0.8]}, path)
    monkeypatch.setattr(TechnicalAgent, "build_prompt", staticmethod(lambda s: "a different prompt"))
    assert rb.read_cache(path) == {}


def test_fingerprint_is_stable():
    assert rb.prompt_fingerprint() == rb.prompt_fingerprint()
