import pytest


@pytest.fixture(autouse=True)
def no_real_test_tracker(monkeypatch):
    """Tests that start the intraday bot must not start its tracker: that would send the owner a real Telegram message."""
    monkeypatch.setattr("src.ops.shadow_watch.ShadowWatch.start", lambda self: None)
