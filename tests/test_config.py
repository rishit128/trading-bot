import pytest

from src.config import RiskLimits, load_settings


def test_defaults_are_valid():
    RiskLimits()


@pytest.mark.parametrize("kwargs", [
    dict(max_position_pct=5),  # someone typed 5 meaning 5%
    dict(max_position_pct=0),
    dict(max_portfolio_exposure_pct=1.5),
    dict(max_daily_loss_pct=-0.01),
    dict(max_drawdown_pct=2),
    dict(stop_loss_pct=0),
    dict(take_profit_pct=0),
    dict(min_confidence=1.5),
    dict(max_position_pct=0.9, max_portfolio_exposure_pct=0.5),
])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskLimits(**kwargs)


def test_env_typo_fails_loudly_instead_of_allowing_500_percent_positions(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_PCT", "5")
    with pytest.raises(ValueError, match="max_position_pct"):
        load_settings()


def test_env_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("MAX_POSITION_PCT", "0.10")
    monkeypatch.setenv("REBUY_COOLDOWN_HOURS", "6")
    monkeypatch.setenv("WATCHLIST", "nvda, amd")
    s = load_settings()
    assert s.risk.max_position_pct == 0.10 and s.rebuy_cooldown_hours == 6 and s.watchlist == ("NVDA", "AMD")


def test_min_position_pct_defaults_to_max_position_pct_flat_sizing_unless_set(monkeypatch):
    monkeypatch.delenv("MIN_POSITION_PCT", raising=False)
    monkeypatch.setenv("MAX_POSITION_PCT", "0.10")
    assert load_settings().risk.min_position_pct == 0.10  # flat: no confidence scaling unless MIN_POSITION_PCT is set
    monkeypatch.setenv("MIN_POSITION_PCT", "0.02")
    assert load_settings().risk.min_position_pct == 0.02 and load_settings().risk.max_position_pct == 0.10


def test_dry_run_is_the_default_and_only_false_disables_it(monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    assert load_settings().dry_run is True
    monkeypatch.setenv("DRY_RUN", "0")
    assert load_settings().dry_run is True
    monkeypatch.setenv("DRY_RUN", "false")
    assert load_settings().dry_run is False


def test_market_scan_is_the_default_universe(monkeypatch):
    for name in ("UNIVERSE", "MAX_CANDIDATES"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.universe == "market" and s.max_candidates == 15


def test_universe_settings_from_env(monkeypatch):
    monkeypatch.setenv("UNIVERSE", "Watchlist")
    monkeypatch.setenv("MAX_CANDIDATES", "5")
    monkeypatch.setenv("MIN_PRICE", "20")
    monkeypatch.setenv("MIN_TRADED_VALUE", "50000000")
    monkeypatch.setenv("MAX_DAILY_VOLATILITY", "0.03")
    s = load_settings()
    assert (s.universe, s.max_candidates, s.min_price, s.min_traded_value, s.max_daily_volatility) == \
        ("watchlist", 5, 20.0, 50_000_000.0, 0.03)


@pytest.mark.parametrize("name,value", [
    ("UNIVERSE", "everything"), ("MAX_CANDIDATES", "0"),
    ("MIN_PRICE", "-1"), ("MAX_DAILY_VOLATILITY", "4"),
])
def test_invalid_universe_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings()


def test_max_open_positions_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        RiskLimits(max_open_positions=0)
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "4")
    assert load_settings().risk.max_open_positions == 4


def test_india_is_the_default_market_with_rupee_thresholds(monkeypatch):
    for name in ("MARKET", "MIN_PRICE", "MIN_TRADED_VALUE", "WATCHLIST", "PAPER_INITIAL_CASH"):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.market == "india" and s.currency == "₹"
    assert s.min_price == 100.0 and s.min_traded_value == 100_000_000.0 and s.paper_initial_cash == 1_000_000.0
    assert s.watchlist == ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")


def test_explicit_env_overrides_beat_market_defaults(monkeypatch):
    monkeypatch.setenv("MARKET", "india")
    monkeypatch.setenv("MIN_PRICE", "250")
    monkeypatch.setenv("PAPER_INITIAL_CASH", "500000")
    s = load_settings()
    assert s.min_price == 250.0 and s.paper_initial_cash == 500_000.0


@pytest.mark.parametrize("name,value", [("MARKET", "uk"), ("PAPER_INITIAL_CASH", "0"), ("PAPER_INITIAL_CASH", "-5")])
def test_invalid_market_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings()


def test_dataclass_defaults_match_the_evidence_backed_exit_settings():
    from src.config import Settings

    s = Settings()
    assert s.trend_exit is True and s.risk.stop_loss_pct == 0.15 and s.risk.take_profit_pct == 1.00


def test_delivery_filter_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("DELIVERY_FILTER", raising=False)
    assert load_settings().delivery_filter is True
    monkeypatch.setenv("DELIVERY_FILTER", "false")
    assert load_settings().delivery_filter is False
