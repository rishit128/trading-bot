from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.angel import bar_row
from src.data.angel_quotes import EntryContext
from src.data.india import IST
from src.data.market_regime import RegimeProvider, live_breadth, risk_of, tone_of, trend_of, with_regime
from src.database import AngelBarRecord, make_session_factory


def test_trend_is_bull_above_both_averages_with_momentum_bear_below_both_and_otherwise_sideways():
    assert trend_of(110, 105, 100, 0.03) == "BULL"
    assert trend_of(110, 105, 100, -0.01) == "SIDEWAYS"  # above both averages but falling over 20 sessions
    assert trend_of(90, 95, 100, -0.03) == "BEAR"
    assert trend_of(99, 105, 100, 0.0) == "SIDEWAYS"


def test_risk_comes_from_the_vix_level_and_its_place_in_its_year():
    assert risk_of(30.0, 0.5) == "STRESS" and risk_of(15.0, 0.95) == "STRESS"
    assert risk_of(15.0, 0.7) == "ELEVATED" and risk_of(12.0, 0.3) == "CALM" and risk_of(None, None) == "CALM"


def test_tone_comes_from_the_share_of_stocks_up_on_the_day():
    assert (tone_of(0.7), tone_of(0.5), tone_of(0.3)) == ("STRONG", "MIXED", "WEAK")


def at(day, h, m):
    return datetime(2026, 10, day, h, m, tzinfo=IST).astimezone(timezone.utc)


def test_live_breadth_compares_each_stocks_latest_price_with_the_previous_sessions_close(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    rows = []
    for i in range(30):
        rows.append(bar_row(f"S{i}", at(21, 15, 25), 100, 100, 100, 100.0, 1, at(21, 15, 30)))
        today = 101.0 if i < 20 else 99.0  # 20 up, 10 down
        rows.append(bar_row(f"S{i}", at(22, 10, 0), 100, 100, 100, today, 1, at(22, 10, 5)))
    with sessions() as s:
        s.bulk_insert_mappings(AngelBarRecord, rows)
        s.commit()
    got = live_breadth(sessions, at(22, 10, 5))
    assert got == {"advancers": 20, "decliners": 10, "breadth": 20 / 30}
    assert live_breadth(sessions, at(23, 10, 5)) is None  # no bars yet today


class Market:
    def __init__(self, nifty, vix=None):
        self.nifty, self.vix = nifty, vix

    def index_series(self):
        return self.nifty

    def vix_series(self):
        return self.vix

    def crude_move(self):
        return (85.0, 0.062)


def series(values):
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=len(values), freq="B"))


def test_the_provider_labels_a_falling_market_bear_and_caches_for_five_minutes():
    falling = series(np.linspace(120.0, 90.0, 260))
    calls = []
    now = [0.0]
    provider = RegimeProvider(Market(falling, series(np.linspace(10, 25, 260))), clock=lambda: now[0])
    regime = provider.current()
    assert regime.trend == "BEAR" and regime.risk == "STRESS" and regime.tone is None
    assert regime.short() == "BEAR/STRESS" and "BEAR" in regime.text()
    assert "Brent crude $85, +6.2%" in regime.text()
    provider.market = SimpleNamespace(index_series=lambda: calls.append(1) or falling, vix_series=lambda: None, crude_move=lambda: None)
    provider.current()
    assert calls == []  # cached
    now[0] = 400.0
    provider.current()
    assert calls == [1]


def test_the_provider_gives_none_when_the_index_cannot_be_read():
    assert RegimeProvider(Market(series([1.0] * 50))).current() is None

    class Broken:
        def index_series(self):
            raise RuntimeError("down")

    assert RegimeProvider(Broken()).current() is None


def test_with_regime_adds_the_label_to_the_record_without_touching_the_decision():
    regime = SimpleNamespace(short=lambda: "BEAR/CALM/WEAK", facts={"crude_1m": 0.0623})
    provider = SimpleNamespace(current=lambda: regime)
    wrapped = with_regime(lambda s, p: EntryContext("near the circuit", {"spread_pct": 0.1}), provider)
    out = wrapped("AAA", 100.0)
    assert out.skip == "near the circuit" and out.note == {"spread_pct": 0.1, "market": "BEAR/CALM/WEAK", "crude_1m": 0.0623}
    none = with_regime(lambda s, p: EntryContext(None, {"a": 1}), SimpleNamespace(current=lambda: None))("AAA", 1.0)
    assert none.note == {"a": 1}
