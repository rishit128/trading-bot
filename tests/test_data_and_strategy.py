import pandas as pd
import pytest

from src.data.indicators import build_snapshot, compute_rsi
from src.engine.strategy import combine
from src.llm import Signal


def bars(closes):
    return pd.DataFrame({"Close": closes, "Volume": [1000] * len(closes)})


def test_rsi_all_gains_is_100():
    assert compute_rsi(pd.Series(range(1, 60), dtype=float)) == 100.0


def test_rsi_all_losses_is_near_0():
    assert compute_rsi(pd.Series(range(60, 1, -1), dtype=float)) < 1.0


def test_rsi_flat_is_50():
    assert compute_rsi(pd.Series([10.0] * 40)) == 50.0


def test_rsi_matches_wilder_reference():
    # Textbook Wilder example data; SMA-seeded RSI(14) over these 14 changes is 70.46
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert compute_rsi(pd.Series(closes)) == pytest.approx(70.46, abs=0.05)


def test_snapshot_moving_averages():
    s = build_snapshot("X", bars([float(i) for i in range(1, 251)]))
    assert s.price == 250.0
    assert s.ma50 == pytest.approx(sum(range(201, 251)) / 50)
    assert s.ma200 == pytest.approx(sum(range(51, 251)) / 200)


def test_snapshot_needs_200_bars():
    with pytest.raises(ValueError):
        build_snapshot("X", bars([1.0] * 150))


def sig(action, conf=0.8):
    return Signal(action=action, confidence=conf, reasoning="r")


def test_technical_hold_stays_hold():
    assert combine(sig("HOLD"), sig("BUY")).action == "HOLD"


def test_missing_sentiment_uses_technical():
    d = combine(sig("BUY", 0.7), None)
    assert d.action == "BUY" and d.confidence == 0.7


def test_neutral_sentiment_uses_technical():
    assert combine(sig("BUY", 0.7), sig("HOLD", 0.0)).action == "BUY"


def test_conflict_vetoes_trade():
    d = combine(sig("BUY"), sig("SELL"))
    assert d.action == "HOLD" and d.confidence == 0.0


def test_agreement_averages_confidence():
    d = combine(sig("BUY", 0.8), sig("BUY", 0.6))
    assert d.action == "BUY" and d.confidence == pytest.approx(0.7)


def test_signal_rejects_bad_values():
    with pytest.raises(ValueError):
        Signal(action="MAYBE", confidence=0.5, reasoning="r")
    with pytest.raises(ValueError):
        Signal(action="BUY", confidence=1.5, reasoning="r")
