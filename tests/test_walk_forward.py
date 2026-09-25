"""Walk-forward validation - rolling non-overlapping hold-out blocks forming one continuous OOS curve.

The strategy tunes nothing, so the honest out-of-sample discipline is structural: blocks never overlap, a block's
decisions never see a later block's data, and every decision date appears in exactly one hold-out block (the OOS
number is one stitched curve, not a picked window)."""
import pytest

from src.research.ablation import DET
from src.research.walk_forward import run_walk_forward, walk_forward_folds
from tests.test_ablation import BARS, COT_GOOD, DATES, llm, route


def test_folds_partition_dates_without_overlap_or_gaps():
    folds = walk_forward_folds(DATES, 4, min_train=0)
    assert len(folds) == 4
    seen = []
    for train, val in folds:
        assert not set(train) & set(val), "a date cannot be both train and hold-out"
        assert tuple(train) == tuple(seen), "train must be exactly the dates before this block"
        assert list(val) == [d for d in DATES if d in val], "a hold-out block must preserve date order"
        seen.extend(val)
    assert seen == list(DATES), "every decision date appears in exactly one hold-out block"
    # a block whose training segment is too thin is dropped so no fold reports a meaningless in-sample number
    short = walk_forward_folds(DATES, 4, min_train=40)
    assert len(short) == 2 and len(short[0][0]) >= 40


def test_walk_forward_offline_det_runs_and_reports_every_block():
    result = run_walk_forward("A", BARS, DATES, phase=DET, n_windows=4)
    assert result.phase == DET and len(result.folds) == 4
    for f in result.folds:
        assert f.val.get("total_return") is not None and f.val.get("trades", 0) >= 0
        assert 0.0 <= f.val_degraded_share <= 1.0
    assert result.oos["decisions"] == len(DATES)
    assert result.oos.get("trades", 0) >= 0
    assert "return" in result.summary()


def test_walk_forward_requires_a_model_for_ai_phases():
    with pytest.raises(ValueError):
        run_walk_forward("A", BARS, DATES, phase="+LLM", n_windows=3, llm=None)


def test_walk_forward_ai_phase_uses_only_its_window():
    phased = llm(lambda p: route(p, COT_GOOD))  # a plain BUY at 0.75 on every decision date

    result = run_walk_forward("A", BARS, DATES, phase="+LLM", n_windows=3, llm=phased)
    assert result.oos["decisions"] == len(DATES)
    for f in result.folds:
        assert f.val.get("trades", 0) + f.val["open_positions"] > 0, "the window's own BUY stream must produce fills"


def test_walk_forward_rejects_starved_windows():
    with pytest.raises(ValueError):
        walk_forward_folds(["d1", "d2"], 3)