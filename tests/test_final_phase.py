"""The final phase: the phase switch, the paired folds and the noise-widened bounds. Pure logic."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from taxiout import cv, paths


def _in_phase(phase: str, code: str) -> dict:
    """Run ``code`` (which prints one JSON object) in a fresh interpreter under ``phase``."""
    env = {**os.environ, "TAXIOUT_PHASE": phase, "PYTHONPATH": str(paths.ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


PROBE = """
import json
from taxiout import paths, postprocess
print(json.dumps({
    "ranking": paths.RANKING.name, "submitting": paths.SUBMITTING.name,
    "months": list(paths.RANK_MONTHS), "noise": paths.TIME_NOISE,
    "dirs": [p.name for p in (paths.SPINE, paths.FEATURES, paths.PREDICTIONS, paths.RANK_PREDS,
                              paths.OVERRIDES)],
    "lobt": postprocess.LOBT_WINDOW, "gap": postprocess.GAP_SLACK,
    "ceilings": postprocess.AIRPORT_CEILINGS,
}))
"""


def test_board_phase_is_the_default_and_unchanged():
    got = _in_phase("board", PROBE)
    assert got["ranking"] == "ranking.parquet" and got["submitting"] == "submitting.parquet"
    assert got["months"] == ["2026-01", "2026-07"]
    assert got["noise"] == 0.0
    assert got["dirs"] == ["spine", "features", "predictions", "rank_preds", "external"]
    assert (got["lobt"], got["gap"]) == (3606.0, 1800.0)
    assert got["ceilings"] == {"LEBL": 3600.0, "LEMD": 3600.0}


def test_final_phase_reads_the_final_files_and_writes_apart():
    got = _in_phase("final", PROBE)
    assert got["ranking"] == "final_ranking.parquet"
    assert got["submitting"] == "final_submitting.parquet"
    assert got["months"] == ["2026-01", "2026-02", "2026-06", "2026-07"]
    assert got["dirs"] == [
        "spine_final",
        "features_final",
        "predictions_final",
        "rank_preds_final",
        "overrides_final",
    ]
    # each bound widens by the organisers' ±12 s noise (twice for a label's two timestamps)
    assert (got["lobt"], got["gap"]) == (3618.0, 1812.0)
    assert got["ceilings"] == {"LEBL": 3624.0, "LEMD": 3624.0}


def test_an_unknown_phase_is_refused():
    with pytest.raises(subprocess.CalledProcessError):
        _in_phase("finale", "import taxiout.paths")


def test_final_pairs_hold_each_pair_out_and_train_on_the_other_ten():
    folds = cv.get_folds("final_pairs")
    assert [f.valid_months for f in folds] == [("2025-01", "2025-07"), ("2025-02", "2025-06")]
    for f in folds:
        assert len(f.train_months) == 10
        assert not set(f.train_months) & set(f.valid_months)
        assert set(f.train_months) | set(f.valid_months) == set(paths.TRAIN_MONTHS)


def test_single_pair_fold_sets_are_the_two_folds_of_final_pairs():
    pairs = {f.name: f for f in cv.get_folds("final_pairs")}
    for name in ("final_pairs_janjul", "final_pairs_febjun"):
        (fold,) = cv.get_folds(name)
        assert pairs[fold.name] == fold
