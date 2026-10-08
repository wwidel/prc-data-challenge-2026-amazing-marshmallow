"""Unit tests for stand_runway_lo — the low-cardinality categorical subset.

The block exists because the ablation showed the full `stand_runway` block *costs* 2.51 trimmed
ex-LIRF, and the cardinalities point at LightGBM's categorical partitioning as the cause. So the
tests assert the two properties that make it a valid test of that reading: it must emit exactly the
low-cardinality codes and nothing above the cut, and its codes must be dense, deterministic and
stable across the training/ranking boundary.
"""

from __future__ import annotations

import polars as pl

from taxiout.features.stand_runway import CATEGORICALS
from taxiout.features.stand_runway_lo import KEEP, MAX_LEVELS, OUTPUTS, featurize


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["EDDF", "EDDF", "EGLL", "EGLL"],
            "runway": ["07L", "25R", "27L", "27L"],
            "wake_cat": ["M", "H", "M", None],
            "market_segment": ["Traditional", "Lowcost", "Traditional", "Cargo"],
            "flight_type": ["S", "S", "N", "S"],
            "flight_rule": ["I", "I", "I", "V"],
        }
    )


def test_emits_exactly_the_declared_outputs():
    out = featurize(frame())
    assert set(out.columns) == {"mvt_id", *OUTPUTS}


def test_drops_every_high_cardinality_code():
    """The point of the block: no stand, ades, operator, ac_type or the 12,053-level pair."""
    dropped = {"stand", "ades", "operator", "ac_type"}
    for source in dropped:
        emitted = f"lo_{CATEGORICALS[source][4:]}"
        assert emitted not in OUTPUTS, f"{source} must not survive the cut"
    assert "cat_stand_runway" not in OUTPUTS
    # and the survivors are the ones the module claims
    assert set(KEEP) == {
        "airport",
        "runway",
        "wake_cat",
        "market_segment",
        "flight_type",
        "flight_rule",
    }


def test_codes_are_dense_and_zero_based():
    out = featurize(frame())
    for column in OUTPUTS:
        values = sorted(out[column].unique().to_list())
        assert values == list(range(len(values))), f"{column} is not a dense 0..n-1 code: {values}"


def test_a_null_level_gets_its_own_code_rather_than_being_dropped():
    out = featurize(frame()).sort("mvt_id")
    # row 4 has wake_cat = None; it must still receive a code, distinct from M and H
    assert out["lo_wake"][3] is not None
    assert out["lo_wake"].n_unique() == 3, "M, H and missing are three levels"


def test_airport_runway_pair_is_within_the_cardinality_cut():
    out = featurize(frame())
    # EDDF|07L, EDDF|25R, EGLL|27L -> three distinct pairs from four rows
    assert out["lo_airport_runway"].n_unique() == 3
    assert out["lo_airport_runway"].n_unique() <= MAX_LEVELS


def test_codes_are_deterministic_across_calls():
    a, b = featurize(frame()), featurize(frame())
    assert a.equals(b)


def test_codes_are_stable_when_rows_are_reordered():
    """Dense rank is alphabetical, so the same level must map to the same code either way."""
    f = frame()
    forward = featurize(f).sort("mvt_id")
    backward = featurize(f.reverse()).sort("mvt_id")
    assert forward.equals(backward)


def test_lo_outputs_are_numeric_to_every_estimator():
    """Documents a fact the x30-x36 ablations rest on (REVIEW-SUMMARY major 38, 2026-10-03).

    The estimators treat a column as categorical only if its name starts with ``cat_``; this
    block's outputs start with ``lo_``, so the codes it keeps reach LightGBM, CatBoost and the rest
    as *numeric* columns. Its arms therefore changed two things at once -- dropping the wide codes
    and re-encoding the kept ones -- and the two effects have not been separated. If the prefix
    ever changes, the arms' meaning changes with it, and this test should be updated deliberately.
    """
    from taxiout.features.stand_runway_lo import OUTPUTS
    from taxiout.models.lgbm import CATEGORICAL_PREFIX

    assert CATEGORICAL_PREFIX == "cat_"
    assert not [c for c in OUTPUTS if c.startswith(CATEGORICAL_PREFIX)]
