"""Tests for the layers after the stack, and the stack's grouped combiner."""

from __future__ import annotations

import polars as pl

from taxiout import paths, postprocess


def test_airport_ceilings_clip_only_the_named_airports():
    """LEBL/LEMD labels are provider-truncated at ~3,600 s; nothing else is touched."""
    frame = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["LEBL", "LEMD", "EGLL", "LEBL"],
            "prediction": [5_000.0, 3_000.0, 9_000.0, 1_200.0],
        }
    )
    out = postprocess.apply_airport_ceilings(frame).sort("mvt_id")
    assert out["prediction"].to_list() == [3_600.0, 3_000.0, 9_000.0, 1_200.0]


def test_airport_ceilings_are_a_no_op_when_empty():
    frame = pl.DataFrame({"airport": ["LEBL"], "prediction": [5_000.0]})
    assert postprocess.apply_airport_ceilings(frame, {})["prediction"].to_list() == [5_000.0]


def test_gap_bound_clips_only_impossible_positive_gap_rows():
    frame = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4, 5],
            # late flight predicted far above its own delay | comfortably inside the bound
            # | negative gap (left alone) | null gap (left alone) | exactly on the bound
            "gap": [1_200.0, 9_000.0, -600.0, None, 1_000.0],
            "prediction": [8_000.0, 1_500.0, 9_999.0, 9_999.0, 2_800.0],
        }
    )
    out = postprocess.apply_gap_bound(frame).sort("mvt_id")
    assert out["prediction"].to_list() == [3_000.0, 1_500.0, 9_999.0, 9_999.0, 2_800.0]


def test_gap_bound_demands_the_gap_column():
    import pytest

    with pytest.raises(ValueError, match="gap column"):
        postprocess.apply_gap_bound(pl.DataFrame({"prediction": [1.0]}))


def _grouped_combiner():
    """Import from the script, which is a thin CLI — the combiner class is the testable part."""
    import sys

    sys.path.insert(0, str(paths.ROOT / "scripts"))
    from build_stack import _GroupedCombiner

    return _GroupedCombiner


def test_grouped_combiner_uses_each_group_s_own_weights():
    """The point of `--weight-groups`: two groups, two weight vectors, selected row by row."""
    import numpy as np

    combiner = _grouped_combiner()(
        {True: np.array([1.0, 0.0]), False: np.array([0.0, 1.0])}, np.array([0.5, 0.5])
    )
    matrix = np.array([[10.0, 100.0], [10.0, 100.0]])
    got = combiner.predict(matrix, [True, False])

    assert got[0] == 10.0, "the unmatched row must use the unmatched vector"
    assert got[1] == 100.0, "the matched row must use the matched vector"


def test_grouped_combiner_falls_back_for_a_group_it_never_saw():
    """A ranking-set group absent from the out-of-fold rows must take the global vector.

    Silently predicting zero, or raising, would both be worse: this is the shipping path.
    """
    import numpy as np

    combiner = _grouped_combiner()({True: np.array([1.0, 0.0])}, np.array([0.0, 1.0]))
    got = combiner.predict(np.array([[10.0, 100.0], [10.0, 100.0]]), [True, False])

    assert got[0] == 10.0
    assert got[1] == 100.0, "the unseen group must fall back to the global vector"


def test_grouped_combiner_without_groups_is_the_global_vector():
    """Called with no groups it must behave exactly like the ungrouped combiner."""
    import numpy as np

    combiner = _grouped_combiner()({True: np.array([1.0, 0.0])}, np.array([0.25, 0.75]))
    got = combiner.predict(np.array([[4.0, 8.0]]))

    assert got[0] == 0.25 * 4.0 + 0.75 * 8.0


def test_lobt_window_projects_both_ways_and_skips_rows_without_lobt():
    frame = pl.DataFrame(
        {
            "prediction": [100.0, 9_000.0, 1_000.0, 50_000.0, 500.0],
            "m_l": [5_000.0, 2_000.0, 1_200.0, None, 1_000.0],
        }
    )
    out = postprocess.apply_lobt_window(frame)["prediction"].to_list()
    w = postprocess.LOBT_WINDOW
    assert out[0] == 5_000.0 - w  # raised to the lower edge
    assert out[1] == 2_000.0 + w  # lowered to the upper edge
    assert out[2] == 1_000.0  # inside, untouched
    assert out[3] == 50_000.0  # no LOBT, untouched
    assert out[4] == 500.0  # lower edge floored at 0, inside


def test_lobt_window_lifts_negative_predictions_to_the_zero_floor():
    frame = pl.DataFrame({"prediction": [-10.0], "m_l": [100.0]})
    assert postprocess.apply_lobt_window(frame)["prediction"].to_list() == [0.0]


def test_lobt_window_demands_the_m_l_column():
    import pytest

    with pytest.raises(ValueError):
        postprocess.apply_lobt_window(pl.DataFrame({"prediction": [1.0]}))


def test_cell_shift_moves_only_the_listed_airport_months():
    frame = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["EDDF", "EDDF", "LTFM", "EGLL"],
            "month": ["2026-01", "2026-07", "2026-01", "2026-01"],
            "prediction": [900.0, 900.0, 1_000.0, 1_200.0],
        }
    )
    shifts = pl.DataFrame(
        {"airport": ["EDDF", "LTFM"], "month": ["2026-01", "2026-01"], "shift": [-21.0, -41]}
    )
    out = postprocess.apply_cell_shift(frame, shifts)
    assert out["mvt_id"].to_list() == [1, 2, 3, 4]  # order kept
    assert out["prediction"].to_list() == [879.0, 900.0, 959.0, 1_200.0]


def test_cell_shift_refuses_a_duplicated_cell_and_a_missing_column():
    import pytest

    frame = pl.DataFrame({"airport": ["EDDF"], "month": ["2026-01"], "prediction": [1.0]})
    twice = pl.DataFrame({"airport": ["EDDF"] * 2, "month": ["2026-01"] * 2, "shift": [1.0, 2.0]})
    with pytest.raises(ValueError):
        postprocess.apply_cell_shift(frame, twice)
    with pytest.raises(ValueError):
        postprocess.apply_cell_shift(frame.drop("month"), twice.head(1))
