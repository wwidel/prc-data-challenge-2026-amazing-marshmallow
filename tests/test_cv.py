"""Cross-validation protocol: the fold sets and the invariants they must satisfy.

Pure logic — no data needed, so these always run.
"""

from __future__ import annotations

from taxiout import cv


def test_quarters_fold_set_keeps_both_evaluation_months_and_costs_a_third_of_all_months():
    """`quarters` exists so the two-fold steering can be checked without 22.7 h of fitting.

    Two properties make it the right middle path, and both are asserted because losing either
    would make it pointless: it must *contain* the evaluation months (so the steering signal is
    retained, not replaced), and it must be materially smaller than `all_months`.
    """
    seasonal = {m for f in cv.FOLD_SETS["seasonal"] for m in f.valid_months}
    quarters = {m for f in cv.FOLD_SETS["quarters"] for m in f.valid_months}
    all_months = {m for f in cv.FOLD_SETS["all_months"] for m in f.valid_months}

    assert seasonal <= quarters, "quarters must keep Jan and Jul, not replace them"
    assert quarters < all_months, "quarters must be a strict subset of all_months"
    assert len(quarters) == 4
    # every fold trains on the other eleven months, so cost scales with the fold count
    assert len(quarters) * 3 == len(all_months), "4 folds against 12: a third of the cost"


def test_no_fold_ever_trains_on_its_own_validation_month():
    """The one invariant that holds for EVERY fold set, whatever its shape."""
    for name, folds in cv.FOLD_SETS.items():
        for fold in folds:
            overlap = set(fold.valid_months) & set(fold.train_months)
            assert not overlap, f"{name}: fold {fold.name} trains on its own validation {overlap}"


def test_leave_one_month_out_sets_train_on_the_other_eleven():
    """Eleven training months is a property of the leave-one-month-out sets specifically.

    `gap6` and `adjacent_dec` deliberately train on six, so asserting eleven universally is wrong —
    the earlier version of this test did, and correctly failed when those sets were added.
    """
    for name in ("seasonal", "all_months", "quick", "winter", "quarters"):
        for fold in cv.FOLD_SETS[name]:
            assert len(fold.train_months) == 11, (
                f"{name}: fold {fold.name} trains on {len(fold.train_months)} months, expected 11"
            )


def test_the_gap_pair_differs_only_in_adjacency():
    """`gap6` and `adjacent_dec` are a matched pair or they measure nothing.

    Same validation month and the same number of training months, so the only difference is whether
    training abuts the validation month. The first version of `adjacent_dec` used
    leave-one-month-out, giving it 11 training months against gap6's 6 — which confounded the gap
    with training size and would have made the experiment uninterpretable.
    """
    gap = cv.FOLD_SETS["gap6"][0]
    adj = cv.FOLD_SETS["adjacent_dec"][0]
    assert gap.valid_months == adj.valid_months, "the pair must share a validation month"
    assert len(gap.train_months) == len(adj.train_months), (
        f"training size must match: gap6 has {len(gap.train_months)}, "
        f"adjacent_dec has {len(adj.train_months)}"
    )
    # adjacency: adjacent_dec's training must reach the month before validation, gap6's must not
    assert "2025-11" in adj.train_months, "adjacent_dec must abut the validation month"
    assert "2025-11" not in gap.train_months, "gap6 must not abut the validation month"
    assert max(gap.train_months) == "2025-06", "gap6's gap should be six months"


def test_steering_rows_applies_both_exclusions_together():
    """The steering metric's population is an intersection, not two filters in sequence.

    This is the definition that `pooled_rmse_trimmed_ex_lirf` rests on, and it had no test
    until 2026-09-28 — which is how the runner came to print the *untrimmed* ex-LIRF under the
    label "steer by this" for the whole project, while the number the ledger actually steers by
    appeared nowhere in its output.
    """
    import numpy as np

    airport = np.array(["EDDF", "LIRF", "EDDF", "LIRF"])
    truth = np.array([100.0, 100.0, cv.TRIM_SECONDS + 1.0, cv.TRIM_SECONDS + 1.0])
    mask = cv.steering_rows(airport, truth)

    assert list(mask) == [True, False, False, False], (
        "only the row that is BOTH away from LIRF and within the trim may survive"
    )
    # Each exclusion alone keeps strictly more rows than the pair — that gap is the whole point.
    assert (airport != cv.STEERING_AIRPORT).sum() == 2
    assert (truth <= cv.TRIM_SECONDS).sum() == 2
    assert mask.sum() == 1


def test_the_trim_boundary_is_inclusive():
    """`<=`, matching TRIM_SECONDS' docstring: 43,200 s is a kept row, 43,201 is not."""
    import numpy as np

    airport = np.array(["EDDF", "EDDF"])
    truth = np.array([float(cv.TRIM_SECONDS), float(cv.TRIM_SECONDS) + 1.0])
    assert list(cv.steering_rows(airport, truth)) == [True, False]


def test_the_runner_labels_the_steering_metric_as_the_one_to_steer_by():
    """Guards the specific regression that prompted all of this.

    The failure was not a wrong computation — `pooled_rmse_ex_lirf` was correct for what it is.
    It was a *label* sitting against the wrong field. So the thing to assert is the pairing:
    whatever "steer by this" points at in the runner must be the field CVResult documents as the
    steering metric.
    """
    from taxiout import paths

    source = (paths.ROOT / "scripts" / "run_experiments.py").read_text()
    # Comment lines mention the marker when explaining it; only emitted lines count.
    marked = [
        line
        for line in source.splitlines()
        if "steer by this" in line and not line.strip().startswith("#")
    ]
    assert len(marked) == 1, f"expected exactly one steering marker, found {len(marked)}"
    assert "pooled_rmse_trimmed_ex_lirf" in marked[0], (
        f"'steer by this' must label the trimmed ex-LIRF, but labels: {marked[0].strip()}"
    )
