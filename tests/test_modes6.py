"""modes6: the two-stage source classifier (L052's remedy for modes5's class imbalance).

These pin the properties the decomposition must keep from modes5 -- no mass on a null stamp, a
never-null mixture, copy-conditional entropy -- plus the two new ones: the joint probabilities are
p(copy) x p(s | copy) and still sum to one, and the balanced posterior never leaks into them.

Pure logic -- no data needed, so these always run.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from taxiout.features import modes6

_TS = pl.Datetime(time_unit="us", time_zone="UTC")


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _rows(**over) -> pl.DataFrame:
    base = {
        "mvt_id": [1],
        "airport": ["EDDF"],
        "operator": ["DLH"],
        "mvt_time": [_utc(2025, 1, 1, 12, 0)],
        "aobt_3": [_utc(2025, 1, 1, 11, 45)],  # t = 900
        "eobt_1": [_utc(2025, 1, 1, 11, 30)],  # t = 1800
        "lobt": [_utc(2025, 1, 1, 11, 20)],
        "iobt": [_utc(2025, 1, 1, 11, 10)],
        "sched_time": [_utc(2025, 1, 1, 11, 0)],
    }
    base.update(over)
    stamps = ("mvt_time", *modes6.SOURCES)
    frame = pl.DataFrame(
        {k: pl.Series(k, v, dtype=_TS) if k in stamps else pl.Series(k, v) for k, v in base.items()}
    )
    return modes6.covariates(frame.lazy()).collect()


PER_AIRPORT = pl.DataFrame({"airport": ["EDDF"], "_none_label": [1000.0]})


def _one(p_copy, cond, cond_b=None, **over):
    rows = _rows(**over)
    cond = np.array([cond], dtype=float)
    cond_b = cond if cond_b is None else np.array([cond_b], dtype=float)
    return modes6.assemble(rows, np.array([p_copy]), cond, cond_b, PER_AIRPORT, 1000.0)


def test_outputs_match_the_declaration():
    out = _one(0.5, [0.5, 0.5, 0.0, 0.0, 0.0])
    assert tuple(out.columns[1:]) == modes6.OUTPUTS


def test_joint_is_pcopy_times_conditional_and_sums_to_one():
    out = _one(0.6, [0.5, 0.25, 0.25, 0.0, 0.0])
    assert out["m6_p_aobt"][0] == pytest.approx(0.30)
    assert out["m6_p_eobt"][0] == pytest.approx(0.15)
    total = sum(out[c][0] for c in modes6._PROB.values()) + out["m6_p_none"][0]
    assert total == pytest.approx(1.0)
    assert out["m6_p_copy"][0] == pytest.approx(0.6)


def test_mixture_is_the_expectation_of_the_implied_labels():
    out = _one(0.6, [0.5, 0.5, 0.0, 0.0, 0.0])
    # 0.3 * 900 + 0.3 * 1800 + 0.4 * 1000
    assert out["m6_label_mix"][0] == pytest.approx(0.3 * 900 + 0.3 * 1800 + 0.4 * 1000)
    assert out["m6_label_copy"][0] == pytest.approx(0.5 * 900 + 0.5 * 1800)


def test_a_null_stamp_carries_no_mass_and_the_conditional_renormalises():
    out = _one(0.8, [0.5, 0.5, 0.0, 0.0, 0.0], eobt_1=[None])
    assert out["m6_p_eobt"][0] == 0.0
    assert out["m6_p_aobt"][0] == pytest.approx(0.8), "eobt's mass moves to the live aobt_3"
    assert out["m6_label_copy"][0] == pytest.approx(900.0)


def test_all_stamps_null_means_not_a_copy_and_the_median_fallback():
    out = _one(0.9, [0.2] * 5, **{s: [None] for s in modes6.SOURCES})
    assert out["m6_p_copy"][0] == 0.0
    assert out["m6_p_none"][0] == 1.0
    assert out["m6_label_copy"][0] is None
    assert out["m6_label_copy_b"][0] is None
    assert out["m6_label_mix"][0] == pytest.approx(1000.0)
    assert all(out[c][0] == 0.0 for c in modes6._PROB_B.values())


def test_the_mixture_is_never_null():
    for over in ({}, {"aobt_3": [None]}, {s: [None] for s in modes6.SOURCES}):
        out = _one(0.5, [0.2] * 5, **over)
        assert out["m6_label_mix"][0] is not None, f"null mixture with {over}"


def test_balanced_posterior_never_enters_the_calibrated_columns():
    a = _one(0.6, [0.9, 0.1, 0.0, 0.0, 0.0], cond_b=[0.1, 0.9, 0.0, 0.0, 0.0])
    b = _one(0.6, [0.9, 0.1, 0.0, 0.0, 0.0], cond_b=[0.5, 0.5, 0.0, 0.0, 0.0])
    for c in (*modes6._PROB.values(), "m6_label_mix", "m6_label_copy", "m6_entropy"):
        assert a[c][0] == b[c][0], f"{c} moved with the balanced posterior"
    assert a["m6_pb_eobt"][0] == pytest.approx(0.9)
    assert a["m6_label_copy_b"][0] == pytest.approx(0.1 * 900 + 0.9 * 1800)


def test_entropy_is_copy_conditional():
    split = _one(0.02, [0.5, 0.5, 0.0, 0.0, 0.0])
    sure = _one(0.02, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert split["m6_entropy"][0] == pytest.approx(np.log(2))
    assert sure["m6_entropy"][0] == pytest.approx(0.0)


def test_balanced_weights_equalise_class_mass():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = modes6.balanced_weights(y, 5)
    mass = [w[y == k].sum() for k in range(3)]
    assert mass == pytest.approx([mass[0]] * 3)
    assert w.sum() == pytest.approx(len(y))


def test_label_class_matches_modes5():
    """The label is modes5's by construction; a divergence would make the two incomparable."""
    from taxiout.features import modes5

    for taxi in (900, 1800, 137, 15 * 60 + 90):
        rows = _rows(taxi_sec=[taxi])
        assert modes6.label_class(rows)["_y"].to_list() == modes5.label_class(rows)["_y"].to_list()


def test_probabilities_crossfit_on_a_toy_frame():
    """End to end on synthetic rows: shapes, ranges, and stage 2 only ever fitted on copies."""
    rng = np.random.default_rng(0)
    n = 3000
    base = _utc(2025, 1, 1, 12, 0)
    mvt = [base.replace(hour=6 + int(h)) for h in rng.integers(0, 12, n)]
    from datetime import timedelta

    stamps = {
        s: [m - timedelta(minutes=10 + 5 * i) for m in mvt] for i, s in enumerate(modes6.SOURCES)
    }
    # label: a third copy aobt_3, a sixth eobt_1, the rest none
    kind = rng.choice([0, 1, 5], n, p=[1 / 3, 1 / 6, 1 / 2])
    taxi = np.where(kind == 0, 600.0, np.where(kind == 1, 900.0, rng.uniform(100, 3000, n)))
    frame = pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "airport": rng.choice(["EDDF", "EGLL"], n),
            "operator": rng.choice(["DLH", "BAW", None], n),
            "month": rng.choice(["2025-01", "2025-02", "2025-03", "2025-04"], n),
            "mvt_time": pl.Series(mvt, dtype=_TS),
            **{s: pl.Series(v, dtype=_TS) for s, v in stamps.items()},
            "taxi_sec": taxi,
        }
    )
    frame = modes6.covariates(frame.lazy()).collect()
    train = modes6.label_class(frame.filter(pl.col("month") != "2025-04"))
    old = dict(modes6._COMMON)
    try:
        modes6._COMMON.update(n_estimators=10, min_child_samples=5, n_jobs=1)
        modes6.STAGE1_PARAMS.update(modes6._COMMON)
        modes6.STAGE2_PARAMS.update(modes6._COMMON)
        p_copy, cond, cond_b = modes6.probabilities(frame, train)
    finally:
        modes6._COMMON.update(old)
        modes6.STAGE1_PARAMS.update(old)
        modes6.STAGE2_PARAMS.update(old)
    assert p_copy.shape == (n,) and cond.shape == (n, 5) and cond_b.shape == (n, 5)
    assert ((p_copy >= 0) & (p_copy <= 1)).all()
    np.testing.assert_allclose(cond.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(cond_b.sum(axis=1), 1.0, atol=1e-9)


def test_no_copy_rows_degrades_to_none_rather_than_failing():
    """The converse leak test sets every training label to 999,999 s, so no row copies a stamp.

    modes6's first real-data run failed exactly there: stage 2 had zero rows to fit. The block
    must instead put all mass on 'none', which is also what makes that test non-vacuous.
    """
    rng = np.random.default_rng(1)
    n = 800
    from datetime import timedelta

    base = _utc(2025, 1, 1, 12, 0)
    mvt = [base.replace(hour=6 + int(h)) for h in rng.integers(0, 12, n)]
    stamps = {
        s: [m - timedelta(minutes=10 + 5 * i) for m in mvt] for i, s in enumerate(modes6.SOURCES)
    }
    frame = pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "airport": rng.choice(["EDDF", "EGLL"], n),
            "operator": ["DLH"] * n,
            "month": rng.choice(["2025-01", "2025-02", "2025-03", "2025-04"], n),
            "mvt_time": pl.Series(mvt, dtype=_TS),
            **{s: pl.Series(v, dtype=_TS) for s, v in stamps.items()},
            "taxi_sec": np.full(n, 999_999.0),
        }
    )
    frame = modes6.covariates(frame.lazy()).collect()
    train = modes6.label_class(frame.filter(pl.col("month") != "2025-04"))
    assert (train["_y"] == modes6.NONE).all()
    old = dict(modes6._COMMON)
    try:
        modes6._COMMON.update(n_estimators=5, min_child_samples=5, n_jobs=1)
        modes6.STAGE1_PARAMS.update(modes6._COMMON)
        modes6.STAGE2_PARAMS.update(modes6._COMMON)
        p_copy, cond, cond_b = modes6.probabilities(frame, train)
    finally:
        modes6._COMMON.update(old)
        modes6.STAGE1_PARAMS.update(old)
        modes6.STAGE2_PARAMS.update(old)
    out = modes6.assemble(frame, p_copy, cond, cond_b, PER_AIRPORT, 1000.0)
    assert (out["m6_p_none"] == 1.0).all()
    assert (out["m6_p_aobt"] == 0.0).all()
