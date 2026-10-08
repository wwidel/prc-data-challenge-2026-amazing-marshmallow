"""modes5: the trained six-way source classifier.

These target the defects `a084_modes4_review.py` found in the predecessor, because those are the
known failure modes of this idea rather than hypothetical ones: a null fallback propagating into the
mixture, a mean where the distribution is skewed, and an entropy taken over the wrong support so
that "confident" came to mean "confidently not a copy".

Pure logic — no data needed, so these always run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from taxiout.features import modes5


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


#: Stamps must keep a real datetime dtype even when the value is null, as they do in the raw data:
#: a bare [None] infers Null dtype and arithmetic against it raises, which is a fixture artifact
#: rather than a module defect.
_TS = pl.Datetime(time_unit="us", time_zone="UTC")


def _rows(**over) -> pl.DataFrame:
    base = {
        "mvt_id": [1],
        "airport": ["EDDF"],
        "mvt_time": [_utc(2025, 1, 1, 12, 0)],
        "aobt_3": [_utc(2025, 1, 1, 11, 45)],
        "eobt_1": [_utc(2025, 1, 1, 11, 30)],
        "lobt": [_utc(2025, 1, 1, 11, 20)],
        "iobt": [_utc(2025, 1, 1, 11, 10)],
        "sched_time": [_utc(2025, 1, 1, 11, 0)],
    }
    base.update(over)
    stamps = ("mvt_time", *modes5.SOURCES)
    return pl.DataFrame(
        {k: pl.Series(k, v, dtype=_TS) if k in stamps else pl.Series(k, v) for k, v in base.items()}
    )


def test_label_class_respects_the_per_stamp_tolerance_and_first_match_wins():
    """aobt_3 carries 120 s, the plan stamps 60 s: an asymmetry that is part of the label.

    See a023 and a084 -- the coarser tolerance moved aobt_3 from 21.0% to 39.1% of the rows.
    """
    # block_time 90 s from aobt_3: inside aobt_3's 120 s, outside a 60 s rule.
    frame = _rows(taxi_sec=[15 * 60 + 90])  # mvt - 16:30 => block at 11:43:30, 90s before aobt_3
    got = modes5.label_class(frame)["_y"].to_list()
    assert got == [modes5.SOURCES.index("aobt_3")], "90 s from aobt_3 must classify as aobt_3"

    # 90 s from eobt_1 only: outside eobt_1's 60 s tolerance, so 'none'.
    frame = _rows(
        aobt_3=[None],
        taxi_sec=[30 * 60 + 90],
    )
    got = modes5.label_class(frame)["_y"].to_list()
    assert got == [len(modes5.SOURCES)], "90 s from eobt_1 is outside its 60 s tolerance"


def test_label_class_marks_a_non_copy_as_none():
    frame = _rows(taxi_sec=[900])  # block at 11:45 exactly == aobt_3
    assert modes5.label_class(frame)["_y"].to_list() == [0]
    frame = _rows(taxi_sec=[137])  # block at 11:57:43, near nothing
    assert modes5.label_class(frame)["_y"].to_list() == [len(modes5.SOURCES)]


def test_the_none_fallback_is_a_median_not_a_mean():
    """`modes4` used the mean of a right-skewed distribution and over-predicted by +68 to +72 s."""
    none_cls = len(modes5.SOURCES)
    train = pl.DataFrame(
        {
            "airport": ["EDDF"] * 5,
            "taxi_sec": [600.0, 700.0, 800.0, 900.0, 60_000.0],  # one monster
            "_y": [none_cls] * 5,
        }
    )
    per_airport, global_median = modes5.none_label_medians(train)
    got = float(per_airport.filter(pl.col("airport") == "EDDF")["_none_label"][0])

    assert got == 800.0, "must be the median"
    assert got != float(np.mean([600, 700, 800, 900, 60_000])), "must NOT be the mean"
    assert global_median == 800.0


def test_the_none_fallback_has_a_global_floor_when_an_airport_has_no_none_rows():
    """A null fallback propagates into the mixture; that bug was caught in modes4."""
    train = pl.DataFrame(
        {"airport": ["EDDF", "EGLL"], "taxi_sec": [500.0, 900.0], "_y": [0, len(modes5.SOURCES)]}
    )
    per_airport, global_median = modes5.none_label_medians(train)
    assert global_median == 900.0, "the global median comes from the 'none' rows when any exist"
    # EDDF has no 'none' row at all, so it is absent from per_airport and must fall back.
    assert "EDDF" not in per_airport["airport"].to_list()


def _assemble_one(proba_row, **over):
    rows = _rows(**over)
    proba = np.array([proba_row], dtype=float)
    per_airport = pl.DataFrame({"airport": ["EDDF"], "_none_label": [1000.0]})
    return modes5.assemble(rows, proba, per_airport, 1000.0)


def test_a_null_stamp_carries_no_probability_and_its_mass_falls_to_none():
    """An unmatched row has all four _flt stamps null together; mass must not sit on them."""
    # all mass on eobt_1, but eobt_1 is null -> must end up as p_none = 1
    out = _assemble_one([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], eobt_1=[None])
    assert out["m5_p_eobt"][0] == 0.0
    assert out["m5_p_none"][0] == 1.0
    assert out["m5_p_copy"][0] == 0.0
    assert out["m5_label_copy"][0] is None, "no evaluable stamp means no copy-conditional label"
    assert out["m5_label_mix"][0] == 1000.0, "the whole mixture falls back on the median"


def test_the_mixture_is_never_null():
    """m5_label_mix is what the downstream model reads; a null there is a missing value."""
    for over in ({}, {"aobt_3": [None]}, {s: [None] for s in modes5.SOURCES}):
        out = _assemble_one([0.2, 0.2, 0.2, 0.1, 0.1, 0.2], **over)
        assert out["m5_label_mix"][0] is not None, f"null mixture with {over}"


def test_the_mixture_weights_the_implied_labels():
    """All mass on aobt_3 must give exactly mvt_time - aobt_3."""
    out = _assemble_one([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert out["m5_label_mix"][0] == 900.0, "12:00 - 11:45 = 900 s"
    assert out["m5_label_copy"][0] == 900.0


def test_entropy_is_copy_conditional_not_over_all_six_classes():
    """The a084 inversion: over six classes, 'confident' meant 'confidently not a copy'.

    A row that is 98% 'none' but splits the remaining 2% evenly between two stamps is *maximally
    unsure about which stamp*, which is what this column must say. Taken over all six classes it
    would instead look highly confident.
    """
    split = _assemble_one([0.01, 0.01, 0.0, 0.0, 0.0, 0.98])
    sure = _assemble_one([0.02, 0.0, 0.0, 0.0, 0.0, 0.98])

    assert split["m5_entropy"][0] > sure["m5_entropy"][0], (
        "an even split between two stamps must be higher-entropy than all mass on one"
    )
    assert abs(split["m5_entropy"][0] - np.log(2)) < 1e-9, "a 50/50 split is exactly log 2"
    assert abs(sure["m5_entropy"][0]) < 1e-9, "all copy-mass on one stamp is zero entropy"
    # And both have the same p_none, so entropy is carrying which-copy, not is-it-a-copy.
    assert abs(split["m5_p_none"][0] - sure["m5_p_none"][0]) < 1e-9


def test_top_prob_is_the_largest_stamp_probability():
    out = _assemble_one([0.1, 0.4, 0.05, 0.05, 0.1, 0.3])
    assert abs(out["m5_top_prob"][0] - 0.4) < 1e-12


def test_the_design_matrix_column_order_is_fixed():
    """The classifier is fitted and predicted in separate calls, so column order must be stable."""
    assert len(modes5.FEATURES) == len(set(modes5.FEATURES)), "no duplicate covariates"
    assert modes5.FEATURES[-len(modes5.CATEGORICAL) :] == modes5.CATEGORICAL
    assert len(modes5.PAIRS) == 10, "five stamps give ten unordered pairs"
    assert len(modes5.NUMERIC) == 5 + 10 + 3


def test_the_six_classes_put_none_last():
    """label_class returns len(SOURCES) for 'none', so the ordering is load-bearing."""
    assert modes5.CLASSES[-1] == "none"
    assert modes5.CLASSES[: len(modes5.SOURCES)] == modes5.SOURCES
    assert modes5.CLF_PARAMS["num_class"] == len(modes5.CLASSES)


def _synthetic(n_per_month: int = 250) -> pl.DataFrame:
    """A small labelled set spanning three months, so the cross-fit groups are all non-empty."""
    rng = np.random.default_rng(7)
    months = ["2025-01", "2025-02", "2025-03"]
    n = n_per_month * len(months)
    takeoff = [_utc(2025, 1, 1, 6, 0) + timedelta(seconds=int(x)) for x in rng.integers(0, 8e5, n)]
    jitter = rng.integers(0, 7, n).astype(int)
    lags = {"aobt_3": 900, "eobt_1": 1500, "lobt": 2100, "iobt": 2700, "sched_time": 3300}

    frame = pl.DataFrame(
        {
            "mvt_id": list(range(n)),
            "airport": [["EDDF", "EGLL", "LIRF"][i % 3] for i in range(n)],
            "month": [months[i // n_per_month] for i in range(n)],
            "operator": [["AAA", "BBB", None][i % 3] for i in range(n)],
            "mvt_time": pl.Series("mvt_time", takeoff, dtype=_TS),
            **{
                k: pl.Series(
                    k,
                    [
                        t - timedelta(seconds=v + int(j))
                        for t, j in zip(takeoff, jitter, strict=True)
                    ],
                    dtype=_TS,
                )
                for k, v in lags.items()
            },
            # half the rows copy aobt_3 exactly, half do not
            "taxi_sec": [
                float(900 + jitter[i]) if i % 2 == 0 else float(500 + (i % 300)) for i in range(n)
            ],
        }
    )
    return modes5.covariates(frame.lazy()).collect()


def test_the_classifier_is_deterministic():
    """`subsample` and `colsample_bytree` make a fit stochastic without a seed.

    A feature block whose output changes between two builds of the same input has a meaningless
    cache key, and the leak test in test_pipeline.py computes the block twice and compares every
    column — so it would fail at random and for the wrong reason.
    """
    covs = _synthetic()
    train = modes5.label_class(covs)
    first = modes5.probabilities(covs, train)
    second = modes5.probabilities(covs, train)

    assert np.array_equal(first, second), "two identical calls gave different probabilities"
    assert first.shape == (covs.height, len(modes5.CLASSES))
    # rows must be distributions
    assert np.allclose(first.sum(axis=1), 1.0, atol=1e-6)


def test_cross_fitting_covers_every_training_row_and_uses_every_group():
    """A training row predicted by a model that saw it would be sharp in-sample and misleading."""
    covs = _synthetic()
    train = modes5.label_class(covs)
    proba = modes5.probabilities(covs, train)

    assert not np.isnan(proba).any(), "every row must get a probability vector"
    assert (proba.sum(axis=1) > 0).all(), "no row may be left at all-zero"
    months = sorted(train["month"].unique().to_list())
    assert len(months) >= modes5.N_INNER, "the fixture must span at least N_INNER months"
