"""The placebo block: deterministic, uniform on [0, 1), independent of row order. Pure."""

from __future__ import annotations

import polars as pl

from taxiout.features.placebo import OUTPUTS, featurize


def test_columns_are_uniform_noise_in_spine_order():
    spine = pl.DataFrame({"mvt_id": list(range(1, 20_001))}, schema={"mvt_id": pl.Int64})
    out = featurize(spine)
    assert out.columns == list(OUTPUTS) and out.height == spine.height
    for c in OUTPUTS:
        v = out[c]
        assert v.dtype == pl.Float32 and v.min() >= 0.0 and v.max() < 1.0
        assert abs(v.mean() - 0.5) < 0.01
    # the three columns are not copies of one another
    assert (out["placebo_1"] != out["placebo_2"]).mean() > 0.99
    assert (out["placebo_2"] != out["placebo_3"]).mean() > 0.99


def test_a_row_keeps_its_values_whatever_the_order():
    spine = pl.DataFrame({"mvt_id": [5, 9, 1, 7]}, schema={"mvt_id": pl.Int64})
    a = featurize(spine).with_columns(spine["mvt_id"])
    b = featurize(spine.reverse()).with_columns(spine.reverse()["mvt_id"])
    assert a.sort("mvt_id").equals(b.sort("mvt_id"))
