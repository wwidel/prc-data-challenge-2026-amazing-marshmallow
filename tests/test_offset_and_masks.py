"""OffsetPredictor, SubsetPredictor and RoutedPredictor's per-route feature masks. Pure logic."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from taxiout import models
from taxiout.models.meta import RoutedPredictor
from taxiout.models.offset import OffsetPredictor

CONSTANT = {"kind": "constant", "params": {}}


def frame(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mvt_id": range(n),
            "airport": ["LIRF" if i % 2 else "EDDF" for i in range(n)],
            "baseline": [100.0 * (i + 1) for i in range(n)],
            "sc_gap_sec": [float(i) for i in range(n)],
            "other": [1.0] * n,
        }
    )


# --- OffsetPredictor -----------------------------------------------------------------------


def test_offset_fits_residual_and_adds_baseline_back():
    X = frame()
    y = X["baseline"].to_numpy() + 50.0  # residual over the baseline is a constant 50
    model = OffsetPredictor(column="baseline", base=CONSTANT).fit(X, y)
    # The constant base learned the residual's mean (50); adding the baseline back must
    # reproduce y exactly.
    assert model.predict(X) == pytest.approx(y)


def test_offset_prediction_tracks_the_baseline_out_of_sample():
    X = frame()
    y = X["baseline"].to_numpy() + 50.0
    model = OffsetPredictor(column="baseline", base=CONSTANT).fit(X, y)
    shifted = X.with_columns((pl.col("baseline") + 1000).alias("baseline"))
    assert model.predict(shifted) == pytest.approx(y + 1000)


def test_offset_null_baseline_contributes_zero_not_nan():
    X = frame().with_columns(
        pl.when(pl.col("mvt_id") == 0).then(None).otherwise(pl.col("baseline")).alias("baseline")
    )
    y = np.full(X.height, 500.0)
    out = OffsetPredictor(column="baseline", base=CONSTANT).fit(X, y).predict(X)
    assert np.isfinite(out).all()


def test_offset_missing_column_raises_before_the_base_is_fitted():
    model = OffsetPredictor(column="nope", base=CONSTANT)
    with pytest.raises(KeyError, match="nope"):
        model.fit(frame(), np.zeros(8))
    assert model.base_ is None


def test_offset_registered_and_satisfies_protocol():
    from taxiout.models.base import Predictor

    built = models.factory("offset", {"column": "baseline", "base": CONSTANT})()
    assert isinstance(built, OffsetPredictor)
    assert isinstance(built.fit(frame(), np.zeros(8)), Predictor)


# --- RoutedPredictor masks -----------------------------------------------------------------


class RecordingBase:
    """A base that records which columns it was fitted and asked to predict with."""

    def __init__(self) -> None:
        self.fit_columns: list[str] | None = None

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> RecordingBase:
        self.fit_columns = X.columns
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert X.columns == self.fit_columns, "predict saw different columns than fit"
        return np.zeros(X.height)


def routed_with_recording(**kwargs) -> RoutedPredictor:
    model = RoutedPredictor(
        by="airport", routes={"rome": ["LIRF"]}, base=CONSTANT, min_rows=1, **kwargs
    )
    model._factory = RecordingBase  # substitute after construction; spec stays valid
    return model


def test_masks_hide_listed_prefixes_from_that_route_only():
    X = frame()
    model = routed_with_recording(drop_for={"rest": ["sc_*"]})
    model.fit(X, np.zeros(X.height))
    assert "sc_gap_sec" not in model.models_["rest"].fit_columns
    assert "sc_gap_sec" in model.models_["rome"].fit_columns
    assert "other" in model.models_["rest"].fit_columns


def test_masks_apply_identically_at_predict_time():
    X = frame()
    model = routed_with_recording(drop_for={"rest": ["sc_*"], "rome": ["other"]})
    model.fit(X, np.zeros(X.height))
    model.predict(X)  # RecordingBase asserts fit/predict column agreement per route


def test_exact_column_names_mask_too():
    X = frame()
    model = routed_with_recording(drop_for={"rome": ["other"]})
    model.fit(X, np.zeros(X.height))
    assert "other" not in model.models_["rome"].fit_columns
    assert "other" in model.models_["rest"].fit_columns


def test_no_masks_is_the_old_behaviour():
    X = frame()
    model = routed_with_recording()
    model.fit(X, np.zeros(X.height))
    assert model.models_["rest"].fit_columns == X.columns


def test_offset_clip_bounds_the_anchor_before_it_is_used():
    X = pl.DataFrame({"anchor": [-50.0, 100.0, 9_000.0, None], "other": [1.0] * 4})
    y = np.array([10.0, 110.0, 7_210.0, 10.0])
    model = OffsetPredictor(column="anchor", base=CONSTANT, clip=[0, 7_200]).fit(X, y)
    # residuals against the clipped, null-filled anchor are 10 on every row, so the constant is 10
    assert np.allclose(model.predict(X), [10.0, 110.0, 7_210.0, 10.0])


def test_offset_without_clip_is_unchanged():
    X = pl.DataFrame({"anchor": [9_000.0], "other": [1.0]})
    model = OffsetPredictor(column="anchor", base=CONSTANT).fit(X, np.array([9_010.0]))
    assert model.predict(X)[0] == 9_010.0


def test_subset_fits_only_the_listed_rows_and_falls_back_elsewhere():
    X = frame(8)
    y = np.where(X["airport"].to_numpy() == "LIRF", 1_000.0, 10.0)
    model = models.factory("subset", {"by": "airport", "values": ["LIRF"], "base": CONSTANT})().fit(
        X, y
    )
    out = model.predict(X)
    lirf = X["airport"].to_numpy() == "LIRF"
    assert np.allclose(out[lirf], 1_000.0)  # the base saw LIRF rows only
    assert np.allclose(out[~lirf], 10.0)  # the fallback saw the others only


def test_subset_composes_with_offset_for_the_gate_delay_form():
    X = frame(8)
    gap = X["sc_gap_sec"].to_numpy()
    y = gap - 3.0  # T = D − G with G = 3 everywhere
    spec = {
        "by": "airport",
        "values": ["LIRF"],
        "base": {"kind": "offset", "params": {"column": "sc_gap_sec", "base": CONSTANT}},
    }
    out = models.factory("subset", spec)().fit(X, y).predict(X)
    lirf = X["airport"].to_numpy() == "LIRF"
    assert np.allclose(out[lirf], gap[lirf] - 3.0)


def test_subset_refuses_an_empty_value_list():
    with pytest.raises(ValueError):
        models.factory("subset", {"by": "airport", "values": []})()
