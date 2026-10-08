"""The early-stopping split of `LGBMPredictor`: pure logic, no data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from taxiout.models.lgbm import LGBMPredictor


def _frame(days: int = 60, per_day: int = 7) -> pl.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=d, hours=h * 3) for d in range(days) for h in range(per_day)]
    return pl.DataFrame({"mvt_time": times}).with_columns(
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month")
    )


def test_last_slice_is_the_latest_rows_by_month():
    X = _frame()
    fit, stop = LGBMPredictor(valid_fraction=0.15)._temporal_split(X)
    assert len(fit) + len(stop) == X.height
    assert X["month"][fit].max() <= X["month"][stop].min()


def test_days_slice_holds_out_whole_days_across_months():
    X = _frame()
    model = LGBMPredictor(valid_fraction=0.15, stop_slice="days", seed=3)
    fit, stop = model._temporal_split(X)
    assert len(fit) + len(stop) == X.height
    assert len(np.intersect1d(fit, stop)) == 0
    day = X["mvt_time"].dt.date()
    # every held-out day is held out entirely
    assert set(day[stop]).isdisjoint(set(day[fit]))
    assert len(set(day[stop])) == round(60 * 0.15)
    # it is not just the end of the period
    assert len(set(X["month"][stop])) > 1


def test_days_slice_is_deterministic_per_seed():
    X = _frame()
    a = LGBMPredictor(stop_slice="days", seed=1)._temporal_split(X)[1]
    b = LGBMPredictor(stop_slice="days", seed=1)._temporal_split(X)[1]
    np.testing.assert_array_equal(a, b)


def test_unknown_stop_slice_is_rejected():
    with pytest.raises(ValueError):
        LGBMPredictor(stop_slice="random")
