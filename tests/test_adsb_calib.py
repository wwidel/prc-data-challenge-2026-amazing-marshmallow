"""Unit tests for `adsb_calib`: the trace against its own airport-month, no data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from taxiout.features.adsb_calib import OUTPUTS, featurize


def deps() -> pl.DataFrame:
    jan = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
    jul = datetime(2025, 7, 10, 12, 0, tzinfo=UTC)
    # airport X, January: three departures, two traced; recorded taxi (mvt - aobt) 900 each.
    # traces start 400 and 600 s after off-block -> taxi_move 500 and 300 -> lags -400, -600.
    # July: one traced departure whose trace starts 100 s after off-block (lag -100).
    rows = [
        (1, "X", jan, jan - timedelta(seconds=900), 500.0, 20.0),
        (2, "X", jan, jan - timedelta(seconds=900), 300.0, 40.0),
        (3, "X", jan, jan - timedelta(seconds=900), None, None),
        (4, "X", jul, jul - timedelta(seconds=900), 800.0, 60.0),
        (5, "Y", jan, jan - timedelta(seconds=900), None, None),
    ]
    return pl.DataFrame(
        {
            "mvt_id": [r[0] for r in rows],
            "airport": [r[1] for r in rows],
            "mvt_time": [r[2] for r in rows],
            "aobt_3": [r[3] for r in rows],
            "adsb_taxi_move": [r[4] for r in rows],
            "adsb_n_ground": [r[5] for r in rows],
        }
    )


def by_id():
    out = featurize(deps())
    assert set(out.columns) == {"mvt_id", *OUTPUTS}
    return {r["mvt_id"]: r for r in out.to_dicts()}


def test_month_statistics_are_per_airport_month():
    r = by_id()
    assert r[1]["ad_match_rate_m"] == pytest.approx(2 / 3)
    assert r[1]["ad_med_lag_m"] == pytest.approx(-500.0)  # median of -400 and -600
    assert r[1]["ad_med_n_ground_m"] == pytest.approx(30.0)
    assert r[4]["ad_match_rate_m"] == pytest.approx(1.0)
    assert r[4]["ad_med_lag_m"] == pytest.approx(-100.0)
    assert r[5]["ad_match_rate_m"] == pytest.approx(0.0) and r[5]["ad_med_lag_m"] is None


def test_lag_adjusted_taxi_removes_the_month_s_typical_lag():
    r = by_id()
    # trace 1 implies 500 s; the month's typical lag is -500 -> adjusted 1000
    # (= the recorded 900 plus this trace's +100 against the typical lag)
    assert r[1]["ad_taxi_move_adj"] == pytest.approx(1000.0)
    assert r[1]["ad_lag_vs_m"] == pytest.approx(100.0)
    assert r[2]["ad_lag_vs_m"] == pytest.approx(-100.0)
    assert r[1]["ad_n_ground_rel"] == pytest.approx(20 / 30)
    # July's single trace equals its month median -> adjusted taxi = recorded taxi
    assert r[4]["ad_taxi_move_adj"] == pytest.approx(900.0) and r[4][
        "ad_lag_vs_m"
    ] == pytest.approx(0.0)


def test_untraced_rows_keep_the_month_columns_only():
    r = by_id()
    assert r[3]["ad_match_rate_m"] == pytest.approx(2 / 3)
    assert (
        r[3]["ad_taxi_move_adj"] is None
        and r[3]["ad_lag_vs_m"] is None
        and r[3]["ad_n_ground_rel"] is None
    )
