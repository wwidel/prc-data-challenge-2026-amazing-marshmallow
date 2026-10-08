"""Pure-logic tests for the weather block's helpers on synthetic observations."""

from __future__ import annotations

import datetime as dt

import polars as pl

from taxiout.features.weather import enrich_obs, featurize, subzero_streaks

UTC = dt.UTC


def _t(hour: int, minute: int = 0, day: int = 2) -> dt.datetime:
    return dt.datetime(2025, 1, day, hour, minute, tzinfo=UTC)


def metar_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "airport": ["EHAM"] * 4 + ["LIRF"],
            "time": [_t(6), _t(7), _t(8), _t(9), _t(6)],
            "temp_c": [-2.0, -1.0, 1.0, 4.0, 12.0],
            "dewpoint_c": [-3.0, -2.0, 0.0, 1.0, 8.0],
            "rh": [90.0, 92.0, 93.0, 80.0, 60.0],
            "wind_dir": [180.0, 190.0, 200.0, 210.0, 90.0],
            "wind_kt": [10.0, 12.0, 14.0, 8.0, 5.0],
            "gust_kt": [None, 20.0, None, None, None],
            "precip_mm": [0.0, 1.0, 2.0, None, 0.0],
            "vis_km": [8.0, 2.0, 1.5, 10.0, 10.0],
            "wx": [None, "-SN", "FZRA BR", None, None],
        }
    )


def test_enrich_flags_and_rolling_sums():
    obs = enrich_obs(metar_frame()).filter(pl.col("airport") == "EHAM").sort("time")
    assert obs["snow"].to_list() == [0, 1, 0, 0]
    assert obs["freezing"].to_list() == [0, 0, 1, 0]
    assert obs["fog"].to_list() == [0, 0, 1, 0]
    # 3h window at 08:00 sees 06:00 (0), 07:00 (1) and 08:00 (2); null precip counts as 0
    assert obs["precip_3h_mm"].to_list() == [0.0, 1.0, 3.0, 3.0]
    # de-icing proxy: cold with precip or frozen codes; 09:00 is 4 degrees and dry
    assert obs["deice_proxy"].to_list() == [0, 1, 1, 0]


def test_frozen_hours_scale_with_cadence():
    obs = enrich_obs(metar_frame()).filter(pl.col("airport") == "EHAM").sort("time")
    # at 09:00, 2 of 4 obs in 24h were sub-zero -> half the elapsed-window estimate
    assert obs["frozen_hours_24h"].to_list()[-1] == 12.0


def test_subzero_streak_is_yesterdays():
    metar = pl.DataFrame(
        {
            "airport": ["EHAM"] * 4,
            "time": [_t(6, day=d) for d in (1, 2, 3, 4)],
            "temp_c": [-5.0, -1.0, 2.0, -3.0],
        }
    ).with_columns(
        dewpoint_c=pl.lit(None, pl.Float64),
        rh=pl.lit(None, pl.Float64),
        wind_dir=pl.lit(None, pl.Float64),
        wind_kt=pl.lit(None, pl.Float64),
        gust_kt=pl.lit(None, pl.Float64),
        precip_mm=pl.lit(None, pl.Float64),
        vis_km=pl.lit(None, pl.Float64),
        wx=pl.lit(None, pl.Utf8),
    )
    st = subzero_streaks(metar).sort("date")
    # day1 streak(yday)=0; day2 sees day1's streak 1; day3 sees 2; day4 sees 0 (day3 warm)
    assert st["streak_yesterday"].to_list() == [0, 1, 2, 0]


def test_featurize_backward_asof_and_age():
    spine = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3],
            "airport": ["EHAM", "EHAM", "LIRF"],
            "mvt_time": [_t(7, 30), _t(6, 10), _t(23)],
        }
    )
    out = featurize(spine, metar_frame()).sort("mvt_id")
    # row 1: latest obs at 07:00 -> temp -1, age 1800s
    assert out["wx_temp_c"].to_list()[0] == -1.0
    assert out["wx_obs_age_s"].to_list()[0] == 1800
    # row 2: 06:10 sees the 06:00 obs
    assert out["wx_temp_c"].to_list()[1] == -2.0
    # row 3: LIRF only has a morning obs; still backward-matched, never EHAM's
    assert out["wx_temp_c"].to_list()[2] == 12.0
    assert out.height == spine.height
