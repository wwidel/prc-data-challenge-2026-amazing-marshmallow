"""The snow block: windows, heavy snow, freezing rain, the day's count, time since snow. Pure."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from taxiout.features.snow import OUTPUTS, featurize

UTC = dt.UTC


def t(h: float) -> dt.datetime:
    return dt.datetime(2026, 2, 2, tzinfo=UTC) + dt.timedelta(hours=h)


def metar() -> pl.DataFrame:
    # half-hourly reports at AAAA from 00:00 to 23:30; snow from 06:00 to 08:30 (6 reports, the
    # first two heavy), freezing rain at 10:00, -2 °C throughout; BBBB has no snow at all
    rows = []
    for i in range(48):
        h = i / 2
        wx = None
        if 6 <= h < 9:
            wx = "+SN" if h < 7 else "-SN"
        if h == 10:
            wx = "FZRA"
        rows.append({"airport": "AAAA", "time": t(h), "wx": wx, "temp_c": -2.0})
        rows.append({"airport": "BBBB", "time": t(h), "wx": None, "temp_c": 5.0})
    return pl.DataFrame(rows, schema_overrides={"wx": pl.Utf8})


def spine() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["AAAA", "AAAA", "AAAA", "BBBB"],
            "mvt_time": [t(5.9), t(9.0), t(12.0), t(12.0)],
        }
    )


def test_outputs_and_rows():
    out = featurize(spine(), metar())
    assert out.columns == ["mvt_id", *OUTPUTS]
    assert sorted(out["mvt_id"].to_list()) == [1, 2, 3, 4]


def test_windows_count_the_reports_before_takeoff():
    out = featurize(spine(), metar()).sort("mvt_id")
    r1, r2, r3, r4 = out.iter_rows(named=True)
    assert r1["sn_3h"] == 0 and r1["sn_last_h"] == pytest.approx(96.0)  # before the snow
    # windows are half-open, (t − h, t]: at 09:00 the 3 h window holds 06:30–08:30, not 06:00
    assert r2["sn_3h"] == 5 and r2["sn_6h"] == 6 and r2["sn_heavy_6h"] == 2
    assert r2["sn_last_h"] == pytest.approx(0.5)
    assert r3["sn_3h"] == 0 and r3["sn_6h"] == 5 and r3["sn_12h"] == 6  # 06:00 is exactly 6 h back
    assert r3["fz_6h"] == 1 and r3["sn_48h_cold"] == 1
    assert r3["sn_day"] == 6 and r1["sn_day"] == 6  # the whole UTC day, post-operational
    assert r4["sn_24h"] == 0 and r4["sn_day"] == 0 and r4["sn_48h_cold"] == 0
