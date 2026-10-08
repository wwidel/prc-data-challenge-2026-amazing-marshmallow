"""Unit tests for the ADS-B block's event detection and matching: pure logic, no data."""

from __future__ import annotations

import polars as pl
import pytest

from taxiout.features.adsb import events, match, to_features


def points() -> pl.DataFrame:
    rows = []
    # A: on the ground at X from t=0 to 600 every 30 s, moving from t=120; airborne at 630
    for t in range(0, 601, 30):
        rows.append(
            (
                "A",
                "X",
                "ABC123" if t == 300 else None,
                float(t),
                1.0,
                2.0,
                True,
                5.0 if t >= 120 else 0.0,
                "adsb_icao",
            )
        )
    rows.append(("A", "X", "ABC123", 630.0, 1.01, 2.01, False, 140.0, "adsb_icao"))
    # B: two ground segments split by a 700 s silence; the second ends in a takeoff at 1050
    for t in (0, 50, 100):
        rows.append(("B", "X", None, float(t), 1.0, 2.0, True, 0.0, "mlat"))
    for t in (800, 900, 1000):
        rows.append(("B", "X", None, float(t), 1.1, 2.1, True, 8.0, "mlat"))
    rows.append(("B", "X", None, 1050.0, 1.2, 2.2, False, 150.0, "mlat"))
    # C: on the ground only (no takeoff)
    rows.append(("C", "X", None, 10.0, 1.0, 2.0, True, 0.0, "adsb_icao"))
    return pl.DataFrame(
        rows,
        schema={
            "icao": pl.Utf8,
            "airport": pl.Utf8,
            "flight": pl.Utf8,
            "t": pl.Float64,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "ground": pl.Boolean,
            "gs": pl.Float64,
            "src": pl.Utf8,
        },
        orient="row",
    )


def test_events_find_one_takeoff_per_departing_segment():
    ev = events(points()).sort("icao")
    assert ev["icao"].to_list() == ["A", "B"]
    a, b = ev.to_dicts()
    assert a["takeoff"] == 630.0 and a["first_ground"] == 0.0 and a["first_move"] == 120.0
    assert a["flight"] == "ABC123" and a["n_ground"] == 21 and a["gap_max"] == 30.0
    assert a["frac_mlat"] == 0.0
    # B's takeoff segment starts after the 700 s silence
    assert b["takeoff"] == 1050.0 and b["first_ground"] == 800.0 and b["n_ground"] == 3
    assert b["frac_mlat"] == 1.0


def deps() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["X", "X", "X", "X"],
            "ts": [640.0, 1060.0, 5000.0, 650.0],
            "cs": ["ABC123", None, None, None],
        }
    )


def test_match_by_callsign_then_time_one_to_one():
    m = match(deps(), events(points())).sort("mvt_id")
    got = {r["mvt_id"]: (r["icao"], r["pri"]) for r in m.to_dicts()}
    assert got[1] == ("A", 0)  # callsign match
    assert got[2] == ("B", 1)  # time match, 10 s
    assert 3 not in got  # nothing within 90 s
    assert 4 not in got  # A is taken by its callsign match; B is 400 s away


def test_features():
    d = deps()
    f = {r["mvt_id"]: r for r in to_features(d, match(d, events(points()))).to_dicts()}
    assert f[1]["adsb_taxi"] == pytest.approx(640.0)
    assert f[1]["adsb_taxi_move"] == pytest.approx(520.0)
    assert f[1]["adsb_takeoff_err"] == pytest.approx(-10.0)
    assert f[1]["adsb_match_pri"] == 0.0
    assert f[2]["adsb_taxi"] == pytest.approx(260.0)
    assert f[3]["adsb_taxi"] is None
