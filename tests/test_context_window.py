"""Unit tests for `context_window`: pure logic on a hand-built day, no data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from taxiout.features.context_window import OUTPUTS, featurize

T0 = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)


def _t(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def movements() -> pl.DataFrame:
    # departures at 12:00, 12:10, 12:20 on runway A; 12:05 on runway B; one at another airport
    # proxies (mvt - aobt_3): 600, 900, 1200, 300, 999; 12:20's aobt is absurd (-> excluded)
    rows = [
        # mvt_id, phase, airport, runway, stand, mvt(min), aobt(min), block(min), taxi_sec
        (1, "DEP", "X", "A", "S1", 0, -10, None, None),
        (2, "DEP", "X", "A", "S2", 10, -5, None, None),
        (3, "DEP", "X", "A", "S3", 20, 20 - 200, None, None),  # proxy 12,000 s -> dropped
        (4, "DEP", "X", "B", "S4", 5, 0, None, None),
        (5, "DEP", "Y", "A", "S1", 0, -16.65, None, None),
        # arrivals: landing at 11:55 (taxi-in 300) and 12:08 (taxi-in 5000 -> clipped 3600)
        (10, "ARR", "X", "A", "S1", -5, None, -10, 300),
        (11, "ARR", "X", "B", "S9", 8, None, 9, 5000),
        # stand S2 got an arrival in-block at 11:00 (age 70 min at departure 2's 12:10)
        (12, "ARR", "X", "A", "S2", -65, None, -60, 420),
    ]
    return pl.DataFrame(
        {
            "mvt_id": [r[0] for r in rows],
            "phase": [r[1] for r in rows],
            "airport": [r[2] for r in rows],
            "runway": [r[3] for r in rows],
            "stand": [r[4] for r in rows],
            "mvt_time": [_t(r[5]) for r in rows],
            "aobt_3": [None if r[6] is None else _t(r[6]) for r in rows],
            "block_time": [None if r[7] is None else _t(r[7]) for r in rows],
            "taxi_sec": [None if r[8] is None else float(r[8]) for r in rows],
        }
    )


def by_id() -> dict[int, dict]:
    out = featurize(movements())
    assert set(out.columns) == {"mvt_id", *OUTPUTS}
    return {r["mvt_id"]: r for r in out.to_dicts()}


def test_only_departures_come_out():
    assert sorted(by_id()) == [1, 2, 3, 4, 5]


def test_neighbour_proxy_windows_exclude_self_and_absurd_values():
    r = by_id()
    # dep 2 (12:10): previous 15 min at X = deps 1 (600) and 4 (300) -> 450
    assert r[2]["cw_nb_prev_apt_15"] == pytest.approx(450.0)
    # next 15 min at X: dep 3 only, whose proxy is absurd -> null
    assert r[2]["cw_nb_next_apt_15"] is None
    # dep 1 (12:00): next 15 min at X = deps 4 (300) and 2 (900) -> 600
    assert r[1]["cw_nb_next_apt_15"] == pytest.approx(600.0)
    # runway scope: dep 1's next 15 on runway A = dep 2 only
    assert r[1]["cw_nb_next_rw_15"] == pytest.approx(900.0)
    # other airport never mixes in
    assert r[5]["cw_nb_prev_apt_60"] is None


def test_own_minus_window_mean():
    r = by_id()
    assert r[2]["cw_own_minus_prev_apt_60"] == pytest.approx(900.0 - 450.0)


def test_arrival_taxi_in_window_and_clip():
    r = by_id()
    # dep 2 (12:10): arrivals landing in (11:55, 12:10] at X = only 12:08 (clipped 3600);
    # 11:55 is on the open edge of the 15-min window
    assert r[2]["cw_arr_tin_apt_15"] == pytest.approx(3600.0)
    assert r[2]["cw_arr_n_apt_15"] == 1
    # 60-min window includes both: (300 + 3600) / 2
    assert r[2]["cw_arr_tin_apt_60"] == pytest.approx(1950.0)
    # runway A only: the 11:55 arrival
    assert r[2]["cw_arr_tin_rw_60"] == pytest.approx(300.0)


def test_last_stand_arrival():
    r = by_id()
    assert r[2]["cw_stand_last_age"] == pytest.approx(70 * 60.0)
    assert r[2]["cw_stand_last_tin"] == pytest.approx(420.0)
    # dep 1 at S1: the 11:50 in-block arrival is 10 minutes old
    assert r[1]["cw_stand_last_age"] == pytest.approx(600.0)
    # dep 4 at S4: no arrival ever
    assert r[4]["cw_stand_last_age"] is None


def test_matches_brute_force_on_random_multi_group_data():
    """Regression for the rolling(group_by) ordering bug: several airports and runways."""
    import numpy as np

    rng = np.random.default_rng(1)
    n = 400
    mins = np.sort(np.round(rng.uniform(0, 600, n) * 60) / 60)  # whole seconds, like the data
    frame = pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "phase": rng.choice(["DEP", "ARR"], n),
            "airport": rng.choice(["Z", "Y", "X"], n),
            "runway": rng.choice(["09", "27", "18"], n),
            "stand": rng.choice(["S1", "S2"], n),
            "mvt_time": [_t(float(m)) for m in mins],
            "aobt_3": [_t(float(m) - round(rng.uniform(1, 40) * 60) / 60) for m in mins],
            "block_time": [_t(float(m) + 5) for m in mins],
            "taxi_sec": rng.uniform(60, 900, n),
        }
    ).sample(fraction=1.0, shuffle=True, seed=3)
    got = {r["mvt_id"]: r for r in featurize(frame).to_dicts()}
    rows = frame.to_dicts()
    for d in (r for r in rows if r["phase"] == "DEP"):
        t = d["mvt_time"]
        for scope, same in (
            ("apt", lambda o, d=d: o["airport"] == d["airport"]),
            ("rw", lambda o, d=d: o["airport"] == d["airport"] and o["runway"] == d["runway"]),
        ):
            for w in (15, 60):
                win = timedelta(minutes=w)
                prev = [
                    (o["mvt_time"] - o["aobt_3"]).total_seconds()
                    for o in rows
                    if o["phase"] == "DEP" and same(o) and t - win <= o["mvt_time"] < t
                ]
                tin = [
                    o["taxi_sec"]
                    for o in rows
                    if o["phase"] == "ARR" and same(o) and t - win < o["mvt_time"] <= t
                ]
                g = got[d["mvt_id"]]
                if prev:
                    assert g[f"cw_nb_prev_{scope}_{w}"] == pytest.approx(np.mean(prev))
                else:
                    assert g[f"cw_nb_prev_{scope}_{w}"] is None
                assert g[f"cw_arr_n_{scope}_{w}"] == len(tin)
                if tin:
                    assert g[f"cw_arr_tin_{scope}_{w}"] == pytest.approx(np.mean(tin))
