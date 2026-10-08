"""OPDI ground-time block: callsign keys, matching, previous leg. Pure logic — always runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from taxiout.features import opdi

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


def deps() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4, 5],
            "airport": ["LIRF", "LIRF", "EHAM", "LIRF", "LIRF"],
            "mvt": [T0, T0 + timedelta(hours=2), T0, T0 + timedelta(hours=5), T0],
            "callsign": [None, "RYR4852", "KLM1", None, None],
            "flight_no": ["ITYTY672", "FR4852", "KL0001", "AZ610", "QFAQFA6"],
        }
    )


def flights() -> pl.DataFrame:
    def ts(h: float) -> datetime:
        return T0 + timedelta(hours=h)

    return pl.DataFrame(
        {
            "icao24": ["aaa", "aaa", "bbb", "ccc", "ddd", "ddd"],
            "cs": ["ITY671", "ITY672", "RYR4852", "KLM1", "QFA5", "QFA6"],
            "adep": ["LIML", "LIRF", "LIRF", "LFPG", "YSSY", "LIRF"],
            "ades": ["LIRF", "LIML", "EGSS", "EHAM", "LIRF", "YSSY"],
            "fs": [
                ts(-4),
                ts(0) + timedelta(seconds=30),
                ts(2) - timedelta(seconds=40),
                ts(0),
                ts(-50),
                ts(0),
            ],
            "ls": [ts(-3), ts(1), ts(4), ts(1), ts(-40), ts(20)],
        }
    )


def test_dedup_prefix_strips_a_doubled_airline_code_only():
    s = pl.DataFrame({"x": ["ITYTY672", "RYRR8830", "QFAQFA6", "RYR4852", "TAY9LA", "FR4852"]})
    out = s.select(opdi.dedup_prefix(pl.col("x"))).to_series().to_list()
    assert out == ["ITY672", "RYR8830", "QFA6", None, None, None]


def test_featurize_matches_by_key_order_airport_and_time():
    out = opdi.featurize(deps(), flights()).sort("mvt_id")
    found = dict(zip(out["mvt_id"], out["op_found"], strict=True))
    # 1: ITYTY672 -> ITY672 via the dedup key; 2: NM callsign RYR4852; 3: KLM1 departs LFPG, not
    # EHAM -> no match; 4: AZ610 is in no flight list; 5: QFAQFA6 -> QFA6
    assert found == {1: 1, 2: 1, 3: 0, 4: 0, 5: 1}


def test_featurize_ground_time_and_same_airport_flag():
    out = opdi.featurize(deps(), flights()).sort("mvt_id")
    row1 = out.filter(pl.col("mvt_id") == 1)
    # previous leg of aaa landed (last seen) at T0 − 3 h; this one lifted off at T0 + 30 s
    assert row1["op_turn_s"][0] == 3 * 3600 + 30
    assert row1["op_same_apt"][0] == 1
    assert row1["op_lag_s"][0] == 30
    # ddd's previous leg ended 36 h + before: outside TURN, so null; RYR has no previous leg
    assert out.filter(pl.col("mvt_id") == 5)["op_turn_s"][0] is None
    assert out.filter(pl.col("mvt_id") == 2)["op_turn_s"][0] is None


def test_match_respects_the_tolerance():
    d = opdi.keys(deps()).filter(pl.col("mvt_id") == 1)
    far = flights().with_columns(pl.col("fs") + timedelta(hours=2))
    assert opdi.match(d, far, tol=1_800).height == 0
