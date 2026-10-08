"""arrival_regime: the day's block_time copy regime, read off the arrivals. Pure logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from taxiout.features import arrival_regime as ar

_TS = pl.Datetime(time_unit="us", time_zone="UTC")


def _t(h: int, m: int = 0, s: int = 0, day: int = 1) -> datetime:
    return datetime(2025, 1, day, h, m, s, tzinfo=UTC)


def _arrivals(rows: list[dict]) -> pl.LazyFrame:
    cols = ("mvt_time", "block_time", "arvt_3", "arvt_1", "sched_time")
    data = {
        "airport": [r.get("airport", "EDDF") for r in rows],
        "operator": [r.get("operator", "DLH") for r in rows],
        **{c: pl.Series(c, [r.get(c) for r in rows], dtype=_TS) for c in cols},
    }
    return pl.DataFrame(data).lazy()


def test_first_match_order_and_tolerance():
    land = _t(10)
    rows = [
        # copies arvt_3 within 60 s (and sched too): arvt_3 wins, first in order
        {
            "mvt_time": land,
            "block_time": _t(10, 5),
            "arvt_3": _t(10, 5, 30),
            "sched_time": _t(10, 5),
        },
        # 61 s from arvt_1: not a copy
        {"mvt_time": land, "block_time": _t(10, 6), "arvt_1": _t(10, 7, 1)},
        # copies its own landing time
        {"mvt_time": land, "block_time": _t(10, 0, 20)},
        # copies sched only
        {"mvt_time": land, "block_time": _t(10, 9), "sched_time": _t(10, 9, 59)},
    ]
    got = ar.arrival_class(_arrivals(rows)).collect()["arr_class"].to_list()
    assert got == ["arvt_3", "none", "mvt_time", "sched_time"]


def test_day_and_operator_month_shares():
    rows = [
        {"mvt_time": _t(8), "block_time": _t(8, 4), "sched_time": _t(8, 4)},  # sched copy
        {"mvt_time": _t(9), "block_time": _t(9, 6), "arvt_1": _t(9, 6, 10)},  # arvt_1 copy
        {"mvt_time": _t(10), "block_time": _t(10, 7)},  # none
        {"mvt_time": _t(11), "block_time": _t(11, 8), "operator": None},  # none, null operator
    ]
    day, opm = ar.regime_tables(_arrivals(rows))
    d = day.collect()
    assert d.height == 1
    assert d["ar_day_copy"][0] == pytest.approx(0.5)
    assert d["ar_day_sched"][0] == pytest.approx(0.25)
    assert d["ar_day_arvt1"][0] == pytest.approx(0.25)
    assert d["ar_day_n"][0] == 4
    o = opm.collect().sort("operator")
    assert o.height == 2, "a null operator is its own group, not dropped"
    dlh = o.filter(pl.col("operator") == "DLH")
    assert dlh["ar_opm_copy"][0] == pytest.approx(2 / 3)
    assert dlh["ar_opm_n"][0] == 3


def test_null_block_time_rows_are_excluded():
    rows = [
        {"mvt_time": _t(8), "block_time": None, "sched_time": _t(8)},
        {"mvt_time": _t(9), "block_time": _t(9, 5), "sched_time": _t(9, 5)},
    ]
    day, _ = ar.regime_tables(_arrivals(rows))
    d = day.collect()
    assert d["ar_day_n"][0] == 1 and d["ar_day_copy"][0] == pytest.approx(1.0)


def test_days_are_separate():
    rows = [
        {"mvt_time": _t(8, day=1), "block_time": _t(8, 4, day=1), "sched_time": _t(8, 4, day=1)},
        {"mvt_time": _t(8, day=2), "block_time": _t(8, 4, day=2)},
    ]
    day, _ = ar.regime_tables(_arrivals(rows))
    d = day.collect().sort("day")
    assert d["ar_day_copy"].to_list() == [1.0, 0.0]
    assert d["day"][1] - d["day"][0] == timedelta(days=1)
