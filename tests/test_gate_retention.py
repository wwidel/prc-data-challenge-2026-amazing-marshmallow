"""Unit tests for `gate_retention`: pure logic on a hand-built hour, no data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from taxiout.features.gate_retention import OUTPUTS, featurize

T0 = datetime(2025, 6, 1, 10, 0, tzinfo=UTC)


def _t(m):
    return None if m is None else T0 + timedelta(minutes=m)


def deps() -> pl.DataFrame:
    # mvt_id, runway, eobt, aobt, mvt, sched   (minutes after 10:00)
    rows = [
        (1, "A", 0, 10, 25, -5),  # pushes at 10:10
        (2, "A", 5, 20, 35, 0),  # overdue from 10:05 to 10:20
        (3, "A", 8, 12, 30, 0),  # overdue 10:08-10:12; taxiing 10:12-10:30
        (4, "B", 2, 15, 28, 0),  # other runway: overdue 10:02-10:15
        (5, "A", None, 30, 40, 0),  # no eobt -> never counted as overdue
    ]
    return pl.DataFrame(
        {
            "mvt_id": [r[0] for r in rows],
            "airport": ["X"] * len(rows),
            "runway": [r[1] for r in rows],
            "eobt_1": [_t(r[2]) for r in rows],
            "aobt_3": [_t(r[3]) for r in rows],
            "mvt_time": [_t(r[4]) for r in rows],
            "sched_time": [_t(r[5]) for r in rows],
        }
    ).with_columns(pl.col("eobt_1").cast(pl.Datetime("us", "UTC")))


def by_id():
    out = featurize(deps())
    assert set(out.columns) == {"mvt_id", *OUTPUTS}
    return {r["mvt_id"]: r for r in out.to_dicts()}


def test_overdue_counts_at_the_push_clock():
    r = by_id()
    # at 10:10 (row 1 pushes): overdue = 2 (since 10:05), 3 (since 10:08), 4 (since 10:02, rwy B)
    assert r[1]["gr_overdue_apt"] == 3
    assert r[1]["gr_overdue_rw"] == 2
    # at 10:20 (row 2 pushes): nobody is overdue any more (2 itself pushes at 10:20)
    assert r[2]["gr_overdue_apt"] == 0


def test_active_on_runway_excludes_self():
    r = by_id()
    # at 10:20 on runway A: 1 (10:10-10:25) and 3 (10:12-10:30) are moving
    assert r[2]["gr_active_rw"] == 2
    # at 10:10: nobody else on A has pushed
    assert r[1]["gr_active_rw"] == 0


def test_recent_delay_to_push():
    r = by_id()
    # pushes in [10:05, 10:20) before row 2: rows 1 (delay 15 min), 3 (12), 4 (15)
    assert r[2]["gr_delay_15m"] == pytest.approx((15 + 12 + 15) * 60 / 3)
    # nothing pushed in the 15 min before 10:10
    assert r[1]["gr_delay_15m"] is None
