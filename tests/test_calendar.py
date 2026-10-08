"""calendar: local time-of-day columns. Pure logic -- a toy spine, no data needed.

Exists because the block shipped for ten days with minute-of-day computed in Int8 (polars
evaluates dt.hour() * 60 in the input's narrow integer type), so the column held minute-of-day
modulo 256 and the cyclic pair encoded a 256-minute period. Nothing caught it: the null-share /
distinct-count screen passes a column that wraps (REVIEW-SUMMARY, 2026-10-03, critical 1).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl

from taxiout.features import get
from taxiout.features.registry import FeatureContext


def _spine(times: list[datetime], airport: str = "EGLL") -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "mvt_id": list(range(n)),
            "airport": [airport] * n,
            "month": ["2025-01"] * n,
            "split": ["train"] * n,
            "mvt_time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "taxi_sec": [600.0] * n,
        }
    )


def test_minute_of_day_spans_the_whole_day():
    # EGLL is UTC+0 in January, so local wall time equals UTC here.
    times = [
        datetime(2025, 1, 15, h, m, tzinfo=UTC) for h, m in ((0, 0), (5, 30), (13, 30), (23, 59))
    ]
    out = get("calendar").compute(FeatureContext(spine=_spine(times)))
    assert out["cal_local_minute_of_day"].to_list() == [0, 330, 810, 1439]


def test_cyclic_pair_has_a_1440_minute_period():
    times = [datetime(2025, 1, 15, 5, 30, tzinfo=UTC), datetime(2025, 1, 15, 18, 0, tzinfo=UTC)]
    out = get("calendar").compute(FeatureContext(spine=_spine(times)))
    for minute, s, c in zip((330, 1080), out["cal_tod_sin"], out["cal_tod_cos"], strict=True):
        assert math.isclose(s, math.sin(minute / 1440 * 2 * math.pi), abs_tol=1e-6)
        assert math.isclose(c, math.cos(minute / 1440 * 2 * math.pi), abs_tol=1e-6)
