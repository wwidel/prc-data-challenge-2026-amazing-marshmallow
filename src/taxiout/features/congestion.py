"""Traffic around the departure: how busy the airport and the runway were, and for how long.

Early on this looked like the biggest untouched lever, but the evidence was split, and this block
exists to settle it rather than to argue it.

Against: an early review built rolling counts ad hoc and measured them at 0.042 correlation with
a baseline model's residual (0.144 with the target), which is about 0.4 s of RMSE. Its
argument for why is the strong one — ``ob_aobt_sec`` is Eurocontrol's reconstruction of the
actual off-block time, so on the 98.4% of rows that carry it the queue is already *inside* the
feature the model leans on hardest, and this block can only restate it.

For: a group-mean oracle over ``airport x runway x 10 min`` has a ceiling near 11.5 s once
corrected for the shrinkage attenuation we measured — the review's own estimate of 1.4 s came
through an estimator that recovers 12% of a
known effect at 3.4 rows per group. That ceiling is for *any* feature resolving the cell exactly,
and counts are a coarse function of it, so the two findings can both be true.

So the columns here are deliberately not all counts. ``cg_prev_gap_rw`` and ``cg_stand_idle_s``
are timings rather than volumes, and the runway-level windows are the ones the 0.318
autocorrelation of taxi-out at a 2-5 minute lag points at.

``CONTEXT``: every input is another row's ``mvt_time`` or ``block_time``, never a departure's
``taxi_sec``. Departure takeoff times and complete arrival records are both present in the
ranking set, so this is computed contemporaneously for 2026 rather than carried over.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Preceding windows for the volume counts, as (label, polars duration).
AIRPORT_WINDOWS = (("5m", "5m"), ("15m", "15m"), ("60m", "60m"))
RUNWAY_WINDOWS = (("5m", "5m"), ("15m", "15m"))

OUTPUTS = (
    "cg_dep_5m",
    "cg_dep_15m",
    "cg_dep_60m",
    "cg_dep_rw_5m",
    "cg_dep_rw_15m",
    "cg_arr_15m",
    "cg_arr_60m",
    "cg_prev_gap_rw",
    "cg_stand_idle_s",
)


def _rolling_count(frame: pl.DataFrame, keys: list[str], period: str, name: str) -> pl.DataFrame:
    """Rows of `frame` in the `period` ending at each timestamp, per key, excluding the row itself.

    `rolling` emits one row per input row, so two movements sharing a timestamp produce two
    identical rows. They see the same window by construction, so one is kept — without that the
    join back onto the departures would be a cartesian product on every tied second.
    """
    return (
        frame.sort("t")
        .rolling(index_column="t", period=period, closed="right", group_by=keys)
        .agg(pl.len().alias(name))
        .unique(subset=[*keys, "t"], keep="first")
        .with_columns(pl.col(name).cast(pl.Int32) - 1)
    )


@feature(name="congestion", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def congestion(ctx: FeatureContext) -> pl.DataFrame:
    """Rolling movement counts, the gap behind the previous departure, and stand turnaround.

    Windows look *backwards* only. The traffic that queues a departure is the traffic already
    airborne or rolling ahead of it, and a forward window would also be unavailable in the sense
    that matters — it would let a late-afternoon row see the evening bank.
    """
    movements = io.scan_movements().select(
        "mvt_id", "phase", "airport", "runway", "stand", "mvt_time", "block_time"
    )

    departures = (
        movements.filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "runway", "stand", pl.col("mvt_time").alias("t"))
        .collect(engine="streaming")
    )
    arrivals = (
        movements.filter(pl.col("phase") == "ARR")
        .select("airport", "stand", pl.col("mvt_time").alias("t"), "block_time")
        .collect(engine="streaming")
    )

    out = departures

    # Departure volume, airport-wide and per runway.
    for label, period in AIRPORT_WINDOWS:
        counts = _rolling_count(departures, ["airport"], period, f"cg_dep_{label}")
        out = out.join(counts, on=["airport", "t"], how="left")
    for label, period in RUNWAY_WINDOWS:
        counts = _rolling_count(departures, ["airport", "runway"], period, f"cg_dep_rw_{label}")
        out = out.join(counts, on=["airport", "runway", "t"], how="left")

    # Arrival volume: landings compete for the same surface, and where runways run mixed-mode
    # they compete for the runway itself. Landing times are not departure times, so this is an
    # as-of lookup of the most recent count rather than an equality join.
    arr_only = arrivals.select("airport", "t")
    out = out.sort(["airport", "t"])
    for label, period in (("15m", "15m"), ("60m", "60m")):
        counted = (
            arr_only.sort("t")
            .rolling(index_column="t", period=period, closed="right", group_by=["airport"])
            .agg(pl.len().cast(pl.Int32).alias(f"cg_arr_{label}"))
            .unique(subset=["airport", "t"], keep="first")
            .sort(["airport", "t"])
        )
        out = out.join_asof(counted, on="t", by="airport", strategy="backward")

    # Seconds since the previous departure off the same runway — the direct throughput measure.
    gaps = (
        departures.sort(["airport", "runway", "t"])
        .with_columns(
            pl.col("t")
            .diff()
            .over(["airport", "runway"])
            .dt.total_seconds()
            .alias("cg_prev_gap_rw")
        )
        .select("mvt_id", "cg_prev_gap_rw")
    )
    out = out.join(gaps, on="mvt_id", how="left")

    # Turnaround: how long the stand has been occupied by the aircraft now leaving it.
    last_on_block = (
        arrivals.filter(pl.col("block_time").is_not_null())
        .select("airport", "stand", pl.col("block_time").alias("on_block"))
        .sort(["airport", "stand", "on_block"])
    )
    out = (
        out.sort(["airport", "stand", "t"])
        .join_asof(
            last_on_block,
            left_on="t",
            right_on="on_block",
            by=["airport", "stand"],
            strategy="backward",
        )
        .with_columns(
            (pl.col("t") - pl.col("on_block")).dt.total_seconds().alias("cg_stand_idle_s")
        )
    )

    return align(ctx.spine, out.select("mvt_id", *OUTPUTS), OUTPUTS)
