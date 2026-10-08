"""The airline schedule, and what it means when nothing else is known.

``sched_time`` is the only time reference that survives on the rows that decide the metric.
Departures with no Network Manager match are 1.6% of rows and 57.6% of the squared error, and
every ``_flt`` column is null together for them — ``aobt_3`` and ``eobt_1``, the two strongest
features in the model, are absent for exactly those rows. ``sched_time`` is a movement-record
field and is present on 100% of them.

The gap between takeoff and schedule behaves completely differently on each side of that flag.
Over 2025, a gap above twelve hours means a mean taxi of 1,262 s and P(taxi > 12 h) = 0.0018
when the flight matched; when it did not, the same gap means a mean of 26,978 s and
P(taxi > 12 h) = 0.387, over 75 rows. Inside the unmatched population the gap correlates 0.442
with the target overall and 0.896 at LIRF, against 0.205 on matched rows.

``sc_gap_nomatch`` exists because of how trees reach that cell. Given the gap and the flag
separately, the model needs two splits to isolate a region holding 22 rows of the seasonal folds,
which its leaf constraint forbids. Carrying the
interaction as its own column turns that into one split on a value no matched row can hold.

Everything here reads the row's own fields, so it is ``ROW_LOCAL``. ``mvt_time`` is the takeoff
time, which is given in the ranking set; only ``block_time`` is blanked.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = (
    "sc_gap_sec",
    "sc_gap_nomatch",
    "sc_has_sched",
    "sc_sched_hour",
)


@feature(name="schedule", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def schedule(ctx: FeatureContext) -> pl.DataFrame:
    """Takeoff minus scheduled departure, plain and crossed with the unmatched flag.

    The gap is *not* clipped. Its extreme values are the signal — the cell this block exists to
    expose sits above twelve hours — so squashing the range would defeat the purpose.
    """
    departures = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "mvt_time", "sched_time", "flight_id")
        .with_columns(
            (pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("sc_gap_sec"),
            pl.col("sched_time").is_not_null().cast(pl.Int32).alias("sc_has_sched"),
            pl.col("sched_time").dt.hour().alias("sc_sched_hour"),
        )
        .with_columns(
            # Zero for every matched row, so one split isolates the unmatched tail.
            pl.when(pl.col("flight_id").is_null())
            .then(pl.col("sc_gap_sec"))
            .otherwise(0)
            .alias("sc_gap_nomatch"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, departures, OUTPUTS)
