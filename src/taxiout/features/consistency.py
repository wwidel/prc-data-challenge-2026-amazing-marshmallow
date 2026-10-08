"""Whether the record agrees with itself, and which plan value the actual was copied from.

`block_time` is written by a recording system, not measured, and on a large minority of rows it
holds a *plan* value rather than an actual one — the schedule, or one of the Network Manager
estimates. Classified by mode: the A-CDM actual on 38.9% of 2025 departures, the schedule
on a further 5.1% where `aobt_3` says something different, a plan value on 3.3%. Those modes have
very different residuals: 145 s on the A-CDM rows against 750 s on the schedule-copied ones.

The model cannot currently tell them apart. It sees `ob_aobt_sec`, `ob_eobt_sec` and `ob_sched_sec`
as separate continuous columns and would have to *subtract* them to notice that two agree
exactly — which a tree cannot do in one split. This block does the subtraction, and states the
agreements as flags.

It also carries the places the movement record and the flight record disagree. A record whose
aircraft type is missing, or whose filed destination differs from the movement's, is a record
that was assembled from partial information, and partial information is where the copied modes
live: every one of the 29 NM-unmatched LIRF departures of 2025 with a schedule gap over 12 h
has a null `ac_type`.

``ROW_LOCAL``: every column is a function of that row's own fields. Nothing reads a label.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Two timestamps "agree" within this many seconds: 60 s for plan values and 120 s for `aobt_3`,
#: whose own resolution is coarser.
TOL_PLAN = 60
TOL_AOBT = 120

OUTPUTS = (
    "cs_eobt_is_sched",
    "cs_lobt_is_sched",
    "cs_iobt_is_sched",
    "cs_aobt_is_sched",
    "cs_any_plan_is_sched",
    "cs_aobt_minus_sched",
    "cs_delay_band",
    "cs_ac_type_null",
    "cs_ac_type_disagrees",
    "cs_ades_disagrees",
    "cs_rule_disagrees",
    "cs_disagreements",
)

#: Departure delay bands in seconds. The schedule-copied mode concentrates in the 1-3 h band at
#: LIRF (33% against an 8% base), so the band is given as a code rather than left to be found.
DELAY_EDGES = [-1800.0, 0.0, 900.0, 3600.0, 10_800.0, 43_200.0]


@feature(name="consistency", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def consistency(ctx: FeatureContext) -> pl.DataFrame:
    """Plan-value agreements, the off-block delay, and record-level disagreements."""

    def agrees(left: str, right: str, tol: int) -> pl.Expr:
        return (
            ((pl.col(left) - pl.col(right)).dt.total_seconds().abs() <= tol)
            .fill_null(False)
            .cast(pl.Int32)
        )

    departures = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select(
            "mvt_id",
            "sched_time",
            "eobt_1",
            "lobt",
            "iobt",
            "aobt_3",
            "ac_type",
            "ac_type_flt",
            "ades",
            "ades_filed",
            "flight_rule",
            "flight_rule_flt",
        )
        .with_columns(
            agrees("eobt_1", "sched_time", TOL_PLAN).alias("cs_eobt_is_sched"),
            agrees("lobt", "sched_time", TOL_PLAN).alias("cs_lobt_is_sched"),
            agrees("iobt", "sched_time", TOL_PLAN).alias("cs_iobt_is_sched"),
            agrees("aobt_3", "sched_time", TOL_AOBT).alias("cs_aobt_is_sched"),
            # The off-block delay itself: how late the actual push was against the schedule.
            (pl.col("aobt_3") - pl.col("sched_time"))
            .dt.total_seconds()
            .alias("cs_aobt_minus_sched"),
            pl.col("ac_type").is_null().cast(pl.Int32).alias("cs_ac_type_null"),
            (
                pl.col("ac_type").is_not_null()
                & pl.col("ac_type_flt").is_not_null()
                & (pl.col("ac_type") != pl.col("ac_type_flt"))
            )
            .cast(pl.Int32)
            .alias("cs_ac_type_disagrees"),
            (
                pl.col("ades").is_not_null()
                & pl.col("ades_filed").is_not_null()
                & (pl.col("ades") != pl.col("ades_filed"))
            )
            .cast(pl.Int32)
            .alias("cs_ades_disagrees"),
            (
                pl.col("flight_rule").is_not_null()
                & pl.col("flight_rule_flt").is_not_null()
                & (pl.col("flight_rule") != pl.col("flight_rule_flt"))
            )
            .cast(pl.Int32)
            .alias("cs_rule_disagrees"),
        )
        .with_columns(
            pl.max_horizontal("cs_eobt_is_sched", "cs_lobt_is_sched", "cs_iobt_is_sched").alias(
                "cs_any_plan_is_sched"
            ),
            pl.sum_horizontal(
                "cs_ac_type_null", "cs_ac_type_disagrees", "cs_ades_disagrees", "cs_rule_disagrees"
            ).alias("cs_disagreements"),
            pl.col("cs_aobt_minus_sched")
            .cut(DELAY_EDGES, labels=[str(i) for i in range(len(DELAY_EDGES) + 1)])
            .cast(pl.Int32)
            .alias("cs_delay_band"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, departures, OUTPUTS)
