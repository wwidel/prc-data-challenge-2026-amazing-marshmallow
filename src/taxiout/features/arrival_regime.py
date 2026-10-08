"""The day's ``block_time`` copy regime, read off the arrivals.

A departure's label is exact whenever its ``block_time`` copies a stamp we hold, and the ranking
set hides ``block_time`` for departures only — arrival rows keep theirs. On 2025, where both
directions carry labels, the share of an airport-day's arrivals whose on-
block time copies a stamp tracks the share of that day's departures whose off-block time does:
correlation 0.50 across airport-days and **0.33 within airport**. So the arrivals expose, label-
free, a day-level property of how the data provider recorded block times, which the departure
classifier (`modes5`) cannot see — it knows the airport, not the day.

An arrival "copies" when its on-block ``block_time`` is within ``TOL`` seconds of ``arvt_3``,
``arvt_1``, ``sched_time`` or its own landing ``mvt_time``, first match in that order. 2025 shares:
``sched_time`` 5.2%, ``arvt_1`` 3.9%, ``arvt_3`` 0.3%, landing 0.03%, none 90.5%.

Whole-day shares are used, as ``runway_history`` uses whole-day statistics: the organisers state the
model is post-operational, and every input is an arrival row the ranking set carries intact.
``CONTEXT``: other rows, never a departure's label.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Seconds within which an arrival's on-block time counts as a copy of a stamp.
TOL = 60.0

#: Candidate stamps for an arrival's on-block time, in match order.
ARR_STAMPS = ("arvt_3", "arvt_1", "sched_time", "mvt_time")

OUTPUTS = (
    "ar_day_copy",
    "ar_day_sched",
    "ar_day_arvt1",
    "ar_day_n",
    "ar_opm_copy",
    "ar_opm_n",
)


def arrival_class(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Which stamp an arrival's on-block time copies, or ``none``. Pure; unit-tested."""
    expr = pl.lit("none")
    for s in reversed(ARR_STAMPS):
        expr = (
            pl.when((pl.col("block_time") - pl.col(s)).dt.total_seconds().abs() <= TOL)
            .then(pl.lit(s))
            .otherwise(expr)
        )
    return frame.with_columns(expr.alias("arr_class"))


def regime_tables(arrivals: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Per airport x UTC day and per airport x operator x month copy shares. Pure; unit-tested."""
    a = (
        arrival_class(arrivals.filter(pl.col("block_time").is_not_null()))
        .with_columns(
            pl.col("mvt_time").dt.date().alias("day"),
            pl.col("mvt_time").dt.strftime("%Y-%m").alias("ym"),
            pl.col("operator").fill_null("\x00none"),
        )
        .with_columns((pl.col("arr_class") != "none").cast(pl.Float64).alias("_copy"))
    )
    day = a.group_by("airport", "day").agg(
        pl.col("_copy").mean().alias("ar_day_copy"),
        (pl.col("arr_class") == "sched_time").cast(pl.Float64).mean().alias("ar_day_sched"),
        (pl.col("arr_class") == "arvt_1").cast(pl.Float64).mean().alias("ar_day_arvt1"),
        pl.len().cast(pl.Int32).alias("ar_day_n"),
    )
    opm = a.group_by("airport", "operator", "ym").agg(
        pl.col("_copy").mean().alias("ar_opm_copy"),
        pl.len().cast(pl.Int32).alias("ar_opm_n"),
    )
    return day, opm


@feature(name="arrival_regime", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def arrival_regime(ctx: FeatureContext) -> pl.DataFrame:
    """The arrivals' copy shares on the departure's airport-day and airport-operator-month."""
    mv = io.scan_movements()
    arrivals = mv.filter(pl.col("phase") == "ARR").select(
        "airport", "operator", "mvt_time", "block_time", *[s for s in ARR_STAMPS if s != "mvt_time"]
    )
    day, opm = regime_tables(arrivals)
    deps = (
        mv.filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "operator", "mvt_time")
        .with_columns(
            pl.col("mvt_time").dt.date().alias("day"),
            pl.col("mvt_time").dt.strftime("%Y-%m").alias("ym"),
            pl.col("operator").fill_null("\x00none"),
        )
        .join(day, on=["airport", "day"], how="left")
        .join(opm, on=["airport", "operator", "ym"], how="left")
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, deps, OUTPUTS)
