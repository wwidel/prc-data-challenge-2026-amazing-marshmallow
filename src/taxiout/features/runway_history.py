"""The runway's own day: who went before, what wake they carried, how the day is flowing.

Community intel made concrete: separation minima make the *pair* of wake categories on a
runway the physical unit of queue spacing (a 241 s gap is congestion for a medium behind a
medium and pure physics for a light behind a super), and the runway's day-so-far tells the
model whether it is living a quiet morning or a pressured evening.

Whole-day statistics are used deliberately. The organisers state the model is **post-operational,
not tactical** — hindsight within the day is licensed — and every input here is a `mvt_time` or
`runway` field, all present in the ranking set. ``CONTEXT``: other rows, never a label.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = (
    "rh_prev_wake",
    "rh_pair",
    "rh_gap_vs_pair",
    "rh_day_seq",
    "rh_day_n_rw",
    "rh_day_median_gap",
    "rh_next15m_rw",
)


@feature(name="runway_history", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def runway_history(ctx: FeatureContext) -> pl.DataFrame:
    """Wake-pair spacing and the runway's whole-day flow statistics."""
    dep = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "runway", "wake_cat", pl.col("mvt_time").alias("t"))
        .collect(engine="streaming")
        .sort(["airport", "runway", "t"])
        .with_columns(
            pl.col("t").dt.date().alias("day"),
            pl.col("wake_cat").fill_null("?").alias("wake"),
        )
        .with_columns(
            pl.col("wake").shift(1).over(["airport", "runway"]).fill_null("?").alias("prev_wake"),
            (pl.col("t").diff().over(["airport", "runway"])).dt.total_seconds().alias("gap"),
        )
        .with_columns((pl.col("prev_wake") + ">" + pl.col("wake")).alias("pair_str"))
    )

    pair_median = dep.group_by("pair_str").agg(pl.col("gap").median().alias("pair_med"))
    day_stats = dep.group_by(["airport", "runway", "day"]).agg(
        pl.len().cast(pl.Int32).alias("rh_day_n_rw"),
        pl.col("gap").median().alias("rh_day_median_gap"),
    )

    out = (
        dep.join(pair_median, on="pair_str", how="left")
        .join(day_stats, on=["airport", "runway", "day"], how="left")
        .with_columns(
            pl.col("prev_wake").rank("dense").cast(pl.Int32).sub(1).alias("rh_prev_wake"),
            pl.col("pair_str").rank("dense").cast(pl.Int32).sub(1).alias("rh_pair"),
            (pl.col("gap") - pl.col("pair_med")).alias("rh_gap_vs_pair"),
            pl.int_range(pl.len())
            .over(["airport", "runway", "day"], order_by="t")
            .cast(pl.Int32)
            .alias("rh_day_seq"),
        )
    )

    # Post-ops licence used directly: departures off the same runway in the 15 minutes AFTER
    # this takeoff — the forward half of the window tactical use would forbid.
    fwd = (
        out.sort("t")
        .rolling(
            index_column="t",
            period="15m",
            offset="0s",
            closed="right",
            group_by=["airport", "runway"],
        )
        .agg(pl.len().alias("rh_next15m_rw"))
        .unique(subset=["airport", "runway", "t"], keep="first")
        .with_columns(pl.col("rh_next15m_rw").cast(pl.Int32) - 1)
    )
    out = out.join(fwd, on=["airport", "runway", "t"], how="left")
    return align(ctx.spine, out.select("mvt_id", *OUTPUTS), OUTPUTS)
