"""Queue depth with the estimated off-block taken from the flight plan first.

The same sweep as :mod:`taxiout.features.queue`, with one parameter changed: the estimated
off-block coalesces ``eobt_1`` before ``aobt_3``. An external review rebuilt the blinded
analysis's ``q_ahead`` observably and found the plan-first ordering worth 5.6 s of held-out
group-mean RMSE over the actual-first one (398.6 against 404.2, baseline 413.9), which is the
opposite of what :mod:`queue` assumes. Kept as its own module so ``f23_queue`` stays
reproducible — the feature cache is keyed on module source.

Original rationale follows.

Queue depth on the tarmac — how many aircraft are taxiing around me, not how many took off.

Every existing congestion column is anchored on ``mvt_time`` and counts *takeoffs* in a
trailing window, which measures throughput. What a taxiing aircraft actually waits behind is
queue **depth**: the number of other aircraft that pushed back before it and have not yet
left the ground. The blinded analysis measured this as the strongest dynamic driver available
(correlation with clean taxi 0.40 at EGLL, 0.39 LTFM, 0.34 LFPG; airport x q_ahead reaches
held-out RMSE 390.3 against 413.9 for airport alone — ``scripts/analysis/fresh_fable/
f03_congestion.py``), and dismissed it as unusable because it needs other flights'
``block_time``. It does not: an *estimated* off-block from the flight-plan columns is enough,
and those are present on every 2026 row.

Computed as a sweep. Each departure contributes +1 at its estimated off-block and −1 at its
takeoff; the running sum over an airport is the number of aircraft on the move at any instant.
Reading that curve at my own off-block gives the queue ahead of me, and at my takeoff the
queue behind. ``CONTEXT``, not target-dependent: no label is touched — ``aobt_3``, ``eobt_1``,
``iobt``, ``lobt`` and ``sched_time`` are all available for 2026 departures.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = (
    "q2_ahead_at_ob",
    "q2_still_at_tko",
    "q2_ahead_rw",
    "q2_est_ob_source",
    "q2_est_taxi_sec",
)

#: Preference order for the estimated off-block. All are present in the ranking set; the
#: ``_flt`` ones are null together on unmatched rows, which is what the source flag records.
OB_SOURCES = ("eobt_1", "aobt_3", "iobt", "lobt", "sched_time")


def estimated_offblock(frame: pl.DataFrame) -> pl.DataFrame:
    """First available off-block estimate, with an integer flag naming which one it was."""
    est = pl.coalesce([pl.col(c) for c in OB_SOURCES]).alias("est_ob")
    source = pl.lit(len(OB_SOURCES), dtype=pl.Int32)
    for i, c in reversed(list(enumerate(OB_SOURCES))):
        source = pl.when(pl.col(c).is_not_null()).then(pl.lit(i, dtype=pl.Int32)).otherwise(source)
    return frame.with_columns(est, source.alias("q2_est_ob_source"))


def occupancy(deps: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Running count of aircraft between estimated off-block and takeoff, per key group.

    Ties are ordered so that a +1 lands before a −1 at the same instant: reading the curve at
    my own off-block therefore includes me (subtracted by the caller), and reading it at my
    takeoff excludes me. Pure; unit-tested.
    """
    events = pl.concat(
        [
            deps.select(*keys, pl.col("est_ob").alias("t"), pl.lit(1, dtype=pl.Int32).alias("d")),
            deps.select(
                *keys, pl.col("mvt_time").alias("t"), pl.lit(-1, dtype=pl.Int32).alias("d")
            ),
        ]
    ).sort([*keys, "t", "d"], descending=[*([False] * len(keys)), False, True])
    events = events.with_columns(pl.col("d").cum_sum().over(keys).alias("occ"))
    return events.group_by([*keys, "t"]).agg(pl.col("occ").last()).sort([*keys, "t"])


def featurize(deps: pl.DataFrame) -> pl.DataFrame:
    """Queue-depth columns for every departure. Pure; unit-tested."""
    deps = estimated_offblock(deps).filter(pl.col("est_ob").is_not_null())
    out = deps.select("mvt_id", "airport", "runway", "est_ob", "mvt_time", "q2_est_ob_source")

    airport_curve = occupancy(deps, ["airport"])
    out = (
        out.sort(["airport", "est_ob"])
        .join_asof(airport_curve, left_on="est_ob", right_on="t", by="airport", strategy="backward")
        .rename({"occ": "_occ_ob"})
        .drop("t")
        .sort(["airport", "mvt_time"])
        .join_asof(
            airport_curve, left_on="mvt_time", right_on="t", by="airport", strategy="backward"
        )
        .rename({"occ": "_occ_tko"})
        .drop("t")
    )

    runway_curve = occupancy(deps, ["airport", "runway"])
    out = (
        out.sort(["airport", "runway", "est_ob"])
        .join_asof(
            runway_curve,
            left_on="est_ob",
            right_on="t",
            by=["airport", "runway"],
            strategy="backward",
        )
        .rename({"occ": "_occ_rw"})
        .drop("t")
    )

    return out.select(
        "mvt_id",
        (pl.col("_occ_ob") - 1).clip(0, None).cast(pl.Int32).alias("q2_ahead_at_ob"),
        pl.col("_occ_tko").clip(0, None).cast(pl.Int32).alias("q2_still_at_tko"),
        (pl.col("_occ_rw") - 1).clip(0, None).cast(pl.Int32).alias("q2_ahead_rw"),
        "q2_est_ob_source",
        (pl.col("mvt_time") - pl.col("est_ob")).dt.total_seconds().alias("q2_est_taxi_sec"),
    )


@feature(name="queue2", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def queue2(ctx: FeatureContext) -> pl.DataFrame:
    """Aircraft taxiing ahead of me at pushback, still out at my takeoff, and on my runway."""
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "runway", "mvt_time", *OB_SOURCES)
        .collect(engine="streaming")
    )
    return align(ctx.spine, featurize(deps), OUTPUTS)
