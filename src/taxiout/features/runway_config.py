"""Which runways the airfield is actually using, around the moment of takeoff.

A stand's distance to the runway is fixed; the runway it is sent to is not. Reconfiguration
turned out to be the largest single driver we have found — it produced the worst day of Rome's
2025 (13 July, +8.52) and a 2.54x swing in Amsterdam's arrival taxi-in *inside the scored month*,
where departing 36L gives 234-332 s and departing 24 gives 545-657 s.

Everything here is built from ``runway`` and ``mvt_time``, both of which the ranking set keeps,
and it uses arrivals as well as departures because arrival rows are intact there too. No labels
are read, so this is ``CONTEXT``: computed once over both sets, no fold-aware handling.

The unit is the airport-hour. Finer buckets would resolve a mid-hour switch but would also make
the smaller airports' shares noisy — LSZH averages about fifteen departures an hour.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: A runway counts as active in an hour if it takes at least this share of that hour's
#: movements in its phase. Below it, a single towed or misrecorded movement would invent a
#: configuration that never existed.
MIN_SHARE = 0.05

#: Bucket width. See the module docstring on why this is not finer.
BUCKET = "1h"

OUTPUTS = (
    "rwy_cfg_code",
    "rwy_dep_active_n",
    "rwy_arr_active_n",
    "rwy_share_in_hour",
    "rwy_is_dominant",
    "rwy_shared_with_arrivals",
    "rwy_cfg_changed",
    "rwy_cfg_seen_n",
)


def _hourly_state() -> pl.LazyFrame:
    """Per airport-hour: the active runway set for each phase, as one configuration string."""
    movements = (
        io.scan_movements()
        .select("airport", "phase", "runway", "mvt_time")
        .with_columns(
            pl.col("mvt_time").dt.truncate(BUCKET).alias("hour"),
            pl.col("runway").fill_null("?"),
        )
    )
    counts = (
        movements.group_by("airport", "hour", "phase", "runway")
        .agg(pl.len().alias("n"))
        .with_columns(
            (pl.col("n") / pl.col("n").sum().over("airport", "hour", "phase")).alias("share")
        )
    )
    active = (
        counts.filter(pl.col("share") >= MIN_SHARE)
        .sort("airport", "hour", "phase", "runway")
        .group_by("airport", "hour", "phase")
        .agg(
            pl.col("runway").str.join("+").alias("set"),
            pl.len().alias("active_n"),
        )
    )
    wide = (
        active.collect(engine="streaming")
        .pivot(on="phase", index=["airport", "hour"], values=["set", "active_n"])
        .lazy()
    )
    # A pivot with one phase absent for an airport-hour leaves nulls; name them so the
    # configuration string stays well-defined rather than becoming null itself.
    return wide.with_columns(
        pl.col("set_DEP").fill_null("none"),
        pl.col("set_ARR").fill_null("none"),
        pl.col("active_n_DEP").fill_null(0),
        pl.col("active_n_ARR").fill_null(0),
    ).with_columns(
        (pl.col("airport") + " D:" + pl.col("set_DEP") + " A:" + pl.col("set_ARR")).alias("config")
    )


@feature(name="runway_config", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def runway_config(ctx: FeatureContext) -> pl.DataFrame:
    """The airfield's configuration in the hour of takeoff, and this movement's place in it."""
    state = _hourly_state()

    # How often each configuration occurs, and whether it differs from the hour before. A rare
    # configuration, or one just switched into, is where the queue disruption lives.
    state = state.with_columns(
        pl.len().over("config").alias("rwy_cfg_seen_n"),
        (pl.col("config") != pl.col("config").shift(1).over("airport"))
        .cast(pl.Int32)
        .alias("rwy_cfg_changed"),
        pl.col("config").rank("dense").cast(pl.Int32).sub(1).alias("rwy_cfg_code"),
    )

    # Per-runway share within the hour, for departures, plus whether arrivals use it too.
    movements = (
        io.scan_movements()
        .select("airport", "phase", "runway", "mvt_time")
        .with_columns(
            pl.col("mvt_time").dt.truncate(BUCKET).alias("hour"), pl.col("runway").fill_null("?")
        )
    )
    dep_share = (
        movements.filter(pl.col("phase") == "DEP")
        .group_by("airport", "hour", "runway")
        .agg(pl.len().alias("n"))
        .with_columns(
            (pl.col("n") / pl.col("n").sum().over("airport", "hour")).alias("rwy_share_in_hour"),
            (pl.col("n") == pl.col("n").max().over("airport", "hour"))
            .cast(pl.Int32)
            .alias("rwy_is_dominant"),
        )
        .select("airport", "hour", "runway", "rwy_share_in_hour", "rwy_is_dominant")
    )
    arr_runways = (
        movements.filter(pl.col("phase") == "ARR")
        .group_by("airport", "hour", "runway")
        .agg(pl.len().alias("n_arr"))
        .filter(pl.col("n_arr") >= 1)
        .select(
            "airport", "hour", "runway", pl.lit(1, dtype=pl.Int32).alias("rwy_shared_with_arrivals")
        )
    )

    departures = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "runway", "mvt_time")
        .with_columns(
            pl.col("mvt_time").dt.truncate(BUCKET).alias("hour"), pl.col("runway").fill_null("?")
        )
        .join(state, on=["airport", "hour"], how="left")
        .join(dep_share, on=["airport", "hour", "runway"], how="left")
        .join(arr_runways, on=["airport", "hour", "runway"], how="left")
        .with_columns(
            pl.col("active_n_DEP").fill_null(0).cast(pl.Int32).alias("rwy_dep_active_n"),
            pl.col("active_n_ARR").fill_null(0).cast(pl.Int32).alias("rwy_arr_active_n"),
            pl.col("rwy_shared_with_arrivals").fill_null(0),
            pl.col("rwy_cfg_changed").fill_null(0),
            pl.col("rwy_cfg_code").fill_null(-1),
            pl.col("rwy_cfg_seen_n").fill_null(0).cast(pl.Int32),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, departures, OUTPUTS)
