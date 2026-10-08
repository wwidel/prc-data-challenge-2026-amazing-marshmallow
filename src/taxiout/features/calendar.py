"""Calendar and time-of-day features, in each airport's local civil time.

Operations follow the local clock — curfews, banks of departures, night restrictions — so the
local hour matters more than the UTC one, and DST has to be handled rather than approximated
with a fixed offset.
"""

from __future__ import annotations

import polars as pl

from taxiout import airports
from taxiout.features.registry import FeatureContext, FeatureKind, feature

OUTPUTS = (
    "cal_local_minute_of_day",
    "cal_local_hour",
    "cal_dow",
    "cal_doy",
    "cal_month",
    "cal_is_weekend",
    "cal_tod_sin",
    "cal_tod_cos",
)


def _local_wall_time(time_col: str = "mvt_time") -> pl.Expr:
    """Wall-clock time at the airport, as a naive datetime, DST included."""
    branches = [
        (
            pl.col("airport").is_in(icaos),
            pl.col(time_col).dt.convert_time_zone(tz).dt.replace_time_zone(None),
        )
        for tz, icaos in airports.by_timezone().items()
    ]
    expr = pl.when(branches[0][0]).then(branches[0][1])
    for cond, value in branches[1:]:
        expr = expr.when(cond).then(value)
    return expr.otherwise(pl.col(time_col).dt.replace_time_zone(None))


@feature(name="calendar", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def calendar(ctx: FeatureContext) -> pl.DataFrame:
    """Local time of day, day of week, day of year, and a cyclic encoding of time of day."""
    # Cast BEFORE multiplying: polars evaluates dt.hour() * 60 in Int8, which wraps, so for most
    # of the challenge this column held minute-of-day modulo 256 as a signed byte and the sin/cos
    # pair encoded a 256-minute period. The range check below fails loudly if anything like it
    # comes back.
    minute_of_day = pl.col("_local").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "_local"
    ).dt.minute().cast(pl.Int32)
    out = (
        ctx.spine.lazy()
        .with_columns(_local_wall_time().alias("_local"))
        .select(
            minute_of_day.cast(pl.Int32).alias("cal_local_minute_of_day"),
            pl.col("_local").dt.hour().cast(pl.Int32).alias("cal_local_hour"),
            pl.col("_local").dt.weekday().cast(pl.Int32).alias("cal_dow"),
            pl.col("_local").dt.ordinal_day().cast(pl.Int32).alias("cal_doy"),
            pl.col("_local").dt.month().cast(pl.Int32).alias("cal_month"),
            (pl.col("_local").dt.weekday() >= 6).cast(pl.Int32).alias("cal_is_weekend"),
            (minute_of_day / 1440 * 2 * 3.141592653589793).sin().alias("cal_tod_sin"),
            (minute_of_day / 1440 * 2 * 3.141592653589793).cos().alias("cal_tod_cos"),
        )
        .collect()
    )
    mod = out["cal_local_minute_of_day"]
    if mod.len() and not (mod.min() >= 0 and mod.max() <= 1439):
        raise AssertionError(f"cal_local_minute_of_day outside [0, 1439]: {mod.min()}..{mod.max()}")
    return out
