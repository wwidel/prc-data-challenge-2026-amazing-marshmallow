"""Whether the movement matched a Network Manager flight record.

Roughly 1% of movements have no NM match, and every ``_flt`` column is null together for them.
That looks like a nuisance and is actually a strong signal: 33 of the 34 departures in 2025 with
a taxi-out over twelve hours are unmatched rows.

The effect is **airport-specific in sign**, so it has to be used as an interaction rather than a
main effect. At LIRF unmatched departures average 6,534 s against 1,145 s for matched ones, with
29.5% exceeding two hours; at EHAM they average 600 s against 788 s, in the other direction.

This reads only the presence of ``flight_id``, never a label, so it is ROW_LOCAL.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = ("flt_no_match", "cat_airport_match")


@feature(name="flight_match", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def flight_match(ctx: FeatureContext) -> pl.DataFrame:
    """The unmatched flag, and its interaction with airport as a single code."""
    source = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "flight_id")
        .with_columns(pl.col("flight_id").is_null().alias("_no_match"))
        .with_columns(
            pl.col("_no_match").cast(pl.Int32).alias("flt_no_match"),
            (pl.col("airport") + "|" + pl.col("_no_match").cast(pl.Utf8))
            .rank("dense")
            .cast(pl.Int32)
            .sub(1)
            .alias("cat_airport_match"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, source, OUTPUTS)
