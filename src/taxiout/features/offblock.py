"""Off-block time proxies.

The target is ``mvt_time - block_time``, and ``block_time`` is what the ranking set removes.
But the dataset keeps four *other* estimates of when the aircraft left the stand, and the gap
between takeoff and each of them is a direct estimate of taxi-out.

``aobt_3`` is by far the best of them, with one caveat:
where Eurocontrol never observed the off-block it back-computed the value by subtracting a
fixed default taxi time from the takeoff, leaving spikes at 20, 25 and 12 minutes. Measured
against a local baseline those account for ~12-13% of departures.

The flags emitted here mark **a band containing excess default mass, not a fabricated row** —
most of the 12-minute band is genuine traffic that happens to taxi for twelve minutes. A model
is expected to learn "trust the value less here", which is the honest reading.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Bands in which ``mvt_time - aobt_3`` is a back-computed constant, not a measurement.
#: A minute-rounded ``aobt_3`` against a second-resolution takeoff spreads each default
#: over a ~60 s window centred on it.
DEFAULT_BANDS: dict[str, tuple[int, int]] = {
    "ob_aobt_is_default_12min": (690, 750),
    "ob_aobt_is_default_20min": (1170, 1230),
    "ob_aobt_is_default_25min": (1470, 1530),
}

OUTPUTS = (
    "ob_aobt_sec",
    "ob_aobt_is_default_12min",
    "ob_aobt_is_default_20min",
    "ob_aobt_is_default_25min",
    "ob_aobt_is_missing",
    "ob_eobt_sec",
    "ob_lobt_sec",
    "ob_iobt_sec",
    "ob_sched_sec",
    "ob_aobt_minus_eobt",
    "ob_aobt_minus_lobt",
)


def _gap(later: str, earlier: str) -> pl.Expr:
    return (pl.col(later) - pl.col(earlier)).dt.total_seconds()


@feature(name="offblock", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def offblock(ctx: FeatureContext) -> pl.DataFrame:
    """Takeoff minus each available off-block estimate, plus AOBT provenance flags."""
    source = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "mvt_time", "aobt_3", "eobt_1", "lobt", "iobt", "sched_time")
        .with_columns(
            _gap("mvt_time", "aobt_3").alias("ob_aobt_sec"),
            _gap("mvt_time", "eobt_1").alias("ob_eobt_sec"),
            _gap("mvt_time", "lobt").alias("ob_lobt_sec"),
            _gap("mvt_time", "iobt").alias("ob_iobt_sec"),
            _gap("mvt_time", "sched_time").alias("ob_sched_sec"),
        )
        .with_columns(
            *[
                pl.col("ob_aobt_sec").is_between(lo, hi).fill_null(False).cast(pl.Int32).alias(col)
                for col, (lo, hi) in DEFAULT_BANDS.items()
            ],
            pl.col("ob_aobt_sec").is_null().cast(pl.Int32).alias("ob_aobt_is_missing"),
            # How far the "actual" reconstruction drifted from the planned times: a delay signal
            # that is independent of the absolute gaps above.
            (pl.col("ob_aobt_sec") - pl.col("ob_eobt_sec")).alias("ob_aobt_minus_eobt"),
            (pl.col("ob_aobt_sec") - pl.col("ob_lobt_sec")).alias("ob_aobt_minus_lobt"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, source, OUTPUTS)
