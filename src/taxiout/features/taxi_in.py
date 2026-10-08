"""Taxi-in statistics as a proxy for the *unimpeded* part of taxi-out.

An arrival taxiing from runway to stand covers much the same ground a departure covers in
reverse, and arrivals barely queue — so the median taxi-in at a stand is close to a measurement
of how far that stand is from the runways, uncontaminated by departure queueing.

Two things make this worth having. It is not target-dependent: it reads ``taxi_sec`` on *arrival*
rows, which are fully present in the ranking set, so it needs no fold-aware treatment and can be
computed contemporaneously for 2026 rather than carried over from 2025. And the residual it
leaves behind — taxi-out minus this — isolates pushback and queueing, which is the part actually
worth modelling.

It is not a drop-in substitute for taxi-out, and the airport interaction is not optional. Where
arrivals and departures share runways it tracks well (EHAM 0.84 at stand level, LEMD 0.78); where
they are segregated it does not (LEBL −0.09, which runs zero shared runways), and at queue-bound
Heathrow stand-to-stand variation is drowned by the queue (0.09).
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Arrival taxi times outside this range are data errors, not long taxis.
PLAUSIBLE_TAXI_SEC = (60, 7200)
#: Below this many observations a group median is noise; the feature falls back.
MIN_OBSERVATIONS = 20

OUTPUTS = (
    "ti_stand_median",
    "ti_stand_n",
    "ti_stand_runway_median",
    "ti_stand_runway_n",
    "ti_airport_median",
    "ti_stand_vs_airport",
)


def _arrivals() -> pl.LazyFrame:
    return (
        io.scan_movements()
        .filter((pl.col("phase") == "ARR") & pl.col("taxi_sec").is_between(*PLAUSIBLE_TAXI_SEC))
        .select("airport", "stand", "runway", "taxi_sec")
    )


@feature(name="taxi_in", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def taxi_in(ctx: FeatureContext) -> pl.DataFrame:
    """Median arrival taxi time at the same stand, and at the same stand-runway pair.

    The stand-runway key is the better physical match but only covers ~41% of departures with
    enough observations, because most of these airports segregate arrival and departure
    runways. The stand-only key covers 99.8%, so both are emitted with their counts and the
    model is left to decide.
    """
    arrivals = _arrivals()

    by_stand = (
        arrivals.group_by("airport", "stand")
        .agg(
            pl.col("taxi_sec").median().alias("ti_stand_median"),
            pl.len().alias("ti_stand_n"),
        )
        .filter(pl.col("ti_stand_n") >= MIN_OBSERVATIONS)
    )
    by_stand_runway = (
        arrivals.group_by("airport", "stand", "runway")
        .agg(
            pl.col("taxi_sec").median().alias("ti_stand_runway_median"),
            pl.len().alias("ti_stand_runway_n"),
        )
        .filter(pl.col("ti_stand_runway_n") >= MIN_OBSERVATIONS)
    )
    by_airport = arrivals.group_by("airport").agg(
        pl.col("taxi_sec").median().alias("ti_airport_median")
    )

    departures = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "stand", "runway")
        .join(by_stand, on=["airport", "stand"], how="left")
        .join(by_stand_runway, on=["airport", "stand", "runway"], how="left")
        .join(by_airport, on="airport", how="left")
        .with_columns(
            pl.col("ti_stand_n").fill_null(0),
            pl.col("ti_stand_runway_n").fill_null(0),
            # How unusual this stand is for its airport: positive means a remote stand.
            (pl.col("ti_stand_median") - pl.col("ti_airport_median")).alias("ti_stand_vs_airport"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, departures, OUTPUTS)
