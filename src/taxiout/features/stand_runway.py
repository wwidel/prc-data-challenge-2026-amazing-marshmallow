"""Categorical identity of the movement: where it started, where it left from, what it was.

Emitted as integer codes against a vocabulary built once from every movement in the spine,
training and ranking together. Building the vocabulary globally is what makes the codes stable
across folds; it involves no labels, so it leaks nothing.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: source column -> emitted code column
CATEGORICALS: dict[str, str] = {
    "airport": "cat_airport",
    "stand": "cat_stand",
    "runway": "cat_runway",
    "ades": "cat_ades",
    "ac_type": "cat_ac_type",
    "wake_cat": "cat_wake",
    "operator": "cat_operator",
    "market_segment": "cat_market_segment",
    "flight_type": "cat_flight_type",
    "flight_rule": "cat_flight_rule",
}

OUTPUTS = (*CATEGORICALS.values(), "cat_stand_runway", "cat_airport_runway")


def _code(column: str) -> pl.Expr:
    """Dense integer code, ``-1`` for null. Order is alphabetical, hence deterministic."""
    return (
        pl.col(column)
        .fill_null("\x00missing")
        .rank("dense")
        .cast(pl.Int32)
        .sub(1)
        .alias(CATEGORICALS.get(column, column))
    )


@feature(name="stand_runway", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def stand_runway(ctx: FeatureContext) -> pl.DataFrame:
    """Integer codes for airport, stand, runway, destination, aircraft, operator and segment.

    ``cat_stand_runway`` is the physical taxi route — the single most structural predictor
    available — and is coded within airport so the same stand name at two airports never
    collides.
    """
    source = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", *CATEGORICALS)
        .with_columns(
            (
                pl.col("airport")
                + "|"
                + pl.col("stand").fill_null("?")
                + "|"
                + pl.col("runway").fill_null("?")
            ).alias("_sr"),
            (pl.col("airport") + "|" + pl.col("runway").fill_null("?")).alias("_ar"),
        )
        .with_columns(
            *[_code(c) for c in CATEGORICALS],
            _code("_sr").alias("cat_stand_runway"),
            _code("_ar").alias("cat_airport_runway"),
        )
        .select("mvt_id", *OUTPUTS)
        .collect(engine="streaming")
    )
    return align(ctx.spine, source, OUTPUTS)
