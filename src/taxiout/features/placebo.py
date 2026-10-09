"""A control block: three columns of noise.

Adding a block moves a member's per-month score even when the block carries nothing: the trees
split differently, and under ``feature_fraction`` every tree draws a different set of columns. A
member with and without this block gives that noise floor, against which the screens of real blocks
(the snow variants first) can be read. The values are hashes of the movement ID, so every rebuild
gives the same columns, and they hold no information about the label.

CONTEXT: no labels.
"""

from __future__ import annotations

import polars as pl

from taxiout.features.registry import FeatureContext, FeatureKind, feature

OUTPUTS = ("placebo_1", "placebo_2", "placebo_3")
SEEDS = (11, 23, 37)
MODULUS = 1_000_003


def featurize(spine: pl.DataFrame) -> pl.DataFrame:
    """One row per spine row, in spine order: uniform values in [0, 1). Pure; unit-tested."""
    return spine.select(
        *[
            ((pl.col("mvt_id").hash(seed=s) % MODULUS) / MODULUS).cast(pl.Float32).alias(c)
            for s, c in zip(SEEDS, OUTPUTS, strict=True)
        ]
    )


@feature(name="placebo", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def placebo(ctx: FeatureContext) -> pl.DataFrame:
    """Three noise columns keyed on the movement ID."""
    return featurize(ctx.spine)
