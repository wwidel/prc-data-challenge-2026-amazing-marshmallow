"""The low-cardinality half of ``stand_runway`` — a direct response to the ablation.

``s02_drop_stand_runway`` scored trimmed ex-LIRF **223.48** against the champion's **225.99**, on a
config identical except for that one block: **dropping all twelve categorical codes improves the
model by 2.51**, which is larger than the training-label cap's 2.36 and second only to the CatBoost
function class (−6.29). Removing features is not supposed to help by that much.

The cardinalities say why:

======================  =============
column                  levels (2025)
======================  =============
``cat_stand_runway``          12,053
``cat_stand``                  1,900
``cat_ades``                   1,569
``cat_operator``                 677
``cat_ac_type``                  270
``cat_airport_runway``            70
``cat_runway``                    53
``cat_airport``                   10
======================  =============

They are handed to LightGBM as ``categorical_feature``, and LightGBM partitions a categorical by
sorting its levels on gradient statistics and cutting — which overfits badly at 12,053 levels
against ``min_data_in_leaf=100``. This converges with two other findings: XGBoost's weakness was
its categorical handling (it earned an NNLS weight of 0.0000 in the stack), and CatBoost's
*strength* is its ordered target statistics on exactly these columns (−6.29 in CV).

So the question the ablation cannot answer is whether the block is harmful *as a whole* or only in
its high-cardinality half. This block keeps the codes at or below ``MAX_LEVELS`` and drops the rest,
which is the smallest change that separates those two readings.

Deliberately a separate module rather than a parameter on ``stand_runway``: the cache key is a hash
of the whole defining module, so editing that file would invalidate the block every shipping member
uses and force a full rebuild.
"""

from __future__ import annotations

import polars as pl

from taxiout.features.registry import FeatureContext, FeatureKind, align, feature
from taxiout.features.stand_runway import CATEGORICALS, _code

#: Keep a code only if its level count is at or below this. The boundary sits in the gap between
#: `cat_airport_runway` (70) and `cat_ac_type` (270), so it is not a knife-edge choice.
MAX_LEVELS = 100

#: Source columns whose codes survive the cut, in `stand_runway`'s own order.
KEEP = ("airport", "runway", "wake_cat", "market_segment", "flight_type", "flight_rule")

OUTPUTS = (*(f"lo_{CATEGORICALS[c][4:]}" for c in KEEP), "lo_airport_runway")


def featurize(rows: pl.DataFrame) -> pl.DataFrame:
    """Dense codes for the low-cardinality categoricals only. Pure; unit-tested."""
    out = rows.select(
        "mvt_id",
        *[_code(c).alias(f"lo_{CATEGORICALS[c][4:]}") for c in KEEP],
        # the airport x runway pair is 70 levels, well inside the cut, and it is the one
        # interaction the location backbone genuinely needs
        pl.concat_str(
            [pl.col("airport").fill_null(""), pl.col("runway").fill_null("")], separator="|"
        )
        .rank("dense")
        .cast(pl.Int32)
        .sub(1)
        .alias("lo_airport_runway"),
    )
    return out


@feature(name="stand_runway_lo", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def stand_runway_lo(ctx: FeatureContext) -> pl.DataFrame:
    """Airport, runway, wake, market segment, flight type, flight rule, and airport x runway."""
    from taxiout import io as tio

    rows = (
        tio.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", *KEEP)
        .collect(engine="streaming")
    )
    rows = ctx.spine.select("mvt_id").join(rows, on="mvt_id", how="left")
    return align(ctx.spine, featurize(rows), OUTPUTS)
