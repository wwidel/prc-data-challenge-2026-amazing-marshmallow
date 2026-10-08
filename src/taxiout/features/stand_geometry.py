"""Stand-to-threshold distance from the X-Plane Scenery Gateway packs.

The stand term of the location backbone is largely geometry (correlation up to 0.895 at
LIRF-25, weighted mean 0.597), and geometry generalises to stands the training data never
saw — EDDM's July-2026 apron is 7.9% unseen stands. Distance joins on the normalized
stand key; rows without coordinates (LTFM entirely, parts of EDDM/LSZH) keep a null distance
plus an explicit flag, which gives tree models the clearly separate bucket.
"""

from __future__ import annotations

import math

import polars as pl

from taxiout import io as tio
from taxiout import paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = ("sg_dist_thr_m", "sg_has_geo")


def norm_stand(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_uppercase()
        .str.replace(r"^(GATE|STAND|RAMP)\s*", "")
        .str.replace_all(r"\s+", "")
        .str.replace(r"^([A-Z]*)0+(\d)", "${1}${2}")
    )


def load_geometry() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stand coordinates: Scenery Gateway, with OSM parking positions filling the gaps.

    OSM closes LTFM (0 -> 99%) and lifts LSZH; Gateway keeps priority where both
    know a stand, so earlier numbers stay comparable.
    """
    gw = paths.DATA / "external" / "gateway"
    gateway = (
        pl.read_parquet(gw / "stands.parquet", glob=False)
        .with_columns(norm_stand(pl.col("stand")).alias("stand_key"))
        .unique(["airport", "stand_key"], keep="first")
        .select("airport", "stand_key", "lat", "lon")
    )
    osm = (
        pl.read_parquet(paths.DATA / "external" / "osm" / "osm_stands.parquet", glob=False)
        .with_columns(norm_stand(pl.col("stand")).alias("stand_key"))
        .unique(["airport", "stand_key"], keep="first")
        .select("airport", "stand_key", "lat", "lon")
    )
    # Third source: OSM A/B stand variants collapsed to their base code (EDDM writes
    # "111A"/"111B" where the movement data says "111"); variants average to the base point.
    osm_base = (
        osm.with_columns(pl.col("stand_key").str.replace(r"(\d)[A-Z]$", "${1}").alias("base"))
        .filter(pl.col("base") != pl.col("stand_key"))
        .group_by("airport", "base")
        .agg(pl.col("lat").mean(), pl.col("lon").mean())
        .rename({"base": "stand_key"})
    )
    stands = pl.concat([gateway, osm, osm_base]).unique(["airport", "stand_key"], keep="first")
    rwys = (
        pl.read_parquet(gw / "runways.parquet", glob=False)
        .with_columns(pl.col("runway").str.replace(r"^0", "").alias("rw_key"))
        .unique(["airport", "rw_key"], keep="first")
        .select("airport", "rw_key", "thr_lat", "thr_lon")
    )
    return stands, rwys


def featurize(spine: pl.DataFrame, stands: pl.DataFrame, rwys: pl.DataFrame) -> pl.DataFrame:
    """Distance from the stand to the departure threshold. Pure; unit-tested."""
    m_lat = 111_320.0
    d = (
        spine.select("mvt_id", "airport", "stand", "runway")
        .with_columns(
            norm_stand(pl.col("stand").fill_null("")).alias("stand_key"),
            pl.col("runway").fill_null("").str.replace(r"^0", "").alias("rw_key"),
        )
        .join(stands, on=["airport", "stand_key"], how="left")
        .join(rwys, on=["airport", "rw_key"], how="left")
    )
    return d.select(
        "mvt_id",
        (
            (
                ((pl.col("lat") - pl.col("thr_lat")) * m_lat) ** 2
                + (
                    (pl.col("lon") - pl.col("thr_lon"))
                    * m_lat
                    * (pl.col("thr_lat") * math.pi / 180).cos()
                )
                ** 2
            ).sqrt()
        ).alias("sg_dist_thr_m"),
        (pl.col("lat").is_not_null() & pl.col("thr_lat").is_not_null())
        .cast(pl.Int32)
        .alias("sg_has_geo"),
    )


@feature(name="stand_geometry", kind=FeatureKind.ROW_LOCAL, outputs=OUTPUTS)
def stand_geometry(ctx: FeatureContext) -> pl.DataFrame:
    """Stand-to-departure-threshold distance, with a has-geometry bucket flag."""
    # The spine is narrow; fetch the raw columns keyed on mvt_id, as other blocks do.
    rows = (
        tio.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "stand", "runway")
        .collect(engine="streaming")
    )
    rows = ctx.spine.select("mvt_id").join(rows, on="mvt_id", how="left")
    stands, rwys = load_geometry()
    return align(ctx.spine, featurize(rows, stands, rwys), OUTPUTS)
