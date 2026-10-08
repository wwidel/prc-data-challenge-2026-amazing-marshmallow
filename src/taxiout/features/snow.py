"""Snow at the departure airport: how much, for how long, and how recently.

`weather` carries the latest METAR's snow flag, which says nothing about a storm's size or about
the day after it. In February 2025 Istanbul's taxi-outs ran 15–16 minutes long on its snow days,
and still 3 minutes long on the freezing day after, and the models missed most of it
(``scripts/analysis/a127_february_error.py``). Here, from the same METAR reports, at the takeoff
time ``t``:

* ``sn_3h`` … ``sn_48h``: snow reports (SN, SG, PL, GS) in the preceding window;
* ``sn_heavy_6h``: heavy snow (+SN) reports in the last 6 h; ``fz_6h``: freezing rain or drizzle;
* ``sn_day``: snow reports over the whole UTC day of the takeoff (post-operational: the organisers
  frame the task as such, and the day's weather is known after the day);
* ``sn_last_h``: hours since the last snow report, capped at 96;
* ``sn_48h_cold``: snow in the last 48 h and the latest temperature at or below 0 °C.

CONTEXT: METAR only, no labels.
"""

from __future__ import annotations

import polars as pl

from taxiout.features.registry import FeatureContext, FeatureKind, align, feature
from taxiout.features.weather import load_metar

OUTPUTS = (
    "sn_3h",
    "sn_6h",
    "sn_12h",
    "sn_24h",
    "sn_48h",
    "sn_heavy_6h",
    "fz_6h",
    "sn_day",
    "sn_last_h",
    "sn_48h_cold",
)
WINDOWS_H = (3, 6, 12, 24, 48)
LAST_CAP_H = 96.0


def observations(metar: pl.DataFrame) -> pl.DataFrame:
    """Per airport, time-sorted reports with cumulative counts. Pure; unit-tested."""
    wx = pl.col("wx").fill_null("")
    return (
        metar.select("airport", "time", "wx", "temp_c")
        .sort("airport", "time")
        .with_columns(
            wx.str.contains("SN|SG|PL|GS").cast(pl.Int32).alias("_sn"),
            wx.str.contains(r"\+SN").cast(pl.Int32).alias("_heavy"),
            wx.str.contains("FZRA|FZDZ").cast(pl.Int32).alias("_fz"),
        )
        .with_columns(
            pl.col("_sn").cum_sum().over("airport").alias("c_sn"),
            pl.col("_heavy").cum_sum().over("airport").alias("c_heavy"),
            pl.col("_fz").cum_sum().over("airport").alias("c_fz"),
            pl.when(pl.col("_sn") == 1)
            .then(pl.col("time"))
            .otherwise(None)
            .forward_fill()
            .over("airport")
            .alias("last_sn"),
        )
    )


def _at(rows: pl.DataFrame, obs: pl.DataFrame, shift_h: float, suffix: str) -> pl.DataFrame:
    """Cumulative counts of the latest report at or before ``t − shift_h`` hours."""
    key = rows.select(
        "mvt_id",
        "airport",
        (pl.col("t") - pl.duration(seconds=int(shift_h * 3600))).alias("_k"),
    ).sort("_k")
    return key.join_asof(
        obs.select(
            "airport",
            pl.col("time").alias("_k"),
            pl.col("c_sn").alias(f"c_sn{suffix}"),
            pl.col("c_heavy").alias(f"c_heavy{suffix}"),
            pl.col("c_fz").alias(f"c_fz{suffix}"),
        ).sort("_k"),
        on="_k",
        by="airport",
        strategy="backward",
    ).drop("_k")


def featurize(spine: pl.DataFrame, metar: pl.DataFrame) -> pl.DataFrame:
    """One row per spine row with the OUTPUTS columns. Pure; unit-tested."""
    obs = observations(metar)
    rows = spine.select("mvt_id", "airport", pl.col("mvt_time").alias("t"))
    now = rows.sort("t").join_asof(
        obs.select(
            "airport",
            pl.col("time").alias("t"),
            "c_sn",
            "c_heavy",
            "c_fz",
            "last_sn",
            "temp_c",
        ).sort("t"),
        on="t",
        by="airport",
        strategy="backward",
    )
    out = now
    for h in WINDOWS_H:
        past = _at(rows, obs, h, f"_{h}")
        out = out.join(past.drop("airport"), on="mvt_id", how="left").with_columns(
            (pl.col("c_sn") - pl.col(f"c_sn_{h}").fill_null(0)).alias(f"sn_{h}h")
        )
    out = out.with_columns(
        (pl.col("c_heavy") - pl.col("c_heavy_6").fill_null(0)).alias("sn_heavy_6h"),
        (pl.col("c_fz") - pl.col("c_fz_6").fill_null(0)).alias("fz_6h"),
        ((pl.col("t") - pl.col("last_sn")).dt.total_seconds() / 3600.0)
        .clip(0.0, LAST_CAP_H)
        .fill_null(LAST_CAP_H)
        .alias("sn_last_h"),
        ((pl.col("sn_48h") > 0) & (pl.col("temp_c") <= 0.0)).cast(pl.Int8).alias("sn_48h_cold"),
    )
    day = (
        obs.with_columns(pl.col("time").dt.date().alias("_d"))
        .group_by("airport", "_d")
        .agg(pl.col("_sn").sum().alias("sn_day"))
    )
    out = out.with_columns(pl.col("t").dt.date().alias("_d")).join(
        day, on=["airport", "_d"], how="left"
    )
    return out.select("mvt_id", *[pl.col(c).cast(pl.Float32) for c in OUTPUTS])


@feature(name="snow", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def snow(ctx: FeatureContext) -> pl.DataFrame:
    """Snow reports before the takeoff, over the day, and since the last one."""
    return align(ctx.spine, featurize(ctx.spine, load_metar()), OUTPUTS)
