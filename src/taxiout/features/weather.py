"""Weather at the departure airport, anchored on takeoff time.

Open METAR from the IEM archive (`scripts/pull_metar.py`). Everything here is backward-looking
from `mvt_time`: the latest observation, short rolling precipitation windows, freezing hours,
and the run of consecutive sub-zero days ending yesterday — the de-icing memory that a single
observation misses. Post-ops licensing would even allow forward windows, but the January story
this block chases (de-icing, snow banks, LVP) is carried by what already happened.
"""

from __future__ import annotations

import math

import polars as pl

from taxiout import paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = (
    "wx_temp_c",
    "wx_dew_spread_c",
    "wx_rh",
    "wx_wind_kt",
    "wx_gust_kt",
    "wx_wind_dir_sin",
    "wx_wind_dir_cos",
    "wx_vis_km",
    "wx_precip_mm",
    "wx_precip_3h_mm",
    "wx_precip_6h_mm",
    "wx_snow",
    "wx_freezing",
    "wx_fog",
    "wx_ts",
    "wx_frozen_hours_24h",
    "wx_deice_proxy",
    "wx_subzero_streak_days",
    "wx_obs_age_s",
)


def load_metar() -> pl.DataFrame:
    path = paths.DATA / "external" / "metar" / "metar.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/pull_metar.py first")
    return pl.read_parquet(path, glob=False)


def enrich_obs(metar: pl.DataFrame) -> pl.DataFrame:
    """Per-observation derived columns, including rolling windows. Pure; unit-tested."""
    out = metar.sort("airport", "time").with_columns(
        (pl.col("temp_c") - pl.col("dewpoint_c")).alias("dew_spread_c"),
        (pl.col("wind_dir") * math.pi / 180).sin().alias("wind_dir_sin"),
        (pl.col("wind_dir") * math.pi / 180).cos().alias("wind_dir_cos"),
        pl.col("wx").fill_null("").str.contains("SN|SG|PL|GS").cast(pl.Int32).alias("snow"),
        pl.col("wx").fill_null("").str.contains("FZ").cast(pl.Int32).alias("freezing"),
        pl.col("wx").fill_null("").str.contains("FG|BR").cast(pl.Int32).alias("fog"),
        pl.col("wx").fill_null("").str.contains("TS").cast(pl.Int32).alias("ts"),
        pl.col("precip_mm").fill_null(0.0).alias("_p"),
        (pl.col("temp_c") <= 0).cast(pl.Int32).fill_null(0).alias("_frz"),
    )
    rolled = []
    for _, g in out.group_by("airport", maintain_order=True):
        g = g.sort("time").with_columns(pl.lit(1, dtype=pl.Int32).alias("_one"))
        g = g.with_columns(
            pl.col("_p").rolling_sum_by("time", window_size="3h").alias("precip_3h_mm"),
            pl.col("_p").rolling_sum_by("time", window_size="6h").alias("precip_6h_mm"),
            pl.col("_frz").rolling_sum_by("time", window_size="24h").alias("frozen_obs_24h"),
            # observations are ~half-hourly; convert the count to hours by local cadence
            pl.col("_one").rolling_sum_by("time", window_size="24h").alias("_obs_24h"),
        )
        rolled.append(g)
    out = pl.concat(rolled).with_columns(
        (24.0 * pl.col("frozen_obs_24h") / pl.col("_obs_24h").clip(1, None)).alias(
            "frozen_hours_24h"
        ),
        (
            (pl.col("temp_c") <= 3)
            & ((pl.col("_p") > 0) | (pl.col("snow") == 1) | (pl.col("freezing") == 1))
        )
        .cast(pl.Int32)
        .alias("deice_proxy"),
    )
    return out.drop("_p", "_frz", "_obs_24h", "_one")


def subzero_streaks(metar: pl.DataFrame) -> pl.DataFrame:
    """Per airport-date: consecutive sub-zero-minimum days ending the *previous* day.

    Anchored on yesterday so the value is constant within a day and never peeks at
    observations later than any departure. Pure; unit-tested.
    """
    daily = (
        metar.with_columns(pl.col("time").dt.date().alias("date"))
        .group_by("airport", "date")
        .agg(pl.col("temp_c").min().alias("tmin"))
        .sort("airport", "date")
        .with_columns((pl.col("tmin") < 0).cast(pl.Int32).alias("below"))
    )
    daily = daily.with_columns(
        (pl.col("below") == 0).cum_sum().over("airport").alias("_grp")
    ).with_columns(
        (pl.col("below").cum_sum().over("airport", "_grp") * pl.col("below")).alias("streak")
    )
    return daily.with_columns(
        pl.col("streak").shift(1, fill_value=0).over("airport").alias("streak_yesterday")
    ).select("airport", "date", "streak_yesterday")


def featurize(spine: pl.DataFrame, metar: pl.DataFrame) -> pl.DataFrame:
    """Join enriched observations onto departures, backward as-of. Pure; unit-tested."""
    obs = enrich_obs(metar)
    streaks = subzero_streaks(metar)
    deps = spine.select("mvt_id", "airport", "mvt_time").with_columns(
        pl.col("mvt_time").dt.date().alias("date")
    )
    joined = (
        deps.sort("airport", "mvt_time")
        .join_asof(
            obs.sort("airport", "time"),
            left_on="mvt_time",
            right_on="time",
            by="airport",
            strategy="backward",
        )
        .join(streaks, on=["airport", "date"], how="left")
    )
    return joined.select(
        "mvt_id",
        pl.col("temp_c").alias("wx_temp_c"),
        pl.col("dew_spread_c").alias("wx_dew_spread_c"),
        pl.col("rh").alias("wx_rh"),
        pl.col("wind_kt").alias("wx_wind_kt"),
        pl.col("gust_kt").fill_null(0.0).alias("wx_gust_kt"),
        pl.col("wind_dir_sin").alias("wx_wind_dir_sin"),
        pl.col("wind_dir_cos").alias("wx_wind_dir_cos"),
        pl.col("vis_km").alias("wx_vis_km"),
        pl.col("precip_mm").alias("wx_precip_mm"),
        pl.col("precip_3h_mm").alias("wx_precip_3h_mm"),
        pl.col("precip_6h_mm").alias("wx_precip_6h_mm"),
        pl.col("snow").alias("wx_snow"),
        pl.col("freezing").alias("wx_freezing"),
        pl.col("fog").alias("wx_fog"),
        pl.col("ts").alias("wx_ts"),
        pl.col("frozen_hours_24h").alias("wx_frozen_hours_24h"),
        pl.col("deice_proxy").alias("wx_deice_proxy"),
        pl.col("streak_yesterday").fill_null(0).alias("wx_subzero_streak_days"),
        (pl.col("mvt_time") - pl.col("time")).dt.total_seconds().alias("wx_obs_age_s"),
    )


@feature(name="weather", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def weather(ctx: FeatureContext) -> pl.DataFrame:
    """Latest METAR before takeoff, short precip windows, freezing memory, sub-zero streaks."""
    return align(ctx.spine, featurize(ctx.spine, load_metar()), OUTPUTS)
