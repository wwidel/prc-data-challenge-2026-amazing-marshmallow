# %% [markdown]
# # a128 — February at Istanbul: snow days, residuals, and the published Feb/Jun levels
#
# `a127`: LTFM carries half of February 2025's trimmed squared error in x36's out-of-fold
# predictions, mostly on a few days (23 and 19 Feb: mean taxi 49 and 46 min, under-predicted by
# ~930 s). APT-DLY records nothing there. This puts beside each other:
#   1. per LTFM day of February 2025 and February 2026: METAR snow reports (weather codes with
#      SN), the day's minimum temperature, and for 2025 x36's mean label and residual;
#   2. EUROCONTROL's published mean taxi-out per airport for February and June, 2025 against
#      2026 (the same statistic as a119/a120: (reference + additional minutes) / flights with a
#      reference).
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
OOF_NAME = "w08_x36_all_months"
AIRPORT = "LTFM"
TRIM = 43_200.0
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")

rec = analysis.Recorder("a128_feb_jun_level_and_snow", "February at Istanbul; Feb/Jun levels")
pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(16)

# %%
rec.section(f"1. {AIRPORT}, February 2025 and 2026: snow, temperature, and 2025's residuals")
metar = (
    pl.read_parquet(paths.DATA / "external" / "metar" / "metar.parquet")
    .filter(pl.col("airport") == AIRPORT)
    .with_columns(pl.col("time").dt.date().alias("day"))
)
wx = (
    metar.filter(pl.col("day").dt.month() == 2)
    .group_by("day")
    .agg(
        pl.len().alias("reports"),
        pl.col("wx").fill_null("").str.contains("SN").sum().alias("snow reports"),
        pl.col("temp_c").min().round(1).alias("min temp"),
        pl.col("vis_km").min().round(1).alias("min vis km"),
    )
)
oof_file = max(
    (f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == OOF_NAME),
    key=lambda f: f.stat().st_mtime,
)
oof = pl.read_parquet(oof_file, glob=False).select("mvt_id", "airport", "y", "yhat")
lab = (
    tio.scan_training()
    .filter((pl.col("phase") == "DEP") & (pl.col("airport") == AIRPORT))
    .select("mvt_id", pl.col("mvt_time").dt.date().alias("day"))
    .collect(engine="streaming")
)
res = (
    oof.join(lab, on="mvt_id", how="inner")
    .filter((pl.col("day").dt.month() == 2) & (pl.col("y") <= TRIM))
    .group_by("day")
    .agg(
        pl.len().alias("departures"),
        pl.col("y").mean().round(0).alias("mean label"),
        (pl.col("y") - pl.col("yhat")).mean().round(0).alias("mean residual"),
        ((pl.col("y") - pl.col("yhat")) ** 2).mean().sqrt().round(0).alias("rmse"),
    )
)
t = wx.join(res, on="day", how="left").sort("day")
rec.show(t.filter(pl.col("day").dt.year() == 2025), "February 2025")
rec.show(t.filter(pl.col("day").dt.year() == 2026), "February 2026 (no labels)")

# %%
rec.section("2. published mean taxi-out (min), February and June, 2025 against 2026")


def published(year: int) -> pl.DataFrame:
    f = paths.DATA / "external" / "eurocontrol" / f"taxi_out_additional_time_{year}.csv"
    return (
        pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy")
        .filter(
            pl.col("APT_ICAO").is_in(AIRPORTS) & pl.col("MONTH_NUM").is_in(["2", "02", "6", "06"])
        )
        .select(
            pl.col("APT_ICAO").alias("airport"),
            pl.col("MONTH_NUM").str.zfill(2).alias("month"),
            (
                (
                    pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64)
                    + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)
                )
                / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64)
            ).alias(str(year)),
            (pl.col("VALID_FL").cast(pl.Float64) / pl.col("TF").cast(pl.Float64)).alias(
                f"valid {year}"
            ),
        )
    )


p = published(2025).join(published(2026), on=["airport", "month"], how="full", coalesce=True)
rec.show(
    p.with_columns(
        (pl.col("2026") - pl.col("2025")).round(2).alias("Δ min"),
        pl.col("2025").round(2),
        pl.col("2026").round(2),
        pl.col("valid 2025").round(3),
        pl.col("valid 2026").round(3),
    ).sort("month", "airport"),
    "mean actual taxi-out over flights with a reference, minutes",
)
rec.done()
