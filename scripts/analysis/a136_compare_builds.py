# %% [markdown]
# # a136 — Two builds of the final file side by side, per month, airport and day
#
# For a candidate against its predecessor (by default F1, the LightGBM snow swaps, against F0,
# v62's members on the final pipeline): where do the predictions move?
#   1. per month: rows, the mean of each build, the mean and RMS of B − A, rows moving > 30 s;
#   2. per month and airport, the same;
#   3. per day for one airport-month (default LTFM, February 2026), with the day's METAR snow
#      reports.
# Builds: A136_A and A136_B (prediction parquets of the final file). Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
A = Path(os.environ.get("A136_A", str(paths.OUTPUT / "F0.parquet")))
B = Path(os.environ.get("A136_B", str(paths.OUTPUT / "F1.parquet")))
DAY_AIRPORT = os.environ.get("A136_DAY_AIRPORT", "LTFM")
DAY_MONTH = os.environ.get("A136_DAY_MONTH", "2026-02")
MOVE = 30.0

rec = analysis.Recorder("a136_compare_builds", "Two builds of the final file side by side")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

# %%
rec.section("inputs")
if not paths.FINAL:
    raise SystemExit("run under TAXIOUT_PHASE=final")


def build(path: Path, col: str) -> pl.DataFrame:
    return pl.read_parquet(path, glob=False).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(col),
    )


rows = (
    tio.scan_ranking()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        pl.col("mvt_time").dt.date().alias("day"),
    )
    .collect(engine="streaming")
)
d = (
    rows.join(build(A, "a"), on="mvt_id")
    .join(build(B, "b"), on="mvt_id")
    .with_columns((pl.col("b") - pl.col("a")).alias("d"))
)
rec.say(f"  A = {A}; B = {B}; rows {d.height:,} of {rows.height:,}")


def summary(by: list[str]) -> pl.DataFrame:
    return (
        d.group_by(by)
        .agg(
            pl.len().alias("rows"),
            pl.col("a").mean().round(1).alias("mean A"),
            pl.col("b").mean().round(1).alias("mean B"),
            pl.col("d").mean().round(2).alias("mean B − A"),
            (pl.col("d") ** 2).mean().sqrt().round(2).alias("rms B − A"),
            (pl.col("d").abs() > MOVE).mean().round(4).alias(f"share |Δ| > {MOVE:.0f} s"),
        )
        .sort(by)
    )


# %%
rec.section("1. per month")
rec.show(summary(["month"]))

# %%
rec.section("2. per month and airport")
rec.show(summary(["month", "airport"]))

# %%
rec.section(f"3. {DAY_AIRPORT}, {DAY_MONTH}, per day")
metar = pl.read_parquet(paths.DATA / "external" / "metar" / "metar.parquet", glob=False)
snow = (
    metar.filter(pl.col("airport") == DAY_AIRPORT)
    .with_columns(pl.col("time").dt.date().alias("day"))
    .group_by("day")
    .agg(pl.col("wx").fill_null("").str.contains("SN|SG|PL|GS").sum().alias("snow reports"))
)
day = d.filter((pl.col("airport") == DAY_AIRPORT) & (pl.col("month") == DAY_MONTH))
rec.show(
    day.group_by("day")
    .agg(
        pl.len().alias("rows"),
        pl.col("a").mean().round(1).alias("mean A"),
        pl.col("b").mean().round(1).alias("mean B"),
        pl.col("d").mean().round(2).alias("mean B − A"),
    )
    .join(snow, on="day", how="left")
    .sort("day")
)
rec.done()
