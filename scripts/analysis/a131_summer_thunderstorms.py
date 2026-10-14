# %% [markdown]
# # a131 — Summer: thunderstorm reports beside the out-of-fold residuals
#
# The summer counterpart of a127/a128. July is the harder of the two board months, and June is a
# quarter of the final phase's combined score. The weather block carries only the latest METAR's
# TS flag. This puts beside each other, for the summer months the out-of-fold file holds:
#   1. per month and airport: departures, trimmed RMSE, share of the month's trimmed squared
#      error, airport-days with a thunderstorm (TS) report, and those days' share of the airport's
#      squared error;
#   2. the June and July airport-days with the largest trimmed squared error, with the day's TS
#      reports, peak gust, precipitation and APT-DLY weather delay (Rome left out);
#   3. per departure (Rome left out), TS reports in the h hours up to takeoff (h = 1, 3, 6) as
#      0 / 1 / 2+, and whether the latest report at takeoff carries TS: rows, mean residual, RMSE.
# The out-of-fold file is A131_OOF (a parquet with mvt_id, airport, y, yhat), by default x36's
# all-months file. Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
OOF_NAME = "w08_x36_all_months"
OOF = os.environ.get("A131_OOF", "")
SUMMER = ("2025-05", "2025-06", "2025-07", "2025-08", "2025-09")
FOCUS = ("2025-06", "2025-07")
TRIM = 43_200.0
TOP_DAYS = 25
WINDOWS_H = (1, 3, 6)
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
EXCLUDE = ("LIRF",)  # left out of sections 2 and 3: the Rome layer serves it, apart from the body

rec = analysis.Recorder("a131_summer_thunderstorms", "Summer: thunderstorm reports and residuals")
pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(16)

# %%
rec.section("inputs")
if OOF:
    oof_file = Path(OOF)
else:
    oof_file = max(
        (f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == OOF_NAME),
        key=lambda f: f.stat().st_mtime,
    )
oof = pl.read_parquet(oof_file, glob=False).select("mvt_id", "airport", "y", "yhat")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", "mvt_time", pl.col("mvt_time").dt.date().alias("day"))
    .collect(engine="streaming")
)
d = (
    oof.join(lab, on="mvt_id", how="inner")
    .filter(pl.col("month").is_in(SUMMER) & (pl.col("y") <= TRIM))
    .with_columns(
        (pl.col("y") - pl.col("yhat")).alias("res"),
        ((pl.col("y") - pl.col("yhat")) ** 2).alias("se"),
    )
)
months = sorted(d["month"].unique().to_list())
rec.say(f"  {oof_file}: {oof.height:,} out-of-fold rows")
rec.say(f"  summer rows with a label <= {TRIM:,.0f} s: {d.height:,}, months {months}")

metar = pl.read_parquet(paths.DATA / "external" / "metar" / "metar.parquet", glob=False)
extra = [c for c in ("gust_kt", "precip_mm") if c in metar.columns]
obs = (
    metar.filter(pl.col("airport").is_in(AIRPORTS))
    .select("airport", "time", "wx", *extra)
    .sort("airport", "time")
    .with_columns(pl.col("wx").fill_null("").str.contains("TS").cast(pl.Int32).alias("ts"))
    .with_columns(pl.col("ts").cum_sum().over("airport").alias("c_ts"))
)
rec.say(f"  METAR reports at the ten airports: {obs.height:,}; with TS: {obs['ts'].sum():,}")
day_aggs = [pl.col("ts").sum().alias("ts reports")]
if "gust_kt" in extra:
    day_aggs.append(pl.col("gust_kt").max().alias("max gust kt"))
if "precip_mm" in extra:
    day_aggs.append(pl.col("precip_mm").sum().round(1).alias("precip mm"))
wx_day = (
    obs.with_columns(pl.col("time").dt.date().alias("day")).group_by("airport", "day").agg(day_aggs)
)

# %%
rec.section("1. per month and airport")
by_ap = (
    d.join(wx_day.select("airport", "day", "ts reports"), on=["airport", "day"], how="left")
    .with_columns((pl.col("ts reports").fill_null(0) > 0).alias("ts day"))
    .group_by("month", "airport")
    .agg(
        pl.len().alias("n"),
        pl.col("se").mean().sqrt().round(1).alias("trimmed rmse"),
        pl.col("se").sum().alias("sse"),
        pl.col("day").filter(pl.col("ts day")).n_unique().alias("ts days"),
        (pl.col("se").filter(pl.col("ts day")).sum() / pl.col("se").sum())
        .round(3)
        .alias("ts days' share of sse"),
        pl.col("res").filter(pl.col("ts day")).mean().round(1).alias("mean res, ts days"),
        pl.col("res").filter(~pl.col("ts day")).mean().round(1).alias("mean res, other days"),
    )
    .with_columns((pl.col("sse") / pl.col("sse").sum().over("month")).round(3).alias("share"))
    .drop("sse")
    .sort("month", "airport")
)
rec.show(by_ap, "trimmed; share = the airport's share of the month's squared error")

# %%
rec.section("2. June and July airport-days by squared error")
apt = (
    pl.read_csv(
        paths.DATA / "external" / "eurocontrol" / "apt_dly_2025.csv",
        infer_schema_length=0,
        encoding="utf8-lossy",
    )
    .filter(pl.col("APT_ICAO").is_in(AIRPORTS))
    .select(
        pl.col("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
        pl.col("APT_ICAO").alias("airport"),
        *[
            pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(a)
            for c, a in (("DLY_APT_ARR_1", "dly total"), ("DLY_APT_ARR_W_1", "dly weather"))
        ],
    )
    .group_by("day", "airport")
    .agg(pl.all().sum())
)
for m in FOCUS:
    if m not in months:
        continue
    mm = d.filter((pl.col("month") == m) & ~pl.col("airport").is_in(EXCLUDE))
    days = (
        mm.group_by("airport", "day")
        .agg(
            pl.len().alias("n"),
            pl.col("se").sum().alias("sse"),
            pl.col("res").mean().round(0).alias("mean residual"),
            pl.col("y").mean().round(0).alias("mean label"),
        )
        .with_columns((pl.col("sse") / pl.col("sse").sum()).alias("share"))
        .sort("sse", descending=True)
        .with_columns(pl.col("share").cum_sum().round(3).alias("cumulative share"))
        .join(wx_day, on=["airport", "day"], how="left")
        .join(apt, on=["day", "airport"], how="left")
        .drop("sse")
    )
    rec.say(
        f"  {m}: {days.height} airport-days; the top 10 carry {days['share'].head(10).sum():.1%} "
        f"of the month's trimmed squared error, the top 25 {days['share'].head(25).sum():.1%}"
    )
    rec.show(days.head(TOP_DAYS).with_columns(pl.col("share").round(4)), f"{m}: top airport-days")

# %%
rec.section("3. per departure: TS reports before takeoff")
rows = (
    d.filter(~pl.col("airport").is_in(EXCLUDE))
    .select("mvt_id", "airport", "month", "res", "se", pl.col("mvt_time").alias("_k"))
    .sort("_k")
)
o = obs.select("airport", pl.col("time").alias("_k"), "c_ts", "ts").sort("_k")
now = rows.join_asof(o, on="_k", by="airport", strategy="backward")
for h in WINDOWS_H:
    past = (
        rows.select("mvt_id", "airport", (pl.col("_k") - pl.duration(hours=h)).alias("_k"))
        .sort("_k")
        .join_asof(
            o.select("airport", "_k", pl.col("c_ts").alias(f"c_ts_{h}")),
            on="_k",
            by="airport",
            strategy="backward",
        )
        .select("mvt_id", f"c_ts_{h}")
    )
    now = now.join(past, on="mvt_id", how="left").with_columns(
        (pl.col("c_ts") - pl.col(f"c_ts_{h}").fill_null(0)).fill_null(0).alias(f"ts_{h}h")
    )
now = now.with_columns(pl.col("ts").fill_null(0).alias("ts at takeoff"))
for h in WINDOWS_H:
    t = (
        now.with_columns(
            pl.when(pl.col(f"ts_{h}h") == 0)
            .then(pl.lit("0"))
            .when(pl.col(f"ts_{h}h") == 1)
            .then(pl.lit("1"))
            .otherwise(pl.lit("2+"))
            .alias(f"TS reports, last {h} h")
        )
        .group_by("month", f"TS reports, last {h} h", "ts at takeoff")
        .agg(
            pl.len().alias("rows"),
            pl.col("res").mean().round(1).alias("mean residual"),
            pl.col("se").mean().sqrt().round(1).alias("rmse"),
            pl.col("se").sum().alias("sse"),
        )
        .with_columns((pl.col("sse") / pl.col("sse").sum().over("month")).round(4).alias("share"))
        .drop("sse")
        .sort("month", f"TS reports, last {h} h", "ts at takeoff")
    )
    rec.show(t, f"window ({h} h before takeoff, takeoff]; ts at takeoff = the latest report's flag")
rec.done()
