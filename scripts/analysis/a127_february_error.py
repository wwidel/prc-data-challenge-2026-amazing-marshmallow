# %% [markdown]
# # a127 — Where February's error sits (x36's all-months out-of-fold predictions, 2025)
#
# x36's leave-one-month-out CV puts February far above every other month (trimmed ex-LIRF: Feb
# 263.65 against Jan 205.79, Jun 196.11, Jul 231.43). February 2026 is a quarter of the final
# phase's combined score and half of its Feb/Jun ranking, so: where does February's squared error
# sit?
#   1. per month: rows, RMSE untrimmed / trimmed, without Rome, and each month's share of the
#      year's squared error;
#   2. February beside January, June and July, split by airport, NM match and label band;
#   3. February's airport-days ranked by squared error — how concentrated is it — with the day's
#      arrival ATFM delay by cause from APT-DLY (de-icing D, weather W, aerodrome capacity G);
#   4. per winter month: residual and RMSE on days with de-icing delay against the other days.
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
OOF_NAME = "w08_x36_all_months"
TRIM = 43_200.0
TOP_DAYS = 25
DEICE_MIN = 1.0  # a "de-icing day": at least this many minutes of APT-DLY cause-D delay
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")

rec = analysis.Recorder("a127_february_error", "Where does February's error sit?")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)

oof_file = max(
    (f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == OOF_NAME),
    key=lambda f: f.stat().st_mtime,
)
oof = pl.read_parquet(oof_file, glob=False).select("mvt_id", "airport", "y", "yhat")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "month",
        pl.col("mvt_time").dt.date().alias("day"),
        pl.col("flight_id").is_null().alias("unmatched"),
    )
    .collect(engine="streaming")
)
d = oof.join(lab, on="mvt_id", how="inner").with_columns(
    (pl.col("y") - pl.col("yhat")).alias("res"),
    ((pl.col("y") - pl.col("yhat")) ** 2).alias("se"),
    pl.col("y")
    .cut([3_600, 10_800, 43_200], labels=["<=1h", "1-3h", "3-12h", ">12h"])
    .cast(pl.Utf8)
    .alias("band"),
)
rec.say(f"  {oof_file.name}: {oof.height:,} OOF rows; joined {d.height:,}")

# %%
rec.section("1. per month")
ex = pl.col("airport") != "LIRF"
trim = pl.col("y") <= TRIM
rec.show(
    d.group_by("month")
    .agg(
        pl.len().alias("rows"),
        pl.col("se").mean().sqrt().round(1).alias("rmse"),
        pl.col("se").filter(trim).mean().sqrt().round(1).alias("trimmed"),
        pl.col("se").filter(ex & trim).mean().sqrt().round(2).alias("trimmed ex-LIRF"),
        pl.col("se").filter(~ex).mean().sqrt().round(1).alias("LIRF"),
        (pl.col("se").sum() / d["se"].sum()).round(4).alias("share of year SSE"),
    )
    .sort("month")
)

# %%
rec.section("2. February beside January, June and July")
four = d.filter(pl.col("month").is_in(["2025-01", "2025-02", "2025-06", "2025-07"]))
for by in ("airport", "unmatched", "band"):
    t = (
        four.filter(trim)
        .group_by(by, "month")
        .agg(pl.len().alias("n"), pl.col("se").mean().sqrt().round(1).alias("rmse"))
        .pivot(on="month", index=by, values=["rmse"], sort_columns=True)
        .sort(by)
    )
    rec.show(t, f"trimmed RMSE by {by} and month")
feb = d.filter(pl.col("month") == "2025-02")
rec.show(
    feb.filter(trim)
    .group_by("airport")
    .agg(
        pl.len().alias("n"),
        pl.col("se").sum().alias("sse"),
        pl.col("res").mean().round(1).alias("mean residual"),
    )
    .with_columns((pl.col("sse") / pl.col("sse").sum()).round(4).alias("share of Feb SSE"))
    .sort("sse", descending=True),
    "February, trimmed: where the squared error is",
)

# %%
rec.section("3. February's airport-days by squared error, with APT-DLY delay by cause")
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
            for c, a in (
                ("FLT_ARR_1", "arrivals"),
                ("DLY_APT_ARR_1", "dly_total"),
                ("DLY_APT_ARR_D_1", "dly_deicing"),
                ("DLY_APT_ARR_W_1", "dly_weather"),
                ("DLY_APT_ARR_G_1", "dly_aerodrome"),
            )
        ],
    )
    .group_by("day", "airport")
    .agg(pl.all().sum())
)
days = (
    feb.filter(trim)
    .group_by("airport", "day")
    .agg(
        pl.len().alias("n"),
        pl.col("se").sum().alias("sse"),
        pl.col("res").mean().round(0).alias("mean residual"),
        pl.col("y").mean().round(0).alias("mean label"),
    )
    .with_columns((pl.col("sse") / pl.col("sse").sum()).alias("share"))
    .sort("sse", descending=True)
    .with_columns(pl.col("share").cum_sum().round(3).alias("cumulative share"))
    .join(apt, on=["day", "airport"], how="left")
)
n_days = days.height
rec.say(
    f"  {n_days} February airport-days; the top 10 carry {days['share'].head(10).sum():.1%} of "
    f"February's trimmed SSE, the top 25 {days['share'].head(25).sum():.1%}"
)
rec.show(days.head(TOP_DAYS).with_columns(pl.col("share").round(4)), "top airport-days")

# %%
rec.section("4. de-icing days against the other days, per winter month")
winter = d.filter(pl.col("month").is_in(["2025-01", "2025-02", "2025-12"]) & trim).join(
    apt.select("day", "airport", "dly_deicing", "dly_weather"),
    on=["day", "airport"],
    how="left",
)
rec.show(
    winter.with_columns((pl.col("dly_deicing").fill_null(0) >= DEICE_MIN).alias("deicing day"))
    .group_by("month", "deicing day")
    .agg(
        pl.len().alias("rows"),
        pl.col("day").n_unique().alias("days"),
        pl.col("res").mean().round(1).alias("mean residual"),
        pl.col("se").mean().sqrt().round(1).alias("rmse"),
        (pl.col("se").sum()).alias("sse"),
    )
    .with_columns((pl.col("sse") / pl.col("sse").sum().over("month")).round(4).alias("share"))
    .drop("sse")
    .sort("month", "deicing day"),
    f"de-icing day = APT-DLY cause-D arrival delay >= {DEICE_MIN:.0f} min at that airport",
)
ad = days.select("share", pl.col("dly_deicing").fill_null(0), pl.col("dly_weather").fill_null(0))
rec.say(
    "  February airport-days: corr(share of SSE, de-icing delay) = "
    f"{np.corrcoef(ad['share'], ad['dly_deicing'])[0, 1]:.3f}; with weather delay = "
    f"{np.corrcoef(ad['share'], ad['dly_weather'])[0, 1]:.3f}"
)
rec.done()
