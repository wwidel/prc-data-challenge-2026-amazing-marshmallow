# %% [markdown]
# # a141 — Snow days in 2025 and 2026, and how a final-phase build responds to them
#
# Was there a storm in the 2026 months, and does the build already raise its predictions on such
# days? Per airport-day of the four 2025 and the four 2026 months, from the METAR reports: snow
# reports (SN, SG, PL, GS), heavy snow (+SN), freezing rain or drizzle, the day's lowest
# temperature. Beside them, for 2025: the mean label (<= 12 h) and the out-of-fold mean prediction
# of A141_MEMBERS' stack (weights fitted on the other pair, as `a129`); for 2026: the mean
# prediction of the build A141_BUILD. Each day's mean prediction is also shown against its
# airport-month's median day ("uplift"). Rome is left out. Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
TRIM = 43_200.0
GROUP_MIN_ROWS = 2000
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
MEMBERS_FILE = os.environ.get("A141_MEMBERS", "")
MEMBERS = Path(MEMBERS_FILE).read_text().split() if MEMBERS_FILE else paths.composition()
HELD = os.environ.get("A141_UNMATCHED_WITHOUT", "02")
BUILD = Path(os.environ.get("A141_BUILD", str(paths.OUTPUT / "final.parquet")))
MIN_SNOW_REPORTS = 6  # an airport-day with at least this many snow reports is listed

rec = analysis.Recorder("a141_snow_days", "Snow days in 2025 and 2026, and the build's response")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)


def newest_oof(name: str) -> Path:
    hits = [f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == name]
    if not hits:
        raise FileNotFoundError(f"no out-of-fold file for {name}")
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
rec.section("inputs")
metar = pl.read_parquet(paths.DATA / "external" / "metar" / "metar.parquet", glob=False)
wx = pl.col("wx").fill_null("")
weather = (
    metar.with_columns(pl.col("time").dt.date().alias("day"))
    .group_by("airport", "day")
    .agg(
        wx.str.contains("SN|SG|PL|GS").sum().alias("snow reports"),
        wx.str.contains(r"\+SN").sum().alias("heavy snow"),
        wx.str.contains("FZRA|FZDZ").sum().alias("freezing rain"),
        pl.col("temp_c").min().round(1).alias("min temp"),
    )
)

frame = None
for m in MEMBERS:
    f = newest_oof(m)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(m))
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "month",
        pl.col("flight_id").is_null().alias("unm"),
        pl.col("mvt_time").dt.date().alias("day"),
    )
    .collect(engine="streaming")
)
frame = frame.join(lab, on="mvt_id", how="inner").filter(pl.col("airport") != "LIRF")
X = frame.select(MEMBERS).to_numpy().astype(float)
y = frame["y"].to_numpy().astype(float)
month = frame["month"].to_numpy()
mm = np.array([m[-2:] for m in month])
unm = frame["unm"].fill_null(False).to_numpy()
pred = np.full(len(y), np.nan)
for ms in PAIRS.values():
    target, fit = np.isin(month, ms), ~np.isin(month, ms)
    w_glob = nnls(X[fit], y[fit])[0]
    for g in (False, True):
        rows = fit & (unm == g)
        w = nnls(X[rows], y[rows])[0] if rows.sum() >= GROUP_MIN_ROWS else w_glob
        sel = target & (unm == g)
        pred[sel] = X[sel] @ w
        if g and HELD:
            others = rows & (mm != HELD)
            w_o = nnls(X[others], y[others])[0] if others.sum() >= GROUP_MIN_ROWS else w
            sel_o = sel & (mm != HELD)
            pred[sel_o] = X[sel_o] @ w_o
oof = frame.select("airport", "month", "day", "y").with_columns(pl.Series("prediction", pred))

rank = (
    tio.scan_ranking()
    .filter((pl.col("phase") == "DEP") & (pl.col("airport") != "LIRF"))
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        pl.col("mvt_time").dt.date().alias("day"),
    )
    .collect(engine="streaming")
)
build = pl.read_parquet(BUILD, glob=False).select(
    pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("prediction"),
)
b26 = rank.join(build, on="mvt_id", how="inner")
rec.say(f"  members ({len(MEMBERS)}) from {MEMBERS_FILE or 'the BASE composition'}; build {BUILD}")
rec.say(
    f"  2025 out-of-fold rows without Rome {oof.height:,}; 2026 rows without Rome {b26.height:,}"
)


def per_day(frame_: pl.DataFrame, labelled: bool) -> pl.DataFrame:
    aggs = [pl.len().alias("rows"), pl.col("prediction").mean().round(0).alias("mean prediction")]
    if labelled:
        trimmed = pl.col("y") <= TRIM
        aggs += [
            pl.col("y").filter(trimmed).mean().round(0).alias("mean label"),
            (pl.col("y") - pl.col("prediction")).filter(trimmed).mean().round(0).alias("residual"),
        ]
    days = frame_.group_by("airport", "month", "day").agg(aggs)
    return days.with_columns(
        (pl.col("mean prediction") - pl.col("mean prediction").median().over("airport", "month"))
        .round(0)
        .alias("uplift")
    ).join(weather, on=["airport", "day"], how="left")


# %%
rec.section(f"1. 2026: airport-days with at least {MIN_SNOW_REPORTS} snow reports")
d26 = per_day(b26, labelled=False)
rec.show(
    d26.filter(pl.col("snow reports") >= MIN_SNOW_REPORTS).sort("airport", "day"),
    "uplift = the day's mean prediction minus its airport-month's median day",
)

# %%
rec.section(f"2. 2025: airport-days with at least {MIN_SNOW_REPORTS} snow reports")
d25 = per_day(oof, labelled=True)
rec.show(
    d25.filter(pl.col("snow reports") >= MIN_SNOW_REPORTS).sort("airport", "day"),
    "residual = mean label − mean out-of-fold prediction (labels <= 12 h)",
)

# %%
rec.section("3. snow reports per airport-month, 2025 against 2026")
months = ("01", "02", "06", "07")
cnt = (
    weather.with_columns(pl.col("day").dt.strftime("%Y-%m").alias("month"))
    .filter(pl.col("month").str.slice(5, 2).is_in(months))
    .filter(pl.col("month").str.slice(0, 4).is_in(["2025", "2026"]))
    .group_by("airport", "month")
    .agg(
        pl.col("snow reports").sum().alias("snow reports"),
        (pl.col("snow reports") >= MIN_SNOW_REPORTS).sum().alias("snow days"),
        pl.col("heavy snow").sum().alias("heavy snow"),
    )
    .filter(pl.col("snow reports") > 0)
    .sort("airport", "month")
)
rec.show(cnt)
rec.done()
