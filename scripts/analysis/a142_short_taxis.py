# %% [markdown]
# # a142 — Short taxis: where they are, and what the off-block evidence said about them
#
# a139: a final-phase build over-predicts taxis under 10 minutes by 60–130 s on average, while its
# own short predictions are calibrated (E[label | prediction] ≈ prediction in the lowest deciles).
# Can short taxis be told apart in advance? On the 2025 out-of-fold rows without Rome, with the
# stack of A142_MEMBERS (weights fitted on the other pair, as `a129`):
#   1. label bands, with and without an ADS-B trace: rows, mean prediction, mean label, residual;
#   2. per airport: the share of departures under 10 minutes, and their residual;
#   3. for taxis under 10 minutes: what `takeoff − aobt_3` and `takeoff − eobt_1` said (bands);
#   4. the other way round: rows whose `takeoff − aobt_3` is under 10 minutes — how many are short;
#   5. short taxis whose label equals `takeoff − aobt_3` within 2 minutes (the stamp was copied).
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, store  # noqa: E402
from taxiout.features import registry  # noqa: E402

# %% knobs
TRIM = 43_200.0
SHORT = 600.0
COPY_TOL = 120.0
GROUP_MIN_ROWS = 2000
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
MEMBERS_FILE = os.environ.get("A142_MEMBERS", "")
MEMBERS = Path(MEMBERS_FILE).read_text().split() if MEMBERS_FILE else paths.composition()
HELD = os.environ.get("A142_UNMATCHED_WITHOUT", "02")
BANDS = [300, 600, 900, 1_200, 1_800, 3_600]
BAND_LABELS = ["<5m", "5-10m", "10-15m", "15-20m", "20-30m", "30-60m", ">60m"]

rec = analysis.Recorder("a142_short_taxis", "Short taxis and the off-block evidence")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)


def newest_oof(name: str) -> Path:
    hits = [f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == name]
    if not hits:
        raise FileNotFoundError(f"no out-of-fold file for {name}")
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
rec.section("inputs")
frame = None
for m in MEMBERS:
    f = newest_oof(m)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(m))
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
raw = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "month",
        pl.col("flight_id").is_null().alias("unm"),
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64).alias("pa"),
        (pl.col("mvt_time") - pl.col("eobt_1")).dt.total_seconds().cast(pl.Float64).alias("pe"),
    )
    .collect(engine="streaming")
)
frame = frame.join(raw, on="mvt_id", how="inner").filter(pl.col("airport") != "LIRF")
block = store.block_dir(registry.get("adsb"))
parts = []
for m in sorted(frame["month"].unique().to_list()):
    ids = pl.read_parquet(paths.SPINE / f"{m}.parquet", columns=["mvt_id"], glob=False)
    v = pl.read_parquet(block / f"{m}.parquet", columns=["adsb_match_pri"], glob=False)
    parts.append(pl.concat([ids, v], how="horizontal"))
cov = pl.concat(parts).select("mvt_id", pl.col("adsb_match_pri").is_not_null().alias("adsb"))
frame = frame.join(cov, on="mvt_id", how="left", maintain_order="left").with_columns(
    pl.col("adsb").fill_null(False)
)
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


def band(col: str, name: str) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit("missing"))
        .otherwise(pl.col(col).cut(BANDS, labels=BAND_LABELS, left_closed=False).cast(pl.Utf8))
        .alias(name)
    )


d = (
    frame.select("airport", "y", "unm", "adsb", "pa", "pe")
    .with_columns(pl.Series("prediction", pred))
    .filter(pl.col("y") <= TRIM)
    .with_columns(
        (pl.col("y") - pl.col("prediction")).alias("res"),
        (pl.col("y") < SHORT).alias("short"),
        ((pl.col("y") - pl.col("pa")).abs() <= COPY_TOL).alias("copied aobt_3"),
        band("y", "label band"),
        band("pa", "takeoff − aobt_3"),
        band("pe", "takeoff − eobt_1"),
    )
)
rec.say(f"  members ({len(MEMBERS)}) from {MEMBERS_FILE or 'the BASE composition'}")
rec.say(
    f"  rows without Rome, labels <= 12 h: {d.height:,}; under 10 min: {int(d['short'].sum()):,}"
)


def summary(frame_: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        frame_.group_by(by)
        .agg(
            pl.len().alias("rows"),
            pl.col("prediction").mean().round(0).alias("mean prediction"),
            pl.col("y").mean().round(0).alias("mean label"),
            pl.col("res").mean().round(1).alias("mean residual"),
            (pl.col("res") ** 2).mean().sqrt().round(1).alias("rmse"),
        )
        .sort(by)
    )


# %%
rec.section("1. label bands, with and without an ADS-B trace")
rec.show(summary(d, ["adsb", "label band"]))

# %%
rec.section("2. per airport: departures under 10 minutes")
rec.show(
    d.group_by("airport")
    .agg(
        pl.len().alias("rows"),
        pl.col("short").mean().round(4).alias("share under 10 min"),
        pl.col("res").filter(pl.col("short")).mean().round(1).alias("residual, short"),
        pl.col("adsb").filter(pl.col("short")).mean().round(3).alias("ADS-B share, short"),
    )
    .sort("airport")
)

# %%
rec.section("3. taxis under 10 minutes: what the off-block stamps said")
short = d.filter(pl.col("short"))
rec.show(summary(short, ["takeoff − aobt_3"]), "by takeoff − aobt_3")
rec.show(summary(short, ["takeoff − eobt_1"]), "by takeoff − eobt_1")

# %%
rec.section("4. rows whose takeoff − aobt_3 is under 10 minutes")
rec.show(
    d.filter(pl.col("pa") < SHORT)
    .group_by("adsb")
    .agg(
        pl.len().alias("rows"),
        pl.col("short").mean().round(3).alias("share actually under 10 min"),
        pl.col("y").mean().round(0).alias("mean label"),
        pl.col("prediction").mean().round(0).alias("mean prediction"),
        pl.col("res").mean().round(1).alias("mean residual"),
    )
    .sort("adsb")
)

# %%
rec.section("5. taxis under 10 minutes: was aobt_3 copied into the label?")
rec.show(summary(short, ["adsb", "copied aobt_3"]))
rec.done()
