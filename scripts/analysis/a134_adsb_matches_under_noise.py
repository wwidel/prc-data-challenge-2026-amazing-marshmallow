# %% [markdown]
# # a134 — ADS-B matches under the organisers' noise
#
# a133: on the January and July 2026 rows, `adsb_taxi` differs between the leaderboard-phase and
# the final-phase stores on 54% of rows, by 26 s RMS, while the takeoff times moved by at most
# 12 s (sd 4.9). The block matches each departure to an ADS-B takeoff by callsign within 300 s
# (priority 0), else by the nearest takeoff within 90 s (priority 1), greedily by |Δt|; its
# features are measured from the record's takeoff time (`adsb_taxi = mvt − first ground point`,
# `adsb_takeoff_err = trace takeoff − mvt`). If a row keeps its event, `adsb_takeoff_err` moves by
# exactly −dt (dt = final − leaderboard takeoff time) and `adsb_taxi` by +dt. So, per month and
# priority:
#   1. rows matched in both stores, in one only, in neither;
#   2. of the rows matched in both: same event, a different event, and the size of the
#      `adsb_taxi` change on each, net of dt.
# Leaderboard-phase store: A134_BOARD_DATA (default this checkout's data/). Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, store  # noqa: E402
from taxiout.features import registry  # noqa: E402

# %% knobs
BOARD_DATA = Path(os.environ.get("A134_BOARD_DATA", str(paths.DATA)))
MONTHS = ("2026-01", "2026-07")
SAME_EVENT_TOL = 1.0  # s: |Δ takeoff_err + dt| within this means the same ADS-B event

rec = analysis.Recorder("a134_adsb_matches_under_noise", "ADS-B matches under the noise")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)

# %%
rec.section("inputs")
if not paths.FINAL:
    raise SystemExit("run under TAXIOUT_PHASE=final")
block = registry.get("adsb")
fdir = store.block_dir(block)
broot = BOARD_DATA / "features" / "adsb"
bdir = max(
    (d for d in broot.iterdir() if all((d / f"{m}.parquet").exists() for m in MONTHS)),
    key=lambda d: d.stat().st_mtime,
)
rec.say(f"  leaderboard store: {bdir}; final store: {fdir}")
COLS = ["adsb_taxi", "adsb_takeoff_err", "adsb_match_pri"]


def load(spine_dir: Path, block_dir: Path, month: str) -> pl.DataFrame:
    ids = pl.read_parquet(spine_dir / f"{month}.parquet", columns=["mvt_id", "airport"])
    return pl.concat(
        [ids, pl.read_parquet(block_dir / f"{month}.parquet", columns=COLS)], how="horizontal"
    )


old_t = (
    tio._normalise(pl.scan_parquet(paths.RAW / "ranking.parquet"), "rank")
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "mvt_time")
    .collect(engine="streaming")
)
new_t = (
    tio._normalise(pl.scan_parquet(paths.RAW / "final_ranking.parquet"), "final")
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", pl.col("mvt_time").alias("mvt_final"))
    .collect(engine="streaming")
)
dt = old_t.join(new_t, on="mvt_id").select(
    "mvt_id", (pl.col("mvt_final") - pl.col("mvt_time")).dt.total_seconds().alias("dt")
)
frames = []
for m in MONTHS:
    b = load(BOARD_DATA / "interim" / "spine", bdir, m)
    f = load(paths.SPINE, fdir, m)
    frames.append(
        b.join(f.drop("airport"), on="mvt_id", suffix="_f")
        .join(dt, on="mvt_id")
        .with_columns(pl.lit(m).alias("month"))
    )
d = pl.concat(frames)
rec.say(f"  rows: {d.height:,} ({', '.join(MONTHS)})")

# %%
rec.section("1. matched in both, one, neither")
mb, mf = pl.col("adsb_match_pri").is_not_null(), pl.col("adsb_match_pri_f").is_not_null()
rec.show(
    d.group_by("month")
    .agg(
        pl.len().alias("rows"),
        (mb & mf).sum().alias("both"),
        (mb & ~mf).sum().alias("leaderboard only"),
        (~mb & mf).sum().alias("final only"),
        (~mb & ~mf).sum().alias("neither"),
    )
    .sort("month")
)

# %%
rec.section("2. rows matched in both: same event or another")
both = d.filter(mb & mf).with_columns(
    (
        (pl.col("adsb_takeoff_err_f") - pl.col("adsb_takeoff_err") + pl.col("dt")).abs()
        <= SAME_EVENT_TOL
    ).alias("same event"),
    (pl.col("adsb_taxi_f") - pl.col("adsb_taxi") - pl.col("dt")).alias("taxi Δ net of dt"),
)
rec.show(
    both.group_by("month", "adsb_match_pri", "same event")
    .agg(
        pl.len().alias("rows"),
        pl.col("taxi Δ net of dt").abs().median().round(1).alias("median |Δ| net"),
        (pl.col("taxi Δ net of dt") ** 2).mean().sqrt().round(1).alias("rms Δ net"),
        (pl.col("adsb_match_pri_f") != pl.col("adsb_match_pri")).sum().alias("priority changed"),
    )
    .with_columns((pl.col("rows") / pl.col("rows").sum().over("month")).round(4).alias("share"))
    .sort("month", "adsb_match_pri", "same event"),
    "leaderboard-phase priority; same event = |Δ takeoff_err + dt| <= 1 s",
)
rec.show(
    both.filter(~pl.col("same event"))
    .group_by("airport")
    .agg(pl.len().alias("rows changing event"))
    .join(both.group_by("airport").agg(pl.len().alias("matched")), on="airport")
    .with_columns((pl.col("rows changing event") / pl.col("matched")).round(4).alias("share"))
    .sort("airport"),
    "per airport",
)
rec.done()
