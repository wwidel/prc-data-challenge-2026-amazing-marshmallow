# %% [markdown]
# # a132 — The final build against v62 on the leaderboard's rows: what moved, member by member
#
# v63 (F0's January and July rows) scored 251.8545 on the leaderboard, v62 251.2454. Between the
# two builds: the organisers' ±12 s on takeoff times (L067); every member refitted with features
# computed from the final file; the stack's weights fitted on four months of out-of-fold
# predictions instead of two; the Rome blend and the override files regenerated on four months;
# a120's shift computed on F0's own build. On the 344,841 January and July 2026 departures:
#   1. the takeoff-time change itself (final − leaderboard file, seconds);
#   2. each member's twelve-month refit, final phase against leaderboard phase: mean and RMS of
#      the difference, overall and for Rome, NM-matched and NM-unmatched rows, and the slope and
#      R² of that difference on the takeoff-time change;
#   3. the two builds the same way, and per airport.
# Leaderboard-phase inputs, from a checkout that built v62: A132_BOARD_PREDS (its refits) and
# A132_BOARD_FILE (the v62 build). Final-phase inputs: this checkout under TAXIOUT_PHASE=final.
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
BOARD_PREDS = Path(os.environ.get("A132_BOARD_PREDS", str(paths.DATA / "rank_preds")))
BOARD_FILE = Path(os.environ.get("A132_BOARD_FILE", str(paths.DATA / "output" / "v62.parquet")))
FINAL_PREDS = paths.DATA / "rank_preds_final"
FINAL_FILE = Path(os.environ.get("A132_FINAL_FILE", str(paths.DATA / "output" / "final.parquet")))
MEMBERS = paths.composition()
EXTRA = ("r14_rules_plus_unmatched_route", "lg01_lirf_gate_delay")  # the Rome layer and override

rec = analysis.Recorder("a132_board_vs_final", "F0 against v62 on the leaderboard's rows")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)

# %%
rec.section("inputs")
cols = ("mvt_id", "airport", "mvt_time", "flight_id")
old = (
    tio._normalise(pl.scan_parquet(paths.RAW / "ranking.parquet"), "rank")
    .filter(pl.col("phase") == "DEP")
    .select(*cols)
    .collect(engine="streaming")
)
new = (
    tio._normalise(pl.scan_parquet(paths.RAW / "final_ranking.parquet"), "final")
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", pl.col("mvt_time").alias("mvt_time_final"))
    .collect(engine="streaming")
)
rows = old.join(new, on="mvt_id", how="inner").select(
    "mvt_id",
    "airport",
    pl.when(pl.col("airport") == "LIRF")
    .then(pl.lit("Rome"))
    .when(pl.col("flight_id").is_null())
    .then(pl.lit("NM-unmatched"))
    .otherwise(pl.lit("NM-matched"))
    .alias("group"),
    (pl.col("mvt_time_final") - pl.col("mvt_time")).dt.total_seconds().cast(pl.Float64).alias("dt"),
)
rec.say(f"  leaderboard departures {old.height:,}; also in the final file {rows.height:,}")
rec.say(f"  members ({len(MEMBERS)}): {', '.join(MEMBERS)}; plus {', '.join(EXTRA)}")
rec.say(f"  leaderboard refits: {BOARD_PREDS}; final refits: {FINAL_PREDS}")
rec.say(f"  builds: {BOARD_FILE} against {FINAL_FILE}")
rec.show(
    rows.group_by("group")
    .agg(
        pl.len().alias("rows"),
        (pl.col("dt") != 0).mean().round(3).alias("moved"),
        pl.col("dt").std().round(2).alias("sd dt"),
        pl.col("dt").min().alias("min"),
        pl.col("dt").max().alias("max"),
    )
    .sort("group"),
    "1. takeoff time, final − leaderboard file (s)",
)


# %%
def compare(a: pl.DataFrame, b: pl.DataFrame, label: str) -> list[dict]:
    """a, b: (mvt_id, v). Per group: mean / RMS of b − a, slope and R² on dt."""
    j = rows.join(a.rename({"v": "a"}), on="mvt_id").join(b.rename({"v": "b"}), on="mvt_id")
    j = j.with_columns((pl.col("b") - pl.col("a")).alias("d"))
    out = []
    for g, part in [("all", j), *sorted(j.group_by("group"), key=lambda t: t[0][0])]:
        name = g if isinstance(g, str) else g[0]
        d, x = part["d"].to_numpy(), part["dt"].to_numpy()
        slope = float(np.polyfit(x, d, 1)[0]) if np.std(x) > 0 else float("nan")
        r2 = float(np.corrcoef(x, d)[0, 1] ** 2) if np.std(d) > 0 and np.std(x) > 0 else 0.0
        out.append(
            {
                "what": label,
                "rows": name,
                "n": part.height,
                "mean Δ": round(float(d.mean()), 2),
                "rms Δ": round(float(np.sqrt((d**2).mean())), 2),
                "share |Δ|>30 s": round(float((np.abs(d) > 30).mean()), 4),
                "slope on dt": round(slope, 3),
                "R² on dt": round(r2, 3),
            }
        )
    return out


def refit(directory: Path, name: str) -> pl.DataFrame:
    f = directory / f"{name}.parquet"
    return pl.read_parquet(f, glob=False).select(
        "mvt_id", pl.col("yhat").cast(pl.Float64).alias("v")
    )


rec.section("2. each member's refit, final phase − leaderboard phase")
table = []
for m in (*MEMBERS, *EXTRA):
    if not (BOARD_PREDS / f"{m}.parquet").exists() or not (FINAL_PREDS / f"{m}.parquet").exists():
        rec.say(f"  {m}: missing in one phase, skipped")
        continue
    table += compare(refit(BOARD_PREDS, m), refit(FINAL_PREDS, m), m)
rec.show(pl.from_dicts(table), "Δ = final-phase refit − leaderboard-phase refit (s)")


# %%
def build(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, glob=False).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v"),
    )


rec.section("3. the builds, F0 − v62")
a, b = build(BOARD_FILE), build(FINAL_FILE)
rec.show(pl.from_dicts(compare(a, b, "F0 − v62")), "by row group")
per_ap = (
    rows.join(a.rename({"v": "v62"}), on="mvt_id")
    .join(b.rename({"v": "F0"}), on="mvt_id")
    .with_columns((pl.col("F0") - pl.col("v62")).alias("d"))
    .group_by("airport")
    .agg(
        pl.len().alias("n"),
        pl.col("d").mean().round(2).alias("mean Δ"),
        (pl.col("d") ** 2).mean().sqrt().round(2).alias("rms Δ"),
        pl.col("v62").mean().round(1).alias("mean v62"),
        pl.col("F0").mean().round(1).alias("mean F0"),
    )
    .sort("airport")
)
rec.show(per_ap, "per airport")
rec.done()
