# %% [markdown]
# # a125 — The LIRF NM-unmatched override from lg01 (the gate-delay model), for the Rome layer
#
# `a124`: on LIRF NM-unmatched rows with a schedule gap ≤ 25,000 s, lg01 alone beats r14 on both
# 2025 months (RMSE Jan 1,954 vs 2,510, Jul 2,961 vs 3,438), and the blend weight chosen on either
# month and scored on the other is λ = 1 (lg01 alone).
# This writes the override: every 2026 LIRF NM-unmatched ranking row that `pop27_gap` does not
# already serve takes max(lg01, 0) from lg01's twelve-month refit, and prints what changes against
# r14's refit on the same rows. Shipped in v58 (board −0.18).

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
GATE = "lg01_lirf_gate_delay"
ROME = "r14_rules_plus_unmatched_route"
EXACT = paths.DATA / "external" / "pop27_gap.parquet"  # rows served the gap; left alone
GAP_COPY = 25_000.0
OUT = paths.DATA / "external" / "lirf_um_lg01.parquet"

rec = analysis.Recorder("a125_lirf_um_override", "LIRF NM-unmatched override from lg01")
pl.Config.set_tbl_rows(30)

# %%
rec.section("inputs")
rows = (
    tio.scan_ranking()
    .filter(
        (pl.col("phase") == "DEP") & (pl.col("airport") == "LIRF") & pl.col("flight_id").is_null()
    )
    .select(
        "mvt_id",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        (pl.col("mvt_time") - pl.col("sched_time"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("gap"),
    )
    .collect(engine="streaming")
)
pop = pl.read_parquet(EXACT, glob=False).select(pl.col("mvt_id").cast(pl.Int64))["mvt_id"].to_list()
g = pl.read_parquet(paths.DATA / "rank_preds" / f"{GATE}.parquet", glob=False).select(
    pl.col("mvt_id").cast(pl.Int64), pl.col("yhat").alias("lg01")
)
r = pl.read_parquet(paths.DATA / "rank_preds" / f"{ROME}.parquet", glob=False).select(
    pl.col("mvt_id").cast(pl.Int64), pl.col("yhat").alias("r14")
)
d = rows.join(g, on="mvt_id", how="left").join(r, on="mvt_id", how="left")
in_pop = d["mvt_id"].is_in(pop)
rec.say(
    f"  2026 LIRF NM-unmatched departures: {d.height}; in pop27_gap: {int(in_pop.sum())}; gap > "
    f"{GAP_COPY:.0f} outside pop27_gap: {int(((d['gap'] > GAP_COPY) & ~in_pop).sum())}; "
    f"lg01 missing: {d['lg01'].null_count()}"
)

# %%
rec.section("1. what the override changes, against r14's refit on the same rows")
o = d.filter(~in_pop).with_columns(pl.col("lg01").clip(0).alias("prediction"))
rec.show(
    o.group_by("month")
    .agg(
        pl.len().alias("rows"),
        pl.col("r14").mean().round(0).alias("r14 mean"),
        pl.col("prediction").mean().round(0).alias("lg01 mean"),
        (pl.col("prediction") - pl.col("r14")).abs().mean().round(0).alias("mean |Δ|"),
        (pl.col("prediction") - pl.col("r14")).abs().max().round(0).alias("max |Δ|"),
        pl.corr("prediction", "r14").round(3).alias("corr"),
    )
    .sort("month"),
    "per month",
)
o.select("mvt_id", "prediction").write_parquet(OUT)
rec.say(f"  wrote {OUT.relative_to(paths.ROOT)}: {o.height} rows")
rec.done()
