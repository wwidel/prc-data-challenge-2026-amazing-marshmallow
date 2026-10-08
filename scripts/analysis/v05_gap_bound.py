# %% [markdown]
# # v05 — The per-row admissible upper bound: taxi-out ≤ (takeoff − schedule) + T
#
# `y = mvt_time − block_time`, so any admissible lower bound on `block_time` is an upper bound on
# `y`, and `sched_time` is on every ranking row. This measures how often `block_time` precedes the
# schedule by more than T — the violation rate of `y ≤ gap + T`. The bound is only meaningful on
# rows with a positive gap (24% of 2025 rows push before their schedule, so `y > gap` there);
# `postprocess.apply_gap_bound` applies it with T = 1,800 s to positive-gap rows only.
#
# Section C of our review script `v05_offset_target.py`, unchanged in substance.

# %%
import polars as pl

from taxiout import io as tio
from taxiout.analysis import Recorder

rec = Recorder("v05_gap_bound", "How often does block_time precede the schedule by more than T?")

dep = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "airport",
        "taxi_sec",
        pl.col("flight_id").is_null().alias("unmatched"),
        (pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("gap"),
        (pl.col("block_time") - pl.col("sched_time")).dt.total_seconds().alias("blk_m_sched"),
    )
    .collect(engine="streaming")
)
full = dep.filter(pl.col("taxi_sec").is_not_null())
rec.say(f"  2025 departures {dep.height:,}; with a label {full.height:,}")

# %%
rec.section("1. how far block_time can precede the schedule (all 2025 departures)")
rec.show(
    full.select(
        pl.len().alias("n"),
        (pl.col("blk_m_sched") < 0).mean().round(4).alias("p_block_before_sched"),
        (pl.col("blk_m_sched") < -300).mean().round(4).alias("p_lt_m300"),
        (pl.col("blk_m_sched") < -900).mean().round(4).alias("p_lt_m900"),
        (pl.col("blk_m_sched") < -1800).mean().round(4).alias("p_lt_m1800"),
        (pl.col("blk_m_sched") < -3600).mean().round(4).alias("p_lt_m3600"),
        pl.col("blk_m_sched").quantile(0.001).alias("q001"),
        pl.col("blk_m_sched").quantile(0.0001).alias("q0001"),
    )
)

# %%
rec.section("2. violation rate of y <= gap + T on rows with a positive gap")
pos = full.filter(pl.col("gap") > 0)
rows = []
for T in (0, 300, 900, 1800, 3600, 7200):
    v = pos.filter(pl.col("taxi_sec") > pl.col("gap") + T)
    rows.append(
        {
            "T": T,
            "n_violating": v.height,
            "rate": round(v.height / pos.height, 5),
            "mean_excess": round(float((v["taxi_sec"] - v["gap"] - T).mean()), 0)
            if v.height
            else None,
            "p_unmatched_among_violators": round(float(v["unmatched"].mean()), 3)
            if v.height
            else None,
        }
    )
rec.show(pl.DataFrame(rows), f"{pos.height:,} rows with a positive gap")
rec.done()
