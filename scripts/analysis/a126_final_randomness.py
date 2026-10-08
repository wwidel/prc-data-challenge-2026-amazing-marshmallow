# %% [markdown]
# # a126 — What did the organisers change in the final file's January and July rows?
#
# `final_ranking.parquet` (final phase: Jan, Feb, Jun, Jul 2026) contains every departure ID of
# `ranking.parquet`, but the organisers introduced randomness, and the earliest and latest Jan/Jul
# timestamps differ between the two files. Several of our exact rules depend on exact times: the
# LOBT window (±3,606 s), the schedule-gap bound (+1,800 s), the one-row `takeoff − aobt_3` rule
# and the 27-row gap rule. So, on the rows the two files share:
#
#   1. rows by phase and month in each file, and the ID overlap per phase;
#   2. per column: how often the value changed, and for timestamps the distribution of the change
#      (final − ranking, seconds);
#   3. whether the change is one shift per row (the same for every timestamp of the row) — then
#      differences between timestamps, which our bounds use, are untouched;
#   4. for arrivals, whose labels are present in both files: does `taxi_sec` still equal
#      `block_time − mvt_time` in the final file?
# Result (2026-10-08): only `mvt_time`, and `block_time` on arrivals, moved: by an integer number
# of seconds in [−12, 12], independently per row and per column; arrival labels were recomputed
# from the moved times. Hence `paths.TIME_NOISE`.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
TIMES = (
    "mvt_time",
    "block_time",
    "sched_time",
    "lobt",
    "iobt",
    "eobt_1",
    "arvt_1",
    "aobt_3",
    "arvt_3",
)
OTHER = (
    "flight_id",
    "flight_no",
    "flight_rule",
    "adep",
    "ades",
    "phase",
    "ac_type",
    "runway",
    "stand",
    "taxi_sec",
    "callsign",
    "adep_flt",
    "ades_flt",
    "ades_filed",
    "market_segment",
    "flight_rule_flt",
    "flight_type",
    "ac_type_flt",
    "wake_cat",
    "operator",
)
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

rec = analysis.Recorder("a126_final_randomness", "What changed in the final file's Jan/Jul rows?")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)

# Both files explicitly, whatever TAXIOUT_PHASE says.
old = tio._normalise(pl.scan_parquet(paths.RAW / "ranking.parquet"), "rank").collect(
    engine="streaming"
)
new = tio._normalise(pl.scan_parquet(paths.RAW / "final_ranking.parquet"), "final").collect(
    engine="streaming"
)

# %%
rec.section("1. rows by phase and month; ID overlap")
rec.show(
    pl.concat(
        [
            old.group_by("phase", "month").len().with_columns(pl.lit("ranking").alias("file")),
            new.group_by("phase", "month").len().with_columns(pl.lit("final").alias("file")),
        ]
    )
    .pivot(on="file", index=["phase", "month"], values="len")
    .sort("phase", "month")
)
for ph in ("DEP", "ARR"):
    a = set(old.filter(pl.col("phase") == ph)["mvt_id"].to_list())
    b = set(new.filter(pl.col("phase") == ph)["mvt_id"].to_list())
    rec.say(
        f"  {ph}: ranking {len(a):,}; final {len(b):,}; shared {len(a & b):,}; "
        f"ranking-only {len(a - b):,}; final-only {len(b - a):,}"
    )

# %%
rec.section("2. per column, on the shared rows")
both = old.join(new, on="mvt_id", how="inner", suffix="_new")
rec.say(f"  shared rows: {both.height:,}")
rows = []
for c in TIMES:
    d = both.select(
        pl.col("phase"),
        pl.col(c).is_null().alias("null_old"),
        pl.col(f"{c}_new").is_null().alias("null_new"),
        (pl.col(f"{c}_new") - pl.col(c)).dt.total_seconds().alias("delta"),
    )
    nn = d.filter(~pl.col("null_old") & ~pl.col("null_new"))
    q = nn["delta"].quantile
    rows.append(
        {
            "column": c,
            "both non-null": nn.height,
            "null changed": int((d["null_old"] != d["null_new"]).sum()),
            "share changed": round(float((nn["delta"] != 0).mean()), 4) if nn.height else None,
            **{f"q{int(p * 100)}": (q(p) if nn.height else None) for p in QUANTILES},
        }
    )
rec.show(pl.from_dicts(rows), "timestamps: delta = final − ranking, seconds")
other = []
for c in OTHER:
    a, b = both[c].cast(pl.Utf8), both[f"{c}_new"].cast(pl.Utf8)
    diff = ~((a == b).fill_null(False) | (a.is_null() & b.is_null()))
    other.append(
        {"column": c, "rows differing": int(diff.sum()), "share": round(float(diff.mean()), 5)}
    )
rec.show(pl.from_dicts(other), "other columns: rows whose value differs (nulls compared too)")

# %%
rec.section("3. one shift per row? delta(column) − delta(mvt_time), seconds")
rel = []
for c in TIMES:
    if c == "mvt_time":
        continue
    x = both.select(
        (
            (pl.col(f"{c}_new") - pl.col(c)).dt.total_seconds()
            - (pl.col("mvt_time_new") - pl.col("mvt_time")).dt.total_seconds()
        ).alias("r")
    ).drop_nulls()["r"]
    rel.append(
        {
            "column": c,
            "rows": x.len(),
            "share zero": round(float((x == 0).mean()), 4) if x.len() else None,
            "min": x.min(),
            "q1": x.quantile(0.01),
            "q50": x.quantile(0.5),
            "q99": x.quantile(0.99),
            "max": x.max(),
        }
    )
rec.show(pl.from_dicts(rel), "zero everywhere = the whole row moved together")
rec.show(
    both.with_columns(
        (pl.col("mvt_time_new") - pl.col("mvt_time")).dt.total_seconds().alias("d_mvt")
    )
    .group_by("phase", "airport")
    .agg(
        pl.len().alias("rows"),
        (pl.col("d_mvt") != 0).mean().round(4).alias("share mvt moved"),
        pl.col("d_mvt").abs().max().alias("max |d_mvt|"),
        pl.col("d_mvt").std().round(1).alias("sd d_mvt"),
    )
    .sort("phase", "airport"),
    "mvt_time change by phase and airport",
)

# %%
rec.section("4. arrivals: is the label still block_time − mvt_time?")
arr = both.filter(pl.col("phase") == "ARR").select(
    "taxi_sec",
    "taxi_sec_new",
    (pl.col("block_time") - pl.col("mvt_time")).dt.total_seconds().alias("diff_old"),
    (pl.col("block_time_new") - pl.col("mvt_time_new")).dt.total_seconds().alias("diff_new"),
)
rec.show(
    arr.select(
        pl.len().alias("arrivals"),
        (pl.col("taxi_sec") == pl.col("diff_old")).mean().round(4).alias("old: label = diff"),
        (pl.col("taxi_sec_new") == pl.col("diff_new")).mean().round(4).alias("final: label = diff"),
        (pl.col("taxi_sec_new") == pl.col("taxi_sec")).mean().round(4).alias("label unchanged"),
        (pl.col("taxi_sec_new") - pl.col("taxi_sec")).abs().max().alias("max |label change|"),
    )
)
rec.done()
