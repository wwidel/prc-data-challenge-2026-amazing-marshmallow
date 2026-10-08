# %% [markdown]
# # a005 — The basics
#
# Inventory of the raw files: what is in them, what type it is, how much of it there is, how
# much is missing, and the summary statistics.
#
# Knobs in the next cell: `SUBSETS` controls which split/phase combinations the null and
# statistic tables are broken down by.

# %%
import polars as pl

from taxiout import io as tio
from taxiout import paths, viz
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder("a005_basics", "What is actually in the raw files?")

#: (label, lazyframe) — every table below is computed for each of these.
SUBSETS = {
    "train DEP": tio.scan_training().filter(pl.col("phase") == "DEP"),
    "train ARR": tio.scan_training().filter(pl.col("phase") == "ARR"),
    "rank DEP": tio.scan_ranking().filter(pl.col("phase") == "DEP"),
    "rank ARR": tio.scan_ranking().filter(pl.col("phase") == "ARR"),
}
COLUMNS = list(tio.COLUMN_MAP.values())

# %% [markdown]
# ## Files on disk

# %%
rec.section("files")
files = sorted(paths.RAW.glob("*.parquet"))
rec.show(
    pl.DataFrame(
        {
            "file": [f.name for f in files],
            "mb": [round(f.stat().st_size / 1e6, 1) for f in files],
            "rows": [pl.scan_parquet(f).select(pl.len()).collect().item() for f in files],
        }
    ).sort("file")
)

# %% [markdown]
# ## Schema
#
# Raw column name, the canonical short name used everywhere downstream, and the dtype.

# %%
rec.section("schema")
raw_schema = pl.scan_parquet(paths.RANKING).collect_schema()
rec.show(
    pl.DataFrame(
        {
            "raw_name": list(tio.COLUMN_MAP.keys()),
            "canonical": list(tio.COLUMN_MAP.values()),
            "dtype": [str(raw_schema[c]) for c in tio.COLUMN_MAP],
            "source": ["movement" if c.endswith("_mvt") else "flight" for c in tio.COLUMN_MAP],
        }
    )
)
rec.say(f"\nsubmitting.parquet: {pl.scan_parquet(paths.SUBMITTING).collect_schema().names()}")

# %% [markdown]
# ## Counts

# %%
rec.section("row counts by subset")
counts = pl.DataFrame(
    {
        "subset": list(SUBSETS),
        "rows": [lf.select(pl.len()).collect(engine="streaming").item() for lf in SUBSETS.values()],
    }
)
rec.show(counts)

# %%
rec.section("rows per month")
per_month = (
    pl.concat(
        [
            lf.select(pl.lit(label).alias("subset"), "month", pl.col("phase").alias("phase"))
            .group_by("subset", "month", "phase")
            .agg(pl.len().alias("n"))
            for label, lf in SUBSETS.items()
        ]
    )
    .collect(engine="streaming")
    .sort("month", "subset")
)
rec.show(per_month.pivot(on="subset", index="month", values="n").fill_null(0).sort("month"))

# %%
monthly = (
    per_month.filter(pl.col("phase") == "DEP")
    .group_by("month")
    .agg(pl.col("n").sum())
    .sort("month")
)
fig, ax = plt.subplots(figsize=(9, 2.8))
labels = monthly["month"].to_list()
ax.bar(
    labels,
    monthly["n"].to_list(),
    color=[viz.ORANGE if m.startswith("2026") else viz.BLUE for m in labels],
    width=0.64,
)
ax.tick_params(axis="x", labelrotation=45)
viz.titled(
    ax,
    "Departures per month",
    "2025 training in blue, 2026 ranking in orange.",
    ylabel="departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "departures_per_month")

# %% [markdown]
# ## Missing data
#
# Null rate per column, for each subset.

# %%
rec.section("null rate per column, as a percentage of that subset's rows")
null_rates = pl.concat(
    [
        lf.select([pl.col(c).null_count().alias(c) for c in COLUMNS])
        .collect(engine="streaming")
        .with_columns(pl.lit(label).alias("subset"))
        for label, lf in SUBSETS.items()
    ]
)
sizes = dict(zip(counts["subset"], counts["rows"], strict=True))
null_pct = null_rates.with_columns(
    [
        (pl.col(c) / pl.col("subset").replace_strict(sizes, return_dtype=pl.Int64) * 100)
        .round(2)
        .alias(c)
        for c in COLUMNS
    ]
)
null_table = (
    null_pct.select("subset", *COLUMNS)
    .transpose(include_header=True, column_names="subset")
    .rename(lambda c: c if c == "column" else f"{c} %")
)
rec.say("  values are percentages of each subset's rows; denominators:")
for label, size in sizes.items():
    rec.say(f"    {label:<10} {size:>9,}")
rec.show(null_table)

# %%
matrix = null_pct.select(COLUMNS).to_numpy().T.astype(float)
fig, ax = plt.subplots(figsize=(7, 8.5))
viz.heatmap(ax, matrix, COLUMNS, list(SUBSETS), fmt="{:.1f}", cbar_label="null rate (%)", vmax=100)
viz.titled(ax, "Null rate per column", "Percent of rows, by split and phase.")
rec.figure(fig, "null_rates")

# %% [markdown]
# ## Cardinality

# %%
rec.section("distinct values per column")
CATEGORICAL = [
    "airport",
    "adep",
    "ades",
    "ades_filed",
    "runway",
    "stand",
    "ac_type",
    "ac_type_flt",
    "operator",
    "callsign",
    "flight_no",
    "wake_cat",
    "market_segment",
    "flight_type",
    "flight_rule",
    "flight_rule_flt",
]
cardinality = pl.concat(
    [
        lf.select([pl.col(c).n_unique().alias(c) for c in CATEGORICAL])
        .collect(engine="streaming")
        .with_columns(pl.lit(label).alias("subset"))
        for label, lf in SUBSETS.items()
    ]
).select("subset", *CATEGORICAL)
rec.show(cardinality.transpose(include_header=True, column_names="subset"))

# %%
rec.section("value counts for the low-cardinality columns (train DEP)")
for column in [
    "phase",
    "flight_rule",
    "flight_rule_flt",
    "flight_type",
    "wake_cat",
    "market_segment",
]:
    rec.show(
        SUBSETS["train DEP"]
        .group_by(column)
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
        .sort("n", descending=True),
        note=f"\n{column}:",
    )

# %% [markdown]
# ## Numeric and datetime ranges

# %%
rec.section("datetime column ranges")
DATETIMES = [
    "mvt_time",
    "block_time",
    "sched_time",
    "lobt",
    "iobt",
    "eobt_1",
    "arvt_1",
    "aobt_3",
    "arvt_3",
]
rec.show(
    pl.concat(
        [
            lf.select(
                [pl.lit(label).alias("subset")]
                + [pl.col(c).min().alias(f"{c}_min") for c in DATETIMES]
                + [pl.col(c).max().alias(f"{c}_max") for c in DATETIMES]
            ).collect(engine="streaming")
            for label, lf in SUBSETS.items()
        ]
    ).transpose(include_header=True, column_names="subset")
)

# %%
rec.section("target statistics (taxi_sec)")
rec.show(
    pl.concat(
        [
            lf.select(
                pl.lit(label).alias("subset"),
                pl.len().alias("n"),
                pl.col("taxi_sec").null_count().alias("nulls"),
                pl.col("taxi_sec").mean().round(1).alias("mean"),
                pl.col("taxi_sec").std().round(1).alias("std"),
                pl.col("taxi_sec").min().alias("min"),
                pl.col("taxi_sec").quantile(0.01).alias("p01"),
                pl.col("taxi_sec").quantile(0.25).alias("p25"),
                pl.col("taxi_sec").median().alias("p50"),
                pl.col("taxi_sec").quantile(0.75).alias("p75"),
                pl.col("taxi_sec").quantile(0.99).alias("p99"),
                pl.col("taxi_sec").max().alias("max"),
            ).collect(engine="streaming")
            for label, lf in SUBSETS.items()
        ]
    )
)

# %% [markdown]
# ## Key checks

# %%
rec.section("identifier checks")
for label, lf in SUBSETS.items():
    frame = lf.select("mvt_id", "flight_id").collect(engine="streaming")
    rec.say(
        f"  {label:<10} rows={frame.height:>9,}  unique mvt_id={frame['mvt_id'].n_unique():>9,}"
        f"  null flight_id={frame['flight_id'].null_count():>7,}"
    )

train_ids = set(
    tio.scan_training().select("mvt_id").collect(engine="streaming")["mvt_id"].to_list()
)
rank_ids = set(tio.scan_ranking().select("mvt_id").collect(engine="streaming")["mvt_id"].to_list())
submit = tio.scan_submitting().collect()
rec.say(f"\n  mvt_id overlap train/ranking      : {len(train_ids & rank_ids):,}")
rec.say(f"  submitting rows                   : {submit.height:,}")
ranking_dep_ids = set(
    tio.scan_ranking().filter(pl.col("phase") == "DEP").select("mvt_id").collect()["mvt_id"]
)
rec.say(f"  submitting ids == ranking DEP ids : {set(submit['mvt_id']) == ranking_dep_ids}")

# %%
rec.done()
