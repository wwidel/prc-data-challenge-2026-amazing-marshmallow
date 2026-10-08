# %% [markdown]
# # a014 — A full profile of the extreme outliers
#
# Everything needed to decide how to treat them: how many there are at each threshold, where
# and when, how large, whether anything visible distinguishes them, and — the question that
# decides the treatment — **whether an outlier means the whole day is bad, or just that row**.
#
# Knobs: `THRESHOLDS` is the ladder examined throughout; `HEADLINE` is the one used wherever a
# single number is needed.

# %%

import numpy as np
import polars as pl

from taxiout import io as tio
from taxiout import viz
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder(
    "a014_outlier_profile",
    "What exactly are the extreme outliers, and are they isolated rows or bad days?",
)

THRESHOLDS = [3_600, 7_200, 14_400, 21_600, 43_200, 86_400]
HEADLINE = 43_200

dep = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        "stand",
        "runway",
        "ades",
        "month",
        "mvt_time",
        "block_time",
        "sched_time",
        "aobt_3",
        "taxi_sec",
        "flight_id",
        "callsign",
        "ac_type",
        "market_segment",
    )
    .with_columns(
        pl.col("mvt_time").dt.date().alias("date"),
        pl.col("mvt_time").dt.hour().alias("hour"),
        pl.col("flight_id").is_null().alias("no_match"),
    )
    .collect(engine="streaming")
)
y = dep["taxi_sec"].to_numpy().astype(float)
rec.say(f"2025 departures: {dep.height:,}   median taxi-out {np.median(y):.0f}s")

# %% [markdown]
# ## 1. How many, at what threshold
#
# Twelve hours is an arbitrary line. This is the whole ladder, with what each row costs the
# metric — the share of squared error against a constant-mean predictor, and the RMSE that
# remains once rows above the threshold are removed.

# %%
rec.section("threshold ladder")
squared = (y - y.mean()) ** 2
ladder = []
for t in THRESHOLDS:
    over = y > t
    ladder.append(
        {
            "threshold_s": t,
            "threshold_h": round(t / 3600, 1),
            "n_rows": int(over.sum()),
            "pct_of_rows": round(float(over.mean() * 100), 5),
            "pct_of_sq_error": round(float(squared[over].sum() / squared.sum() * 100), 2),
            "rmse_if_removed": round(float(np.sqrt(squared[~over].mean())), 1),
            "n_airports": int(dep.filter(pl.Series(over))["airport"].n_unique()),
            "n_days": int(dep.filter(pl.Series(over))["date"].n_unique()),
        }
    )
rec.show(pl.DataFrame(ladder))
rec.say(f"  RMSE with everything: {np.sqrt(squared.mean()):.1f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
frame = pl.DataFrame(ladder)
axes[0].plot(frame["threshold_h"], frame["pct_of_sq_error"], marker="o", color=viz.BLUE)
axes[0].set_xscale("log")
viz.titled(
    axes[0],
    "Share of squared error above each threshold",
    "Against a constant-mean predictor, 2025 departures.",
    xlabel="threshold (hours, log scale)",
    ylabel="% of squared error",
)
viz.only_y_grid(axes[0])
axes[1].plot(frame["threshold_h"], frame["n_rows"], marker="o", color=viz.ORANGE)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
viz.titled(
    axes[1],
    "Rows above each threshold",
    "Note both axes are logarithmic.",
    xlabel="threshold (hours, log scale)",
    ylabel="rows",
)
viz.only_y_grid(axes[1])
fig.tight_layout()
rec.figure(fig, "threshold_ladder")

# %% [markdown]
# ## 2. Where they are

# %%
rec.section("rows above each threshold, by airport")
per_airport = dep.group_by("airport").agg(
    pl.len().alias("n_dep"),
    *[(pl.col("taxi_sec") > t).sum().alias(f"gt_{t // 3600}h") for t in THRESHOLDS],
)
rec.show(
    per_airport.with_columns(
        (pl.col(f"gt_{HEADLINE // 3600}h") / pl.col("n_dep") * 1e6)
        .round(1)
        .alias("per_million_12h")
    ).sort("per_million_12h", descending=True)
)

# %% [markdown]
# ## 3. The roster
#
# Every departure above the headline threshold, with everything that might identify it.

# %%
rec.section(f"every departure over {HEADLINE:,}s")
extreme = dep.filter(pl.col("taxi_sec") > HEADLINE).sort("taxi_sec", descending=True)
rec.show(
    extreme.select(
        "airport",
        "date",
        "hour",
        "taxi_sec",
        (pl.col("taxi_sec") / 3600).round(2).alias("hours"),
        (pl.col("taxi_sec") - 86_400).alias("minus_1day"),
        "stand",
        "runway",
        "ades",
        "no_match",
        pl.col("callsign").str.slice(0, 3).alias("airline"),
    )
)

# %% [markdown]
# ## 4. Is the whole day bad, or just the row?
#
# **This is the question that decides the treatment.** For every airport-day containing an
# extreme row: how many departures that day, how many were extreme, what the day's mean looked
# like against that airport's own year, and how the rest of the day behaved once the extreme
# rows are set aside.

# %%
rec.section("the days that contain an extreme row")
daily = dep.group_by("airport", "date").agg(
    pl.len().alias("n_day"),
    pl.col("taxi_sec").mean().alias("day_mean"),
    (pl.col("taxi_sec") > HEADLINE).sum().alias("n_extreme"),
    pl.col("taxi_sec").filter(pl.col("taxi_sec") <= HEADLINE).mean().alias("day_mean_clean"),
    pl.col("taxi_sec").filter(pl.col("taxi_sec") <= HEADLINE).quantile(0.95).alias("day_p95_clean"),
)
norms = daily.group_by("airport").agg(
    pl.col("day_mean").mean().alias("_m"),
    pl.col("day_mean").std().alias("_s"),
    pl.col("day_mean_clean").mean().alias("_mc"),
    pl.col("day_mean_clean").std().alias("_sc"),
)
affected = (
    daily.filter(pl.col("n_extreme") > 0)
    .join(norms, on="airport")
    .with_columns(
        ((pl.col("day_mean") - pl.col("_m")) / pl.col("_s")).round(2).alias("z_raw"),
        ((pl.col("day_mean_clean") - pl.col("_mc")) / pl.col("_sc")).round(2).alias("z_clean"),
        (pl.col("n_extreme") / pl.col("n_day") * 100).round(3).alias("pct_of_day"),
    )
    .sort("z_clean", descending=True)
)
rec.show(
    affected.select(
        "airport",
        "date",
        "n_day",
        "n_extreme",
        "pct_of_day",
        pl.col("day_mean").round(0),
        pl.col("day_mean_clean").round(0),
        pl.col("day_p95_clean").round(0),
        "z_raw",
        "z_clean",
    )
)
rec.say(
    "\n  z_raw is the day's standing including the extreme rows; z_clean excludes them."
    "\n  If z_clean is unremarkable, the day was normal and the rows are isolated defects."
)
rec.say(
    f"  days with an extreme row that are still above z_clean = 2: "
    f"{affected.filter(pl.col('z_clean') > 2).height} of {affected.height}"
)

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.scatter(affected["z_raw"], affected["z_clean"], s=40, color=viz.BLUE, alpha=0.7, linewidths=0)
lim = [min(affected["z_clean"].min(), -1) - 0.5, affected["z_raw"].max() + 0.5]
ax.plot(lim, lim, color=viz.GRID, linewidth=1.2, zorder=1)
ax.axhline(2, color=viz.ORANGE, linewidth=1.2)
viz.annotate(ax, lim[1], 2, " day genuinely disrupted", dx=-140, dy=5, color=viz.ORANGE)
for row in affected.filter(pl.col("z_clean") > 2).iter_rows(named=True):
    viz.annotate(ax, row["z_raw"], row["z_clean"], f" {row['airport']} {row['date']}", dx=4, dy=2)
viz.titled(
    ax,
    "Does an extreme row mean a bad day?",
    "Each dot is an airport-day containing at least one extreme row. Grey line is equality.",
    xlabel="z of the day's mean, including extreme rows",
    ylabel="z with the extreme rows removed",
)
rec.figure(fig, "day_vs_row")

# %% [markdown]
# ## 5. Are they clustered in time within the day?

# %%
rec.section("spacing of extreme rows on days that have more than one")
multi = affected.filter(pl.col("n_extreme") > 1).select("airport", "date")
if multi.height:
    spans = []
    for row in multi.iter_rows(named=True):
        same = extreme.filter(
            (pl.col("airport") == row["airport"]) & (pl.col("date") == row["date"])
        ).sort("mvt_time")
        times = same["mvt_time"].to_list()
        spans.append(
            {
                "airport": row["airport"],
                "date": row["date"],
                "n": same.height,
                "first": times[0].strftime("%H:%M"),
                "last": times[-1].strftime("%H:%M"),
                "span_h": round((times[-1] - times[0]).total_seconds() / 3600, 1),
            }
        )
    rec.show(pl.DataFrame(spans))
rec.show(
    extreme.group_by("hour").agg(pl.len().alias("n")).sort("hour"),
    "\nhour of takeoff (UTC) across all extreme rows:",
)

# %% [markdown]
# ## 6. What distinguishes them, ex ante
#
# Anything a model could see *before* knowing the target.

# %%
rec.section("signals available at prediction time, by threshold")
signals = []
for t in THRESHOLDS + [0]:
    subset = dep.filter(pl.col("taxi_sec") > t) if t else dep
    signals.append(
        {
            "group": f"> {t // 3600}h" if t else "all departures",
            "n": subset.height,
            "no_nm_match_%": round(subset["no_match"].mean() * 100, 2),
            "aobt_3_null_%": round(subset["aobt_3"].null_count() / subset.height * 100, 2),
            "lirf_%": round((subset["airport"] == "LIRF").mean() * 100, 1),
            "remote_stand_%": round(
                subset.select(
                    pl.col("stand").str.starts_with("6")
                    | pl.col("stand").str.starts_with("7")
                    | pl.col("stand").str.starts_with("8")
                )
                .to_series()
                .mean()
                * 100,
                1,
            ),
        }
    )
rec.show(pl.DataFrame(signals))

# %%
rec.section("do the same stands and runways recur?")
rec.show(
    extreme.group_by("airport", "stand")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    .head(12),
    "stands:",
)
rec.show(
    extreme.group_by("airport", "runway")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    .head(8),
    "runways:",
)
rec.show(
    extreme.group_by(pl.col("callsign").str.slice(0, 3).alias("airline"))
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    .head(8),
    "airlines:",
)

# %% [markdown]
# ## 7. Concentration in the evaluation months

# %%
rec.section("January and July against the rest of the year")
rows = []
for t in THRESHOLDS:
    over = dep.filter(pl.col("taxi_sec") > t)
    in_jj = over.filter(pl.col("month").str.slice(5, 2).is_in(["01", "07"])).height
    dep_jj = dep.filter(pl.col("month").str.slice(5, 2).is_in(["01", "07"])).height
    rest_n, rest_d = over.height - in_jj, dep.height - dep_jj
    rows.append(
        {
            "threshold_h": t // 3600,
            "n_total": over.height,
            "n_jan_jul": in_jj,
            "pct_in_jan_jul": round(in_jj / over.height * 100, 1) if over.height else None,
            "rate_jan_jul_ppm": round(in_jj / dep_jj * 1e6, 2),
            "rate_rest_ppm": round(rest_n / rest_d * 1e6, 2),
            "concentration": round((in_jj / dep_jj) / (rest_n / rest_d), 2) if rest_n else None,
        }
    )
rec.show(pl.DataFrame(rows))

# %% [markdown]
# ## 8. What trimming the training set would cost

# %%
rec.section("rows lost if training is trimmed at each threshold")
rec.show(
    pl.DataFrame(
        [
            {
                "threshold_h": t // 3600,
                "rows_dropped": int((y > t).sum()),
                "pct_dropped": round(float((y > t).mean() * 100), 4),
                "airports_touched": int(dep.filter(pl.col("taxi_sec") > t)["airport"].n_unique()),
                "days_touched": int(dep.filter(pl.col("taxi_sec") > t)["date"].n_unique()),
            }
            for t in THRESHOLDS
        ]
    )
)

# %%
rec.done()
