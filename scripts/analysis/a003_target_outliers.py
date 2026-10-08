# %% [markdown]
# # a003 — How much of the metric is corrupt records?
#
# RMSE squares errors, so large outliers carry weight out of proportion to their count. Cells
# below cover monthly spread, the concentration of squared error, the worst rows, and the target
# distribution. `TRIM` sets the threshold used throughout.

# %%
import numpy as np
import polars as pl

from taxiout import paths, spine, viz
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder("a003_target_outliers", "How concentrated is the squared error in bad rows?")

TRIM = 43_200  # 12 hours
sp = spine.load(paths.TRAIN_MONTHS)
target = sp["taxi_sec"].to_numpy().astype(np.float64)

# %% [markdown]
# ## Monthly spread

# %%
rec.section("monthly spread of the target")
monthly = (
    sp.group_by("month")
    .agg(
        pl.len().alias("n"),
        pl.col("taxi_sec").mean().round(1).alias("mean"),
        pl.col("taxi_sec").std().round(1).alias("std"),
        pl.col("taxi_sec").quantile(0.999).round(0).alias("p999"),
        (pl.col("taxi_sec") > 3600).mean().round(5).alias("frac_gt_1h"),
        (pl.col("taxi_sec") > TRIM).sum().alias("n_gt_12h"),
        pl.col("taxi_sec").max().alias("max"),
    )
    .sort("month")
)
rec.show(monthly)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
months = [m[5:] for m in monthly["month"].to_list()]
evaluation = {"01", "07"}

axes[0].bar(
    months,
    monthly["std"].to_list(),
    color=[viz.BLUE if m in evaluation else viz.MUTED_FILL for m in months],
    width=0.64,
)
viz.titled(
    axes[0],
    "Standard deviation of taxi-out, by month",
    "2025. The two evaluation months are highlighted.",
    ylabel="std (seconds)",
)
viz.only_y_grid(axes[0])

axes[1].bar(
    months,
    (monthly["frac_gt_1h"] * 100).to_list(),
    color=[viz.BLUE if m in evaluation else viz.MUTED_FILL for m in months],
    width=0.64,
)
viz.titled(
    axes[1],
    "Share of departures taking over an hour, by month",
    "2025. The two evaluation months are highlighted.",
    ylabel="% over one hour",
)
viz.only_y_grid(axes[1])
fig.tight_layout()
rec.figure(fig, "monthly_spread")

# %% [markdown]
# ## Concentration of squared error
#
# Departures sorted by squared error against a constant-mean prediction, accumulated worst
# first.

# %%
rec.section(f"squared-error concentration (constant-mean predictor, threshold {TRIM}s)")
squared = (target - target.mean()) ** 2
ordered = np.sort(squared)[::-1]
cumulative = np.cumsum(ordered) / squared.sum()
rank_share = np.arange(1, ordered.size + 1) / ordered.size

bad = target > TRIM
n_bad = int(bad.sum())
total, part = float(squared.sum()), float(squared[bad].sum())
rec.say(f"  rows over 12h            : {n_bad:,}  ({bad.mean() * 100:.4f}% of 2025)")
rec.say(f"  their share of sq. error : {part / total * 100:.1f}%")
rec.say(f"  RMSE including them      : {(total / sp.height) ** 0.5:.1f}")
rec.say(f"  RMSE excluding them      : {((total - part) / (sp.height - n_bad)) ** 0.5:.1f}")

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.plot(rank_share * 100, cumulative * 100, color=viz.BLUE, linewidth=2)
ax.plot([0, 100], [0, 100], color=viz.GRID, linewidth=1.2, zorder=1)
ax.set_xscale("log")
ax.set_xlim(1e-4, 100)
ax.set_ylim(0, 100)

x_bad = n_bad / ordered.size * 100
y_bad = cumulative[n_bad - 1] * 100
ax.scatter([x_bad], [y_bad], s=42, color=viz.ORANGE, zorder=3, linewidths=0)
viz.annotate(
    ax,
    x_bad,
    y_bad,
    f"the {n_bad} rows over 12 h\ncarry {y_bad:.0f}% of the error",
    dx=8,
    dy=-6,
    color=viz.ORANGE,
)
viz.titled(
    ax,
    "Cumulative share of squared error, worst departures first",
    "Against a constant-mean prediction. Grey line is a uniform spread.",
    xlabel="share of departures, worst first (%, log scale)",
    ylabel="cumulative share of squared error (%)",
)
rec.figure(fig, "error_concentration")

# %% [markdown]
# ## The worst rows
#
# Testing one hypothesis about them: a one-day timestamp slip, i.e. whether subtracting
# 86,400 s lands them back in a plausible range.

# %%
rec.section("are the worst rows a one-day timestamp error?")
worst = sp.filter(pl.col("taxi_sec") > TRIM).with_columns(
    (pl.col("taxi_sec") - 86_400).alias("minus_1day")
)
rec.show(
    worst.select(
        pl.col("minus_1day").is_between(-60, 7200).mean().round(3).alias("frac_plausible"),
        pl.col("minus_1day").min().alias("min"),
        pl.col("minus_1day").max().alias("max"),
    )
)
rec.show(
    worst.group_by("airport").agg(pl.len().alias("n")).sort("n", descending=True), "by airport:"
)

# %% [markdown]
# ## The short end

# %%
rec.section("the short end")
rec.show(
    sp.select(
        (pl.col("taxi_sec") < 0).sum().alias("negative"),
        (pl.col("taxi_sec") < 60).sum().alias("under_60s"),
        (pl.col("taxi_sec") < 120).sum().alias("under_120s"),
    )
)

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
positive = target[target > 0]
ax.hist(positive, bins=np.logspace(1, np.log10(positive.max()), 90), color=viz.BLUE, linewidth=0)
ax.set_xscale("log")
for value, label in [(np.median(positive), "median"), (3600, "1 hour"), (TRIM, "12 hours")]:
    ax.axvline(value, color=viz.ORANGE, linewidth=1.4, zorder=3)
    viz.annotate(ax, value, ax.get_ylim()[1] * 0.55, f" {label}", dx=3, dy=0, color=viz.ORANGE)
viz.titled(
    ax,
    "Distribution of taxi-out times",
    "2025 departures, log scale, positive values only.",
    xlabel="taxi-out (seconds, log scale)",
    ylabel="departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "target_distribution")

# %%
