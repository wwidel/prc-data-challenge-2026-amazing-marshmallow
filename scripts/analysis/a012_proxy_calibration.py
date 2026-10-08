# %% [markdown]
# # a012 — Calibrated proxy comparison, and the skew question
#
# Two checks on claims made elsewhere in this repo.
#
# **1. Raw RMSE is the wrong yardstick for a proxy.** `mvt_time − eobt_1` carries a large fixed
# offset, and raw RMSE punishes that. A model fits an intercept and a slope on its first split,
# so the fair comparison gives every proxy the same courtesy.
#
# **2. Does subtracting a group baseline really make the residual more skewed?** Raw skewness is
# `E[(r−r̄)³]/σ³`, so shrinking σ raises it mechanically. Skewness is nonetheless
# scale-*invariant* — a pure rescaling cannot change it — so the question is whether the *shape*
# changed. Trimming to central quantiles answers that directly.

# %%
import numpy as np
import polars as pl

from taxiout import cv, viz
from taxiout import io as tio
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder(
    "a012_proxy_calibration",
    "Which off-block proxy is best once calibrated, and is the residual genuinely more skewed?",
)

TRIM_ABOVE = 43_200
PROXIES = {
    "aobt_3": "aobt_3",
    "eobt_1": "eobt_1",
    "iobt": "iobt",
    "lobt": "lobt",
    "sched_time": "sched_time",
}

frame = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id", "airport", "stand", "runway", "month", "taxi_sec", "mvt_time", *PROXIES.values()
    )
    .filter(pl.col("taxi_sec") <= TRIM_ABOVE)
    .collect(engine="streaming")
)
y = frame["taxi_sec"].to_numpy().astype(float)
rec.say(f"rows: {frame.height:,}   sd(target) = {y.std():.1f}")

# %% [markdown]
# ## Raw versus calibrated

# %%
rec.section("each proxy: raw, then with a fitted intercept and slope")
rows = []
for label, column in PROXIES.items():
    gap = (frame["mvt_time"] - frame[column]).dt.total_seconds().to_numpy().astype(float)
    ok = ~np.isnan(gap)
    g, t = gap[ok], y[ok]
    slope, intercept = np.polyfit(g, t, 1)
    rows.append(
        {
            "proxy": label,
            "n": int(ok.sum()),
            "raw_rmse": round(float(np.sqrt(np.mean((g - t) ** 2))), 1),
            "bias": round(float(np.mean(g - t)), 1),
            "corr": round(float(np.corrcoef(g, t)[0, 1]), 3),
            "intercept": round(float(intercept), 1),
            "slope": round(float(slope), 3),
            "calibrated_rmse": round(float(np.sqrt(np.mean((slope * g + intercept - t) ** 2))), 1),
        }
    )
rec.show(pl.DataFrame(rows).sort("calibrated_rmse"))
rec.say(
    "  A model fits intercept and slope for free, so calibrated_rmse is the comparison that"
    "\n  matters when deciding which proxy to build features from."
)

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
table = pl.DataFrame(rows).sort("calibrated_rmse", descending=True)
labels = table["proxy"].to_list()
positions = range(len(labels))
ax.barh(
    [p + 0.2 for p in positions],
    table["raw_rmse"].to_list(),
    0.38,
    label="raw",
    color=viz.MUTED_FILL,
)
ax.barh(
    [p - 0.2 for p in positions],
    table["calibrated_rmse"].to_list(),
    0.38,
    label="calibrated",
    color=viz.BLUE,
)
ax.set_yticks(list(positions), labels=labels)
ax.set_xscale("log")
ax.legend(loc="lower right")
viz.titled(
    ax,
    "Off-block proxies, raw against calibrated",
    "RMSE as a prediction of taxi-out. Calibrated fits an intercept and slope. Log scale.",
    xlabel="RMSE (s, log scale)",
)
viz.only_x_grid(ax)
rec.figure(fig, "proxy_calibrated")

# %% [markdown]
# ## Out of sample, and on top of the structural baseline
#
# A single global line is generous to a proxy with a stable offset. This repeats the comparison
# per fold, and then asks what each proxy adds *after* the airport × stand × runway mean, which
# is what a model would actually have left to explain.

# %%
rec.section("held-out: each proxy calibrated on train months, scored on the held-out month")
rows = []
for fold in cv.get_folds("seasonal"):
    fit = frame.filter(pl.col("month").is_in(list(fold.train_months)))
    held = frame.filter(pl.col("month").is_in(list(fold.valid_months)))
    base = (
        fit.group_by("airport", "stand", "runway")
        .agg(pl.col("taxi_sec").mean().alias("base"), pl.len().alias("n"))
        .filter(pl.col("n") >= 30)
        .drop("n")
    )
    fit_b = fit.join(base, on=["airport", "stand", "runway"], how="left").with_columns(
        pl.col("base").fill_null(fit["taxi_sec"].mean())
    )
    held_b = held.join(base, on=["airport", "stand", "runway"], how="left").with_columns(
        pl.col("base").fill_null(fit["taxi_sec"].mean())
    )
    for label, column in PROXIES.items():
        gf = (fit_b["mvt_time"] - fit_b[column]).dt.total_seconds().to_numpy().astype(float)
        gh = (held_b["mvt_time"] - held_b[column]).dt.total_seconds().to_numpy().astype(float)
        yf, yh = (
            fit_b["taxi_sec"].to_numpy().astype(float),
            held_b["taxi_sec"].to_numpy().astype(float),
        )
        bf, bh = fit_b["base"].to_numpy().astype(float), held_b["base"].to_numpy().astype(float)
        ok_f, ok_h = ~np.isnan(gf), ~np.isnan(gh)

        slope, intercept = np.polyfit(gf[ok_f], yf[ok_f], 1)
        pred = np.where(ok_h, slope * np.nan_to_num(gh) + intercept, bh)

        # and: proxy on top of the structural baseline, as a 2-variable least squares
        design_f = np.column_stack([bf[ok_f], gf[ok_f], np.ones(ok_f.sum())])
        coef, *_ = np.linalg.lstsq(design_f, yf[ok_f], rcond=None)
        pred2 = np.where(ok_h, coef[0] * bh + coef[1] * np.nan_to_num(gh) + coef[2], bh)

        rows.append(
            {
                "fold": fold.name,
                "proxy": label,
                "calibrated_alone": round(float(np.sqrt(np.mean((pred - yh) ** 2))), 1),
                "with_structural_base": round(float(np.sqrt(np.mean((pred2 - yh) ** 2))), 1),
            }
        )
rec.show(pl.DataFrame(rows).sort("fold", "with_structural_base"))

# %% [markdown]
# ## The skew question
#
# Residual after each group baseline, then the same residual trimmed to central quantiles. If
# the rise in raw skew were purely the shrinking σ, the central-quantile skew would not move.

# %%
rec.section("skewness of the residual, full and trimmed to central quantiles")
KEYS = {
    "global": [],
    "airport": ["airport"],
    "airport x stand": ["airport", "stand"],
    "airport x stand x runway": ["airport", "stand", "runway"],
}


def skew(values: np.ndarray) -> float:
    centred = values - values.mean()
    return float((centred**3).mean() / values.std() ** 3)


rows = []
for fold in cv.get_folds("seasonal"):
    fit = frame.filter(pl.col("month").is_in(list(fold.train_months)))
    held = frame.filter(pl.col("month").is_in(list(fold.valid_months)))
    global_median = fit["taxi_sec"].median()
    for label, keys in KEYS.items():
        if not keys:
            residual = held["taxi_sec"].to_numpy().astype(float) - global_median
        else:
            table = fit.group_by(keys).agg(pl.col("taxi_sec").median().alias("b"))
            joined = held.join(table, on=keys, how="left").with_columns(
                pl.col("b").fill_null(global_median)
            )
            residual = (joined["taxi_sec"] - joined["b"]).to_numpy().astype(float)
        entry = {
            "fold": fold.name,
            "baseline": label,
            "std": round(residual.std(), 1),
            "skew_full": round(skew(residual), 2),
        }
        for q in (0.999, 0.99, 0.95):
            lo, hi = np.quantile(residual, [(1 - q) / 2, 1 - (1 - q) / 2])
            entry[f"skew_c{int(q * 1000) / 10:g}"] = round(
                skew(residual[(residual >= lo) & (residual <= hi)]), 2
            )
        # how much of the raw skew change is the denominator alone?
        entry["m3"] = float(((residual - residual.mean()) ** 3).mean())
        rows.append(entry)
skew_table = pl.DataFrame(rows)
rec.show(skew_table.drop("m3"))

# %%
rec.section("decomposing the raw skew rise: denominator versus third moment")
for fold_name in ["2025-01", "2025-07"]:
    part = skew_table.filter(pl.col("fold") == fold_name)
    base = part.filter(pl.col("baseline") == "global").row(0, named=True)
    fine = part.filter(pl.col("baseline") == "airport x stand x runway").row(0, named=True)
    predicted = base["m3"] / fine["std"] ** 3
    share = (predicted - base["skew_full"]) / (fine["skew_full"] - base["skew_full"]) * 100
    rec.say(
        f"  {fold_name}: skew {base['skew_full']:.2f} -> {fine['skew_full']:.2f}."
        f"  Holding the third moment fixed predicts {predicted:.2f}"
        f"  (share of the rise: {share:.0f}%)."
        f"  Third moment ratio {fine['m3'] / base['m3']:.3f}."
    )

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, fold_name in zip(axes, ["2025-01", "2025-07"], strict=True):
    part = skew_table.filter(pl.col("fold") == fold_name)
    labels = part["baseline"].to_list()
    for column, colour, name in [
        ("skew_full", viz.MUTED_FILL, "all rows"),
        ("skew_c99.9", viz.BLUE, "central 99.9%"),
        ("skew_c95", viz.ORANGE, "central 95%"),
    ]:
        ax.plot(labels, part[column].to_list(), marker="o", color=colour, label=name)
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.legend(loc="center left")
    viz.titled(
        ax, f"Fold {fold_name}", "Skewness of the residual after each baseline.", ylabel="skewness"
    )
    viz.only_y_grid(ax)
fig.tight_layout()
rec.figure(fig, "skew_by_baseline")

# %%
rec.done()
