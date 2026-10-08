# %% [markdown]
# # a009 — The Network Manager flight match
#
# About 1% of movements have no matched Network Manager flight record, and all their `_flt`
# columns are null together. This asks whether that condition carries signal about taxi-out.
#
# Knobs: `EXTREME_ABOVE` sets what counts as an extreme row; `MIN_GROUP` is the support floor
# for the offset baseline.

# %%
import numpy as np
import polars as pl

from taxiout import cv, viz
from taxiout import io as tio
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder("a009_flight_match", "Does a missing NM flight match predict taxi-out?")

EXTREME_ABOVE = 43_200
MIN_GROUP = 30

frame = (
    pl.concat([tio.scan_training(), tio.scan_ranking()])
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "airport", "stand", "runway", "month", "split", "taxi_sec", "flight_id")
    .with_columns(pl.col("flight_id").is_null().alias("no_match"))
    .collect(engine="streaming")
)
train = frame.filter(pl.col("split") == "train")
rank = frame.filter(pl.col("split") == "rank")

# %% [markdown]
# ## Do the extreme rows share this condition?

# %%
rec.section("the extreme rows against the base rate")
extreme = train.filter(pl.col("taxi_sec") > EXTREME_ABOVE)
rec.say(f"  extreme rows (> {EXTREME_ABOVE:,}s)     : {extreme.height}")
rec.say(f"  of those, no NM flight match     : {extreme['no_match'].sum()}")
rec.say(f"  base rate across all departures  : {train['no_match'].mean() * 100:.3f}%")

# %% [markdown]
# ## What does the condition look like at each airport?

# %%
rec.section("taxi-out by airport and match status (2025)")
profile = (
    train.group_by("airport", "no_match")
    .agg(
        pl.len().alias("n"),
        pl.col("taxi_sec").mean().round(0).alias("mean"),
        pl.col("taxi_sec").median().alias("median"),
        (pl.col("taxi_sec") > 3600).mean().mul(100).round(2).alias("pct_gt_1h"),
        (pl.col("taxi_sec") > 7200).mean().mul(100).round(2).alias("pct_gt_2h"),
        (pl.col("taxi_sec") > EXTREME_ABOVE).sum().alias("n_extreme"),
    )
    .sort("airport", "no_match")
)
rec.show(profile)

# %%
wide = profile.pivot(on="no_match", index="airport", values="mean").rename(
    {"false": "matched", "true": "unmatched"}
)
wide = wide.with_columns((pl.col("unmatched") - pl.col("matched")).alias("gap")).sort("gap")
fig, ax = plt.subplots(figsize=(8, 3.4))
positions = range(wide.height)
ax.barh(
    [p + 0.2 for p in positions], wide["matched"].to_list(), 0.38, label="NM match", color=viz.BLUE
)
ax.barh(
    [p - 0.2 for p in positions],
    wide["unmatched"].to_list(),
    0.38,
    label="no NM match",
    color=viz.ORANGE,
)
ax.set_yticks(list(positions), labels=wide["airport"].to_list())
ax.legend(loc="lower right")
viz.titled(
    ax,
    "Mean taxi-out by Network Manager match status",
    "2025 departures, by airport.",
    xlabel="mean taxi-out (s)",
)
viz.only_x_grid(ax)
rec.figure(fig, "match_status_by_airport")

# %% [markdown]
# ## Is the rate stable into 2026?

# %%
rec.section("no-match rate, 2025 training vs 2026 ranking, January and July only")
seasonal = frame.filter(pl.col("month").str.slice(5, 2).is_in(["01", "07"]))
rec.show(
    seasonal.group_by("airport", "split")
    .agg(pl.len().alias("n"), pl.col("no_match").mean().mul(100).round(3).alias("pct_no_match"))
    .pivot(on="split", index="airport", values=["n", "pct_no_match"])
    .sort("airport")
)
rec.say(f"\n  expected unmatched rows in the ranking set: {rank['no_match'].sum():,}")

# %% [markdown]
# ## Is it worth anything out of sample?
#
# Take the structural baseline `b02` uses — the airport × stand × runway mean, backing off —
# and add one additive offset per (airport, no_match), learned on the fold's training months.
# Scored with the competition metric on the held-out month.

# %%
rec.section("held-out gain from a single (airport x no_match) offset")


def group_mean(data: pl.DataFrame, keys: list[str], value: str, min_rows: int) -> pl.DataFrame:
    return (
        data.group_by(keys)
        .agg(pl.col(value).mean().alias("_v"), pl.len().alias("_n"))
        .filter(pl.col("_n") >= min_rows)
        .drop("_n")
    )


def structural(fit: pl.DataFrame, apply_to: pl.DataFrame) -> pl.Series:
    """airport x stand x runway mean, backing off to stand, then airport, then global."""
    out = apply_to.with_columns(pl.lit(None, dtype=pl.Float64).alias("pred"))
    for keys in (["airport", "stand", "runway"], ["airport", "stand"], ["airport"]):
        table = group_mean(fit, keys, "taxi_sec", MIN_GROUP).rename({"_v": "_lookup"})
        out = (
            out.join(table, on=keys, how="left", maintain_order="left")
            .with_columns(pl.col("pred").fill_null(pl.col("_lookup")))
            .drop("_lookup")
        )
    return out["pred"].fill_null(fit["taxi_sec"].mean())


results = []
for fold in cv.get_folds("seasonal"):
    fit = train.filter(pl.col("month").is_in(list(fold.train_months)))
    held = train.filter(pl.col("month").is_in(list(fold.valid_months)))

    base_fit = structural(fit, fit)
    base_held = structural(fit, held)

    offsets = (
        fit.with_columns((pl.col("taxi_sec") - base_fit).alias("resid"))
        .group_by("airport", "no_match")
        .agg(pl.col("resid").mean().alias("offset"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_GROUP)
        .drop("n")
    )
    adjusted = (
        held.with_columns(base_held.alias("base"))
        .join(offsets, on=["airport", "no_match"], how="left", maintain_order="left")
        .with_columns((pl.col("base") + pl.col("offset").fill_null(0.0)).alias("adjusted"))
    )
    y = held["taxi_sec"].to_numpy().astype(float)
    keep = y <= EXTREME_ABOVE
    for label, pred in [
        ("structural", adjusted["base"].to_numpy()),
        ("+ match offset", adjusted["adjusted"].to_numpy()),
    ]:
        results.append(
            {
                "fold": fold.name,
                "model": label,
                "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 1),
                "rmse_trimmed": round(float(np.sqrt(np.mean((y[keep] - pred[keep]) ** 2))), 1),
            }
        )
rec.show(pl.DataFrame(results))

# %% [markdown]
# ## Is the gain bigger than the noise?
#
# A paired day-blocked bootstrap on the difference in squared error. Blocking by day because
# taxi-out is heavily autocorrelated within an airport-day, so a row bootstrap would understate
# the interval.

# %%
rec.section("paired day-block bootstrap on the RMSE gain (2000 resamples)")
rng = np.random.default_rng(0)
for fold in cv.get_folds("seasonal"):
    fit = train.filter(pl.col("month").is_in(list(fold.train_months)))
    held = train.filter(pl.col("month").is_in(list(fold.valid_months)))
    base_fit, base_held = structural(fit, fit), structural(fit, held)
    offsets = (
        fit.with_columns((pl.col("taxi_sec") - base_fit).alias("resid"))
        .group_by("airport", "no_match")
        .agg(pl.col("resid").mean().alias("offset"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_GROUP)
        .drop("n")
    )
    scored = (
        held.with_columns(base_held.alias("base"))
        .join(offsets, on=["airport", "no_match"], how="left", maintain_order="left")
        .with_columns(
            (pl.col("base") + pl.col("offset").fill_null(0.0)).alias("adjusted"),
            pl.col("month").alias("block"),
        )
    )
    y = scored["taxi_sec"].to_numpy().astype(float)
    e0 = (y - scored["base"].to_numpy()) ** 2
    e1 = (y - scored["adjusted"].to_numpy()) ** 2
    # block by day within the fold month
    days = held["mvt_id"].to_numpy() * 0  # placeholder replaced below
    day_index = (
        tio.scan_training()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", pl.col("mvt_time").dt.ordinal_day().alias("doy"))
        .collect(engine="streaming")
    )
    days = scored.select("mvt_id").join(day_index, on="mvt_id", how="left")["doy"].to_numpy()
    unique_days = np.unique(days)
    buckets = {d: np.flatnonzero(days == d) for d in unique_days}
    gains = []
    for _ in range(2000):
        picked = rng.choice(unique_days, size=unique_days.size, replace=True)
        idx = np.concatenate([buckets[d] for d in picked])
        gains.append(np.sqrt(e0[idx].mean()) - np.sqrt(e1[idx].mean()))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    rec.say(
        f"  {fold.name}: gain {np.sqrt(e0.mean()) - np.sqrt(e1.mean()):+.1f}s   "
        f"95% CI [{lo:+.1f}, {hi:+.1f}]   ({unique_days.size} day blocks)"
    )

# %%
rec.done()
