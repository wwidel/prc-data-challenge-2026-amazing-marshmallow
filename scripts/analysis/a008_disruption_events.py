# %% [markdown]
# # a008 — Extreme outliers and disruption events
#
# Two questions that turned out to be connected:
#
# 1. What are the most extreme taxi-out values, and are they real events or bad records?
# 2. Which days show a genuine airport-wide excursion, and what caused them?
#
# Knobs in the next cell. `EVENTS` is the list of externally-dated incidents to probe — add a
# date and re-run to test any hypothesis about a particular day.

# %%
import datetime as dt

import numpy as np
import polars as pl

from taxiout import io as tio
from taxiout import viz
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder(
    "a008_disruption_events",
    "Are the extreme values real events or bad records, and which days were genuinely disrupted?",
)

EXTREME_ABOVE = 43_200  # 12 hours
#: A day counts as part of a disruption episode while its z stays at or above this.
EPISODE_Z = 3.0
MIN_FLIGHTS_PER_DAY = 100

#: Externally sourced incident dates to test against the data. (airport, date, description)
EVENTS = [
    ("EDDM", "2025-10-02", "drone sightings, overnight closure"),
    ("EDDM", "2025-10-03", "drone sightings, morning after"),
    ("EDDM", "2025-10-04", "control: day after"),
    ("LTFM", "2025-02-19", "Siberian snowstorm, ~150 flights cancelled"),
    ("LTFM", "2025-02-23", "same snow week"),
    ("EDDM", "2025-12-24", "Christmas Eve"),
    ("LFPG", "2025-01-17", "winter weather"),
    ("LIRF", "2025-07-13", "unexplained"),
]

departures = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        "ades",
        "mvt_time",
        "block_time",
        "taxi_sec",
        "stand",
        "runway",
        "month",
    )
    .with_columns(pl.col("mvt_time").dt.date().alias("date"))
    .collect(engine="streaming")
)
rec.say(f"2025 departures: {departures.height:,}")

# %% [markdown]
# ## The extreme values
#
# If these were a disruption, they would cluster on a handful of days. If they were a
# timestamp bug, they would cluster on a value.

# %%
rec.section(f"departures with taxi-out over {EXTREME_ABOVE:,}s")
extreme = departures.filter(pl.col("taxi_sec") > EXTREME_ABOVE).sort("taxi_sec", descending=True)
rec.show(
    extreme.select(
        "airport",
        "mvt_time",
        "block_time",
        "taxi_sec",
        (pl.col("taxi_sec") / 3600).round(1).alias("hours"),
        "stand",
        "runway",
        "ades",
    )
)
rec.say(
    f"\n  rows: {extreme.height}   distinct dates: {extreme['date'].n_unique()}"
    f"   distinct airports: {extreme['airport'].n_unique()}"
)
rec.show(
    extreme.group_by("airport").agg(pl.len().alias("n")).sort("n", descending=True), "by airport:"
)

# %%
rec.section("test: is block_time recorded exactly one day early?")
offset = extreme.with_columns((pl.col("taxi_sec") - 86_400).alias("minus_1day"))
clean = offset.filter(pl.col("minus_1day").is_between(0, 3600))
rec.say(f"  rows landing in [86400, 90000]s : {clean.height} of {extreme.height}")
if clean.height:
    rec.say(
        f"  after removing exactly 24h        : min={clean['minus_1day'].min()}s "
        f"median={clean['minus_1day'].median():.0f}s max={clean['minus_1day'].max()}s"
    )
    rec.say(f"  a normal taxi-out is ~{departures['taxi_sec'].median():.0f}s")
rec.show(
    offset.filter(~pl.col("minus_1day").is_between(0, 3600))
    .select(
        "airport", "mvt_time", "block_time", (pl.col("taxi_sec") / 3600).round(1).alias("hours")
    )
    .sort("hours", descending=True),
    "\nthe rows the one-day hypothesis does NOT explain:",
)

# %% [markdown]
# ## Is this one airport's problem, at every scale?

# %%
rec.section("outlier rate by airport, departures")
by_airport = (
    departures.group_by("airport")
    .agg(
        pl.len().alias("n"),
        (pl.col("taxi_sec") > 3600).mean().mul(100).round(3).alias("pct_gt_1h"),
        (pl.col("taxi_sec") > 7200).mean().mul(100).round(4).alias("pct_gt_2h"),
        (pl.col("taxi_sec") > EXTREME_ABOVE).sum().alias("n_extreme"),
    )
    .sort("pct_gt_2h", descending=True)
)
rec.show(by_airport)

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
labels = by_airport["airport"].to_list()
ax.bar(
    labels,
    by_airport["pct_gt_2h"].to_list(),
    color=viz.bar_gap_colors(labels, {"LIRF"}),
    width=0.62,
)
viz.titled(
    ax,
    "Share of departures with taxi-out over two hours",
    "2025, by airport.",
    ylabel="% of departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "outlier_rate_by_airport")

# %% [markdown]
# ## Does the contamination reach 2026?
#
# Departure targets are blanked in the ranking set, but **arrivals are not**. If the same
# airport shows the same pathology in its 2026 arrival taxi times, its 2026 departures are
# very likely affected too — and nothing we model can fix that.

# %%
rec.section("arrival taxi-in outlier rate, 2025 training vs 2026 ranking")
arrivals = (
    pl.concat([tio.scan_training(), tio.scan_ranking()])
    .filter(pl.col("phase") == "ARR")
    .select("split", "airport", "month", "taxi_sec")
    .collect(engine="streaming")
)
rec.show(
    arrivals.group_by("airport", "split")
    .agg(
        pl.len().alias("n"),
        (pl.col("taxi_sec") > 7200).mean().mul(100).round(4).alias("pct_gt_2h"),
    )
    .pivot(on="split", index="airport", values=["n", "pct_gt_2h"])
    .sort("airport")
)

# %% [markdown]
# ## Which days were genuinely disrupted?
#
# Daily mean taxi-out per airport, standardised against that airport's own year. A real
# airport-wide event moves every departure, so it shows as a large z on the daily mean rather
# than as a few huge rows.

# %%
rec.section("most disrupted days, by z-score of the daily mean within each airport")
daily = (
    departures.group_by("airport", "date")
    .agg(pl.len().alias("n"), pl.col("taxi_sec").mean().alias("mean"))
    .filter(pl.col("n") >= MIN_FLIGHTS_PER_DAY)
)
scored = daily.join(
    daily.group_by("airport").agg(
        pl.col("mean").mean().alias("_m"), pl.col("mean").std().alias("_s")
    ),
    on="airport",
).with_columns(((pl.col("mean") - pl.col("_m")) / pl.col("_s")).round(2).alias("z"))
rec.show(
    scored.sort("z", descending=True)
    .head(25)
    .select("airport", "date", "n", pl.col("mean").round(0), "z")
)

# %%
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.scatter(
    scored["date"].to_list(),
    scored["z"].to_list(),
    s=7,
    color=viz.BLUE,
    alpha=0.45,
    linewidths=0,
)
flagged = scored.filter(pl.col("z") > 5)
ax.scatter(
    flagged["date"].to_list(),
    flagged["z"].to_list(),
    s=26,
    color=viz.ORANGE,
    linewidths=0,
    zorder=3,
)
for row in flagged.sort("z", descending=True).head(6).iter_rows(named=True):
    viz.annotate(
        ax, row["date"], row["z"], f" {row['airport']} {row['date']}", dx=4, dy=-2, color=viz.ORANGE
    )
ax.axhline(0, color=viz.GRID, linewidth=1.0)
viz.titled(
    ax,
    "Daily mean taxi-out, standardised within each airport",
    "Each dot is one airport-day in 2025. Orange: more than five standard deviations.",
    ylabel="z-score of daily mean",
)
viz.only_y_grid(ax)
rec.figure(fig, "disrupted_days")

# %% [markdown]
# ## Probing specific externally-dated incidents
#
# Edit `EVENTS` above to test any date.
#
# **On the statistic.** The obvious test — compare the day's mean to the airport's overall mean
# and divide by the standard error of the mean — assumes departures are independent within a
# day. They are badly not: weather, queueing and runway configuration persist for hours, and
# measuring the design effect at these airports gives 6x to 150x, so a per-row standard error
# overstates significance by roughly an order of magnitude.
#
# The day is the unit of analysis, so the null is the empirical distribution of *that airport's
# own daily means*, which the previous cell already computed. Both are shown below so the
# difference is visible.

# %%
rec.section("named events: per-row null (inflated) vs the day-level null (correct)")
probes = []
for airport, date, description in EVENTS:
    airport_rows = departures.filter(pl.col("airport") == airport)
    day = airport_rows.filter(pl.col("date").cast(pl.Utf8) == date)
    base = airport_rows.filter(pl.col("date").cast(pl.Utf8) != date)
    if day.height == 0:
        continue
    # the airport's own distribution of daily means, from the previous cell
    own_days = scored.filter(pl.col("airport") == airport)
    match = own_days.filter(pl.col("date").cast(pl.Utf8) == date)
    z_day = match["z"][0] if match.height else float("nan")
    rank = (own_days["mean"] > day["taxi_sec"].mean()).sum() + 1 if own_days.height else None
    probes.append(
        {
            "airport": airport,
            "date": date,
            "event": description,
            "n": day.height,
            "mean": round(day["taxi_sec"].mean(), 0),
            "baseline_mean": round(base["taxi_sec"].mean(), 0),
            "z_per_row": round(
                (day["taxi_sec"].mean() - base["taxi_sec"].mean())
                / base["taxi_sec"].std()
                * np.sqrt(day.height),
                1,
            ),
            "z_day_level": z_day,
            "rank_of_days": f"{rank}/{own_days.height}" if rank else "",
        }
    )
rec.show(pl.DataFrame(probes))
rec.say(
    "  z_per_row treats departures as independent within a day and is inflated by roughly an"
    "\n  order of magnitude. z_day_level and rank_of_days are the ones to read."
)

# %%
fig, ax = plt.subplots(figsize=(9, 3.4))
probe_frame = pl.DataFrame(probes).sort("z_day_level")
labels = [f"{r['airport']} {r['date']}" for r in probe_frame.iter_rows(named=True)]
ax.barh(labels, probe_frame["z_day_level"].to_list(), color=viz.BLUE, height=0.6)
ax.axvline(0, color=viz.INK_MUTED, linewidth=1.0)
for label, value in zip(labels, probe_frame["z_day_level"].to_list(), strict=True):
    ax.text(value + 0.15, label, f"{value:+.1f}", va="center", fontsize=8, color=viz.INK_2)
viz.titled(
    ax,
    "Named incidents against the airport's own distribution of daily means",
    "Standardised on the day as the unit, not the individual departure.",
    xlabel="z-score of the daily mean",
)
viz.only_x_grid(ax)
rec.figure(fig, "named_events")

# %% [markdown]
# ## Runway mix over time
#
# A runway closed for maintenance changes taxi distances across the airfield for weeks or
# months. It would show as a runway's share of departures collapsing and another's rising.

# %%
rec.section("runway share of departures, by airport and month")
mix = (
    departures.group_by("airport", "month", "runway")
    .agg(pl.len().alias("n"))
    .with_columns(
        (pl.col("n") / pl.col("n").sum().over("airport", "month")).round(3).alias("share")
    )
)
swing = (
    mix.group_by("airport", "runway")
    .agg(
        pl.col("share").min().alias("min_share"),
        pl.col("share").max().alias("max_share"),
        pl.col("n").sum().alias("total"),
    )
    .filter(pl.col("total") >= 2000)
    .with_columns((pl.col("max_share") - pl.col("min_share")).round(3).alias("swing"))
    .sort("swing", descending=True)
)
rec.show(swing.head(20), "runways whose monthly share moves most:")

# %%
top_swing = swing.head(6)
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for ax, row in zip(axes.ravel(), top_swing.iter_rows(named=True), strict=False):
    series = mix.filter(
        (pl.col("airport") == row["airport"]) & (pl.col("runway") == row["runway"])
    ).sort("month")
    ax.plot(
        [m[5:] for m in series["month"]],
        series["share"].to_list(),
        color=viz.BLUE,
        marker="o",
        markersize=4,
    )
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{row['airport']} runway {row['runway']}", fontsize=9.5, loc="left", color=viz.INK
    )
    viz.only_y_grid(ax)
fig.suptitle(
    "Monthly share of departures, for the runways whose usage moves most",
    fontsize=11,
    fontweight="bold",
    color=viz.INK,
    x=0.005,
    ha="left",
    y=1.0,
)
fig.tight_layout()
rec.figure(fig, "runway_mix_over_time")

# %% [markdown]
# ## How long does a disruption last?
#
# A z-score names a day; a feature or an exclusion window needs the whole episode. This walks
# outward from each flagged day while the airport stays above `EPISODE_Z`.

# %%
rec.section(f"episode extent around each flagged day (contiguous days above z = {EPISODE_Z})")
episodes = []
for airport, date, description in EVENTS:
    own = scored.filter(pl.col("airport") == airport).sort("date")
    if own.is_empty():
        continue
    dates = own["date"].to_list()
    zs = own["z"].to_list()
    try:
        centre = dates.index(dt.date.fromisoformat(date))
    except ValueError:
        continue
    if zs[centre] < EPISODE_Z:
        episodes.append(
            {
                "airport": airport,
                "seed": date,
                "event": description,
                "days": 0,
                "from": None,
                "to": None,
                "peak_z": zs[centre],
                "mean_z": zs[centre],
            }
        )
        continue
    lo = hi = centre
    while lo - 1 >= 0 and zs[lo - 1] >= EPISODE_Z:
        lo -= 1
    while hi + 1 < len(zs) and zs[hi + 1] >= EPISODE_Z:
        hi += 1
    window = zs[lo : hi + 1]
    episodes.append(
        {
            "airport": airport,
            "seed": date,
            "event": description,
            "days": hi - lo + 1,
            "from": dates[lo].isoformat(),
            "to": dates[hi].isoformat(),
            "peak_z": round(max(window), 2),
            "mean_z": round(sum(window) / len(window), 2),
        }
    )
rec.show(pl.DataFrame(episodes))

# %%
rec.section(f"every airport-day above z = {EPISODE_Z}, grouped into episodes")
flagged = scored.filter(pl.col("z") >= EPISODE_Z).sort("airport", "date")
runs = []
for airport in sorted(flagged["airport"].unique()):
    own = flagged.filter(pl.col("airport") == airport).sort("date")
    dates = own["date"].to_list()
    zs = own["z"].to_list()
    start = 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or (dates[i] - dates[i - 1]).days > 1:
            block = zs[start:i]
            runs.append(
                {
                    "airport": airport,
                    "from": dates[start].isoformat(),
                    "to": dates[i - 1].isoformat(),
                    "days": i - start,
                    "peak_z": round(max(block), 2),
                }
            )
            start = i
rec.show(pl.DataFrame(runs).sort("peak_z", descending=True))
rec.say(
    f"  {len(runs)} episodes over {sum(r['days'] for r in runs)} airport-days,"
    f" out of {scored.height:,} airport-days in 2025."
)

# %% [markdown]
# ## Does the 2026 configuration match 2025?
#
# `runway` is given in the ranking set, so this is directly checkable. A runway heavily used in
# 2026 but closed during the corresponding 2025 month has thin training support behind any
# stand-runway statistic.

# %%
rec.section("runway share, 2026 ranking vs the same month of 2025")
ranking = (
    tio.scan_ranking()
    .filter(pl.col("phase") == "DEP")
    .select("airport", "month", "runway")
    .collect(engine="streaming")
)
both = pl.concat(
    [
        departures.select("airport", "month", "runway").with_columns(pl.lit("2025").alias("year")),
        ranking.with_columns(pl.lit("2026").alias("year")),
    ]
).with_columns(pl.col("month").str.slice(5, 2).alias("mm"))

compared = (
    both.filter(pl.col("mm").is_in(["01", "07"]))
    .group_by("airport", "mm", "year", "runway")
    .agg(pl.len().alias("n"))
    .with_columns(
        (pl.col("n") / pl.col("n").sum().over("airport", "mm", "year")).round(3).alias("share")
    )
    .pivot(on="year", index=["airport", "mm", "runway"], values="share")
    .fill_null(0.0)
    .with_columns((pl.col("2026") - pl.col("2025")).round(3).alias("delta"))
    .filter((pl.col("2026") > 0.02) | (pl.col("2025") > 0.02))
    .sort("delta", descending=True)
)
rec.show(compared.head(20), "runways gaining the most share in 2026:")
rec.show(compared.tail(15), "runways losing the most share in 2026:")
rec.show(
    compared.filter((pl.col("2025") < 0.005) & (pl.col("2026") > 0.02)),
    "\nused in 2026 but essentially absent from the same 2025 month:",
)

# %%
rec.done()
