# %% [markdown]
# # a001 — Off-block proxies
#
# The ranking set blanks `block_time`, but keeps other estimates of when the aircraft left the
# stand: `aobt_3`, `eobt_1`, `lobt`, `iobt`, plus `sched_time`. Takeoff minus each of them is a
# candidate estimate of taxi-out.
#
# Run top to bottom, or step through cell by cell.

# %%
import polars as pl

from taxiout import io as tio
from taxiout import viz
from taxiout.analysis import Recorder

viz.style()
plt = viz.plt

rec = Recorder("a001_offblock_proxies", "Which off-block proxy best reconstructs taxi-out?")

PROXIES = ["aobt_3", "eobt_1", "lobt", "iobt", "sched_time"]
# Candidate default taxi times, as centre seconds. The bands are +/-30 s because aobt_3 is
# minute-rounded against a second-resolution takeoff.
CANDIDATES = [300, 420, 600, 720, 900, 1200, 1500, 1800]
DEFAULT_CENTRES = {"default_12min": 720, "default_20min": 1200, "default_25min": 1500}
DEFAULT_BANDS = {name: (c - 30, c + 30) for name, c in DEFAULT_CENTRES.items()}

train = tio.scan_training().filter(pl.col("phase") == "DEP")
rank = tio.scan_ranking().filter(pl.col("phase") == "DEP")

# %% [markdown]
# ## Timestamp resolution
#
# Share of each timestamp whose seconds component is zero.

# %%
rec.section("timestamp resolution: share of values landing exactly on a minute")
resolution = train.select(
    [
        (pl.col(c).dt.second() == 0).mean().round(4).alias(c)
        for c in ["mvt_time", "block_time", "sched_time", "aobt_3", "eobt_1", "lobt", "iobt"]
    ]
).collect()
rec.show(resolution)

# %% [markdown]
# ## Each proxy, used raw as the prediction
#
# No fitting at all: just `mvt_time - <estimate>` against the true taxi-out.

# %%
rec.section("each proxy used raw as the prediction (whole of 2025, in-sample)")
scored = pl.concat(
    [
        train.with_columns((pl.col("mvt_time") - pl.col(source)).dt.total_seconds().alias("px"))
        .filter(pl.col("px").is_not_null())
        .select(
            pl.lit(source).alias("proxy"),
            pl.len().alias("n"),
            pl.corr("px", "taxi_sec").round(3).alias("corr"),
            ((pl.col("px") - pl.col("taxi_sec")) ** 2).mean().sqrt().round(1).alias("rmse"),
        )
        .collect(engine="streaming")
        for source in PROXIES
    ]
).sort("rmse")
rec.show(scored)

# %%
fig, ax = plt.subplots(figsize=(7, 2.8))
order = scored.sort("rmse", descending=True)
labels, values = order["proxy"].to_list(), order["rmse"].to_list()
colors = viz.bar_gap_colors(labels, highlight={"aobt_3"})
ax.barh(labels, values, color=colors, height=0.62)
for label, value in zip(labels, values, strict=True):
    ax.text(value + 30, label, f"{value:,.0f}", va="center", fontsize=8, color=viz.INK_2)
viz.titled(
    ax,
    "RMSE of each off-block proxy, used raw",
    "Takeoff minus the estimate, taken directly as the prediction. 2025 departures, in-sample.",
    xlabel="RMSE (seconds)",
)
viz.only_x_grid(ax)
ax.set_xlim(0, max(values) * 1.15)
rec.figure(fig, "proxy_rmse")

# %% [markdown]
# ## Distribution of the aobt_3 gap
#
# `mvt_time - aobt_3` over 2025 departures, 5-second bins, with `DEFAULT_BANDS` shaded.

# %%
proxy_values = (
    train.with_columns((pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("px"))
    .filter(pl.col("px").is_between(0, 3000))
    .select("px")
    .collect(engine="streaming")["px"]
    .to_numpy()
)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(proxy_values, bins=range(0, 3001, 5), color=viz.BLUE, linewidth=0)
for label, (lo, hi) in DEFAULT_BANDS.items():
    ax.axvspan(lo, hi, color=viz.ORANGE, alpha=0.16, linewidth=0)
    minutes = 20 if "20" in label else 12
    ax.annotate(
        f"{minutes}-min default",
        xy=((lo + hi) / 2, ax.get_ylim()[1] * 0.92),
        ha="center",
        fontsize=8,
        color=viz.ORANGE,
    )
viz.titled(
    ax,
    "Takeoff minus aobt_3",
    "2025 departures, 5-second bins. Shaded: the bands listed in DEFAULT_BANDS.",
    xlabel="takeoff − aobt_3 (seconds)",
    ylabel="departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "proxy_distribution")

# %% [markdown]
# ## Excess mass over a flanking baseline
#
# Each band's row count against the mean of two flanking bands of equal width. Edit
# `CANDIDATES` to test other centres.
#
# Method note: correlating the proxy against the target *inside* a band is not informative,
# because restricting a variable to a 60-second window removes its variance. The next cell runs
# that as a control, at both band and non-band centres.

# %%
rec.section("excess mass over a flanking baseline (band width 61 s)")
proxy = (
    train.with_columns((pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("px"))
    .filter(pl.col("px").is_between(0, 3600))
    .select("px", "taxi_sec")
    .collect(engine="streaming")
)
total = proxy.height


def band(lo: int, hi: int) -> pl.DataFrame:
    return proxy.filter(pl.col("px").is_between(lo, hi))


excess = pl.DataFrame(
    {
        "centre_s": CANDIDATES,
        "band_n": [band(c - 30, c + 30).height for c in CANDIDATES],
        "flank_n": [
            (band(c - 150, c - 90).height + band(c + 90, c + 150).height) / 2 for c in CANDIDATES
        ],
    }
).with_columns(
    (pl.col("band_n") - pl.col("flank_n")).round(0).alias("excess_n"),
    ((pl.col("band_n") - pl.col("flank_n")) / total * 100).round(2).alias("excess_pct"),
)
rec.show(excess)

# %%
rec.section("control: correlation inside a band, at default and non-default centres")
restricted = pl.DataFrame(
    {
        "band_centre": [600, 720, 900, 1200, 1500],
        "corr": [
            round(band(c - 30, c + 30).select(pl.corr("px", "taxi_sec")).item(), 3)
            for c in [600, 720, 900, 1200, 1500]
        ],
        "is_default": [False, True, False, True, True],
    }
)
rec.show(restricted)

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
counts, edges = None, None
values = proxy["px"].to_numpy()
counts, edges, _ = ax.hist(
    values[values <= 3000], bins=range(0, 3001, 30), color=viz.BLUE, linewidth=0
)
for centre in DEFAULT_CENTRES.values():
    ax.axvline(centre, color=viz.ORANGE, linewidth=1.5, zorder=3)
    height = counts[min(int(centre // 30), len(counts) - 1)]
    viz.annotate(ax, centre, height, f"{centre // 60} min", dx=5, dy=4, color=viz.ORANGE)
viz.titled(
    ax,
    "Takeoff minus aobt_3, with DEFAULT_CENTRES marked",
    "2025 departures, 30-second bins.",
    xlabel="takeoff − aobt_3 (seconds)",
    ylabel="departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "proxy_distribution_defaults")


# %%
def classify(frame):
    """Label each departure by which DEFAULT_BANDS entry its aobt_3 gap falls in."""
    kind = pl.lit("measured")
    for label, (lo, hi) in DEFAULT_BANDS.items():
        kind = pl.when(pl.col("px").is_between(lo, hi)).then(pl.lit(label)).otherwise(kind)
    return (
        frame.with_columns((pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("px"))
        .filter(pl.col("px").is_not_null())
        .with_columns(kind.alias("kind"))
    )


train_kind, rank_kind = classify(train), classify(rank)

# %% [markdown]
# ## Band rate by airport, train against ranking

# %%
rec.section("default rate by airport — train vs ranking")
default_rate = (
    train_kind.group_by("airport")
    .agg((pl.col("kind") != "measured").mean().round(3).alias("train"))
    .collect(engine="streaming")
    .join(
        rank_kind.group_by("airport")
        .agg((pl.col("kind") != "measured").mean().round(3).alias("rank"))
        .collect(engine="streaming"),
        on="airport",
    )
    .sort("train", descending=True)
)
rec.show(default_rate)

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
airports = default_rate["airport"].to_list()
positions = range(len(airports))
width = 0.38
ax.bar(
    [p - width / 2 - 0.01 for p in positions],
    default_rate["train"].to_list(),
    width,
    label="2025 training",
    color=viz.BLUE,
)
ax.bar(
    [p + width / 2 + 0.01 for p in positions],
    default_rate["rank"].to_list(),
    width,
    label="2026 ranking",
    color=viz.ORANGE,
)
ax.set_xticks(list(positions))
ax.set_xticklabels(airports)
ax.set_ylim(0, 0.5)
ax.legend(loc="upper right")
viz.titled(
    ax,
    "Share of departures whose aobt_3 falls in a default band",
    "By airport, 2025 training against the 2026 ranking set.",
    ylabel="share of departures",
)
viz.only_y_grid(ax)
rec.figure(fig, "default_rate_by_airport")

# %%
