# %% [markdown]
# # a112 — Is `block_time` within a fixed window of `lobt`, and what does projecting onto it do?
#
# Team outgoing-boat's public write-up (https://github.com/EnioAguiar/prc-taxiout-2026,
# `docs/research/2026-09-27-janela-lobt.md`) states that in 100% of 2025 departures with a LOBT,
# |BLOCK − LOBT| ≤ 3606 s, and projects every prediction onto `mvt − lobt ± 3606`. Our only
# admissible bound until then was the one-sided `gap + 1800` (`postprocess.apply_gap_bound`).
# This measures, on our data:
#
#   1. the distribution of `block_time − lobt` on 2025 departures (share inside ±W, extremes);
#   2. the RMSE effect of projecting each member's Jan + Jul OOF onto the window
#      (ex-LIRF, trimmed and untrimmed, per fold) -- no fitting, so no selection.
# Shipped from v44 (board −0.77).

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
WINDOW = float(os.environ.get("A112_WINDOW", "3606"))
MEMBERS = paths.composition()  # v62's members
TRIM_ABOVE = 43_200.0

rec = analysis.Recorder("a112_lobt_window", "Is block_time inside a fixed window of lobt?")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
rec.say(f"  window ±{WINDOW:.0f} s; {len(MEMBERS)} members")


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


# %%
rec.section("1. block_time − lobt on 2025 departures")
dep = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        (pl.col("block_time") - pl.col("lobt")).dt.total_seconds().cast(pl.Float64).alias("b_l"),
        (pl.col("mvt_time") - pl.col("lobt")).dt.total_seconds().cast(pl.Float64).alias("m_l"),
        pl.col("flight_id").is_null().alias("unm"),
    )
    .collect(engine="streaming")
)
has = dep.filter(pl.col("b_l").is_not_null())
rec.say(
    f"  2025 departures {dep.height:,}; with lobt and block_time {has.height:,} "
    f"({has.height / dep.height:.2%}); unmatched among those {int(has['unm'].sum()):,}"
)
rec.show(
    has.select(
        pl.len().alias("n"),
        (pl.col("b_l").abs() <= WINDOW).mean().alias("share_inside"),
        (pl.col("b_l").abs() > WINDOW).sum().alias("n_outside"),
        pl.col("b_l").min().alias("min"),
        pl.col("b_l").quantile(0.001).alias("p0.1"),
        pl.col("b_l").median().alias("median"),
        pl.col("b_l").quantile(0.999).alias("p99.9"),
        pl.col("b_l").max().alias("max"),
    ),
    "block_time − lobt, seconds",
)
rec.show(
    has.group_by("airport")
    .agg(
        pl.len().alias("n"),
        (pl.col("b_l").abs() > WINDOW).sum().alias("n_outside"),
        pl.col("b_l").min().alias("min"),
        pl.col("b_l").max().alias("max"),
    )
    .sort("airport"),
    "per airport",
)
rec.show(
    dep.group_by("unm")
    .agg(pl.len().alias("n"), pl.col("b_l").is_not_null().mean().alias("share_with_lobt"))
    .sort("unm"),
    "lobt availability by NM match",
)

# %%
rec.section("2. projecting each member's OOF onto the window (Jan + Jul 2025)")
win = dep.select("mvt_id", "m_l", "unm")
rows = []
for m in MEMBERS:
    f = newest_oof(m)
    o = (
        pl.read_parquet(f, glob=False)
        .select("mvt_id", "airport", "fold", "y", "yhat")
        .join(win, on="mvt_id", how="left")
    )
    o = o.filter(pl.col("airport") != "LIRF").with_columns(
        pl.when(pl.col("m_l").is_null())
        .then(pl.col("yhat"))
        .otherwise(pl.col("yhat").clip(pl.col("m_l") - WINDOW, pl.col("m_l") + WINDOW))
        .clip(0.0)
        .alias("proj")
    )
    y, a, b = (o[c].to_numpy().astype(float) for c in ("y", "yhat", "proj"))
    keep = y <= TRIM_ABOVE
    fold = o["fold"].to_numpy()
    row = {
        "member": m,
        "moved": int((np.abs(a - b) > 1e-9).sum()),
        "untrim": round(float(np.sqrt(np.mean((y - a) ** 2))), 2),
        "untrim_proj": round(float(np.sqrt(np.mean((y - b) ** 2))), 2),
        "trim": round(float(np.sqrt(np.mean((y[keep] - a[keep]) ** 2))), 3),
        "trim_proj": round(float(np.sqrt(np.mean((y[keep] - b[keep]) ** 2))), 3),
    }
    for fo in np.unique(fold):
        sel = keep & (fold == fo)
        before = np.sqrt(np.mean((y[sel] - a[sel]) ** 2))
        row[f"d_trim_{fo}"] = round(float(np.sqrt(np.mean((y[sel] - b[sel]) ** 2)) - before), 3)
    rows.append(row)
    rec.say(f"  {m}: {f.name}")
rec.show(pl.from_dicts(rows), "ex-LIRF; proj = clip(yhat, m_l ± W) then ≥ 0; no-lobt rows kept")
rec.done()
