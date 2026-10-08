# %% [markdown]
# # a115 — ADS-B drift 2025 → 2026 (label-free) and where a member's error sits
#
# Two questions to answer before spending more on ADS-B:
#
# 1. **Drift.** Trace coverage changed between the training year and the scored months (EGLL,
#    LEMD and LIRF most; flagged by team outgoing-boat and on the challenge Discord). If 2026's
#    denser receivers see the stand where 2025's did not, the trace starts *earlier* relative to
#    the recorded AOBT, and offsets learnt on 2025 over-predict. Measured without labels: per
#    airport × month, the match rate and the median of `adsb_taxi_move − (mvt − aobt_3)` — the
#    observed start of movement relative to the recorded off-block, both available on every
#    ranking row.
# 2. **Error budget.** A member's Jan + Jul 2025 OOF squared error split by airport, NM match,
#    trace presence and label band, so the next block is aimed at the largest cell, not the next
#    idea. Our run used `ab02` (ab07 without adsb_calib, not in this repository); the default here
#    is ab07.
#
# Reads the cached `adsb` block, the raw `aobt_3`, and the member's OOF file. Section 1 is the
# measurement behind the `adsb_calib` block.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, store  # noqa: E402

# %% knobs
MEMBER = os.environ.get("A115_MEMBER", "ab07_ab02_adsb_calib")
MONTHS_SHOWN = ("2025-01", "2025-07", "2026-01", "2026-07")
BANDS = [(0, 3600, "≤1h"), (3600, 10800, "1–3h"), (10800, 43200, "3–12h"), (43200, 10**9, ">12h")]

rec = analysis.Recorder("a115_adsb_drift_and_error_budget", "ADS-B drift and an error budget")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


# %%
rec.section("inputs")
months = tuple(sorted(f.stem for f in paths.SPINE.glob("*.parquet")))
adsb = (
    store.assemble(["adsb"], months)
    .select(
        "mvt_id",
        "airport",
        "month",
        "adsb_taxi_move",
        "adsb_taxi",
        "adsb_gs0",
        "adsb_n_ground",
        "adsb_match_pri",
    )
    .collect()
)
raw = (
    tio.scan_movements()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64).alias("m_a"),
        pl.col("flight_id").is_null().alias("unm"),
    )
    .collect(engine="streaming")
)
frame = adsb.join(raw, on="mvt_id", how="left").with_columns(
    pl.col("adsb_match_pri").is_not_null().alias("has_trace"),
    (pl.col("adsb_taxi_move") - pl.col("m_a")).alias("start_vs_aobt"),
)
rec.say(f"  {frame.height:,} departures over {len(months)} months; adsb block hash in use")

# %%
rec.section(
    "1. per airport × month: coverage and where the trace starts relative to the recorded AOBT"
)
drift = (
    frame.filter(pl.col("month").is_in(MONTHS_SHOWN))
    .group_by("airport", "month")
    .agg(
        pl.len().alias("n"),
        pl.col("has_trace").mean().round(3).alias("match_rate"),
        (pl.col("adsb_match_pri") == 0).mean().round(3).alias("by_callsign"),
        pl.col("adsb_taxi_move").median().alias("med_taxi_move"),
        pl.col("start_vs_aobt").median().alias("med_start_vs_aobt"),
        pl.col("start_vs_aobt").quantile(0.25).alias("q25_start_vs_aobt"),
        pl.col("start_vs_aobt").quantile(0.75).alias("q75_start_vs_aobt"),
        (pl.col("adsb_gs0") <= 1).filter(pl.col("has_trace")).mean().round(3).alias("seen_stopped"),
        pl.col("adsb_n_ground").median().alias("med_n_ground"),
    )
    .sort("airport", "month")
)
rec.show(
    drift,
    "start_vs_aobt = adsb_taxi_move − (mvt − aobt_3), seconds: negative = the trace starts after "
    "the recorded off-block; a shift between 2025 and 2026 at the same airport is drift",
)
wide = drift.select("airport", "month", "match_rate", "med_start_vs_aobt").pivot(
    on="month", index="airport", values=["match_rate", "med_start_vs_aobt"]
)
rec.show(wide.sort("airport"), "the same, side by side")

# %%
rec.section(f"2. error budget of {MEMBER} (Jan + Jul 2025 OOF)")
f = newest_oof(MEMBER)
rec.say(f"  OOF file: {f.name}")
oof = (
    pl.read_parquet(f, glob=False)
    .select("mvt_id", "airport", "fold", "y", "yhat")
    .join(frame.select("mvt_id", "has_trace", "unm"), on="mvt_id", how="left")
    .with_columns(((pl.col("y") - pl.col("yhat")) ** 2).alias("se"))
)
band = pl.lit("?")
for lo, hi, name in reversed(BANDS):
    band = pl.when((pl.col("y") > lo) & (pl.col("y") <= hi)).then(pl.lit(name)).otherwise(band)
oof = oof.with_columns(band.alias("band"))
total = float(oof["se"].sum())
ex = oof.filter(pl.col("airport") != "LIRF")
total_ex = float(ex["se"].sum())
rec.say(
    f"  rows {oof.height:,}; untrimmed RMSE all {float((total / oof.height) ** 0.5):.1f}, "
    f"ex-LIRF {float((total_ex / ex.height) ** 0.5):.1f}"
)


def budget(frame: pl.DataFrame, keys: list[str], denom: float, label: str) -> None:
    rec.show(
        frame.group_by(keys)
        .agg(
            pl.len().alias("n"),
            (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
            (pl.col("se").sum() / denom).round(4).alias("share_of_sq_error"),
        )
        .sort("share_of_sq_error", descending=True),
        label,
    )


budget(oof, ["airport"], total, "by airport (incl. LIRF; share of ALL squared error)")
budget(ex, ["unm", "has_trace"], total_ex, "ex-LIRF: by NM match × trace presence")
budget(ex, ["band"], total_ex, "ex-LIRF: by label band")
budget(ex, ["airport", "has_trace"], total_ex, "ex-LIRF: by airport × trace presence")
budget(
    oof.filter(pl.col("airport") == "LIRF"),
    ["unm", "band"],
    float(oof.filter(pl.col("airport") == "LIRF")["se"].sum()),
    "LIRF only (this member, not the Rome layer): by NM match × label band",
)
rec.done()
