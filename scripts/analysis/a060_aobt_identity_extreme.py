# %% [markdown]
# # a060 — The `onerow_aobt` override: "matched ∧ huge aobt gap ⇒ the label is that gap"
#
# For an NM-matched departure whose `takeoff − aobt_3` is enormous, is the recorded label that
# gap? Two separate questions:
#
# 1. **Is the aobt identity grounded at all?** `block_time` *is* the A-CDM actual off-block on a
#    large share of 2025 departures, and on those rows the bare identity beats any model (the
#    basis of the `modes` block). Quantified here per gap band.
# 2. **Does it survive at the extreme?** Measured by band, with the sample size stated, so "well
#    grounded" and "grounded on two rows" cannot be confused.
#
# Then it writes the override the analysis grounds: every 2026 matched row with an implied taxi
# above 12 h takes `takeoff − aobt_3` (`data/external/onerow_aobt.parquet`; one row in 2026).

# %%
import polars as pl

from taxiout import io as tio
from taxiout import paths
from taxiout.analysis import Recorder

rec = Recorder("a060_aobt_identity_extreme", "Does the aobt identity hold in the tail?")

# %% knobs
BANDS = [1_800.0, 3_600.0, 7_200.0, 14_400.0, 43_200.0]
AGREE = 120.0
#: Raise it and the file is empty; lower it into the 1–4 h bands and it picks up rows whose
#: labels section 1 shows are ordinary.
OVERRIDE_THRESHOLD = 43_200.0
OUT = paths.DATA / "external" / "onerow_aobt.parquet"

# %%
rec.section("1. 2025 matched departures, by implied-taxi band")
tr = (
    tio.scan_training()
    .filter(
        (pl.col("phase") == "DEP")
        & pl.col("flight_id").is_not_null()
        & pl.col("aobt_3").is_not_null()
    )
    .select(
        "mvt_id",
        "airport",
        "taxi_sec",
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("ob_aobt_sec"),
    )
    .collect(engine="streaming")
)
tr = tr.with_columns(
    pl.col("ob_aobt_sec").cut(BANDS).cast(pl.Utf8).alias("band"),
    (pl.col("taxi_sec") - pl.col("ob_aobt_sec")).alias("resid"),
)
rec.show(
    tr.group_by("band")
    .agg(
        pl.len().alias("n"),
        (pl.col("resid").abs() <= AGREE).mean().round(3).alias("P_identity_holds"),
        pl.col("resid").median().round(0).alias("median_resid"),
        (pl.col("resid") ** 2).mean().sqrt().round(0).alias("rmse_of_identity"),
        pl.col("taxi_sec").median().round(0).alias("median_label"),
    )
    .sort("band")
)
rec.say(
    "\n  P_identity_holds = share of rows where the label equals takeoff - aobt_3 within 2 min."
)

# %%
rec.section("2. the extreme band, row by row (this is the whole evidence base)")
extreme = tr.filter(pl.col("ob_aobt_sec") > 14_400).sort("ob_aobt_sec", descending=True)
rec.say(f"  2025 matched rows with an implied taxi over 4 h: {extreme.height}")
rec.show(extreme.select("airport", "ob_aobt_sec", "taxi_sec", "resid").head(20))

# %%
rec.section("3. the 2026 population the rule would fire on")
rk = (
    tio.scan_ranking()
    .filter(
        (pl.col("phase") == "DEP")
        & pl.col("flight_id").is_not_null()
        & pl.col("aobt_3").is_not_null()
    )
    .select(
        "mvt_id",
        "airport",
        "month",
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("ob_aobt_sec"),
    )
    .collect(engine="streaming")
)
for threshold in (7_200, 14_400, 43_200):
    sub = rk.filter(pl.col("ob_aobt_sec") > threshold)
    rec.say(f"  2026 matched rows with implied taxi > {threshold:>6}s: {sub.height}")

# %%
rec.section("4. the override file")
out = (
    rk.filter(pl.col("ob_aobt_sec") > OVERRIDE_THRESHOLD)
    .select("mvt_id", pl.col("ob_aobt_sec").alias("prediction"))
    .sort("mvt_id")
)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(OUT)
rec.say(
    f"  wrote {OUT.relative_to(paths.ROOT)} ({out.height} rows, rule > {OVERRIDE_THRESHOLD:,.0f} s)"
)

# %%
rec.done()
