# %% [markdown]
# # a050 — The `pop27_gap` override: Rome's 27 NM-unmatched rows with a huge schedule gap
#
# The 27 LIRF NM-unmatched ranking departures whose takeoff came more than 20,000 s after the
# schedule. On such rows in 2025 the label is, overwhelmingly, the takeoff − schedule gap itself
# (the aircraft is recorded as leaving the stand at its scheduled time), so the override predicts
# exactly that gap. Writes `data/external/pop27_gap.parquet` (mvt_id, prediction) for
# `build_stack.py --overrides`. Uses only columns the ranking set carries; no labels.

# %%
import polars as pl

from taxiout import io as tio
from taxiout import paths
from taxiout.analysis import Recorder

rec = Recorder("a050_pop27_values", "The 27 huge-gap NM-unmatched Rome rows: predict the gap")

# %% knobs
GAP_MIN = 20_000.0
OUT = paths.DATA / "external" / "pop27_gap.parquet"

# %%
rec.section("the population")
pop = (
    tio.scan_ranking()
    .filter(
        (pl.col("phase") == "DEP") & (pl.col("airport") == "LIRF") & pl.col("flight_id").is_null()
    )
    .select("mvt_id", "month", "mvt_time", "sched_time")
    .collect(engine="streaming")
    .with_columns((pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("gap"))
    .filter(pl.col("gap") > GAP_MIN)
)
rec.say(
    f"  rows: {pop.height} | Jan: {pop.filter(pl.col('month') == '2026-01').height} | "
    f"Jul: {pop.filter(pl.col('month') == '2026-07').height}"
)

# %%
rec.section("the override file")
OUT.parent.mkdir(parents=True, exist_ok=True)
pop.select("mvt_id", pl.col("gap").cast(pl.Float64).alias("prediction")).write_parquet(OUT)
rec.say(
    f"  {OUT.name}: {pop.height} rows, sum {pop['gap'].sum():,.0f} s, min {pop['gap'].min():,.0f}, "
    f"max {pop['gap'].max():,.0f}"
)

# %%
rec.done()
