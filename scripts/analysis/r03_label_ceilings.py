# %% [markdown]
# # r03 — Per-airport label maxima: where the provider cuts long taxis
#
# Section 1 of our review script `r03_mechanisms.py`, unchanged. Per airport, the largest
# departure label and the counts just below and above one and two hours, then the same for
# arrivals. LEBL and LEMD never exceed ~3,600 s in either direction, which is what the 3,600 s
# prediction ceiling in `postprocess.AIRPORT_CEILINGS` rests on.

# %%
import polars as pl

from taxiout import io as tio
from taxiout.analysis import Recorder

rec = Recorder("r03_label_ceilings", "Per-airport label maxima: is there a cut at 1 h or 2 h?")

tr = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("airport", "taxi_sec")
    .collect(engine="streaming")
)
arr = (
    tio.scan_training()
    .filter(pl.col("phase") == "ARR")
    .select("airport", "taxi_sec")
    .collect(engine="streaming")
)
rec.say(f"  train DEP {tr.height:,}; train ARR {arr.height:,}")

# %%
rec.section("1. departures: label maxima and the tail just below/above 1 h and 2 h")
rec.show(
    tr.group_by("airport")
    .agg(
        pl.col("taxi_sec").max().alias("max"),
        pl.col("taxi_sec").is_between(3000, 3300).sum().alias("3000-3300"),
        pl.col("taxi_sec").is_between(3300, 3610).sum().alias("3300-3610"),
        (pl.col("taxi_sec") > 3610).sum().alias(">3610"),
        pl.col("taxi_sec").is_between(6000, 6600).sum().alias("6000-6600"),
        pl.col("taxi_sec").is_between(6600, 7210).sum().alias("6600-7210"),
        (pl.col("taxi_sec") > 7210).sum().alias(">7210"),
    )
    .sort("airport")
)

# %%
rec.section("2. arrivals (taxi-in)")
rec.show(
    arr.group_by("airport")
    .agg(
        pl.col("taxi_sec").max().alias("max"),
        pl.col("taxi_sec").is_between(3000, 3610).sum().alias("3000-3610"),
        (pl.col("taxi_sec") > 3610).sum().alias(">3610"),
    )
    .sort("airport")
)
rec.done()
