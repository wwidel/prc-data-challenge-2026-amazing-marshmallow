"""EUROCONTROL's daily series per airport: regulation, slot adherence, pre-departure delay.

Team outgoing-boat joined three public EUROCONTROL CSVs per airport × UTC day (bundled with other
changes, −7.5 board to them; https://github.com/EnioAguiar/prc-taxiout-2026). The code is ours; the
files come from ``scripts/pull_eurocontrol.py`` into ``data/external/eurocontrol/``.

Per airport × UTC day of the takeoff:

* ``ec_dep_n``: departures; ``ec_reg_frac``: share ATFM-regulated; ``ec_reg_n``;
* ``ec_late_frac`` / ``ec_early_frac``: share of regulated departures off-slot late / early;
* ``ec_pre_min``: all pre-departure delay per IFR departure (minutes);
* ``ec_atc_min``: ATC pre-departure delay per departure (minutes).

These are day-level aggregates the provider publishes from operational data, including the day
being predicted; the rules allow open, documented data and nothing restricts its timing
(outgoing-boat's reading of the rules). ``CONTEXT``: external, never a row's label. **The cache key
does not see the CSV files**: rebuild with ``--overwrite`` if they change.
"""

from __future__ import annotations

import polars as pl

from taxiout import io, paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

SRC = paths.DATA / "external" / "eurocontrol"
OUTPUTS = (
    "ec_dep_n",
    "ec_reg_n",
    "ec_reg_frac",
    "ec_late_frac",
    "ec_early_frac",
    "ec_pre_min",
    "ec_atc_min",
)


def _read(name: str, cols: list[str]) -> pl.DataFrame:
    frames = []
    for f in sorted(SRC.glob(f"{name}_*.csv")):
        frames.append(
            pl.read_csv(f, infer_schema_length=0)
            .select("FLT_DATE", "APT_ICAO", *cols)
            .with_columns(
                pl.col("FLT_DATE").str.to_date().alias("day"),
                *[pl.col(c).cast(pl.Float64, strict=False) for c in cols],
            )
        )
    return (
        pl.concat(frames)
        .group_by("APT_ICAO", "day")
        .agg(*[pl.col(c).sum() for c in cols])
        .rename({"APT_ICAO": "airport"})
    )


def daily(slot: pl.DataFrame, pre: pl.DataFrame, atc: pl.DataFrame) -> pl.DataFrame:
    """The feature table per airport × day from the three series. Pure; unit-tested."""

    def ratio(a: str, b: str) -> pl.Expr:
        return pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b))

    return (
        slot.join(pre, on=["airport", "day"], how="full", coalesce=True)
        .join(atc, on=["airport", "day"], how="full", coalesce=True)
        .select(
            "airport",
            "day",
            pl.col("FLT_DEP_1").alias("ec_dep_n"),
            pl.col("FLT_DEP_REG_1").alias("ec_reg_n"),
            ratio("FLT_DEP_REG_1", "FLT_DEP_1").alias("ec_reg_frac"),
            ratio("FLT_DEP_OUT_LATE_1", "FLT_DEP_REG_1").alias("ec_late_frac"),
            ratio("FLT_DEP_OUT_EARLY_1", "FLT_DEP_REG_1").alias("ec_early_frac"),
            ratio("DLY_ALL_PRE_2", "FLT_DEP_IFR_2").alias("ec_pre_min"),
            ratio("DLY_ATC_PRE_3", "FLT_DEP_3").alias("ec_atc_min"),
        )
    )


@feature(name="eurocontrol_daily", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def eurocontrol_daily(ctx: FeatureContext) -> pl.DataFrame:
    """EUROCONTROL's daily regulation and pre-departure-delay series at the departure airport."""
    slot = _read(
        "atfm_slot_adherence",
        ["FLT_DEP_1", "FLT_DEP_REG_1", "FLT_DEP_OUT_LATE_1", "FLT_DEP_OUT_EARLY_1"],
    )
    pre = _read("all_pre_departure_delays", ["FLT_DEP_IFR_2", "DLY_ALL_PRE_2"])
    atc = _read("atc_pre_departure_delays", ["FLT_DEP_3", "DLY_ATC_PRE_3"])
    table = daily(slot, pre, atc)
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", pl.col("mvt_time").dt.date().alias("day"))
        .collect(engine="streaming")
        .join(table, on=["airport", "day"], how="left")
        .select("mvt_id", *OUTPUTS)
    )
    return align(ctx.spine, deps, OUTPUTS)
