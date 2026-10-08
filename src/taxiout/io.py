"""Lazy scanners over the raw competition files.

The raw column names are long and suffixed (``TAXITIME_SEC_mvt``). Everything downstream
works with the short lowercase names in :data:`COLUMN_MAP` instead; the mapping is applied
once, here, and is the only place the raw spelling appears.
"""

from __future__ import annotations

import polars as pl

from taxiout import paths

#: raw column name -> canonical short name used everywhere else.
COLUMN_MAP: dict[str, str] = {
    # movement record
    "MVT_ID_mvt": "mvt_id",
    "FLIGHT_ID_mvt": "flight_id",
    "FLIGHT_mvt": "flight_no",
    "FLIGHT_RULE_mvt": "flight_rule",
    "ADEP_mvt": "adep",
    "ADES_mvt": "ades",
    "PHASE_mvt": "phase",
    "MVT_TIME_UTC_mvt": "mvt_time",
    "BLOCK_TIME_UTC_mvt": "block_time",
    "SCHED_TIME_UTC_mvt": "sched_time",
    "AIRCRAFT_TYPE_mvt": "ac_type",
    "RUNWAY_mvt": "runway",
    "STAND_mvt": "stand",
    "TAXITIME_SEC_mvt": "taxi_sec",
    # flight record (Network Manager)
    "LOBT_flt": "lobt",
    "CALLSIGN_flt": "callsign",
    "ADEP_flt": "adep_flt",
    "ADES_flt": "ades_flt",
    "ADES_FILED_flt": "ades_filed",
    "MARKET_SEGMENT_flt": "market_segment",
    "IOBT_flt": "iobt",
    "FLIGHT_RULE_flt": "flight_rule_flt",
    "FLIGHT_TYPE_flt": "flight_type",
    "AIRCRAFT_TYPE_flt": "ac_type_flt",
    "WK_TBL_CAT_flt": "wake_cat",
    "AIRCRAFT_OPERATOR_flt": "operator",
    "EOBT_1_flt": "eobt_1",
    "ARVT_1_flt": "arvt_1",
    "AOBT_3_flt": "aobt_3",
    "ARVT_3_flt": "arvt_3",
}

INVERSE_COLUMN_MAP: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}


def _normalise(lf: pl.LazyFrame, split: str) -> pl.LazyFrame:
    """Rename to canonical names and add the derived columns every caller wants."""
    return lf.rename(COLUMN_MAP).with_columns(
        # Float64 in the files, but integral. Int64 is exact and joins/groups faster.
        pl.col("mvt_id").cast(pl.Int64),
        pl.lit(split).alias("split"),
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        # The airport the movement physically happened at: origin for a departure,
        # destination for an arrival.
        pl.when(pl.col("phase") == "DEP")
        .then(pl.col("adep"))
        .otherwise(pl.col("ades"))
        .alias("airport"),
    )


def scan_training() -> pl.LazyFrame:
    """All 2025 movements, both phases."""
    return _normalise(pl.scan_parquet(paths.TRAINING_GLOB), "train")


def scan_ranking() -> pl.LazyFrame:
    """Jan + Jul 2026 movements. Departures have ``block_time``/``taxi_sec`` blanked."""
    return _normalise(pl.scan_parquet(paths.RANKING), "rank")


def scan_movements() -> pl.LazyFrame:
    """Training and ranking stacked — the full universe of movements we may look at.

    Use this for context features (congestion, taxi-in statistics): arrivals in the
    ranking window are complete, so they may legitimately inform 2026 predictions.
    """
    return pl.concat([scan_training(), scan_ranking()], how="vertical")


def scan_submitting() -> pl.LazyFrame:
    """The submission template, in the exact row order a submission must use."""
    return pl.scan_parquet(paths.SUBMITTING).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id")
    )
