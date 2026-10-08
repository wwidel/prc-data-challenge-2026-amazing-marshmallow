"""Per-cell rates of the block_time recording modes — the first target-dependent block.

On roughly 39% of 2025 departures ``block_time`` is the A-CDM actual off-block, and on those
rows ``ob_aobt_sec`` *is* the taxi: the bare identity scores RMSE 70 where the model scores 145,
because the model hedges against the rows where the recording took something else. The model
cannot un-hedge on its own — whether a row is in
the A mode is a fact about ``block_time``, which the ranking set withholds.

What *can* travel is the rate. Which mode gets recorded is strongly patterned by airport, by how
late the push was, and by whether the plan times collapse onto the schedule — and those
covariates are present on every 2026 row. So this block estimates, per cell of admissible
covariates, the probability that a row's ``block_time`` is the A-CDM actual (``md_p_a``) and the
probability it is the schedule (``md_p_s``), and hands the model those rates.

**Why this is TARGET_DEPENDENT.** A training row's mode is computed from its ``block_time``,
reconstructed as ``mvt_time − taxi_sec``. Knowing ``block_time`` is knowing the label. So the
rates must be estimated per fold from that fold's training months only — a globally cached
version would leak every validation month's labels into its own features. This is the first real
user of the per-fold path in
``cv.build_matrix``; ``tests/test_pipeline.py`` carries a leak test that corrupts the held-out
labels and asserts these columns do not move.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Tolerances: the plan values are exact copies, the A-CDM actual is coarser.
TOL_SAME = 60
TOL_AOBT = 120
#: Cell rates are shrunk toward the airport's own rate by this many pseudo-rows.
SHRINK_K = 200.0
#: Off-block delay bands (aobt_3 − sched_time, seconds); null aobt gets its own band.
DELAY_EDGES = [-1800.0, 0.0, 900.0, 3600.0, 10_800.0, 43_200.0]

OUTPUTS = ("md_p_a", "md_p_s", "md_cell_n")

#: The covariates a cell is keyed on. Everything here is present on 2026 rows.
CELL_KEY = ("airport", "md_delay_band", "md_eobt_is_sched", "md_aobt_missing")


def _covariates(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Admissible cell covariates for every departure, matched or not."""
    return frame.with_columns(
        (pl.col("aobt_3") - pl.col("sched_time"))
        .dt.total_seconds()
        .cut(DELAY_EDGES, labels=[str(i) for i in range(len(DELAY_EDGES) + 1)])
        .cast(pl.Utf8)
        .fill_null("missing")
        .alias("md_delay_band"),
        ((pl.col("eobt_1") - pl.col("sched_time")).dt.total_seconds().abs() <= TOL_SAME)
        .fill_null(False)
        .alias("md_eobt_is_sched"),
        pl.col("aobt_3").is_null().alias("md_aobt_missing"),
    )


@feature(name="modes", kind=FeatureKind.TARGET_DEPENDENT, outputs=OUTPUTS)
def modes(ctx: FeatureContext) -> pl.DataFrame:
    """P(block_time is the A-CDM actual) and P(it is the schedule), per covariate cell."""
    raw = _covariates(
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "sched_time", "aobt_3", "eobt_1")
    ).collect(engine="streaming")

    rows = ctx.spine.select("mvt_id", "airport").join(raw, on="mvt_id", how="left")

    # --- label the training rows, and only them -------------------------------------------
    train = (
        ctx.training_rows()
        .select("mvt_id", "airport", "mvt_time", "taxi_sec")
        .join(raw, on="mvt_id", how="left")
        .with_columns(
            (pl.col("mvt_time") - pl.duration(seconds=pl.col("taxi_sec"))).alias("_block")
        )
        .with_columns(
            ((pl.col("_block") - pl.col("aobt_3")).dt.total_seconds().abs() <= TOL_AOBT)
            .fill_null(False)
            .alias("_is_a"),
        )
        .with_columns(
            (
                ((pl.col("_block") - pl.col("sched_time")).dt.total_seconds().abs() <= TOL_SAME)
                & ~pl.col("_is_a")
            )
            .fill_null(False)
            .alias("_is_s"),
        )
    )

    prior = train.group_by("airport").agg(
        pl.col("_is_a").mean().alias("_pa_prior"),
        pl.col("_is_s").mean().alias("_ps_prior"),
    )
    cells = (
        train.group_by(list(CELL_KEY))
        .agg(
            pl.len().alias("md_cell_n"),
            pl.col("_is_a").sum().alias("_na"),
            pl.col("_is_s").sum().alias("_ns"),
        )
        .join(prior, on="airport", how="left")
        .with_columns(
            (
                (pl.col("_na") + SHRINK_K * pl.col("_pa_prior")) / (pl.col("md_cell_n") + SHRINK_K)
            ).alias("md_p_a"),
            (
                (pl.col("_ns") + SHRINK_K * pl.col("_ps_prior")) / (pl.col("md_cell_n") + SHRINK_K)
            ).alias("md_p_s"),
        )
        .select(*CELL_KEY, "md_p_a", "md_p_s", "md_cell_n")
    )

    # --- hand every row its cell's rate; an unseen cell falls back to the airport prior ---
    out = (
        rows.join(cells, on=list(CELL_KEY), how="left")
        .join(prior, on="airport", how="left")
        .with_columns(
            pl.col("md_p_a").fill_null(pl.col("_pa_prior")),
            pl.col("md_p_s").fill_null(pl.col("_ps_prior")),
            pl.col("md_cell_n").fill_null(0),
        )
        .select("mvt_id", *OUTPUTS)
    )
    return align(ctx.spine, out, OUTPUTS)
