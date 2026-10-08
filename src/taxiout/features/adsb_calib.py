"""The trace, re-expressed against this airport-month's own receiver behaviour (label-free).

``scripts/analysis/a115_adsb_drift_and_error_budget.py`` measured the drift the challenge Discord
warned about: adsb.lol's receivers changed between 2025 and 2026. At EGLL the median of
``adsb_taxi_move − (mvt − aobt_3)`` — how much later than the recorded off-block the trace first
shows movement — went from −842 / −1,093 s
(Jan / Jul 2025) to −538 / −244 s (Jan / Jul 2026), while coverage went from 21% / 8% to 60% / 98%.
A model that learnt "an EGLL trace starts ~1,000 s after off-block" from 2025 over-predicts every
traced EGLL row in 2026. Nothing in the 2025 folds can show this.

The remedy is to hand the model the trace **relative to its own month**. Per airport × UTC month,
from that month's departures alone, with no label involved:

* ``ad_match_rate_m``: share of departures with a matched trace (the receiver coverage);
* ``ad_med_lag_m``: median of ``adsb_taxi_move − (mvt − aobt_3)`` over traced rows with an AOBT
  (the typical lag of the trace behind the recorded off-block, this month);
* ``ad_med_n_ground_m``: median point count of the month's traces (receiver density).

And per departure:

* ``ad_taxi_move_adj = adsb_taxi_move − ad_med_lag_m``: the trace-implied taxi with this month's
  typical lag removed — comparable across months and airports;
* ``ad_lag_vs_m = (adsb_taxi_move − (mvt − aobt_3)) − ad_med_lag_m``: how unusual this trace's
  start is for the month (a late-starting trace, or an off-block record that disagrees with it);
* ``ad_n_ground_rel = adsb_n_ground / ad_med_n_ground_m``: trace density relative to the month.

The statistics use the whole month, including rows after the departure; the organisers state the
model is post-operational and whole-day statistics are already in use (`runway_history`). Reads
the cached ``adsb`` block (``CONTEXT``; rebuild with ``--overwrite`` when ``adsb`` is rebuilt).
"""

from __future__ import annotations

import polars as pl

from taxiout import io, paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

OUTPUTS = (
    "ad_match_rate_m",
    "ad_med_lag_m",
    "ad_med_n_ground_m",
    "ad_taxi_move_adj",
    "ad_lag_vs_m",
    "ad_n_ground_rel",
)


def featurize(deps: pl.DataFrame) -> pl.DataFrame:
    """All outputs. `deps`: mvt_id, airport, mvt_time, aobt_3, adsb_taxi_move, adsb_n_ground."""
    d = deps.with_columns(
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("_m"),
        (
            pl.col("adsb_taxi_move").cast(pl.Float64)
            - (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64)
        ).alias("_lag"),
    )
    month = d.group_by("airport", "_m").agg(
        pl.col("adsb_taxi_move").is_not_null().mean().alias("ad_match_rate_m"),
        pl.col("_lag").median().alias("ad_med_lag_m"),
        pl.col("adsb_n_ground").cast(pl.Float64).median().alias("ad_med_n_ground_m"),
    )
    out = d.join(month, on=["airport", "_m"], how="left").with_columns(
        (pl.col("adsb_taxi_move").cast(pl.Float64) - pl.col("ad_med_lag_m")).alias(
            "ad_taxi_move_adj"
        ),
        (pl.col("_lag") - pl.col("ad_med_lag_m")).alias("ad_lag_vs_m"),
        pl.when(pl.col("ad_med_n_ground_m") > 0)
        .then(pl.col("adsb_n_ground").cast(pl.Float64) / pl.col("ad_med_n_ground_m"))
        .alias("ad_n_ground_rel"),
    )
    return out.select("mvt_id", *OUTPUTS)


@feature(name="adsb_calib", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def adsb_calib(ctx: FeatureContext) -> pl.DataFrame:
    """The trace against its own airport-month: coverage, typical lag, lag-adjusted taxi."""
    from taxiout import store  # local: the store imports the feature registry

    months = tuple(sorted(f.stem for f in paths.SPINE.glob("*.parquet")))
    traces = (
        store.assemble(["adsb"], months)
        .select("mvt_id", "adsb_taxi_move", "adsb_n_ground")
        .collect()
    )
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "mvt_time", "aobt_3")
        .collect(engine="streaming")
        .join(traces, on="mvt_id", how="left")
    )
    return align(ctx.spine, featurize(deps), OUTPUTS)
