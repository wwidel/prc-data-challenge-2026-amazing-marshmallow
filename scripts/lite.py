"""The light tier: about thirty numbers, no feature store, no model library, minutes not days.

Per airport, least squares of the label on the two off-block stamps every NM-matched departure
carries — ``takeoff − aobt_3`` and ``takeoff − eobt_1`` — fitted on rows with both stamps and a
label of at most 12 h. Rows without both fall back to the stand's median taxi (stands with at
least 30 departures), then the airport's. Then the two exact rules and the bounds of the full
build: an NM-unmatched LIRF departure more than 12 h after its schedule takes the gap; an
NM-matched one whose ``takeoff − aobt_3`` exceeds 4 h takes that; the schedule-gap bound and the
LEBL/LEMD ceilings; and, for the 2026 file, the LOBT window.

This is variant "S0b" of ``scripts/analysis/a104_minimal_variants.py`` in our working repository,
extracted to stand alone. a104 measured, on the Jan + Jul 2025 folds (each fitted on the other
eleven months): trimmed RMSE without Rome 297.84 (Jan 280.52, Jul 311.18), all rows untrimmed
440.49, LIRF untrimmed 1,035.1 — against 216.53 / 320.82 / 625.1 on the same folds for our
stack at the time (v39's 13 members and the r14 Rome layer; leaderboard 276.79).

    uv run python scripts/lite.py [--out data/output/lite.parquet]

Needs only data/raw/ and the spine (``scripts/build_spine.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from taxiout import io, paths, postprocess, spine

FOLDS = ("2025-01", "2025-07")
TRIM_ABOVE = 43_200.0
STAND_MIN_ROWS = 30
RULE_GAP = 43_200.0  # NM-unmatched LIRF with schedule gap above this -> the gap
RULE_AOBT = 14_400.0  # NM-matched with takeoff - aobt_3 above this -> that value


def load() -> pl.DataFrame:
    """Every departure of the spine (2025 and 2026) with the stamps the fit reads."""
    full = spine.load()
    raw = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select(
            "mvt_id",
            "stand",
            "eobt_1",
            "aobt_3",
            "sched_time",
            "lobt",
            pl.col("flight_id").is_null().alias("unm"),
        )
        .collect(engine="streaming")
    )
    d = full.join(raw, on="mvt_id", how="left").with_columns(
        (pl.col("mvt_time") - pl.col("eobt_1")).dt.total_seconds().cast(pl.Float64).alias("pe"),
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64).alias("pa"),
        (pl.col("mvt_time") - pl.col("sched_time"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("gap"),
        (pl.col("mvt_time") - pl.col("lobt")).dt.total_seconds().cast(pl.Float64).alias("m_l"),
        pl.col("taxi_sec").cast(pl.Float64).alias("y"),
    )
    assert d.height == full.height
    return d


def fit(train: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Per-airport coefficients (b0, b_pa, b_pe), stand medians and airport medians."""
    coefs = []
    for ap in sorted(train["airport"].unique().to_list()):
        t = train.filter(
            (pl.col("airport") == ap)
            & pl.col("pa").is_not_null()
            & pl.col("pe").is_not_null()
            & (pl.col("y") <= TRIM_ABOVE)
        )
        A = np.column_stack([np.ones(t.height), t["pa"].to_numpy(), t["pe"].to_numpy()])
        c = np.linalg.lstsq(A, t["y"].to_numpy(), rcond=None)[0]
        coefs.append({"airport": ap, "b0": c[0], "b_pa": c[1], "b_pe": c[2]})
    stand_med = (
        train.group_by("airport", "stand")
        .agg(pl.col("y").median().alias("st_med"), pl.len().alias("st_n"))
        .filter(pl.col("st_n") >= STAND_MIN_ROWS)
        .select("airport", "stand", "st_med")
    )
    ap_med = train.group_by("airport").agg(pl.col("y").median().alias("ap_med"))
    return pl.from_dicts(coefs), stand_med, ap_med


def predict(frame: pl.DataFrame, model) -> pl.DataFrame:
    """The S0b prediction, then the two rules and the bounds. Adds a ``prediction`` column."""
    coefs, stand_med, ap_med = model
    out = (
        frame.join(coefs, on="airport", how="left")
        .join(stand_med, on=["airport", "stand"], how="left")
        .join(ap_med, on="airport", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("b0") + pl.col("b_pa") * pl.col("pa") + pl.col("b_pe") * pl.col("pe"),
                pl.col("st_med"),
                pl.col("ap_med"),
            )
            .clip(0.0, None)
            .alias("prediction")
        )
        .with_columns(
            pl.when(pl.col("unm") & (pl.col("airport") == "LIRF") & (pl.col("gap") > RULE_GAP))
            .then(pl.col("gap"))
            .when(~pl.col("unm") & (pl.col("pa") > RULE_AOBT))
            .then(pl.col("pa"))
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
    )
    out = postprocess.apply_airport_ceilings(postprocess.apply_gap_bound(out))
    return out.with_columns(pl.col("prediction").clip(*postprocess.CLIP))


def rmse(frame: pl.DataFrame, sel: pl.Expr, trimmed: bool) -> float:
    s = frame.filter(sel & pl.col("y").is_not_null())
    if trimmed:
        s = s.filter(pl.col("y") <= TRIM_ABOVE)
    return float(np.sqrt(np.mean((s["y"].to_numpy() - s["prediction"].to_numpy()) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(paths.OUTPUT / "lite.parquet"))
    ap.add_argument("--dry-run", action="store_true", help="score and validate, write nothing")
    args = ap.parse_args()

    d = load()
    print(f"departures: {d.height:,}")

    parts = []
    for held in FOLDS:
        train = d.filter(pl.col("month").is_in([m for m in paths.TRAIN_MONTHS if m != held]))
        parts.append(predict(d.filter(pl.col("month") == held), fit(train)))
    ev = pl.concat(parts)
    body, rome = pl.col("airport") != "LIRF", pl.col("airport") == "LIRF"
    print("cross-validated on Jan and Jul 2025, each fitted on the other eleven months:")
    print(f"  trimmed RMSE without Rome  {rmse(ev, body, True):8.2f}")
    for held in FOLDS:
        held_rmse = rmse(ev, body & (pl.col("month") == held), True)
        print(f"    {held}                  {held_rmse:8.2f}")
    print(f"  untrimmed, without Rome    {rmse(ev, body, False):8.2f}")
    print(f"  untrimmed, Rome only       {rmse(ev, rome, False):8.1f}")
    print(f"  untrimmed, all rows        {rmse(ev, pl.lit(True), False):8.2f}")

    train = d.filter(pl.col("month").is_in(list(paths.TRAIN_MONTHS)))
    rank = predict(d.filter(pl.col("month").is_in(list(paths.RANK_MONTHS))), fit(train))
    rank = postprocess.apply_lobt_window(rank).select("mvt_id", "prediction")
    frame = postprocess.write(rank, Path(args.out), dry_run=args.dry_run)
    print(
        f"{'validated (dry run)' if args.dry_run else 'wrote ' + args.out}: {frame.height:,} rows, "
        f"mean {frame['TAXITIME_SEC_mvt'].mean():,.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
