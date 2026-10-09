"""Build the final prediction: NNLS over the members' refits, the Rome layer, then the layers.

In this order, which is part of the result:

1. **Body.** Each member's out-of-fold predictions on the non-LIRF rows fit a non-negative least
   squares combiner, with ``--weight-groups unmatched`` one weight vector for NM-matched rows and
   one for NM-unmatched rows (a group with fewer than ``--group-min-rows`` fitting rows inherits
   the global vector). The weights are applied to the members' twelve-month refits in
   ``data/rank_preds/<member>.parquet``, written by ``scripts/refit_predict.py``. Members come
   from ``config/compositions/`` (``paths.composition()``) unless ``--members`` names a file.
   With ``--unmatched-without-month MM`` the NM-unmatched vector is fitted without the
   out-of-fold rows of calendar month MM and applied to every other month, while that month's
   unmatched rows keep the vector fitted on all months (the final phase uses 02: two storm days
   at Istanbul in February 2025 pull the all-month vector up; ``a137``).
2. **Rome.** LIRF rows take ``--rome-from <experiment>``'s twelve-month refit verbatim.
3. **Cell shift.** ``--cell-shift`` adds a fixed number of seconds to every row of each listed
   airport-month (``postprocess.apply_cell_shift``).
4. **Overrides.** Each ``--overrides`` parquet of (mvt_id, prediction) replaces the rows it lists,
   in the order given; later files win.
5. **Bounds.** The schedule-gap bound and the per-airport ceilings (``postprocess``).
6. **LOBT window.** With ``--lobt-window``, each prediction is projected onto
   ``mvt − lobt ± 3606 s``, last, so it wins over every earlier layer.

The best submission (v62) is ``make stack``; REPRODUCE.md has the full command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

from taxiout import io, paths, postprocess, spine


class _LinearCombiner:
    """Minimal predictor wrapping a fixed non-negative weight vector."""

    def __init__(self, weights) -> None:
        self.weights = weights

    def predict(self, matrix, groups=None):
        return matrix @ self.weights


class _GroupedCombiner:
    """One weight vector per group, selected row by row.

    ``groups`` is the group value of each row, in the same order as the design matrix. A value
    with no fitted vector falls back to the global one, which is what happens to a ranking-set
    group that never appeared in the out-of-fold rows.
    """

    def __init__(self, weights_by_group: dict, fallback) -> None:
        self.weights_by_group = weights_by_group
        self.fallback = fallback

    def predict(self, matrix, groups=None):
        if groups is None:
            return matrix @ self.fallback
        out = np.empty(matrix.shape[0], dtype=float)
        groups = np.asarray(groups)
        for value in set(groups.tolist()):
            weights = self.weights_by_group.get(value, self.fallback)
            mask = groups == value
            out[mask] = matrix[mask] @ weights
        return out


def _month_held_vectors(
    matrix: np.ndarray, y: np.ndarray, unmatched: np.ndarray, month: np.ndarray, held: str
) -> tuple[np.ndarray, np.ndarray]:
    """The NM-unmatched rows' two NNLS vectors for ``--unmatched-without-month``.

    Returns (fitted on the unmatched rows of every month except ``held``, fitted on all unmatched
    rows). ``month`` is each row's calendar month as ``MM``. Pure; unit-tested.
    """
    unmatched = np.asarray(unmatched, dtype=bool)
    others = unmatched & (np.asarray(month) != held)
    without, _ = nnls(matrix[others], y[others])
    every, _ = nnls(matrix[unmatched], y[unmatched])
    return without, every


def _apply_held_month(
    stacked: np.ndarray,
    matrix: np.ndarray,
    unmatched: np.ndarray,
    month: np.ndarray,
    held: str,
    weights_all: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Give the unmatched rows of month ``held`` the all-month vector; leave every other row.

    Returns the new predictions and the number of rows changed. Pure; unit-tested.
    """
    sel = np.asarray(unmatched, dtype=bool) & (np.asarray(month) == held)
    out = np.asarray(stacked, dtype=float).copy()
    out[sel] = matrix[sel] @ weights_all
    return out, int(sel.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--members",
        default=None,
        help="file listing the members, one per line (default: the BASE composition)",
    )
    ap.add_argument(
        "--rome-from",
        required=True,
        help="take LIRF rows from data/rank_preds/<name>.parquet",
    )
    ap.add_argument(
        "--overrides",
        action="append",
        default=[],
        help="parquet of (mvt_id, prediction) applied after the Rome layer (repeatable)",
    )
    ap.add_argument(
        "--cell-shift",
        default=None,
        help=(
            "CSV of (airport, month YYYY-MM, shift seconds) added to every row of that "
            "airport-month before the overrides, so exact override rows keep their values"
        ),
    )
    ap.add_argument(
        "--weight-groups",
        default=None,
        choices=("unmatched",),
        help=(
            "fit one NNLS weight vector per group instead of one globally; `unmatched` splits on "
            "whether the NM flight match failed"
        ),
    )
    ap.add_argument(
        "--unmatched-without-month",
        default=None,
        metavar="MM",
        help=(
            "with --weight-groups unmatched: fit the unmatched vector without the out-of-fold rows "
            "of calendar month MM (e.g. 02) and apply it to every other month; that month's "
            "unmatched rows keep the vector fitted on all months"
        ),
    )
    ap.add_argument(
        "--group-min-rows",
        type=int,
        default=2000,
        help="a group with fewer fitting rows than this inherits the global vector",
    )
    ap.add_argument(
        "--lobt-window",
        action="store_true",
        help="after every other layer, project each prediction onto mvt - lobt +- 3606 s",
    )
    ap.add_argument(
        "--out",
        default=str(paths.OUTPUT / ("final.parquet" if paths.FINAL else "v62.parquet")),
        help="where to write the prediction file (competition format)",
    )
    ap.add_argument("--dry-run", action="store_true", help="build and validate, write nothing")
    args = ap.parse_args()
    if args.unmatched_without_month and args.weight_groups != "unmatched":
        ap.error("--unmatched-without-month needs --weight-groups unmatched")

    if args.members:
        members = Path(args.members).read_text().split()
        print(f"members from {args.members}")
    else:
        members = paths.composition()
        base = (
            paths.BASE_COMPOSITION.read_text().strip()
            if paths.BASE_COMPOSITION.exists()
            else "newest"
        )
        print(f"members from config/compositions/ ({base})")
    print(f"members ({len(members)}): {', '.join(members)}")

    # --- out-of-fold training matrix -------------------------------------------------------
    # No Path.glob: grid-expanded member names contain brackets, which glob reads as
    # character classes. A config rerun under a new hash leaves two files for one name; take
    # the newest and say which won.
    oof_paths: dict[str, Path] = {}
    for f in sorted(paths.PREDICTIONS.iterdir()):
        if f.suffix != ".parquet":
            continue
        name = f.name.split("__")[0]
        if name not in oof_paths or f.stat().st_mtime > oof_paths[name].stat().st_mtime:
            oof_paths[name] = f
    oof = None
    for name in members:
        print(f"  OOF {name}: {oof_paths[name].name}")
        f = pl.read_parquet(oof_paths[name], glob=False).select(
            "mvt_id", "airport", "y", pl.col("yhat").alias(name)
        )
        oof = f if oof is None else oof.join(f.select("mvt_id", name), on="mvt_id")

    # Match status: a null flight_id on the DEP row. It is available on the ranking rows too,
    # because only block_time and taxi_sec are blanked there.
    groups_oof = groups_rank = None
    if args.weight_groups == "unmatched":
        cols = (
            "mvt_id",
            pl.col("flight_id").is_null().alias("grp"),
            pl.col("mvt_time").dt.strftime("%m").alias("mm"),
        )
        groups_oof = (
            io.scan_training()
            .filter(pl.col("phase") == "DEP")
            .select(*cols)
            .collect(engine="streaming")
        )
        groups_rank = (
            io.scan_ranking()
            .filter(pl.col("phase") == "DEP")
            .select(*cols)
            .collect(engine="streaming")
        )
        print(
            f"  match status: {groups_oof['grp'].sum():,} of {groups_oof.height:,} unmatched in "
            f"2025, {groups_rank['grp'].sum():,} of {groups_rank.height:,} in the ranking set"
        )
        oof = oof.join(groups_oof, on="mvt_id", how="left")

    # Rome is served by its own layer, so the combiner is fitted on the other nine airports.
    train = oof.filter(pl.col("airport") != "LIRF")
    X = np.column_stack([train[m].to_numpy().astype(float) for m in members])
    y_train = train["y"].to_numpy().astype(float)
    weights, _ = nnls(X, y_train)
    combiner = _LinearCombiner(weights)
    print(
        "  nnls weights: "
        + ", ".join(f"{m}={w:.4f}" for m, w in zip(members, weights, strict=True))
    )
    if args.weight_groups:
        # One vector per group, each fitted only on that group's rows. A group too thin to fit
        # on inherits the global vector rather than an overfitted one.
        group_weights: dict[object, np.ndarray] = {}
        for value in sorted(train["grp"].unique().to_list(), key=str):
            rows = train.filter(pl.col("grp") == value)
            if rows.height < args.group_min_rows:
                group_weights[value] = weights
                print(f"  group {value!r}: {rows.height:,} rows, too thin -> global vector")
                continue
            gx = np.column_stack([rows[m].to_numpy().astype(float) for m in members])
            gw, _ = nnls(gx, rows["y"].to_numpy().astype(float))
            group_weights[value] = gw
            print(
                f"  group {value!r}: {rows.height:,} rows, weights "
                + ", ".join(f"{m}={w:.4f}" for m, w in zip(members, gw, strict=True))
            )
        held, w_unmatched_all = args.unmatched_without_month, None
        if held:
            unm = train["grp"].fill_null(False).to_numpy()
            mm = train["mm"].to_numpy()
            n_others = int((unm & (mm != held)).sum())
            if n_others < args.group_min_rows:
                ap.error(f"only {n_others:,} unmatched rows outside month {held}")
            w_without, w_unmatched_all = _month_held_vectors(X, y_train, unm, mm, held)
            group_weights[True] = w_without
            print(
                f"  group True without month {held}: {n_others:,} rows, weights "
                + ", ".join(f"{m}={w:.4f}" for m, w in zip(members, w_without, strict=True))
                + f" (sum {w_without.sum():.4f}; all months {w_unmatched_all.sum():.4f})"
            )
        combiner = _GroupedCombiner(group_weights, weights)
    print(f"combiner fitted on {train.height:,} non-LIRF OOF rows")

    # --- ranking-side member predictions ---------------------------------------------------
    rank = None
    for name in members:
        f = pl.read_parquet(paths.RANK_PREDS / f"{name}.parquet", glob=False).rename({"yhat": name})
        rank = f if rank is None else rank.join(f, on="mvt_id")
    rank_groups = None
    if args.weight_groups:
        # Join BEFORE building the matrix, so the matrix, the groups and the ids all come from
        # one frame rather than being paired positionally across a join.
        rank = rank.join(groups_rank, on="mvt_id", how="left", maintain_order="left")
        # A ranking row whose match status is unknown takes the matched vector, not a guess.
        rank_groups = rank["grp"].fill_null(False).to_numpy()
    rank_matrix = np.column_stack([rank[m].to_numpy().astype(float) for m in members])
    stacked = combiner.predict(rank_matrix, rank_groups)
    if rank_groups is not None:
        print(f"  grouped predict: {int(rank_groups.sum()):,} of {len(rank_groups):,} unmatched")
    if args.weight_groups and args.unmatched_without_month:
        stacked, n_held = _apply_held_month(
            stacked, rank_matrix, rank_groups, rank["mm"].to_numpy(), held, w_unmatched_all
        )
        print(f"  unmatched rows of month {held}: {n_held:,} keep the all-month vector")
    out = rank.select("mvt_id").with_columns(pl.Series("prediction", stacked))

    # --- Rome takes a single model's values verbatim ----------------------------------------
    rank_spine = spine.load(paths.RANK_MONTHS)
    rome_ids = rank_spine.filter(pl.col("airport") == "LIRF")["mvt_id"].implode()
    rome = pl.read_parquet(paths.RANK_PREDS / f"{args.rome_from}.parquet", glob=False).rename(
        {"yhat": "rome"}
    )
    print(f"Rome layer: {args.rome_from} (twelve-month refit)")
    rome = rome.with_columns(pl.col("mvt_id").cast(pl.Int64))
    out = (
        out.join(rome, on="mvt_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("mvt_id").is_in(rome_ids))
            .then(pl.col("rome").cast(pl.Float64))
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        .select("mvt_id", "prediction")
    )
    n_rome = int(out.filter(pl.col("mvt_id").is_in(rome_ids)).height)
    print(f"rows: {out.height:,} ({n_rome:,} Rome rows)")

    # --- per airport-month shifts, before the overrides: exact override rows win -------------
    if args.cell_shift:
        shifts = pl.read_csv(args.cell_shift)
        cells = (
            io.scan_ranking()
            .filter(pl.col("phase") == "DEP")
            .select("mvt_id", "airport", pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"))
            .collect(engine="streaming")
        )
        before = out["prediction"].to_numpy().copy()
        out = (
            out.join(cells, on="mvt_id", how="left", maintain_order="left")
            .pipe(postprocess.apply_cell_shift, shifts)
            .select("mvt_id", "prediction")
        )
        delta = out["prediction"].to_numpy() - before
        print(
            f"  cell shift {Path(args.cell_shift).name}: {shifts.height} airport-months, "
            f"{int((delta != 0).sum()):,} rows moved, sum of moves {float(delta.sum()):+,.0f} s"
        )

    # --- explicit overrides, in order; later files win -------------------------------------
    for path in args.overrides:
        vals = pl.read_parquet(path, glob=False).select(
            pl.col("mvt_id").cast(pl.Int64), pl.col("prediction").cast(pl.Float64).alias("_ovr")
        )
        before_rows = out.height
        matched = int(out.join(vals, on="mvt_id", how="inner").height)
        out = (
            out.join(vals, on="mvt_id", how="left", maintain_order="left")
            .with_columns(pl.col("_ovr").fill_null(pl.col("prediction")).alias("prediction"))
            .drop("_ovr")
        )
        assert out.height == before_rows, "an override file duplicated ids"
        print(
            f"  override {Path(path).name}: {vals.height:,} rows in file, "
            f"{matched:,} matched into the prediction"
        )

    # --- admissible bounds: the per-row gap bound, then the per-airport ceilings -------------
    gaps = (
        io.scan_ranking()
        .filter(pl.col("phase") == "DEP")
        .select(
            "mvt_id",
            (pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("gap"),
        )
        .collect(engine="streaming")
    )
    before = out["prediction"].to_numpy().copy()
    out = (
        out.join(gaps, on="mvt_id", how="left", maintain_order="left")
        .pipe(postprocess.apply_gap_bound)
        .drop("gap")
        .join(
            rank_spine.select("mvt_id", "airport"), on="mvt_id", how="left", maintain_order="left"
        )
        .pipe(postprocess.apply_airport_ceilings)
        .select("mvt_id", "prediction")
    )
    moved = int((out["prediction"].to_numpy() != before).sum())
    print(
        f"  bounds applied: gap+{postprocess.GAP_SLACK:.0f}s and {postprocess.AIRPORT_CEILINGS}"
        f" -> {moved} predictions lowered"
    )

    # --- the LOBT window, last: it is exact, so it overrides everything above ----------------
    if args.lobt_window:
        windows = (
            io.scan_ranking()
            .filter(pl.col("phase") == "DEP")
            .select(
                "mvt_id",
                (pl.col("mvt_time") - pl.col("lobt"))
                .dt.total_seconds()
                .cast(pl.Float64)
                .alias("m_l"),
            )
            .collect(engine="streaming")
        )
        before = out["prediction"].to_numpy().copy()
        out = (
            out.join(windows, on="mvt_id", how="left", maintain_order="left")
            .pipe(postprocess.apply_lobt_window)
            .select("mvt_id", "prediction")
        )
        after = out["prediction"].to_numpy()
        print(
            f"  LOBT window ±{postprocess.LOBT_WINDOW:.0f}s: {int((after > before).sum())} raised, "
            f"{int((after < before).sum())} lowered, sum of squared moves "
            f"{float(((after - before) ** 2).sum()):,.0f}"
        )

    target = Path(args.out)
    frame = postprocess.write(out, target, dry_run=args.dry_run)
    values = frame["TAXITIME_SEC_mvt"]
    print(
        f"{'validated (dry run)' if args.dry_run else 'wrote ' + str(target)}: {frame.height:,} "
        f"rows, mean {values.mean():,.1f} s, max {values.max():,} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
