"""Refit one recorded experiment on all twelve 2025 months and save its ranking predictions.

Thin CLI over the tested pieces (`cv.build_matrix`, `models.factory`) — the member-refit step
of a stacked submission, where the combiner needs each member's predictions for the ranking
rows rather than a finished submission file.

    uv run python scripts/refit_predict.py <experiment-name> [...]

Writes data/rank_preds/<experiment>.parquet with columns mvt_id, yhat.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from taxiout import cv, models, paths, profiles
from taxiout.experiment import load_all


def catalogue() -> dict:
    """Every experiment by *name*, grid expansions included.

    A grid config expands into names like ``t01_leaf_sweep[min_data_in_leaf=5,...]`` that have
    no YAML file of their own, so resolving by filename cannot find them.
    """
    return {e.name: e for e in load_all(str(paths.EXPERIMENTS))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+", help="experiment names (config/experiments/<n>.yaml)")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    out_dir = paths.DATA / "rank_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = profiles.load(args.profile) if args.profile else None

    known = catalogue()
    for name in args.experiments:
        out = out_dir / f"{name}.parquet"
        if out.exists():
            print(f"{name}: already on disk, skipping")
            continue
        exp = known.get(name)
        if exp is None:
            print(f"{name}: no experiment with this name in config/experiments/", file=sys.stderr)
            return 1
        months = tuple(sorted(set(paths.TRAIN_MONTHS) | set(paths.RANK_MONTHS)))
        matrix = cv.build_matrix(exp.features, months, train_months=paths.TRAIN_MONTHS)
        is_train = matrix["month"].is_in(list(paths.TRAIN_MONTHS))
        train, rank = matrix.filter(is_train), matrix.filter(~is_train)
        params = exp.resolved_params(profile) if profile else exp.model_params
        model = models.factory(exp.model_kind, params)()
        # cv.fit_and_predict, not a bare fit: the label and target treatments must match the
        # ones the experiment was cross-validated with (see its docstring for the bug this
        # replaced -- capped models used to refit on raw labels).
        predicted = cv.fit_and_predict(
            model,
            train,
            rank,
            train_label_cap=exp.train_label_cap,
            target_transform=exp.target_transform,
            train_relabel=exp.train_relabel,
            clip=exp.clip,
        )
        pl.DataFrame({"mvt_id": rank["mvt_id"], "yhat": predicted}).write_parquet(out)
        print(f"{name}: wrote {out.name} ({rank.height:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
