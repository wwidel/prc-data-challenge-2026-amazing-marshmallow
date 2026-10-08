"""Join the out-of-fold files of a CV run one fold at a time into one file per experiment.

A long CV can run as two jobs, one per fold of ``final_pairs`` (fold sets ``final_pairs_janjul``
and ``final_pairs_febjun``), so a spot preemption costs one fold, not two. For each name, this
takes the newest file covering each of the two folds and writes their union as
``<name>__merged.parquet``, which is then the newest file the stacker and the analyses pick.

    uv run python scripts/merge_oof.py NAME [NAME ...]
"""

from __future__ import annotations

import sys

import polars as pl

from taxiout import cv, paths


def merge(name: str) -> pl.DataFrame:
    want = [f.name for f in cv.get_folds("final_pairs")]
    files = sorted(
        (
            f
            for f in paths.PREDICTIONS.iterdir()
            if f.suffix == ".parquet"
            and f.name.split("__")[0] == name
            and not f.name.endswith("__merged.parquet")
        ),
        key=lambda f: f.stat().st_mtime,
    )
    newest: dict[str, object] = {}
    for f in files:  # oldest first, so the newest file covering a fold wins
        for fold in pl.read_parquet(f, columns=["fold"], glob=False)["fold"].unique().to_list():
            newest[fold] = f
    missing = [w for w in want if w not in newest]
    if missing:
        raise SystemExit(f"{name}: no out-of-fold file covers {missing}")
    out = pl.concat(
        [pl.read_parquet(newest[w], glob=False).filter(pl.col("fold") == w) for w in want]
    )
    if out["mvt_id"].n_unique() != out.height:
        raise SystemExit(f"{name}: duplicated rows across folds")
    return out


def main(names: list[str]) -> int:
    for name in names:
        out = merge(name)
        dest = paths.PREDICTIONS / f"{name}__merged.parquet"
        out.write_parquet(dest, compression="zstd")
        print(f"{name}: {out.height:,} rows from {out['fold'].n_unique()} folds -> {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
