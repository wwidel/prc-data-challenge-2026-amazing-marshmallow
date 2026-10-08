"""The canonical spine: one row per scored-or-scoreable departure, in a fixed order.

Everything in the feature store is stored *aligned to this row order*, which is what lets an
experiment matrix be assembled by horizontal concatenation instead of a join. The order is an
invariant: once the spine is built, rebuilding it with different contents invalidates every
cached feature. Never append to it — rebuild.

Layout: ``data/interim/spine/<month>.parquet``, one file per month, rows sorted by
``(airport, mvt_time, mvt_id)``. ``mvt_id`` is unique, so the order is total and deterministic.
"""

from __future__ import annotations

import polars as pl

from taxiout import io, paths

#: Columns carried by the spine itself. Features supply everything else.
SPINE_COLUMNS: tuple[str, ...] = ("mvt_id", "airport", "month", "split", "mvt_time", "taxi_sec")

#: The sort key that defines the immutable row order.
SORT_KEY: tuple[str, ...] = ("airport", "mvt_time", "mvt_id")


def _departures() -> pl.LazyFrame:
    return (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select(
            "mvt_id",
            "airport",
            "month",
            "split",
            "mvt_time",
            pl.col("taxi_sec").cast(pl.Int32),
        )
    )


def build(months: tuple[str, ...] = paths.ALL_MONTHS, *, overwrite: bool = False) -> dict[str, int]:
    """Build the spine, one parquet per month. Returns ``{month: row_count}``."""
    paths.ensure_dirs()
    deps = _departures()
    heights: dict[str, int] = {}
    for month in months:
        out = paths.SPINE / f"{month}.parquet"
        if out.exists() and not overwrite:
            heights[month] = pl.scan_parquet(out).select(pl.len()).collect().item()
            continue
        frame = (
            deps.filter(pl.col("month") == month)
            .sort(SORT_KEY)
            .collect(engine="streaming")
            .select(SPINE_COLUMNS)
        )
        frame.write_parquet(out, compression="zstd")
        heights[month] = frame.height
    return heights


def months_available() -> tuple[str, ...]:
    """Months for which a spine file exists, in chronological order."""
    return tuple(sorted(p.stem for p in paths.SPINE.glob("*.parquet")))


def scan(months: tuple[str, ...] | None = None) -> pl.LazyFrame:
    """Lazily scan the spine for the given months, in chronological month order."""
    months = months or months_available()
    missing = [m for m in months if not (paths.SPINE / f"{m}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"spine missing for {missing}; run `uv run python scripts/build_spine.py`"
        )
    return pl.concat(
        [pl.scan_parquet(paths.SPINE / f"{m}.parquet") for m in sorted(months)], how="vertical"
    )


def load(months: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Materialise the spine. ~2.4M rows over 6 narrow columns — cheap."""
    return scan(months).collect()


def heights(months: tuple[str, ...] | None = None) -> dict[str, int]:
    """Row count per month, used to validate that a feature block is aligned."""
    months = months or months_available()
    return {
        m: pl.scan_parquet(paths.SPINE / f"{m}.parquet").select(pl.len()).collect().item()
        for m in sorted(months)
    }
