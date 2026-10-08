"""Independent check of a built prediction file. Exits non-zero on any fault.

Structural validation already happens inside `build_stack.py`, and it is not enough: a file
missing an entire override layer is perfectly well-formed. One of our builds passed every
structural check and would have scored ~317 instead of 281.98, because the override files were
omitted. What caught it was a *bound*: the maximum prediction was 20,000, which is impossible for
a correct build.

    uv run python scripts/verify_stack.py data/output/v62.parquet [--overrides FILE ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from taxiout import io, paths

#: The one-row override predicts 86,340 s (that row's takeoff − aobt_3) and the pop27 rows take
#: schedule gaps near 87,500 s; a correct v62 build has a maximum of 111,654. A maximum at or below
#: this means an override layer did not land.
MIN_EXPECTED_MAX = 86_340.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="the prediction parquet written by build_stack.py")
    ap.add_argument(
        "--overrides",
        action="append",
        default=[],
        help="override parquet to join back and require every row of (repeatable)",
    )
    ap.add_argument(
        "--min-max",
        type=float,
        default=MIN_EXPECTED_MAX,
        help="fail if the maximum prediction is at or below this (default 86,340)",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    sub = pl.read_parquet(path, glob=False)
    truth = pl.read_parquet(paths.RAW / "submitting.parquet", glob=False)
    problems: list[str] = []

    print(f"  {path.name}: {sub.height:,} rows, columns {sub.columns}")
    if sub.columns != truth.columns:
        problems.append(f"columns {sub.columns} != {truth.columns}")
    if [str(d) for d in sub.dtypes] != [str(d) for d in truth.dtypes]:
        got, exp = [str(d) for d in sub.dtypes], [str(d) for d in truth.dtypes]
        problems.append(f"dtypes {got} != {exp}")
    if sub.height != truth.height:
        problems.append(f"row count {sub.height} != {truth.height}")

    ids, want = set(sub["MVT_ID_mvt"].to_list()), set(truth["MVT_ID_mvt"].to_list())
    if len(ids) != sub.height:
        problems.append(f"{sub.height - len(ids)} duplicate ids")
    if want - ids:
        problems.append(f"{len(want - ids)} ids missing")
    if ids - want:
        problems.append(f"{len(ids - want)} ids not in the ranking set")

    # A third, independent count.
    n_dep = (
        io.scan_ranking()
        .filter(pl.col("phase") == "DEP")
        .select(pl.len())
        .collect(engine="streaming")
    ).item()
    if n_dep != sub.height:
        problems.append(f"row count {sub.height} != ranking DEP {n_dep}")

    pred = sub["TAXITIME_SEC_mvt"]
    if pred.null_count():
        problems.append(f"{pred.null_count()} null predictions")
    if int((pred < 0).sum()):
        problems.append(f"{int((pred < 0).sum())} negative predictions")
    print(f"  prediction: min {pred.min():,.0f}  mean {pred.mean():,.1f}  max {pred.max():,.0f}")
    if float(pred.max()) <= args.min_max:
        problems.append(
            f"maximum {pred.max():,.0f} is not above {args.min_max:,.0f}, so an override layer "
            "did not land"
        )

    for spec in args.overrides:
        ov = pl.read_parquet(spec, glob=False)
        col = next(c for c in ov.columns if c != "mvt_id")
        j = sub.select(
            pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("shipped"),
        ).join(ov.select("mvt_id", pl.col(col).alias("intended")), on="mvt_id", how="inner")
        if j.height != ov.height:
            problems.append(f"{spec}: {ov.height - j.height} of {ov.height} rows did not land")
        d = j["shipped"] - j["intended"]
        # Bounds run AFTER the overrides, so shipped BELOW intended is legitimate; above is not.
        above = int((d > 1.0).sum())
        lowered = int((d < -1.0).sum())
        print(
            f"  {spec.split('/')[-1]:24s} {j.height:4d} joined, {j.height - above - lowered:4d} "
            f"exact, {lowered:3d} lowered by the bounds, {above:3d} ABOVE their override"
        )
        if above:
            problems.append(f"{spec}: {above} rows ship above their override value")

    if problems:
        print("\n  FAILED:", *(f"\n    - {p}" for p in problems), sep="")
        return 1
    print("\n  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
