# %% [markdown]
# # a138 — Two pulls of the open data, file by file
#
# The pull scripts are the only way open data enters this repository, and the sources are live
# (REPRODUCE.md, "Why a rebuild will be close"). This compares a second pull with the one a build
# read: for every file under B's `data/external/`, its counterpart under A's, by rows, columns, and
# the rows found in one pull only. METAR differences are also counted per month, to show whether
# a source changed history or only its recent months.
# A138_A: the data/external directory the build read (default: this checkout's); A138_B: the new
# pull's data/external directory (required). Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from taxiout import analysis, paths  # noqa: E402

# %% knobs
A = Path(os.environ.get("A138_A", str(paths.DATA / "external")))
B = Path(os.environ["A138_B"])
GLOBS = (
    "metar/metar.parquet",
    "gateway/stands.parquet",
    "gateway/runways.parquet",
    "osm/osm_stands.parquet",
    "eurocontrol/*.csv",
    "opdi/flight_list_*.parquet",
    "adsb/cut/*.parquet",
)

rec = analysis.Recorder("a138_compare_pulls", "Two pulls of the open data, file by file")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(10)
pl.Config.set_fmt_str_lengths(60)


def load(f: Path) -> pl.DataFrame:
    if f.suffix == ".csv":
        return pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy")
    return pl.read_parquet(f, glob=False)


# %%
rec.section("inputs")
rec.say(f"  A (the build's): {A}")
rec.say(f"  B (the new pull): {B}")

# %%
rec.section("1. file by file")
rows, metar = [], None
for pattern in GLOBS:
    for fb in sorted(B.glob(pattern)):
        rel = fb.relative_to(B)
        fa = A / rel
        if not fa.exists():
            rows.append({"file": str(rel), "rows A": None, "rows B": None, "note": "not in A"})
            continue
        a, b = load(fa), load(fb)
        same_cols = a.columns == b.columns
        if same_cols and a.schema == b.schema:
            only_a = a.join(b, on=a.columns, how="anti", nulls_equal=True).height
            only_b = b.join(a, on=a.columns, how="anti", nulls_equal=True).height
            note = "identical" if only_a == only_b == 0 and a.height == b.height else ""
        else:
            only_a = only_b = None
            note = "columns or types differ"
        rows.append(
            {
                "file": str(rel),
                "rows A": a.height,
                "rows B": b.height,
                "only in A": only_a,
                "only in B": only_b,
                "note": note,
            }
        )
        if str(rel) == "metar/metar.parquet" and same_cols:
            metar = (a, b)
rec.show(
    pl.from_dicts(rows, infer_schema_length=None),
    "rows only in A / only in B: anti-joins on all columns",
)

# %%
if metar is not None:
    rec.section("2. METAR: rows found in one pull only, per month")
    a, b = metar
    month = pl.col("time").dt.strftime("%Y-%m").alias("month")
    only_a = a.join(b, on=a.columns, how="anti", nulls_equal=True).group_by(month).len("only in A")
    only_b = b.join(a, on=a.columns, how="anti", nulls_equal=True).group_by(month).len("only in B")
    rec.show(only_a.join(only_b, on="month", how="full", coalesce=True).sort("month"))
rec.done()
