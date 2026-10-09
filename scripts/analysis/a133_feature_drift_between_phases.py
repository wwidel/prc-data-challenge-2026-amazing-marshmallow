# %% [markdown]
# # a133 — The cached feature blocks of the two phases, column by column
#
# a132: each member's final-phase refit differs from its leaderboard-phase refit by ~31 s RMS on
# NM-matched January and July 2026 rows, and the organisers' ±12 s on takeoff times explains only
# 2–4% of that variance. The training months are the same in both phases. So: which cached
# blocks differ between the two feature stores, on 2025 rows (what the refits train on) and on
# the shared January and July 2026 rows (what they predict)? Per block and month: rows compared,
# and per column the share of rows that differ (beyond 1e-4 relative, or null in one store only)
# and the RMS of the difference.
# The leaderboard-phase store is A133_BOARD_DATA (a data/ directory with features/ and
# interim/spine/; default this checkout's); the final-phase store is this checkout's under
# TAXIOUT_PHASE=final. Label-dependent blocks are recomputed per fit and not cached, so they are
# not compared here. Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from taxiout import analysis, paths, store  # noqa: E402
from taxiout.experiment import load_all  # noqa: E402
from taxiout.features import registry  # noqa: E402

# %% knobs
BOARD_DATA = Path(os.environ.get("A133_BOARD_DATA", str(paths.DATA)))
MONTHS = tuple(os.environ.get("A133_MONTHS", "2025-01,2025-07,2026-01,2026-07").split(","))
REL_TOL = 1e-4
TOP = 3  # columns listed per block

rec = analysis.Recorder("a133_feature_drift_between_phases", "Feature stores of the two phases")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(90)

# %%
rec.section("inputs")
if not paths.FINAL:
    raise SystemExit("run under TAXIOUT_PHASE=final: the final-phase store is this checkout's")
members = paths.composition()
known = {e.name: e for e in load_all(str(paths.EXPERIMENTS))}
blocks = sorted({b for m in members for b in known[m].features})
cached = [registry.get(b) for b in blocks if registry.get(b).kind.cacheable]
rec.say(f"  members ({len(members)}): {', '.join(members)}")
rec.say(f"  blocks they read: {len(blocks)}; cached (compared here): {len(cached)}")
rec.say(f"  leaderboard store: {BOARD_DATA}; final store: {paths.FEATURES}; months {MONTHS}")


def board_dir(name: str) -> Path | None:
    """The leaderboard store's newest hash directory for a block holding every month."""
    root = BOARD_DATA / "features" / name
    if not root.is_dir():
        return None
    full = [d for d in root.iterdir() if all((d / f"{m}.parquet").exists() for m in MONTHS)]
    return max(full, key=lambda d: d.stat().st_mtime) if full else None


def frame(spine_dir: Path, block_dir: Path, month: str) -> pl.DataFrame:
    ids = pl.read_parquet(spine_dir / f"{month}.parquet", columns=["mvt_id"], glob=False)
    values = pl.read_parquet(block_dir / f"{month}.parquet", glob=False)
    if values.height != ids.height:
        raise SystemExit(f"{block_dir}/{month}: {values.height} rows, spine {ids.height}")
    return pl.concat([ids, values], how="horizontal")


# %%
rec.section("per block and month")
summary, detail = [], []
for block in cached:
    bdir, fdir = board_dir(block.name), store.block_dir(block)
    if bdir is None or not all((fdir / f"{m}.parquet").exists() for m in MONTHS):
        rec.say(f"  {block.name}: not in both stores for every month, skipped")
        continue
    same_hash = bdir.name == fdir.name
    for month in MONTHS:
        a = frame(BOARD_DATA / "interim" / "spine", bdir, month)
        b = frame(paths.SPINE, fdir, month)
        cols = [c for c in a.columns if c != "mvt_id" and c in b.columns]
        j = a.select("mvt_id", *cols).join(
            b.select("mvt_id", *cols), on="mvt_id", how="inner", suffix="__f"
        )
        worst = []
        for c in cols:
            x, y = pl.col(c).cast(pl.Float64), pl.col(f"{c}__f").cast(pl.Float64)
            diff = (x.is_null() != y.is_null()) | (
                x.is_not_null() & y.is_not_null() & ((x - y).abs() > REL_TOL * (1 + x.abs()))
            )
            s = j.select(
                diff.mean().alias("share"),
                ((x - y) ** 2).mean().sqrt().alias("rms"),
            ).row(0, named=True)
            worst.append((s["share"] or 0.0, s["rms"], c))
        worst.sort(reverse=True)
        summary.append(
            {
                "block": block.name,
                "same hash": same_hash,
                "month": month,
                "rows": j.height,
                "columns": len(cols),
                "columns differing": sum(1 for w in worst if w[0] > 0),
                "max share": round(worst[0][0], 4) if worst else 0.0,
                "worst column": worst[0][2] if worst else "",
            }
        )
        for share, rms, c in worst[:TOP]:
            if share > 0:
                detail.append(
                    {
                        "block": block.name,
                        "month": month,
                        "column": c,
                        "share differing": round(share, 4),
                        "rms Δ": None if rms is None else round(rms, 3),
                    }
                )
rec.show(pl.from_dicts(summary), "share of rows differing in the block's worst column")
if detail:
    rec.show(pl.from_dicts(detail), f"the {TOP} columns that differ most, per block and month")
rec.done()
