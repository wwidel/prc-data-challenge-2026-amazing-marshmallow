# %% [markdown]
# # a111 — The Rome matched-row blend, as an override file
#
# Rome (LIRF) takes the r14 model verbatim, except on NM-matched rows, where a non-negative least
# squares blend of r14 and the body members does better cross-fold. The blend is fitted here on
# both folds' LIRF matched out-of-fold rows and applied to the members' twelve-month refits.
#
# v62 uses the blend fitted over a 19-column candidate set (r14 and 18 body models); NNLS gave
# every column but two a weight of exactly zero (r14 0.8946, ab07 0.1285). An NNLS solution with
# full-rank support is also the solution restricted to that support, so the two columns below
# reproduce it. Unmatched LIRF rows keep r14 (they get their own override, a125).
#
# Writes `data/external/rome_matched_blend.parquet`: (mvt_id, prediction) for the matched LIRF
# ranking rows. Pass it as the FIRST `--overrides` file, so the later overrides still win on their
# rows and the bounds still apply after all of them.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
ROME_BASE = "r14_rules_plus_unmatched_route"
MEMBERS = ["ab07_ab02_adsb_calib"]
COLS = [ROME_BASE, *MEMBERS]
#: r14 carries its own rule on matched rows with takeoff − aobt_3 > 4 h (predict that gap
#: exactly); a blend would dilute an exact value, so those rows keep r14's.
RULE_AOBT = 14_400.0
OUT = paths.DATA / "external" / "rome_matched_blend.parquet"

rec = analysis.Recorder("a111_rome_matched_blend", "Rome matched-row blend override")
rec.say(f"  columns: {', '.join(COLS)}; output {OUT.name}")
pl.Config.set_tbl_rows(40)


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


# %%
rec.section("fit the matched LIRF vector on both folds")
frame = None
for m in COLS:
    o = pl.read_parquet(newest_oof(m), glob=False).select(
        "mvt_id", "airport", "y", pl.col("yhat").alias(m)
    )
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
match = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
fit = frame.filter(pl.col("airport") == "LIRF").join(match, on="mvt_id").filter(~pl.col("unm"))
X = np.column_stack([fit[c].to_numpy().astype(float) for c in COLS])
w = nnls(X, fit["y"].to_numpy().astype(float))[0]
rec.say(f"  fitted on {fit.height:,} matched LIRF OOF rows (Jan + Jul); v62's: 26,131")
rec.show(
    pl.from_dicts(
        [{"member": c, "weight": round(float(v), 4)} for c, v in zip(COLS, w, strict=True)]
    ),
    f"weights, sum {w.sum():.4f} (v62's: r14 0.8946, ab07 0.1285)",
)

# %%
rec.section("the override file")
rank = None
for c in COLS:
    r = pl.read_parquet(paths.DATA / "rank_preds" / f"{c}.parquet", glob=False).rename({"yhat": c})
    rank = r if rank is None else rank.join(r, on="mvt_id", how="inner")
rm = (
    tio.scan_ranking()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("flight_id").is_null().alias("unm"),
        (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64).alias("pa"),
    )
    .collect(engine="streaming")
)
lirf_matched = rm.filter((pl.col("airport") == "LIRF") & ~pl.col("unm"))
rule_rows = lirf_matched.filter(pl.col("pa") > RULE_AOBT)
rec.say(
    f"  matched LIRF ranking rows {lirf_matched.height:,}; "
    f"r14 aobt-rule rows excluded {rule_rows.height:,}"
)
rank = rank.join(rm, on="mvt_id", how="inner").filter(
    (pl.col("airport") == "LIRF") & ~pl.col("unm") & ~(pl.col("pa") > RULE_AOBT).fill_null(False)
)
pred = np.column_stack([rank[c].to_numpy().astype(float) for c in COLS]) @ w
out = rank.select("mvt_id").with_columns(pl.Series("prediction", np.clip(pred, 0.0, None)))
OUT.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(OUT)
diff = pred - rank[ROME_BASE].to_numpy().astype(float)
rec.say(
    f"  wrote {OUT.relative_to(paths.ROOT)}: {out.height:,} matched LIRF ranking rows; "
    f"vs r14: mean {diff.mean():+.1f} s, mean |Δ| {np.abs(diff).mean():.1f}, "
    f"max |Δ| {np.abs(diff).max():.0f}"
)
rec.done()
