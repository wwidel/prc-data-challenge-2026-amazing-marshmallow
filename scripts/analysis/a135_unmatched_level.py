# %% [markdown]
# # a135 — The NM-unmatched rows: their weights and their level, per pair
#
# F0 (the final build on v62's members) gives the NM-unmatched rows one NNLS vector fitted on
# the four months' out-of-fold predictions: cb03 alone, weight 1.151. On the 4,900 unmatched
# January and July 2026 rows outside Rome that puts F0 +110.65 s above v62 on average (a132),
# against +2.07 s on matched rows. The fit uses every label, as the build does; our CV scores
# are trimmed at 12 h, the leaderboard's are not, and the unmatched rows hold most labels above
# 4 h. On the unmatched non-Rome out-of-fold rows of the four months:
#   1. rows per month, labels above 4 h and 12 h, and each member's mean residual and RMSE
#      (all labels and trimmed);
#   2. the unmatched NNLS vector fitted on Jan+Jul, on Feb+Jun and on all four, with all labels
#      and with labels <= 12 h: weights, their sum, the fitted level;
#   3. each fit scored on the other pair (and the four-month fits in sample): RMSE with all
#      labels and trimmed, and the mean residual.
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
TRIM = 43_200.0
LONG = 14_400.0
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
MEMBERS = paths.composition()

rec = analysis.Recorder("a135_unmatched_level", "The NM-unmatched rows: weights and level")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)


def newest_oof(name: str):
    hits = [f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == name]
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
rec.section("inputs")
frame = None
for m in MEMBERS:
    f = newest_oof(m)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(m))
    rec.say(f"  {m}: {f.name}")
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
u = frame.join(lab, on="mvt_id").filter(pl.col("unm") & (pl.col("airport") != "LIRF"))
rec.say(f"  members ({len(MEMBERS)}): {', '.join(MEMBERS)}")
rec.say(f"  unmatched non-Rome out-of-fold rows: {u.height:,}")

# %%
rec.section("1. rows and members, per month")
rec.show(
    u.group_by("month")
    .agg(
        pl.len().alias("rows"),
        (pl.col("y") > LONG).sum().alias("labels > 4 h"),
        (pl.col("y") > TRIM).sum().alias("labels > 12 h"),
        pl.col("y").mean().round(1).alias("mean label"),
        pl.col("y").filter(pl.col("y") <= TRIM).mean().round(1).alias("mean label, trimmed"),
    )
    .sort("month")
)
rows = []
for m in MEMBERS:
    for month, g in sorted(u.group_by("month"), key=lambda t: t[0][0]):
        r = (g["y"] - g[m]).to_numpy()
        t = g.filter(pl.col("y") <= TRIM)
        rt = (t["y"] - t[m]).to_numpy()
        rows.append(
            {
                "member": m,
                "month": month[0],
                "mean residual": round(float(r.mean()), 1),
                "rmse": round(float(np.sqrt((r**2).mean())), 1),
                "mean residual, trimmed": round(float(rt.mean()), 1),
                "rmse, trimmed": round(float(np.sqrt((rt**2).mean())), 1),
            }
        )
rec.show(pl.from_dicts(rows), "residual = label − prediction")

# %%
rec.section("2–3. the unmatched vector, fitted per pair and on all four, scored on the other pair")
X = u.select(MEMBERS).to_numpy().astype(float)
y = u["y"].to_numpy().astype(float)
month = u["month"].to_numpy()
trim = y <= TRIM
fits = {"Jan+Jul": np.isin(month, PAIRS["Jan+Jul"]), "Feb+Jun": np.isin(month, PAIRS["Feb+Jun"])}
fits["four"] = np.ones_like(trim)
out, weights = [], []
for fit_name, on in fits.items():
    for labels, keep in (("all labels", np.ones_like(trim)), ("labels <= 12 h", trim)):
        w = nnls(X[on & keep], y[on & keep])[0]
        weights.append(
            {
                "fitted on": fit_name,
                "labels": labels,
                "rows": int((on & keep).sum()),
                "weights": ", ".join(f"{m}={v:.3f}" for m, v in zip(MEMBERS, w, strict=True) if v),
                "sum": round(float(w.sum()), 3),
            }
        )
        for score_name, ms in (*PAIRS.items(), ("four", None)):
            sel = np.ones_like(trim) if ms is None else np.isin(month, ms)
            pred = X[sel] @ w
            r, rt = y[sel] - pred, (y[sel] - pred)[trim[sel]]
            out.append(
                {
                    "fitted on": fit_name,
                    "labels": labels,
                    "scored on": score_name,
                    "in sample": score_name == fit_name
                    or fit_name == "four"
                    or (score_name == "four"),
                    "rmse": round(float(np.sqrt((r**2).mean())), 1),
                    "rmse, trimmed": round(float(np.sqrt((rt**2).mean())), 1),
                    "mean residual": round(float(r.mean()), 1),
                    "mean residual, trimmed": round(float(rt.mean()), 1),
                }
            )
rec.show(pl.from_dicts(weights), "the vectors")
rec.show(pl.from_dicts(out), "scores (in sample = the fit saw some of the scored rows)")
rec.done()
