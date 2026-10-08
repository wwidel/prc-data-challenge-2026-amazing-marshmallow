# %% [markdown]
# # a130 — Members alone, per month: trimmed RMSE without Rome on the final phase's folds
#
# The quick read for one change at a time: each listed member's out-of-fold predictions scored per
# held-out month (trimmed RMSE without Rome), with deltas against the first member listed, which
# is the baseline. A130_MEMBERS: comma-separated experiment names, baseline first. Numbers only;
# a member's gate is read in the stack (a129).

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
MEMBERS = [m for m in os.environ.get("A130_MEMBERS", "").split(",") if m]
TRIM = 43_200.0

rec = analysis.Recorder("a130_member_months", "Members alone, per month")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)
if len(MEMBERS) < 1:
    raise SystemExit("set A130_MEMBERS=baseline,candidate[,candidate...]")


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    if not hits:
        raise FileNotFoundError(f"no out-of-fold file for {name} in {paths.PREDICTIONS}")
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
rows = []
for m in MEMBERS:
    f = newest_oof(m)
    o = (
        pl.read_parquet(f, glob=False)
        .select("mvt_id", "airport", "y", "yhat")
        .join(lab, on="mvt_id")
    )
    b = o.filter((pl.col("airport") != "LIRF") & (pl.col("y") <= TRIM))
    r = {"member": m, "file": f.name.split("__")[1][:12]}
    for month, g in sorted(b.group_by("month"), key=lambda kv: kv[0][0]):
        r[month[0]] = float(np.sqrt(((g["y"] - g["yhat"]) ** 2).mean()))
    r["four"] = float(np.sqrt(((b["y"] - b["yhat"]) ** 2).mean()))
    um = b.filter(pl.col("unm"))
    r["unmatched"] = float(np.sqrt(((um["y"] - um["yhat"]) ** 2).mean())) if um.height else None
    rows.append(r)
t = pl.from_dicts(rows)
num = [c for c in t.columns if c not in ("member", "file")]
rec.show(t.with_columns(pl.col(c).round(2) for c in num), "trimmed RMSE without Rome")
base = rows[0]
rec.show(
    pl.from_dicts(
        [
            {
                "member": r["member"],
                **{c: round(r[c] - base[c], 3) for c in num if r[c] is not None},
            }
            for r in rows[1:]
        ]
    ),
    f"Δ against {base['member']}",
)
rec.done()
