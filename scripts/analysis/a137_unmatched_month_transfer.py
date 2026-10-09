# %% [markdown]
# # a137 — The NM-unmatched vector, month by month: how well does its level carry over?
#
# a135: the unmatched rows' NNLS vector fitted on the four months is cb03 × 1.151, pulled up by
# February 2025 (every member under-predicts February's unmatched rows by 430–710 s); fitted on
# Jan+Jul alone it sums to 1.047. Which way of fitting it holds on a month it has not seen? On the
# four months' out-of-fold predictions, each month is held out in turn. The matched rows' vector is
# fitted on the other three months and is the same for every option; the unmatched rows take:
#   pooled         — the unmatched vector fitted on the other three months (F0's and F1's rule);
#   pooled ≤ 12 h  — the same, on labels of at most 12 h;
#   without Feb    — fitted on the other months except February (for February: the other three);
#   same pair      — fitted on the other month of the held-out month's pair (Jan–Jul, Feb–Jun);
#   no group       — the matched rows' vector (no separate unmatched vector);
#   in sample      — fitted on the held-out month itself (a floor no rule can reach).
# Per month and option: the unmatched rows' RMSE (all labels and trimmed), their mean residual,
# the vector's sum, and the month's trimmed RMSE without Rome — the number the stack is judged by —
# with its change against "pooled"; then the four months together. Members: A137_MEMBERS (a file,
# one member per line; default the BASE composition). Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
TRIM = 43_200.0
MONTHS = ("2025-01", "2025-02", "2025-06", "2025-07")
PAIR = {"2025-01": "2025-07", "2025-07": "2025-01", "2025-02": "2025-06", "2025-06": "2025-02"}
FEB = "2025-02"
MEMBERS_FILE = os.environ.get("A137_MEMBERS", "")
MEMBERS = Path(MEMBERS_FILE).read_text().split() if MEMBERS_FILE else paths.composition()

rec = analysis.Recorder("a137_unmatched_month_transfer", "The unmatched vector, month by month")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)


def newest_oof(name: str) -> Path:
    hits = [f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == name]
    if not hits:
        raise FileNotFoundError(f"no out-of-fold file for {name}")
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
rec.section("inputs")
frame = None
for m in MEMBERS:
    f = newest_oof(m)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(m))
    rec.say(f"  {m}: {f.name}, {o.height:,} rows")
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
d = frame.join(lab, on="mvt_id", how="inner").filter(pl.col("airport") != "LIRF")
rec.say(f"  members ({len(MEMBERS)}) from {MEMBERS_FILE or 'the BASE composition'}")
rec.say(f"  non-Rome out-of-fold rows {d.height:,}; NM-unmatched {int(d['unm'].sum()):,}")

X = d.select(MEMBERS).to_numpy().astype(float)
y = d["y"].to_numpy().astype(float)
month = d["month"].to_numpy()
unm = d["unm"].fill_null(False).to_numpy()
trim = y <= TRIM


def fit(rows: np.ndarray) -> np.ndarray:
    return nnls(X[rows], y[rows])[0]


# %%
rec.section("1. each month held out")
rows, sse = [], {}
for m in MONTHS:
    held, other = month == m, month != m
    w_matched = fit(other & ~unm)
    base = X @ w_matched
    options = {
        "pooled": other & unm,
        "pooled ≤ 12 h": other & unm & trim,
        "without Feb": other & unm & (month != FEB),
        "same pair": (month == PAIR[m]) & unm,
        "no group": None,
        "in sample": held & unm,
    }
    for name, fit_rows in options.items():
        w = w_matched if fit_rows is None else fit(fit_rows)
        pred = base.copy()
        hu = held & unm
        pred[hu] = X[hu] @ w
        r = y[hu] - pred[hu]
        rt = r[trim[hu]]
        sel = held & trim
        month_sse = float(((y[sel] - pred[sel]) ** 2).sum())
        sse[(name, m)] = (month_sse, int(sel.sum()))
        rows.append(
            {
                "held out": m,
                "option": name,
                "fit rows": int(hu.sum()) if fit_rows is None else int(fit_rows.sum()),
                "vector sum": round(float(w.sum()), 3),
                "unmatched rmse": round(float(np.sqrt((r**2).mean())), 1),
                "unmatched rmse, trimmed": round(float(np.sqrt((rt**2).mean())), 1),
                "mean residual": round(float(r.mean()), 1),
                "mean residual, trimmed": round(float(rt.mean()), 1),
                "month trimmed rmse": round(float(np.sqrt(month_sse / sel.sum())), 3),
            }
        )
t = pl.from_dicts(rows)
t = t.join(
    t.filter(pl.col("option") == "pooled").select(
        "held out", pl.col("month trimmed rmse").alias("_p")
    ),
    on="held out",
).with_columns((pl.col("month trimmed rmse") - pl.col("_p")).round(3).alias("Δ vs pooled"))
rec.show(t.drop("_p"), "unmatched rows of the held-out month; month = all its non-Rome rows ≤ 12 h")

# %%
rec.section("2. the four months together")
four = []
for name in dict.fromkeys(k[0] for k in sse):
    s = sum(sse[(name, m)][0] for m in MONTHS)
    n = sum(sse[(name, m)][1] for m in MONTHS)
    four.append({"option": name, "four-month trimmed rmse": round(float(np.sqrt(s / n)), 3)})
f4 = pl.from_dicts(four)
p = f4.filter(pl.col("option") == "pooled")["four-month trimmed rmse"].item()
rec.show(f4.with_columns((pl.col("four-month trimmed rmse") - p).round(3).alias("Δ vs pooled")))
rec.done()
