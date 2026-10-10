# %% [markdown]
# # a140 — A monotone stretch of the stack's top predictions, cross-fitted
#
# a139: a final-phase build's out-of-fold predictions are calibrated up to their 95th percentile
# (mean residual −8 to −3 s), but its top 1% sits under the labels: +167 s between the 99th and
# 99.9th percentiles, +528 s above. Does a stretch of the stack's output, fitted on one pair of
# months and applied to the other, carry over? Two forms, each fitted on rows without Rome with
# labels <= 12 h of the fitting pair:
#   hinge     — prediction + a × max(prediction − T, 0), T the fitting pair's quantile Q of the
#               prediction, a by least squares on the rows above T;
#   isotonic  — an increasing step function of the prediction (scikit-learn), clipped outside
#               the fitting range.
# The body is the stack of A140_MEMBERS (default: the BASE composition) with the weights of
# `a129` (fitted on the other pair, NM-matched / unmatched, the unmatched vector without
# A140_UNMATCHED_WITHOUT for the other months); Rome is left out (its own layer). Reading, fixed
# before running: the gate of `a129` — trimmed RMSE without Rome better in at least 3 of the 4
# months and by at least 0.3 s over the four. The untrimmed RMSE without Rome is shown beside it.
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
TRIM = 43_200.0
GROUP_MIN_ROWS = 2000
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
MEMBERS_FILE = os.environ.get("A140_MEMBERS", "")
MEMBERS = Path(MEMBERS_FILE).read_text().split() if MEMBERS_FILE else paths.composition()
HELD = os.environ.get("A140_UNMATCHED_WITHOUT", "02")
QUANTILES = (0.95, 0.99, 0.995)
GATE_MONTHS, GATE_POOLED = 3, -0.3

rec = analysis.Recorder("a140_tail_calibration", "A stretch of the stack's top predictions")
pl.Config.set_tbl_rows(40)
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
    rec.say(f"  {m}: {f.name}")
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
frame = frame.join(lab, on="mvt_id", how="inner").filter(pl.col("airport") != "LIRF")
rec.say(f"  members ({len(MEMBERS)}) from {MEMBERS_FILE or 'the BASE composition'}")
rec.say(f"  rows without Rome {frame.height:,}; unmatched vector without month {HELD}")

X = frame.select(MEMBERS).to_numpy().astype(float)
y = frame["y"].to_numpy().astype(float)
month = frame["month"].to_numpy()
mm = np.array([m[-2:] for m in month])
unm = frame["unm"].fill_null(False).to_numpy()
trim = y <= TRIM

# the stack, out of fold: weights fitted on the other pair
stack = np.full(len(y), np.nan)
for ms in PAIRS.values():
    target, fit = np.isin(month, ms), ~np.isin(month, ms)
    w_glob = nnls(X[fit], y[fit])[0]
    for g in (False, True):
        rows = fit & (unm == g)
        w = nnls(X[rows], y[rows])[0] if rows.sum() >= GROUP_MIN_ROWS else w_glob
        sel = target & (unm == g)
        stack[sel] = X[sel] @ w
        if g and HELD:
            others = rows & (mm != HELD)
            w_o = nnls(X[others], y[others])[0] if others.sum() >= GROUP_MIN_ROWS else w
            sel_o = sel & (mm != HELD)
            stack[sel_o] = X[sel_o] @ w_o
stack = np.clip(stack, 0.0, None)


# %%
def calibrated(kind: str, q: float | None) -> tuple[np.ndarray, list[str]]:
    """The stretched stack, each pair's map fitted on the other pair's trimmed rows."""
    out, notes = stack.copy(), []
    for name, ms in PAIRS.items():
        target, fit = np.isin(month, ms), ~np.isin(month, ms) & trim
        p, t = stack[fit], y[fit]
        if kind == "hinge":
            thr = float(np.quantile(p, q))
            above = p > thr
            h = p[above] - thr
            a = float(((t[above] - p[above]) * h).sum() / (h * h).sum())
            out[target] = stack[target] + a * np.clip(stack[target] - thr, 0.0, None)
            notes.append(f"{name}: T {thr:,.0f} s, a {a:+.3f} ({int(above.sum()):,} rows)")
        else:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p, t)
            out[target] = iso.predict(stack[target])
            notes.append(f"{name}: isotonic on {int(fit.sum()):,} rows")
    return out, notes


def scores(pred: np.ndarray) -> dict[str, float]:
    r = y - pred
    out = {}
    for m in sorted(set(month)):
        sel = (month == m) & trim
        out[m] = float(np.sqrt((r[sel] ** 2).mean()))
    out["four"] = float(np.sqrt((r[trim] ** 2).mean()))
    out["untrimmed"] = float(np.sqrt((r**2).mean()))
    return out


rec.section("1. the stack and its stretched versions (RMSE without Rome; months trimmed)")
base = scores(stack)
months = sorted(set(month))
rows = [{"variant": "stack", **{k: round(v, 2) for k, v in base.items()}}]
notes = {}
variants = [("hinge", q) for q in QUANTILES] + [("isotonic", None)]
for kind, q in variants:
    label = f"hinge above q{q}" if kind == "hinge" else "isotonic"
    pred, notes[label] = calibrated(kind, q)
    s = scores(pred)
    d = {k: s[k] - base[k] for k in s}
    better = sum(d[m] < 0 for m in months)
    rows.append(
        {
            "variant": label,
            **{k: round(v, 2) for k, v in s.items()},
            "Δ four": round(d["four"], 3),
            "Δ untrimmed": round(d["untrimmed"], 3),
            "months better": better,
            "gate": "PASS" if better >= GATE_MONTHS and d["four"] <= GATE_POOLED else "fail",
        }
    )
rec.show(pl.from_dicts(rows, infer_schema_length=None))
for label, ns in notes.items():
    rec.say(f"  {label}: " + "; ".join(ns))
rec.done()
