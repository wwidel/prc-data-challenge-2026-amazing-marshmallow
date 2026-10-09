# %% [markdown]
# # a129 — The stack on the final phase's folds, per month, weights fitted on the other pair
#
# The final phase ranks January + July, February + June, and all four together. Our CV folds are
# those two pairs (`final_pairs`, each trained on the other ten months of 2025). This scores the
# stack the way the final file is built — NNLS with one vector for NM-matched and one for
# NM-unmatched rows, Rome left out of the fit — but with the weights fitted on the OTHER pair, so
# each score is out of sample twice: the members never saw the month, the weights never saw the
# pair.
#   1. each member alone, per month (trimmed RMSE without Rome);
#   2. the base stack (v62's seven members) per month and over the four months;
#   3. every candidate set in A129_SETS beside it, with the deltas and the gate fixed in advance:
#      better on at least 3 of the 4 months and by at least 0.3 s over the four.
# A129_SETS: "name=-member,+member;name2=+member" — changes relative to the base.
# A129_BASE: a file listing the base's members, one per line (default: the BASE composition).

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
GROUP_MIN_ROWS = 2000
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
GATE_MONTHS, GATE_POOLED = 3, -0.3
BASE_FILE = os.environ.get("A129_BASE", "")
BASE = Path(BASE_FILE).read_text().split() if BASE_FILE else paths.composition()
SPEC = os.environ.get("A129_SETS", "")

rec = analysis.Recorder("a129_stack_cv", "The stack on the final phase's folds, cross-pair weights")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)


def parse_sets(spec: str) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, mods = part.split("=", 1)
        members = list(BASE)
        for m in (x.strip() for x in mods.split(",")):
            if m.startswith("-"):
                members.remove(m[1:])
            elif m.startswith("+"):
                members.append(m[1:])
        sets[name] = members
    return sets


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
rec.section("inputs")
sets = {"base": list(BASE), **parse_sets(SPEC)}
names = sorted({m for s in sets.values() for m in s})
frame = None
for n in names:
    f = newest_oof(n)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(n))
    rec.say(f"  {n}: {f.name}, {o.height:,} rows")
    frame = o if frame is None else frame.join(o.select("mvt_id", n), on="mvt_id", how="inner")
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select("mvt_id", "month", pl.col("flight_id").is_null().alias("unm"))
    .collect(engine="streaming")
)
frame = frame.join(lab, on="mvt_id", how="inner")
months = sorted(frame["month"].unique().to_list())
rec.say(f"  base from {BASE_FILE or 'the BASE composition'} ({len(BASE)} members)")
rec.say(f"  joined rows {frame.height:,}; months {months}; base: {', '.join(BASE)}")
for k, v in sets.items():
    if k != "base":
        rec.say(f"  set {k}: {', '.join(v)}")

body = (pl.col("airport") != "LIRF") & (pl.col("y") <= TRIM)
y_all = frame["y"].to_numpy().astype(float)
unm_all = frame["unm"].fill_null(False).to_numpy()
month_all = frame["month"].to_numpy()
not_lirf = (frame["airport"] != "LIRF").to_numpy()


def per_month(col: str) -> dict[str, float]:
    b = frame.filter(body)
    out = {
        r["month"]: r["rmse"]
        for r in b.group_by("month")
        .agg(((pl.col("y") - pl.col(col)) ** 2).mean().sqrt().alias("rmse"))
        .iter_rows(named=True)
    }
    out["four"] = float(np.sqrt(((b["y"] - b[col]) ** 2).mean()))
    for pair, ms in PAIRS.items():
        p = b.filter(pl.col("month").is_in(list(ms)))
        out[pair] = float(np.sqrt(((p["y"] - p[col]) ** 2).mean()))
    return out


def stack(members: list[str]) -> np.ndarray:
    X = frame.select(members).to_numpy().astype(float)
    pred = np.full(frame.height, np.nan)
    for ms in PAIRS.values():
        target = np.isin(month_all, ms)
        fit = ~target & not_lirf
        w_glob = nnls(X[fit], y_all[fit])[0]
        for g in (False, True):
            rows = fit & (unm_all == g)
            w = nnls(X[rows], y_all[rows])[0] if rows.sum() >= GROUP_MIN_ROWS else w_glob
            sel = target & (unm_all == g)
            pred[sel] = X[sel] @ w
    return pred


# %%
rec.section("1. each member alone (trimmed RMSE without Rome)")
cols = [*months, "Jan+Jul", "Feb+Jun", "four"]
rec.show(
    pl.from_dicts(
        [{"member": n, **{c: round(per_month(n)[c], 2) for c in cols}} for n in names]
    ).sort("four"),
)

# %%
rec.section("2–3. the stack, weights fitted on the other pair")
scores = {}
for k, members in sets.items():
    frame = frame.with_columns(pl.Series(f"stack_{k}", stack(members)))
    scores[k] = per_month(f"stack_{k}")
rows = []
for k, s in scores.items():
    d = {c: s[c] - scores["base"][c] for c in cols}
    better = sum(d[m] < 0 for m in months)
    rows.append(
        {
            "set": k,
            **{c: round(s[c], 2) for c in cols},
            **{f"Δ {c}": round(d[c], 3) for c in ("Jan+Jul", "Feb+Jun", "four")},
            "months better": better,
            "gate": "—"
            if k == "base"
            else ("PASS" if better >= GATE_MONTHS and d["four"] <= GATE_POOLED else "fail"),
        }
    )
rec.show(
    pl.from_dicts(rows), f"gate: at least {GATE_MONTHS} months better and Δ four <= {GATE_POOLED}"
)

# %%
rec.section("4. weights fitted on all four months (what the final build would use), per set")
for k, members in sets.items():
    X = frame.select(members).to_numpy().astype(float)
    fit = not_lirf
    lines = []
    for g in (False, True):
        r = fit & (unm_all == g)
        w = nnls(X[r], y_all[r])[0]
        lines.append(
            f"{'unmatched' if g else 'matched'}: "
            + ", ".join(f"{m}={v:.3f}" for m, v in zip(members, w, strict=True) if v > 0)
        )
    rec.say(f"  {k}: " + " | ".join(lines))
rec.done()
