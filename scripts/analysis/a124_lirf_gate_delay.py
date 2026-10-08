# %% [markdown]
# # a124 — LIRF NM-unmatched rows: the gate-delay member (lg01) against r14, cross-fold
#
# Team OpenAir's Rome steps (https://github.com/ChandanHegde07/OpenAir) moved their board only as
# *soft* blends and reversed as hard swaps. So the question is not "is lg01 better" but "which
# blend weight, chosen on one month, still wins on the other":
#   1. out-of-fold RMSE on LIRF NM-unmatched rows, Jan and Jul 2025, of r14, lg01 and
#      T = (1 − λ) r14 + λ lg01 for λ on a grid; λ picked on one month and scored on the other;
#      OpenAir's λ = 0.25 shown as a fixed reference, decided before running;
#   2. the long-hold rule (OPDI ground time ≥ LONG_HOLD: pull halfway to the long-hold median
#      label, the median learned on the other month), on top of the chosen blend;
#   3. rows with a gap above GAP_COPY are reported apart: the submission serves them the gap
#      (`pop27_gap`), so neither member reaches them.
# A Rome-layer candidate is built only from a blend that wins on both months (a125).

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, store  # noqa: E402

# %% knobs
ROME = "r14_rules_plus_unmatched_route"
GATE = "lg01_lirf_gate_delay"
LAMBDAS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
LAMBDA_REF = 0.25  # OpenAir's v21–v24 weight, fixed in advance
GAP_COPY = 25_000.0  # above this the submission serves the gap (pop27_gap)
LONG_HOLD = 7_200.0  # OpenAir's long-hold threshold
PULL = 0.5
TRIM = 43_200.0

rec = analysis.Recorder("a124_lirf_gate_delay", "LIRF unmatched: lg01 vs r14, cross-fold")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


# %%
rec.section("inputs")
months = tuple(sorted(f.stem for f in paths.SPINE.glob("*.parquet")))
op = store.assemble(["opdi"], months).select("mvt_id", "op_turn_s", "op_found").collect()
lab = (
    tio.scan_training()
    .filter((pl.col("phase") == "DEP") & (pl.col("airport") == "LIRF"))
    .select(
        "mvt_id",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        pl.col("flight_id").is_null().alias("unm"),
        pl.col("taxi_sec").cast(pl.Float64).alias("y"),
        (pl.col("mvt_time") - pl.col("sched_time"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("gap"),
    )
    .collect(engine="streaming")
)
fr, fg = newest_oof(ROME), newest_oof(GATE)
rec.say(f"  {ROME}: {fr.name}; {GATE}: {fg.name}")
d = (
    lab.join(
        pl.read_parquet(fr, glob=False).select("mvt_id", pl.col("yhat").alias("r14")), on="mvt_id"
    )
    .join(pl.read_parquet(fg, glob=False).select("mvt_id", pl.col("yhat").alias("lg")), on="mvt_id")
    .join(op, on="mvt_id", how="left")
    .with_columns(pl.col("lg").clip(0))
)
u = d.filter(pl.col("unm"))
served = u.filter(pl.col("gap") <= GAP_COPY)
rec.say(
    f"  LIRF departures with both OOFs: {d.height:,}; NM-unmatched {u.height}; of which gap ≤ "
    f"{GAP_COPY:.0f}: {served.height} (the rows a Rome-layer change would reach); OPDI ground "
    f"time on {served['op_turn_s'].is_not_null().sum()} of them"
)

# %%
rec.section("1. RMSE on LIRF NM-unmatched rows (gap ≤ GAP_COPY), per month")
rows = []
for m in sorted(served["month"].unique().to_list()):
    s = served.filter(pl.col("month") == m)
    y, a, b = s["y"].to_numpy(), s["r14"].to_numpy(), s["lg"].to_numpy()
    keep = y <= TRIM
    row = {"month": m, "rows": s.height, "r14": round(rmse(y, a), 1), "lg01": round(rmse(y, b), 1)}
    for lam in LAMBDAS:
        row[f"λ={lam}"] = round(rmse(y, (1 - lam) * a + lam * b), 1)
    row["r14 trimmed"] = round(rmse(y[keep], a[keep]), 1)
    row["lg01 trimmed"] = round(rmse(y[keep], b[keep]), 1)
    rows.append(row)
rec.show(pl.from_dicts(rows), "untrimmed unless marked; λ is lg01's weight against r14")

cross = []
ms = sorted(served["month"].unique().to_list())
for fit_m in ms:
    for score_m in ms:
        if fit_m == score_m:
            continue
        f = served.filter(pl.col("month") == fit_m)
        s = served.filter(pl.col("month") == score_m)
        yf, af, bf = f["y"].to_numpy(), f["r14"].to_numpy(), f["lg"].to_numpy()
        best = min(LAMBDAS, key=lambda lam: rmse(yf, (1 - lam) * af + lam * bf))
        y, a, b = s["y"].to_numpy(), s["r14"].to_numpy(), s["lg"].to_numpy()
        cross.append(
            {
                "λ chosen on": fit_m,
                "λ": best,
                "scored on": score_m,
                "r14": round(rmse(y, a), 1),
                "blend at chosen λ": round(rmse(y, (1 - best) * a + best * b), 1),
                f"blend at λ={LAMBDA_REF} (fixed)": round(
                    rmse(y, (1 - LAMBDA_REF) * a + LAMBDA_REF * b), 1
                ),
                "SSE change at chosen λ": round(
                    float(((y - ((1 - best) * a + best * b)) ** 2).sum() - ((y - a) ** 2).sum()), 0
                ),
            }
        )
rec.show(pl.from_dicts(cross), "cross-fold: the weight is learned on the other month")

# %%
rec.section("2. the long-hold rule on top of the fixed-λ blend, cross-fold")
lh = []
for fit_m in ms:
    for score_m in ms:
        if fit_m == score_m:
            continue
        f = served.filter((pl.col("month") == fit_m) & (pl.col("op_turn_s") >= LONG_HOLD))
        typ = float(f["y"].median()) if f.height else float("nan")
        s = served.filter(pl.col("month") == score_m)
        y = s["y"].to_numpy()
        base = (1 - LAMBDA_REF) * s["r14"].to_numpy() + LAMBDA_REF * s["lg"].to_numpy()
        long = (s["op_turn_s"].fill_null(0).to_numpy() >= LONG_HOLD) & np.isfinite(typ)
        pulled = np.where(long, (1 - PULL) * base + PULL * typ, base)
        lh.append(
            {
                "median learned on": fit_m,
                "long-hold rows there": f.height,
                "median label (s)": round(typ, 0),
                "scored on": score_m,
                "long-hold rows": int(long.sum()),
                "blend": round(rmse(y, base), 1),
                "blend + long-hold pull": round(rmse(y, pulled), 1),
            }
        )
rec.show(
    pl.from_dicts(lh),
    f"pull {PULL} toward the other month's long-hold median, ground time ≥ {LONG_HOLD:.0f} s",
)

# %%
rec.section("3. rows with gap above GAP_COPY (served the gap in the submission), for the record")
big = u.filter(pl.col("gap") > GAP_COPY)
rec.show(
    big.group_by("month")
    .agg(
        pl.len().alias("rows"),
        ((pl.col("y") - pl.col("gap")) ** 2).mean().sqrt().round(0).alias("RMSE of the gap"),
        ((pl.col("y") - pl.col("r14")) ** 2).mean().sqrt().round(0).alias("r14"),
        ((pl.col("y") - pl.col("lg")) ** 2).mean().sqrt().round(0).alias("lg01"),
    )
    .sort("month"),
    "LIRF NM-unmatched, gap > GAP_COPY",
)
rec.done()
