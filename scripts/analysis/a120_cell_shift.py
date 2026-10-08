# %% [markdown]
# # a120 — A build moved towards EUROCONTROL's published monthly taxi-out, per airport-month
#
# EUROCONTROL publishes the mean taxi-out per airport and month (the APDF-based "taxi-out
# additional time" indicator, ansperformance.eu; its use was approved by the organisers), for
# 2026 too. The published mean leaves flights out — those EUROCONTROL marks invalid (in winter, at
# five airports, a share no taxi-time bound reproduces) and those without a reference time — so the
# true mean of the flights we predict sits above or below it by an **offset**, which the 2025
# labels measure:
#
#   1. per 2025 airport-month, on our rows that are not on a rare stand-runway pair:
#      offset = label mean − published mean = W (labels inside the rule) + O (labels outside it);
#   2. W is modelled as b0[airport] + E[airport] × s_w + F × s_r, where s_w is the share of
#      flights EUROCONTROL excludes beyond the label rule (the winter exclusion), E the excess
#      taxi of such a flight, s_r the no-reference share; O is the airport's 2025 average;
#   3. leave-one-month-out on 2025 (every airport's month m held out together) gives the error
#      of the predicted offset, per airport and regime (winter: s_w >= WINTER; else summer);
#   4. for each 2026 airport-month: offset from the published 2026 shares, hence the build's error
#      b = build mean − (published + offset), and its error bar;
#   5. a cell moves only if |b| > MIN_SNR × its error bar, by −K × b, on every row (LIRF and the
#      exact-override rows excepted).
# K and MIN_SNR were fixed before any board result. An earlier version (a119) subtracted the whole
# 2025 comparison, which also removed our own 2025 out-of-fold error; only the label-measured
# offset belongs there.
#
# v62 ships the shifts this computed on an earlier build, v54 (`config/cellshift_v54.csv`; on the
# board, v54 252.05 -> v56 251.68, −0.37). The shifts are relative, so they sit on any base; v54
# itself needs members that are not in this repository, so on another build this prints that
# build's cells, not v54's.
# Point it at a build with A120_PRED (default: data/output/v62_noshift.parquet, `make
# stack-noshift`).

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, postprocess  # noqa: E402

# %% knobs
SRC = paths.DATA / "external" / "eurocontrol"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
SKIP = ("LIRF",)  # labels above 2 h move Rome's mean by ~100 s; the Rome layer serves it
PRED = Path(os.environ.get("A120_PRED", str(paths.OUTPUT / "v62_noshift.parquet")))
BOARD = float(os.environ.get("A120_BOARD", "252.05"))  # the build's board score (arithmetic only)
K = float(os.environ.get("A120_K", "0.5"))  # a half step towards the estimate
MIN_SNR = 1.0  # a cell moves only if |b| exceeds this many leave-one-month-out errors
RULE = (120.0, 7_200.0)  # the label rule closest to EUROCONTROL's VALID_FL count (a119)
WINTER = 0.005  # an airport-month is "winter" when s_w >= this
MIN_WINTER = 2  # an airport gets its own E with at least this many winter months, else the pooled E
RARE_MIN = 20  # stand-runway pairs with fewer departures in the reference months are "rare"
EXACT = (
    paths.DATA / "external" / "pop27_gap.parquet",
    paths.DATA / "external" / "onerow_aobt.parquet",
)
OUT_CSV = paths.DATA / "external" / "cellshift.csv"  # `build_stack.py --cell-shift`
SCORED = ("01", "07")

rec = analysis.Recorder("a120_cell_shift", "A build moved towards the published monthly taxi-out")
pl.Config.set_tbl_rows(130)
pl.Config.set_tbl_cols(20)


def published(year: int) -> pl.DataFrame:
    f = SRC / f"taxi_out_additional_time_{year}.csv"
    rec.say(f"  {f.name}: {f.stat().st_size:,} bytes")
    num = (
        "TF",
        "VALID_FL",
        "NO_REF",
        "TOTAL_REF_NB_FL",
        "TOTAL_REF_TIME_MIN",
        "TOTAL_ADD_TIME_MIN",
    )
    return (
        pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy")
        .filter(pl.col("APT_ICAO").is_in(AIRPORTS))
        .select(
            pl.col("APT_ICAO").alias("airport"),
            (pl.col("YEAR") + "-" + pl.col("MONTH_NUM").str.zfill(2)).alias("month"),
            *[pl.col(c).cast(pl.Float64) for c in num],
        )
        .with_columns(
            (
                (pl.col("TOTAL_REF_TIME_MIN") + pl.col("TOTAL_ADD_TIME_MIN"))
                / pl.col("TOTAL_REF_NB_FL")
                * 60
            ).alias("pub"),
            (1 - pl.col("VALID_FL") / pl.col("TF")).alias("invalid"),
            (pl.col("NO_REF") / pl.col("TF")).alias("s_r"),
        )
    )


def present(c: str) -> pl.Expr:
    return pl.col(c).is_not_null() & (pl.col(c).cast(pl.Utf8).str.strip_chars() != "")


# %%
rec.section("inputs")
pub = pl.concat([published(2025), published(2026)])
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        "stand",
        "runway",
        pl.col("taxi_sec").cast(pl.Float64),
    )
    .collect(engine="streaming")
    .with_columns(
        (
            pl.col("taxi_sec").is_not_null()
            & (pl.col("taxi_sec") >= RULE[0])
            & (pl.col("taxi_sec") <= RULE[1])
        ).alias("rule")
    )
)
rk = (
    tio.scan_ranking()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        "stand",
        "runway",
        pl.col("flight_id").is_null().alias("unm"),
        (pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("gap"),
        (pl.col("mvt_time") - pl.col("lobt")).dt.total_seconds().cast(pl.Float64).alias("m_l"),
    )
    .collect(engine="streaming")
)
sub_f = PRED
sub = pl.read_parquet(sub_f).select(
    pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"),
)
rk = rk.join(sub, on="mvt_id", how="left")
assert rk["pred"].null_count() == 0, "ranking departures without a predicted value"
# rare pairs, as a119: a 2025 month against the other eleven, a 2026 month against all of 2025
pm = (
    lab.filter(present("stand") & present("runway"))
    .group_by("airport", "stand", "runway", "month")
    .agg(pl.len().alias("n_m"))
)
ptot = pm.group_by("airport", "stand", "runway").agg(pl.col("n_m").sum().alias("n_tot"))
lab = (
    lab.join(pm, on=["airport", "stand", "runway", "month"], how="left")
    .join(ptot, on=["airport", "stand", "runway"], how="left")
    .with_columns(((pl.col("n_tot") - pl.col("n_m")) < RARE_MIN).fill_null(True).alias("rare"))
    .drop("n_m", "n_tot")
)
rk = (
    rk.join(ptot, on=["airport", "stand", "runway"], how="left")
    .with_columns((pl.col("n_tot") < RARE_MIN).fill_null(True).alias("rare"))
    .drop("n_tot")
)
rec.say(
    f"  2025 departures {lab.height:,}; 2026 ranking departures {rk.height:,}; build "
    f"{sub_f.name}; K = {K}, MIN_SNR = {MIN_SNR}, rule {RULE}, WINTER {WINTER}, skip {SKIP}"
)

# %%
rec.section("1. the 2025 offsets: label mean of our rows − published mean, per airport-month")
c25 = (
    lab.group_by("airport", "month")
    .agg(
        pl.len().alias("n"),
        pl.col("rule").sum().alias("n_rule"),
        pl.col("taxi_sec").filter(~pl.col("rare")).mean().alias("mean_nr"),
        pl.col("taxi_sec").filter(pl.col("rule") & ~pl.col("rare")).mean().alias("mean_rule_nr"),
    )
    .join(pub, on=["airport", "month"], how="inner")
    .with_columns(
        ((pl.col("n_rule") - pl.col("VALID_FL")) / pl.col("TF")).alias("s_w"),
        (1 - pl.col("n_rule") / pl.col("TF")).alias("r"),
        (pl.col("mean_rule_nr") - pl.col("pub")).alias("W"),
        (pl.col("mean_nr") - pl.col("mean_rule_nr")).alias("O"),
    )
    .with_columns((pl.col("W") + pl.col("O")).alias("offset"))
    .filter(~pl.col("airport").is_in(list(SKIP)))
    .sort("airport", "month")
)
for col, what, scale in (
    ("s_w", "s_w: excluded beyond the label rule, % of TF", 100),
    ("s_r", "s_r: no reference, % of TF", 100),
    ("W", "W: label mean inside the rule − published (s)", 1),
    ("O", "O: label mean, all − inside the rule (s)", 1),
):
    rec.show(
        c25.select("airport", "month", (pl.col(col) * scale).round(1).alias(col))
        .pivot(on="month", index="airport", values=col, sort_columns=True)
        .sort("airport"),
        what,
    )


# %%
def fit(frame: pl.DataFrame) -> dict:
    """OLS of W on airport intercepts, own E slopes (>= MIN_WINTER winter months), pooled F."""
    aps = sorted(frame["airport"].unique().to_list())
    nwin = frame.filter(pl.col("s_w") >= WINTER).group_by("airport").len()
    own = sorted(a for a, k in zip(nwin["airport"], nwin["len"], strict=True) if k >= MIN_WINTER)
    # winter months of airports without their own E stay out of the fit (they take the pooled E)
    fr = frame.filter(pl.col("airport").is_in(own) | (pl.col("s_w") < WINTER))
    cols = [f"b0_{a}" for a in aps] + [f"E_{a}" for a in own] + ["F"]
    X = np.zeros((fr.height, len(cols)))
    for i, (a, sw, sr) in enumerate(zip(fr["airport"], fr["s_w"], fr["s_r"], strict=True)):
        X[i, cols.index(f"b0_{a}")] = 1.0
        if a in own:
            X[i, cols.index(f"E_{a}")] = sw
        X[i, -1] = sr
    beta, *_ = np.linalg.lstsq(X, fr["W"].to_numpy(), rcond=None)
    p = dict(zip(cols, beta, strict=True))
    mass = {a: float(frame.filter(pl.col("airport") == a)["s_w"].clip(0).sum()) for a in own}
    e_bar = sum(p[f"E_{a}"] * mass[a] for a in own) / sum(mass.values()) if own else 0.0
    return {
        "b0": {a: p[f"b0_{a}"] for a in aps},
        "E": {a: (p[f"E_{a}"] if a in own else e_bar) for a in aps},
        "F": p["F"],
        "O": {a: float(frame.filter(pl.col("airport") == a)["O"].mean()) for a in aps},
        "own": own,
        "e_bar": e_bar,
    }


def predict(model: dict, airport: str, s_w: float, s_r: float) -> float:
    return model["b0"][airport] + model["E"][airport] * s_w + model["F"] * s_r + model["O"][airport]


rec.section("2. the offset model, fitted on all of 2025")
full = fit(c25)
rec.say(
    f"  own E: {full['own']}; pooled E for the rest: {full['e_bar']:.0f} s; "
    f"F (no-reference slope, pooled): {full['F']:.0f} s per unit share"
)
rec.show(
    pl.from_dicts(
        [
            {
                "airport": a,
                "b0 (s)": round(full["b0"][a], 1),
                "E: excess of an excluded flight (s)": round(full["E"][a], 0),
                "own E": a in full["own"],
                "O: 2025 mean (s)": round(full["O"][a], 1),
            }
            for a in sorted(full["b0"])
        ]
    ),
    "W = b0 + E × s_w + F × s_r; offset = W + O",
)

# %%
rec.section("3. leave-one-month-out on 2025: the error of the predicted offset")
lomo = []
for m in sorted(c25["month"].unique().to_list()):
    model_m = fit(c25.filter(pl.col("month") != m))
    for row in c25.filter(pl.col("month") == m).iter_rows(named=True):
        pred = predict(model_m, row["airport"], row["s_w"], row["s_r"])
        lomo.append(
            {
                "airport": row["airport"],
                "month": m,
                "winter": row["s_w"] >= WINTER,
                "offset": row["offset"],
                "predicted": pred,
                "err": pred - row["offset"],
            }
        )
lomo = pl.from_dicts(lomo)
sig = lomo.group_by("airport", "winter").agg(
    (pl.col("err") ** 2).mean().sqrt().alias("sigma"), pl.len().alias("months")
)
pooled_sig = {
    w: float(np.sqrt((lomo.filter(pl.col("winter") == w)["err"] ** 2).mean()))
    for w in (True, False)
}
rec.show(
    lomo.select("airport", "month", pl.col("err").round(0))
    .pivot(on="month", index="airport", values="err", sort_columns=True)
    .sort("airport"),
    "predicted − actual offset (s), each month predicted from a fit without it",
)
rec.show(
    sig.with_columns(pl.col("sigma").round(1)).sort("airport", "winter"),
    f"RMS error by airport and regime; pooled: winter {pooled_sig[True]:.1f} s, "
    f"summer {pooled_sig[False]:.1f} s",
)

# %%
rec.section("4. 2026: the build's estimated error per airport-month")
r25 = c25.select("airport", pl.col("month").str.slice(5, 2).alias("mm"), "r")
c26 = (
    rk.filter(pl.col("month").str.slice(5, 2).is_in(SCORED) & ~pl.col("airport").is_in(list(SKIP)))
    .group_by("airport", "month")
    .agg(
        pl.len().alias("n"),
        pl.col("pred").filter(~pl.col("rare")).mean().alias("pred_nr"),
        pl.col("unm").mean().alias("unmatched"),
    )
    .join(pub, on=["airport", "month"], how="inner")
    .with_columns(pl.col("month").str.slice(5, 2).alias("mm"))
    .join(r25, on=["airport", "mm"], how="left")
    .with_columns((pl.col("invalid") - pl.col("r")).alias("s_w"))
    .sort("airport", "month")
)
rows26 = []
for row in c26.iter_rows(named=True):
    a = row["airport"]
    off = predict(full, a, row["s_w"], row["s_r"])
    winter = row["s_w"] >= WINTER
    s = sig.filter((pl.col("airport") == a) & (pl.col("winter") == winter))["sigma"]
    sigma = float(s[0]) if s.len() else pooled_sig[winter]
    b = float(row["pred_nr"] - (row["pub"] + off))
    move = bool(abs(b) > MIN_SNR * sigma)
    rows26.append(
        {
            "airport": a,
            "month": row["month"],
            "n": row["n"],
            "published (min)": round(row["pub"] / 60, 2),
            "s_w %": round(100 * row["s_w"], 2),
            "s_r %": round(100 * row["s_r"], 1),
            "offset (s)": round(off, 1),
            "build − published (s)": round(row["pred_nr"] - row["pub"], 1),
            "b: build error (s)": round(b, 1),
            "sigma (s)": round(sigma, 1),
            "regime": "winter" if winter else "summer",
            "moves": move,
            "shift (s)": round(-K * b, 1) if move else 0.0,
            "_b": b,
        }
    )
t26 = pl.from_dicts(rows26)
rec.show(
    t26.drop("_b"), "b > 0: build above the estimated true mean; shift = −K × b where it moves"
)
N = rk.height
moved = t26.filter(pl.col("moves"))
for k in (K, 1.0):
    gain = float((moved["n"] * moved["_b"] ** 2 * (2 * k - k * k)).sum()) / N
    rec.say(
        f"  arithmetic, if every moved cell's b were exact: step {k} changes the board by "
        f"{np.sqrt(max(BOARD**2 - gain, 0.0)) - BOARD:+.2f} s ({moved.height} cells, "
        f"{int(moved['n'].sum()):,} rows)"
    )
everything = float((t26["n"] * t26["_b"] ** 2).sum()) / N
rec.say(
    f"  arithmetic, every cell moved by its full b: {np.sqrt(BOARD**2 - everything) - BOARD:+.2f} s"
)

# %%
rec.section("5. the shift file")
exact_ids = (
    pl.concat(
        [pl.read_parquet(f, glob=False).select(pl.col("mvt_id").cast(pl.Int64)) for f in EXACT]
    )["mvt_id"]
    .unique()
    .to_list()
)
shift = moved.select("airport", "month", pl.col("shift (s)").alias("shift"))
in_cells = rk.join(shift, on=["airport", "month"], how="inner")
n_exact = int(in_cells["mvt_id"].is_in(exact_ids).sum())
ov = in_cells.filter(~pl.col("mvt_id").is_in(exact_ids)).with_columns(
    (pl.col("pred") + pl.col("shift")).alias("prediction")
)
raw = ov["prediction"].to_numpy().copy()
# the layers that run after the shift in build_stack.py, to count the rows they move back
ov = (
    ov.pipe(postprocess.apply_gap_bound)
    .pipe(postprocess.apply_airport_ceilings)
    .pipe(postprocess.apply_lobt_window)
    .with_columns(pl.col("prediction").clip(*postprocess.CLIP).round(0))
)
n_back = int((np.abs(ov["prediction"].to_numpy() - np.round(raw)) > 0.5).sum())
rec.say(
    f"  rows: {ov.height:,} in {moved.height} cells (exact-override rows left alone: {n_exact}); "
    f"moved back by the bounds / window / clip: {n_back}"
)
if shift.height:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    shift.write_csv(OUT_CSV)
    rec.say(f"  wrote {OUT_CSV.relative_to(paths.ROOT)} ({shift.height} airport-months)")
else:
    rec.say("  no cell moves: no file written")
rec.done()
