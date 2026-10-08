# %% [markdown]
# # a119 — EUROCONTROL's published monthly taxi-out against our labels, OOF and a build
#
# The APDF taxi-out indicator (`taxi_out_additional_time_{year}.csv`, ansperformance.eu; the
# organisers confirmed it may be used) publishes per airport-month: `TF`
# (flights), `VALID_FL`, `NO_REF` (valid flights whose stand-runway group has no unimpeded
# reference), `TOTAL_REF_NB_FL` (= VALID_FL − NO_REF), `TOTAL_REF_TIME_MIN` and
# `TOTAL_ADD_TIME_MIN`. The published mean actual taxi-out is (REF + ADD) / REF_NB, over the
# flights **with a reference only**.
#
# Its 2026 rows are a published level for the two scored months. Before our 2026 means are read
# against them, the same comparison is measured where the answer is known (2025):
#   1. counts: TF against our departures, per airport-month, both years;
#   2. the published invalid and no-reference shares, every month of 2025 and 2026;
#   3. validity rules on our 2025 labels: which reproduce VALID_FL;
#   4. the published mean against our 2025 label mean, three populations;
#   5. Jan + Jul 2025: our out-of-fold prediction mean against the published mean — the
#      comparison's own error where the answer is known;
#   6. 2026: a build's mean against the published mean, with section 5's error and the
#      label-free signs of a moved population beside it.
# outgoing-boat's diary records one failed use of these means (a year-over-year delta applied to
# predictions, with the valid share moving between years); sections 2, 3 and 6 show where the
# populations move. a120 is the method that shipped.
#
# Sections 1–4 read only the labels and the published files. Sections 5–6 depend on the members
# and the build: our run used v54 and an OOF stand-in that included `ab02` (not in this
# repository), so with the defaults here those two sections print different numbers.

# %%
import math
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths  # noqa: E402

# %% knobs
SRC = paths.DATA / "external" / "eurocontrol"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
PRED = Path(os.environ.get("A119_PRED", str(paths.OUTPUT / "v62_noshift.parquet")))
BOARD = float(os.environ.get("A119_BOARD", "252.05"))  # that build's board RMSE (arithmetic)
#: stand-in for the stack's OOF on Jan + Jul 2025: v54's matched-row weights without ab02 (and
#: t05's 0.024), renormalised; LIRF takes the Rome layer's OOF, as the build does
MEMBERS = {
    k: float(v)
    for k, v in (
        x.split(":")
        for x in os.environ.get(
            "A119_MEMBERS",
            "cb03_k20_combo_adsb:0.4334,ab07_ab02_adsb_calib:0.2702",
        ).split(",")
    )
}
ROME = "r14_rules_plus_unmatched_route"
LO_GRID = (0.0, 60.0, 120.0, 180.0, 300.0)  # validity rule candidates: label >= lo (s)
HI_GRID = (3_600.0, 5_400.0, 7_200.0, 10_800.0, 21_600.0, math.inf)  # ... and label <= hi
NEEDS = ("none", "stand", "runway", "both")  # ... and these fields present
RULE = os.environ.get("A119_RULE", "")  # "lo,hi,need" to fix the rule; empty = best in section 3
RARE_MIN = 20  # a stand-runway pair with fewer departures in the reference months is "rare"
SCORED = ("01", "07")

rec = analysis.Recorder("a119_published_taxi_out", "Published monthly taxi-out vs ours")
pl.Config.set_tbl_rows(130)
pl.Config.set_tbl_cols(20)


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


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
        pl.read_csv(f, infer_schema_length=0)
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
            ).alias("pub_mean_s"),
            (1 - pl.col("VALID_FL") / pl.col("TF")).alias("invalid_share"),
            (pl.col("NO_REF") / pl.col("VALID_FL")).alias("noref_share"),
        )
    )


def present(c: str) -> pl.Expr:
    return pl.col(c).is_not_null() & (pl.col(c).cast(pl.Utf8).str.strip_chars() != "")


def rule_expr(lo: float, hi: float, need: str) -> pl.Expr:
    e = pl.col("taxi_sec").is_not_null() & (pl.col("taxi_sec") >= lo)
    if math.isfinite(hi):
        e = e & (pl.col("taxi_sec") <= hi)
    return e & label_free(need)


def label_free(need: str) -> pl.Expr:
    """The part of a rule that needs no label (and so applies to 2026)."""
    e = pl.lit(True)
    if need in ("stand", "both"):
        e = e & present("stand")
    if need in ("runway", "both"):
        e = e & present("runway")
    return e


def rule_name(lo: float, hi: float, need: str) -> str:
    return f"lo={lo:.0f} hi={'inf' if not math.isfinite(hi) else f'{hi:.0f}'} need={need}"


# %%
rec.section("inputs")
pub = pl.concat([published(2025), published(2026)])
dup = pub.group_by("airport", "month").len().filter(pl.col("len") > 1)
rec.say(
    f"  published airport-months: {pub.height} ({pub['month'].min()} .. {pub['month'].max()}); "
    f"duplicated cells: {dup.height}"
)
lab = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        "stand",
        "runway",
        pl.col("flight_id").is_null().alias("unm"),
        pl.col("taxi_sec").cast(pl.Float64),
    )
    .collect(engine="streaming")
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
    )
    .collect(engine="streaming")
)
sub_f = PRED
sub = pl.read_parquet(sub_f).select(
    pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"),
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"),
)
rk = rk.join(sub, on="mvt_id", how="left")
rec.say(
    f"  2025 departures {lab.height:,}; 2026 ranking departures {rk.height:,}; submission "
    f"{sub_f.name} ({sub.height:,} rows, {rk['pred'].null_count()} ranking rows without one)"
)
rec.say(f"  OOF stand-in: {MEMBERS} (renormalised), LIRF from {ROME}; board for arithmetic {BOARD}")

# pair counts for the "rare stand-runway pair" proxy of NO_REF: a 2025 month against the other
# eleven months of 2025, a 2026 month against all of 2025 (label-free in both years)
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

# %%
rec.section("1. counts: TF against our departures")


def ours(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("airport", "month").agg(
        pl.len().alias("ours"),
        (~present("stand")).mean().round(4).alias("no_stand"),
        (~present("runway")).mean().round(4).alias("no_runway"),
        pl.col("unm").mean().round(4).alias("unmatched"),
        pl.col("rare").mean().round(4).alias("rare_pair"),
    )


cnt = pl.concat([ours(lab), ours(rk)]).join(pub, on=["airport", "month"], how="inner")
cnt = cnt.with_columns((pl.col("ours") - pl.col("TF")).alias("ours − TF"))
rec.show(
    cnt.filter(pl.col("month").str.slice(5, 2).is_in(SCORED))
    .select(
        "airport",
        "month",
        pl.col("TF").cast(pl.Int64),
        "ours",
        "ours − TF",
        "no_stand",
        "no_runway",
        "unmatched",
        "rare_pair",
    )
    .sort("airport", "month"),
    "Jan and Jul of both years; rare_pair = share on a stand-runway pair with < RARE_MIN "
    "departures in the reference months",
)
rec.show(
    cnt.filter(pl.col("month").str.starts_with("2025"))
    .group_by("airport")
    .agg(
        pl.col("TF").sum().cast(pl.Int64),
        pl.col("ours").sum(),
        pl.col("ours − TF").abs().sum().alias("sum |ours − TF| over months"),
    )
    .sort("airport"),
    "all twelve months of 2025",
)

# %%
rec.section("2. the published invalid and no-reference shares, by month")
for col, what in (("invalid_share", "1 − VALID_FL / TF"), ("noref_share", "NO_REF / VALID_FL")):
    rec.show(
        pub.select("airport", "month", (pl.col(col) * 100).round(1).alias(col))
        .pivot(on="month", index="airport", values=col, sort_columns=True)
        .sort("airport"),
        f"{what}, percent",
    )

# %%
rec.section("3. validity rules on the 2025 labels: which reproduce VALID_FL")
rules = [(lo, hi, need) for lo in LO_GRID for hi in HI_GRID for need in NEEDS]
rc = (
    lab.group_by("airport", "month")
    .agg(*[rule_expr(*r).sum().alias(rule_name(*r)) for r in rules])
    .join(pub.select("airport", "month", "TF", "VALID_FL"), on=["airport", "month"], how="inner")
)
fit = []
for r in rules:
    err = (rc[rule_name(*r)] - rc["VALID_FL"]).to_numpy()
    tf = rc["TF"].to_numpy()
    fit.append(
        {
            "rule": rule_name(*r),
            "sum |count − VALID_FL|": int(np.abs(err).sum()),
            "max |err| (flights)": int(np.abs(err).max()),
            "cells within 0.5% of TF": int((np.abs(err) <= 0.005 * tf).sum()),
            "cells": len(err),
        }
    )
fit = pl.from_dicts(fit).sort("sum |count − VALID_FL|")
rec.show(fit.head(12), "the twelve closest rules over the 2025 airport-months")
if RULE:
    lo_, hi_, need_ = RULE.split(",")
    best = (float(lo_), float(hi_), need_)
else:
    best = next(r for r in rules if rule_name(*r) == fit["rule"][0])
rec.say(f"  rule used below: {rule_name(*best)} ({'fixed by A119_RULE' if RULE else 'closest'})")
rec.show(
    rc.select(
        "airport",
        "month",
        ((pl.col(rule_name(*best)) - pl.col("VALID_FL")) / pl.col("TF") * 100).round(2).alias("e"),
    )
    .pivot(on="month", index="airport", values="e", sort_columns=True)
    .sort("airport"),
    "that rule's count − VALID_FL, percent of TF",
)
lab = lab.with_columns(rule_expr(*best).alias("valid"), label_free(best[2]).alias("lf"))
rk = rk.with_columns(label_free(best[2]).alias("lf"))

# %%
rec.section("4. the published mean against our 2025 label mean, three populations")
lm = (
    lab.group_by("airport", "month")
    .agg(
        pl.col("taxi_sec").drop_nulls().mean().alias("all labels"),
        pl.col("taxi_sec").filter(pl.col("valid")).mean().alias("rule"),
        pl.col("taxi_sec").filter(pl.col("valid") & ~pl.col("rare")).mean().alias("rule, not rare"),
    )
    .join(pub.select("airport", "month", "pub_mean_s"), on=["airport", "month"], how="inner")
)
variants = ("all labels", "rule", "rule, not rare")
rec.show(
    pl.from_dicts(
        [
            {
                "population": v,
                "mean |Δ| (s)": round(float((lm[v] - lm["pub_mean_s"]).abs().mean()), 1),
                "median |Δ| (s)": round(float((lm[v] - lm["pub_mean_s"]).abs().median()), 1),
                "max |Δ| (s)": round(float((lm[v] - lm["pub_mean_s"]).abs().max()), 1),
                "cells": lm.height,
            }
            for v in variants
        ]
    ),
    "label mean − published mean over the 2025 airport-months",
)
rec.show(
    lm.select(
        "airport",
        "month",
        (pl.col("pub_mean_s") / 60).round(2).alias("published (min)"),
        *[(pl.col(v) - pl.col("pub_mean_s")).round(1).alias(f"Δ {v} (s)") for v in variants],
    )
    .filter(pl.col("month").str.slice(5, 2).is_in(SCORED))
    .sort("airport", "month"),
    "Jan and Jul 2025",
)
rec.show(
    lm.select(
        "airport", "month", (pl.col("rule, not rare") - pl.col("pub_mean_s")).round(0).alias("d")
    )
    .pivot(on="month", index="airport", values="d", sort_columns=True)
    .sort("airport"),
    "Δ (rule, not rare), seconds, every month of 2025",
)

# %%
rec.section("5. Jan + Jul 2025: out-of-fold prediction mean against the published mean")
oof = None
wsum = sum(MEMBERS.values())
for m, w in MEMBERS.items():
    f = newest_oof(m)
    rec.say(f"  {m} (w {w / wsum:.3f}): {f.name}")
    o = pl.read_parquet(f, glob=False).select("mvt_id", (pl.col("yhat") * w / wsum).alias(m))
    oof = o if oof is None else oof.join(o, on="mvt_id", how="inner")
fr = newest_oof(ROME)
rec.say(f"  {ROME}: {fr.name}")
oof = (
    oof.with_columns(pl.sum_horizontal(list(MEMBERS)).alias("p_members"))
    .select("mvt_id", "p_members")
    .join(
        pl.read_parquet(fr, glob=False).select("mvt_id", pl.col("yhat").alias("p_rome")),
        on="mvt_id",
        how="left",
    )
)
l5 = (
    lab.join(oof, on="mvt_id", how="inner")
    .with_columns(
        pl.when(pl.col("airport") == "LIRF")
        .then(pl.col("p_rome"))
        .otherwise(pl.col("p_members"))
        .alias("pred")
    )
    .filter(pl.col("month").str.slice(5, 2).is_in(SCORED))
)
rec.say(f"  2025 rows with an OOF prediction: {l5.height:,}")


def pred_means(frame: pl.DataFrame) -> pl.DataFrame:
    """Label-free population means — the quantities that exist in 2026 too."""
    pop = pl.col("lf") & ~pl.col("rare")
    return frame.group_by("airport", "month").agg(
        pl.len().alias("n"),
        pl.col("pred").mean().alias("pred, all rows"),
        pl.col("pred").filter(pop).mean().alias("pred, label-free pop."),
        pl.col("pred").filter(pop & ~pl.col("unm")).mean().alias("pred, label-free, matched"),
    )


p25 = (
    pred_means(l5)
    .join(
        l5.group_by("airport", "month").agg(
            pl.col("pred")
            .filter(pl.col("valid") & ~pl.col("rare"))
            .mean()
            .alias("pred, rule pop."),
            pl.col("taxi_sec")
            .filter(pl.col("valid") & ~pl.col("rare"))
            .mean()
            .alias("label, rule pop."),
        ),
        on=["airport", "month"],
    )
    .join(pub.select("airport", "month", "pub_mean_s"), on=["airport", "month"], how="inner")
)
pcols = ("pred, all rows", "pred, label-free pop.", "pred, label-free, matched", "pred, rule pop.")
rec.show(
    p25.select(
        "airport",
        "month",
        "n",
        (pl.col("pub_mean_s") / 60).round(2).alias("published (min)"),
        (pl.col("label, rule pop.") - pl.col("pub_mean_s")).round(1).alias("Δ label, rule pop."),
        *[(pl.col(c) - pl.col("pub_mean_s")).round(1).alias(f"Δ {c}") for c in pcols],
    ).sort("airport", "month"),
    "Δ = mean − published mean, seconds; 'rule pop.' needs labels, 'label-free pop.' does not",
)

# %%
rec.section("6. 2026: the submission's mean against the published mean")
p26 = (
    pred_means(rk.filter(pl.col("month").str.slice(5, 2).is_in(SCORED)))
    .join(pub, on=["airport", "month"], how="inner")
    .join(
        cnt.select("airport", "month", "rare_pair", "unmatched"),
        on=["airport", "month"],
        how="left",
    )
)
prev = (
    p25.join(
        pub.select("airport", "month", "invalid_share", "noref_share"),
        on=["airport", "month"],
        how="left",
    )
    .join(cnt.select("airport", "month", "rare_pair"), on=["airport", "month"], how="left")
    .select(
        "airport",
        pl.col("month").str.slice(5, 2).alias("mm"),
        (pl.col("pred, label-free pop.") - pl.col("pub_mean_s")).alias("d25"),
        (pl.col("pred, label-free, matched") - pl.col("pub_mean_s")).alias("d25m"),
        pl.col("pred, label-free pop.").alias("p25"),
        pl.col("pub_mean_s").alias("pub25"),
        pl.col("invalid_share").alias("inv25"),
        pl.col("noref_share").alias("noref25"),
        pl.col("rare_pair").alias("rare25"),
    )
)
n_total = rk.height
t6 = (
    p26.with_columns(pl.col("month").str.slice(5, 2).alias("mm"))
    .join(prev, on=["airport", "mm"], how="left")
    .with_columns(
        (pl.col("pred, label-free pop.") - pl.col("pub_mean_s")).alias("d26"),
        (pl.col("pred, label-free, matched") - pl.col("pub_mean_s")).alias("d26m"),
    )
    .with_columns((pl.col("d26") - pl.col("d25")).alias("b"))
    .with_columns(
        ((BOARD**2 - pl.col("n") * pl.col("b") ** 2 / n_total).clip(0).sqrt() - BOARD).alias(
            "arith"
        )
    )
)
rec.show(
    t6.select(
        "airport",
        "month",
        "n",
        (pl.col("pub25") / 60).round(2).alias("pub 2025 (min)"),
        (pl.col("pub_mean_s") / 60).round(2).alias("pub 2026 (min)"),
        (pl.col("pub_mean_s") - pl.col("pub25")).round(0).alias("published Δyear (s)"),
        (pl.col("pred, label-free pop.") - pl.col("p25")).round(0).alias("ours Δyear (s)"),
        pl.col("d25").round(0).alias("Δ 2025 (s)"),
        pl.col("d26").round(0).alias("Δ 2026 (s)"),
        pl.col("b").round(0).alias("Δ26 − Δ25 (s)"),
        pl.col("d26m").round(0).alias("Δ 2026 matched (s)"),
        pl.col("d25m").round(0).alias("Δ 2025 matched (s)"),
    ).sort("airport", "month"),
    "Δ = our mean − published mean on the label-free population; 'ours Δyear' = submission "
    "2026 against OOF 2025, both on that population",
)
rec.show(
    t6.select(
        "airport",
        "month",
        (pl.col("inv25") * 100).round(1).alias("invalid 2025 %"),
        (pl.col("invalid_share") * 100).round(1).alias("invalid 2026 %"),
        (pl.col("noref25") * 100).round(1).alias("no-ref 2025 %"),
        (pl.col("noref_share") * 100).round(1).alias("no-ref 2026 %"),
        (pl.col("rare25") * 100).round(1).alias("rare pair 2025 %"),
        (pl.col("rare_pair") * 100).round(1).alias("rare pair 2026 %"),
        (pl.col("unmatched") * 100).round(1).alias("unmatched 2026 %"),
        pl.col("arith").round(2).alias("uniform-shift arithmetic (board s)"),
    ).sort("airport", "month"),
    "population signs; the last column is arithmetic only: the board change if every row of the "
    "cell carried a uniform bias of (Δ26 − Δ25) and were shifted by it, at the stated board RMSE",
)
rec.done()
