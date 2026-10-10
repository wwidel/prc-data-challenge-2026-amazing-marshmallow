# %% [markdown]
# # a139 — Where a final-phase build errs, out of fold: month, airport, week, hour, rows
#
# An out-of-fold version of a final-phase build on the four months of 2025:
#   - the body is the stack of A139_MEMBERS (default: the BASE composition), weights fitted on the
#     other pair, one vector for NM-matched and one for NM-unmatched rows, the unmatched vector for
#     every month but A139_UNMATCHED_WITHOUT fitted without it (as `a129`);
#   - Rome takes r14's out-of-fold predictions verbatim;
#   - then the schedule-gap bound, the LEBL/LEMD ceilings and the LOBT window (`postprocess`).
# Left out: a120's cell shift (2026 only), the Rome matched-row blend and the lg01 override.
# Every breakdown gives rows, RMSE over all labels and over labels <= 12 h ("trimmed"), the mean
# residual (label − prediction) on trimmed rows, and each group's share of the squared error.
# Numbers only.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, postprocess, store  # noqa: E402
from taxiout.features import registry  # noqa: E402

# %% knobs
TRIM = 43_200.0
GROUP_MIN_ROWS = 2000
PAIRS = {"Jan+Jul": ("2025-01", "2025-07"), "Feb+Jun": ("2025-02", "2025-06")}
MEMBERS_FILE = os.environ.get("A139_MEMBERS", "")
MEMBERS = Path(MEMBERS_FILE).read_text().split() if MEMBERS_FILE else paths.composition()
HELD = os.environ.get("A139_UNMATCHED_WITHOUT", "02")
ROME = "r14_rules_plus_unmatched_route"
COUNTRY = {
    "EDDF": "Germany",
    "EDDM": "Germany",
    "EGLL": "United Kingdom",
    "EHAM": "Netherlands",
    "LEBL": "Spain",
    "LEMD": "Spain",
    "LFPG": "France",
    "LIRF": "Italy",
    "LSZH": "Switzerland",
    "LTFM": "Türkiye",
}
BANDS = [300, 600, 1_200, 1_800, 3_600, 7_200, 14_400, 43_200]
BAND_LABELS = ["<5m", "5-10m", "10-20m", "20-30m", "30-60m", "1-2h", "2-4h", "4-12h", ">12h"]
TOP = 20

rec = analysis.Recorder("a139_where_the_stack_errs", "Where a final-phase build errs, out of fold")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)


def newest_oof(name: str) -> Path:
    hits = [f for f in paths.PREDICTIONS.iterdir() if f.name.split("__")[0] == name]
    if not hits:
        raise FileNotFoundError(f"no out-of-fold file for {name}")
    return max(hits, key=lambda f: f.stat().st_mtime)


# %%
rec.section("inputs")
frame = None
for m in [*MEMBERS, ROME]:
    f = newest_oof(m)
    o = pl.read_parquet(f, glob=False).select("mvt_id", "airport", "y", pl.col("yhat").alias(m))
    rec.say(f"  {m}: {f.name}")
    frame = o if frame is None else frame.join(o.select("mvt_id", m), on="mvt_id", how="inner")
raw = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "month",
        pl.col("flight_id").is_null().alias("unm"),
        pl.col("mvt_time").dt.date().alias("day"),
        pl.col("mvt_time").dt.hour().alias("hour_utc"),
        pl.col("mvt_time").dt.week().alias("iso_week"),
        (pl.col("mvt_time") - pl.col("sched_time")).dt.total_seconds().alias("gap"),
        (pl.col("mvt_time") - pl.col("lobt")).dt.total_seconds().cast(pl.Float64).alias("m_l"),
    )
    .collect(engine="streaming")
)
frame = frame.join(raw, on="mvt_id", how="inner")
block = store.block_dir(registry.get("adsb"))
parts = []
for m in sorted(frame["month"].unique().to_list()):
    ids = pl.read_parquet(paths.SPINE / f"{m}.parquet", columns=["mvt_id"], glob=False)
    v = pl.read_parquet(block / f"{m}.parquet", columns=["adsb_match_pri"], glob=False)
    parts.append(pl.concat([ids, v], how="horizontal"))
cov = pl.concat(parts).select("mvt_id", pl.col("adsb_match_pri").is_not_null().alias("adsb"))
frame = frame.join(cov, on="mvt_id", how="left", maintain_order="left").with_columns(
    pl.col("adsb").fill_null(False), pl.col("unm").fill_null(False)
)
rec.say(f"  members ({len(MEMBERS)}) from {MEMBERS_FILE or 'the BASE composition'}")
rec.say(f"  unmatched vector fitted without month {HELD} for the other months; Rome = {ROME}")
rec.say(f"  out-of-fold rows {frame.height:,}; months {sorted(frame['month'].unique().to_list())}")

# %%
X = frame.select(MEMBERS).to_numpy().astype(float)
y = frame["y"].to_numpy().astype(float)
month = frame["month"].to_numpy()
mm = np.array([m[-2:] for m in month])
unm = frame["unm"].to_numpy()
body = (frame["airport"] != "LIRF").to_numpy()
pred = np.full(frame.height, np.nan)
for ms in PAIRS.values():
    target = np.isin(month, ms)
    fit = ~target & body
    w_glob = nnls(X[fit], y[fit])[0]
    for g in (False, True):
        rows = fit & (unm == g)
        w = nnls(X[rows], y[rows])[0] if rows.sum() >= GROUP_MIN_ROWS else w_glob
        sel = target & (unm == g) & body
        pred[sel] = X[sel] @ w
        if g and HELD:
            others = rows & (mm != HELD)
            w_o = nnls(X[others], y[others])[0] if others.sum() >= GROUP_MIN_ROWS else w
            sel_o = sel & (mm != HELD)
            pred[sel_o] = X[sel_o] @ w_o
pred[~body] = frame[ROME].to_numpy()[~body]
d = (
    frame.with_columns(pl.Series("prediction", pred))
    .pipe(postprocess.apply_gap_bound)
    .pipe(postprocess.apply_airport_ceilings)
    .pipe(postprocess.apply_lobt_window)
    .with_columns(pl.col("prediction").clip(*postprocess.CLIP))
    .with_columns(
        (pl.col("y") - pl.col("prediction")).alias("res"),
        ((pl.col("y") - pl.col("prediction")) ** 2).alias("se"),
        (pl.col("y") <= TRIM).alias("trim"),
        pl.col("airport").replace_strict(COUNTRY).alias("country"),
        pl.col("y")
        .cut(BANDS, labels=BAND_LABELS, left_closed=False)
        .cast(pl.Utf8)
        .alias("label band"),
    )
)
SSE_ALL = float(d["se"].sum())
SSE_TRIM_BODY = float(d.filter(pl.col("trim") & (pl.col("airport") != "LIRF"))["se"].sum())


def table(frame_: pl.DataFrame, by: list[str], sort: str | None = None) -> pl.DataFrame:
    t = frame_.group_by(by).agg(
        pl.len().alias("rows"),
        pl.col("se").mean().sqrt().round(1).alias("rmse"),
        pl.col("se").filter(pl.col("trim")).mean().sqrt().round(1).alias("rmse trimmed"),
        pl.col("res").filter(pl.col("trim")).mean().round(1).alias("mean res, trimmed"),
        (pl.col("se").sum() / SSE_ALL).round(4).alias("share of all SSE"),
        (pl.col("se").filter(pl.col("trim") & (pl.col("airport") != "LIRF")).sum() / SSE_TRIM_BODY)
        .round(4)
        .alias("share of trimmed ex-Rome SSE"),
    )
    return t.sort(sort, descending=True) if sort else t.sort(by)


# %%
rec.section("1. overall")
ex = pl.col("airport") != "LIRF"
for name, sel in (("all rows", pl.lit(True)), ("without Rome", ex), ("Rome", ~ex)):
    s = d.filter(sel)
    rec.say(
        f"  {name:13s} rows {s.height:>7,}  rmse {np.sqrt(s['se'].mean()):7.2f}  trimmed "
        f"{np.sqrt(s.filter(pl.col('trim'))['se'].mean()):7.2f}  labels > 12 h "
        f"{int((~s['trim']).sum()):>3}  share of all SSE {s['se'].sum() / SSE_ALL:.3f}"
    )
se_sorted = np.sort(d["se"].to_numpy())[::-1]
for q in (0.001, 0.01, 0.1):
    k = int(len(se_sorted) * q)
    rec.say(
        f"  the worst {q:.1%} of rows ({k:,}) carry {se_sorted[:k].sum() / SSE_ALL:.1%} of all SSE"
    )

# %%
rec.section("2. by month")
rec.show(table(d, ["month"]))
rec.show(table(d.filter(ex), ["month"]), "without Rome")

# %%
rec.section("3. by airport and country")
rec.show(table(d, ["country", "airport"]))
rec.show(
    table(d.filter(ex), ["country"], sort="share of trimmed ex-Rome SSE"),
    "per country, without Rome",
)

# %%
rec.section("4. airport-months, without Rome, by trimmed squared error")
rec.show(table(d.filter(ex), ["airport", "month"], sort="share of trimmed ex-Rome SSE").head(15))

# %%
rec.section("5. ISO weeks and hours (UTC), without Rome")
rec.show(table(d.filter(ex), ["iso_week"]))
rec.show(table(d.filter(ex), ["hour_utc"]))

# %%
rec.section("6. NM match and ADS-B coverage")
rec.show(table(d.filter(ex), ["unm", "adsb"]), "without Rome")
rec.show(table(d.filter(ex), ["airport", "adsb"]), "without Rome, per airport")

# %%
rec.section("7. label bands")
rec.show(table(d.filter(ex), ["label band"]), "without Rome")
rec.show(table(d.filter(~ex), ["label band"]), "Rome")

# %%
rec.section(f"8. the {TOP} airport-days with the most trimmed squared error, without Rome")
days = (
    d.filter(ex & pl.col("trim"))
    .group_by("airport", "day")
    .agg(
        pl.len().alias("rows"),
        pl.col("y").mean().round(0).alias("mean label"),
        pl.col("res").mean().round(0).alias("mean residual"),
        (pl.col("se").sum() / SSE_TRIM_BODY).alias("share"),
    )
    .sort("share", descending=True)
    .with_columns(pl.col("share").cum_sum().round(3).alias("cumulative"), pl.col("share").round(4))
)
rec.say(f"  {days.height} airport-days; the top {TOP} carry {days['share'].head(TOP).sum():.1%}")
rec.show(days.head(TOP))
rec.done()
