# %% [markdown]
# # a118 — EUROCONTROL's delay-by-cause file (APT-DLY) against taxi-out, 2025
#
# APT-DLY gives, per airport and day, the arrival ATFM delay minutes **by cause**: weather (W),
# de-icing (D), aerodrome capacity (G), ATC capacity (C), staffing (S), routing (R), airspace (M),
# ATC equipment (T), other equipment (E), industrial action by ATC (I) and others (N), special
# event (P), accident/incident (A), environment (V), other (O), unspecified (NA). It is the
# structured, open record of the disruptions a news calendar would list. Analysed on Jan + Jul
# 2025 against taxi-out, to see whether it could help in 2026. Not used by the best submission.
#
# Measured, per airport-day (UTC date of the takeoff / of the flight date in APT-DLY):
#   1. how often each cause occurs at our ten airports in Jan/Jul 2025 and 2026;
#   2. whether it tracks the day's mean taxi-out (level) and, more to the point, the day's mean
#      **residual of our best members** (`MEMBERS`' out-of-fold predictions): only the residual says
#      whether it carries information the model does not already have;
#   3. event days by cause: the residual on them against the airport's other days;
#   4. a cross-fold ceiling: a day-level linear correction fitted on January, scored on July and
#      vice versa (the only honest way to price a selection);
#   5. overlap with what the models already see (`eurocontrol_daily`'s pre-departure delay);
#   6. 2026: the scored months' disrupted days (label-free);
#   7. the en-route side (`ert_dly_ansp`: en-route ATFM delay per ANSP-day by the same causes): the
#      home ANSP's and the network's delay per flight against the daily residual, and the residual
#      on the industrial-action days.

# %%
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import taxiout.io as tio  # noqa: E402
from taxiout import analysis, paths, store  # noqa: E402

# %% knobs
SRC = paths.DATA / "external" / "eurocontrol"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
MONTHS_2025 = ("2025-01", "2025-07")
MONTHS_2026 = ("2026-01", "2026-07")
MEMBERS = os.environ.get("A118_MEMBERS", "cb03_k20_combo_adsb,ab07_ab02_adsb_calib").split(",")
BODY_MAX = (
    3_600.0  # daily residuals on the body (labels <= 1 h): the part a day-level effect can move
)
EVENT_MIN = 1_000.0  # an "event day" for a cause: at least this many delay minutes from it
#: section 7: the ANSP whose en-route delay a departure from the airport meets first
HOME_ANSP = {
    "EDDF": "DFS",
    "EDDM": "DFS",
    "EGLL": "NATS (Continental)",
    "EHAM": "LVNL",
    "LEBL": "ENAIRE",
    "LEMD": "ENAIRE",
    "LFPG": "DSNA",
    "LIRF": "ENAV",
    "LSZH": "Skyguide",
    "LTFM": "DHMI",
}
NETWORK = "EUROCONTROL Area (MS)"  # the whole-network row of ert_dly_ansp (ENTITY_TYPE AREA)
#: cause code -> group
GROUPS = {
    "W": "weather",
    "D": "deicing",
    "G": "aerodrome_cap",
    "C": "atc_cap",
    "S": "atc_staff",
    "R": "atc_other",
    "M": "atc_other",
    "T": "atc_other",
    "I": "strike",
    "N": "strike",
    "P": "special_event",
    "A": "accident",
    "E": "equipment",
    "V": "environment",
    "O": "other",
    "NA": "other",
}

rec = analysis.Recorder("a118_apt_dly", "APT-DLY (delay by cause) against taxi-out, 2025")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(18)


def newest_oof(name: str):
    hits = [
        f
        for f in paths.PREDICTIONS.iterdir()
        if f.suffix == ".parquet" and f.name.split("__")[0] == name
    ]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


# %%
rec.section("inputs")
frames = []
for year in (2025, 2026):
    f = SRC / f"apt_dly_{year}.csv"
    rec.say(f"  {f.name}: {f.stat().st_size:,} bytes")
    frames.append(pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy"))
raw = pl.concat(frames, how="diagonal")
cause_cols = {
    c: c.removeprefix("DLY_APT_ARR_").removesuffix("_1")
    for c in raw.columns
    if c.startswith("DLY_APT_ARR_") and c.endswith("_1") and c != "DLY_APT_ARR_1"
}
apt = (
    raw.filter(pl.col("APT_ICAO").is_in(AIRPORTS))
    .with_columns(
        pl.col("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
        pl.col("APT_ICAO").alias("airport"),
        *[
            pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0)
            for c in ["FLT_ARR_1", "DLY_APT_ARR_1", *cause_cols]
        ],
    )
    .group_by("airport", "day")
    .agg(
        pl.col("FLT_ARR_1").sum().alias("arr_n"),
        pl.col("DLY_APT_ARR_1").sum().alias("dly_total"),
        *[pl.col(c).sum() for c in cause_cols],
    )
)
for group in sorted(set(GROUPS.values())):
    cols = [c for c, code in cause_cols.items() if GROUPS.get(code) == group]
    apt = apt.with_columns(
        pl.sum_horizontal(cols).alias(f"g_{group}") if cols else pl.lit(0.0).alias(f"g_{group}")
    )
groups = sorted(set(GROUPS.values()))
apt = apt.with_columns(
    pl.col("day").dt.strftime("%Y-%m").alias("month"),
    (pl.col("dly_total") / pl.col("arr_n").clip(1)).alias("dly_per_arr"),
    *[(pl.col(f"g_{g}") / pl.col("arr_n").clip(1)).alias(f"pa_{g}") for g in groups],
)
rec.say(
    f"  airport-days at our ten airports: {apt.height:,} ({apt['day'].min()} .. {apt['day'].max()})"
)

# %%
rec.section("1. how often each cause occurs (event day = at least EVENT_MIN delay minutes from it)")
freq = (
    apt.filter(pl.col("month").is_in([*MONTHS_2025, *MONTHS_2026]))
    .group_by("month")
    .agg(
        pl.len().alias("airport_days"),
        (pl.col("dly_total") >= EVENT_MIN).sum().alias("any_cause"),
        *[(pl.col(f"g_{g}") >= EVENT_MIN).sum().alias(g) for g in groups],
    )
    .sort("month")
)
rec.show(freq, f"airport-days with >= {EVENT_MIN:.0f} min from a cause, at our ten airports")
rec.show(
    apt.filter(pl.col("month").is_in([*MONTHS_2025, *MONTHS_2026]))
    .group_by("airport", "month")
    .agg(
        pl.col("dly_total").sum().round(0).alias("dly_min"),
        (pl.col("dly_total") >= EVENT_MIN).sum().alias("event_days"),
    )
    .pivot(on="month", index="airport", values=["dly_min", "event_days"])
    .sort("airport"),
    "per airport: total arrival ATFM delay minutes and event days, by month",
)

# %%
rec.section("2. day level, Jan + Jul 2025: taxi-out and our members' residuals against the causes")
dep = (
    tio.scan_training()
    .filter(pl.col("phase") == "DEP")
    .select(
        "mvt_id",
        "airport",
        pl.col("mvt_time").dt.date().alias("day"),
        pl.col("mvt_time").dt.strftime("%Y-%m").alias("month"),
        "taxi_sec",
    )
    .filter(pl.col("month").is_in(MONTHS_2025))
    .collect(engine="streaming")
)
res = None
for m in MEMBERS:
    f = newest_oof(m)
    rec.say(f"  {m}: {f.name}")
    o = pl.read_parquet(f, glob=False).select("mvt_id", pl.col("yhat").alias(f"p_{m}"))
    res = o if res is None else res.join(o, on="mvt_id", how="inner")
rows = dep.join(res, on="mvt_id", how="inner").with_columns(
    pl.mean_horizontal([f"p_{m}" for m in MEMBERS]).alias("pred")
)
rows = rows.with_columns((pl.col("taxi_sec") - pl.col("pred")).alias("resid"))
body = rows.filter(pl.col("taxi_sec") <= BODY_MAX)
daily = (
    body.group_by("airport", "day", "month")
    .agg(
        pl.len().alias("n_dep"),
        pl.col("taxi_sec").mean().alias("taxi_mean"),
        pl.col("resid").mean().alias("resid_mean"),
        (pl.col("resid") ** 2).mean().sqrt().alias("rmse"),
    )
    .join(
        apt.select(
            "airport",
            "day",
            "dly_total",
            "dly_per_arr",
            *[f"pa_{g}" for g in groups],
            *[f"g_{g}" for g in groups],
        ),
        on=["airport", "day"],
        how="left",
    )
    .with_columns(
        pl.col(c).fill_null(0.0)
        for c in [
            "dly_total",
            "dly_per_arr",
            *[f"pa_{g}" for g in groups],
            *[f"g_{g}" for g in groups],
        ]
    )
)
# within airport-month: subtract the airport-month mean (airports differ in level)
dm = daily.with_columns(
    *[
        (pl.col(c) - pl.col(c).mean().over("airport", "month")).alias(f"{c}_w")
        for c in ["taxi_mean", "resid_mean", "dly_per_arr", *[f"pa_{g}" for g in groups]]
    ]
)
rec.say(f"  airport-days with departures and OOF: {daily.height:,}; body rows {body.height:,}")


def corr(a: str, b: str, frame: pl.DataFrame = dm) -> float:
    x, y = frame[a].to_numpy(), frame[b].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


ct = [
    {
        "cause (delay min per arrival)": c.removeprefix("pa_")
        if c != "dly_per_arr"
        else "ALL causes",
        "corr with daily taxi (within apt-month)": round(corr(f"{c}_w", "taxi_mean_w"), 3),
        "corr with daily residual (within apt-month)": round(corr(f"{c}_w", "resid_mean_w"), 3),
        "days with any": int((dm[c] > 0).sum()),
    }
    for c in ["dly_per_arr", *[f"pa_{g}" for g in groups]]
]
rec.show(
    pl.from_dicts(ct),
    "correlations across airport-days, Jan + Jul 2025, both series demeaned within airport-month",
)

# %%
rec.section("3. event days: the members' residual on them against the airport's other days")
ev = []
for g in ["total", *groups]:
    col = "dly_total" if g == "total" else f"g_{g}"
    flag = dm[col].to_numpy() >= EVENT_MIN
    if flag.sum() == 0:
        continue
    ev.append(
        {
            "cause": g,
            "event days": int(flag.sum()),
            "taxi mean vs apt-month (s)": round(
                float(dm["taxi_mean_w"].to_numpy()[flag].mean()), 1
            ),
            "residual on event days (s)": round(float(dm["resid_mean"].to_numpy()[flag].mean()), 1),
            "residual other days (s)": round(float(dm["resid_mean"].to_numpy()[~flag].mean()), 1),
            "body RMSE event days": round(
                float(
                    np.sqrt(
                        (dm["rmse"].to_numpy()[flag] ** 2 * dm["n_dep"].to_numpy()[flag]).sum()
                        / dm["n_dep"].to_numpy()[flag].sum()
                    )
                ),
                1,
            ),
            "body RMSE other days": round(
                float(
                    np.sqrt(
                        (dm["rmse"].to_numpy()[~flag] ** 2 * dm["n_dep"].to_numpy()[~flag]).sum()
                        / dm["n_dep"].to_numpy()[~flag].sum()
                    )
                ),
                1,
            ),
        }
    )
rec.show(
    pl.from_dicts(ev),
    f"event day = >= {EVENT_MIN:.0f} min from the cause; residual = label − members' OOF, body",
)
rec.show(
    dm.sort("dly_total", descending=True)
    .head(15)
    .select(
        "airport",
        "day",
        "n_dep",
        pl.col("dly_total").round(0),
        pl.col("taxi_mean_w").round(0),
        pl.col("resid_mean").round(1),
        *[
            pl.col(f"g_{g}").round(0)
            for g in ("weather", "deicing", "aerodrome_cap", "atc_staff", "strike")
        ],
    ),
    "the fifteen most-delayed airport-days of Jan + Jul 2025",
)

# %%
rec.section(
    "4. cross-fold ceiling: a day-level residual correction, fit on one month, scored on the other"
)
feat = ["dly_per_arr", *[f"pa_{g}" for g in groups]]
out = []
rowsb = body.join(
    dm.select("airport", "day", *feat), on=["airport", "day"], how="left"
).with_columns(pl.col(c).fill_null(0.0) for c in feat)
for fit_m, score_m in ((MONTHS_2025[0], MONTHS_2025[1]), (MONTHS_2025[1], MONTHS_2025[0])):
    tr = dm.filter(pl.col("month") == fit_m)
    X = np.column_stack([np.ones(tr.height), *[tr[c].to_numpy() for c in feat]])
    w = tr["n_dep"].to_numpy().astype(float)
    y = tr["resid_mean"].to_numpy()
    lam = 1e3
    A = X.T @ (X * w[:, None]) + lam * np.eye(X.shape[1])
    A[0, 0] -= lam  # no penalty on the intercept
    beta = np.linalg.solve(A, X.T @ (w * y))
    te = rowsb.filter(pl.col("month") == score_m)
    Xt = np.column_stack([np.ones(te.height), *[te[c].to_numpy() for c in feat]])
    corr_t = Xt @ beta - beta[0]  # the cause-driven part only; the intercept is a month-level bias
    yt, pt = te["taxi_sec"].to_numpy(), te["pred"].to_numpy()
    ex = (te["airport"] != "LIRF").to_numpy()

    def rmse(p, m, yt=yt):
        return float(np.sqrt(np.mean((yt[m] - p[m]) ** 2)))

    out.append(
        {
            "fit on": fit_m,
            "scored on": score_m,
            "body ex-LIRF RMSE, members": round(rmse(pt, ex), 3),
            "with day correction": round(rmse(pt + corr_t, ex), 3),
            "rows corrected by > 10 s": int((np.abs(corr_t) > 10).sum()),
            "largest correction (s)": round(float(np.abs(corr_t).max()), 1),
        }
    )
rec.show(
    pl.from_dicts(out),
    "body rows (labels <= 1 h), ex-LIRF; ridge on day-level delay per arrival by cause",
)

# %%
rec.section("5. overlap with what the models already see: eurocontrol_daily's pre-departure delay")
months_all = tuple(sorted(f.stem for f in paths.SPINE.glob("*.parquet")))
ec = (
    store.assemble(["eurocontrol_daily"], months_all)
    .select(
        "mvt_id", "airport", pl.col("mvt_time").dt.date().alias("day"), "ec_pre_min", "ec_reg_frac"
    )
    .collect()
    .group_by("airport", "day")
    .agg(pl.col("ec_pre_min").first(), pl.col("ec_reg_frac").first())
)
ov = dm.join(ec, on=["airport", "day"], how="left").with_columns(
    *[
        (pl.col(c) - pl.col(c).mean().over("airport", "month")).alias(f"{c}_w")
        for c in ("ec_pre_min", "ec_reg_frac")
    ]
)
rec.show(
    pl.from_dicts(
        [
            {
                "pair (within airport-month)": "APT-DLY total vs pre-departure delay (ec_pre_min)",
                "corr": round(corr("dly_per_arr_w", "ec_pre_min_w", ov), 3),
            },
            {
                "pair (within airport-month)": "APT-DLY weather vs ec_pre_min",
                "corr": round(corr("pa_weather_w", "ec_pre_min_w", ov), 3),
            },
            {
                "pair (within airport-month)": "APT-DLY total vs regulated share (ec_reg_frac)",
                "corr": round(corr("dly_per_arr_w", "ec_reg_frac_w", ov), 3),
            },
            {
                "pair (within airport-month)": "ec_pre_min vs daily residual",
                "corr": round(corr("ec_pre_min_w", "resid_mean_w", ov), 3),
            },
        ]
    ),
    "how much of APT-DLY the models' existing EUROCONTROL inputs already carry",
)

# %%
rec.section("6. 2026: the scored months' disrupted days (label-free)")
rk = (
    tio.scan_ranking()
    .filter(pl.col("phase") == "DEP")
    .select("airport", pl.col("mvt_time").dt.date().alias("day"))
    .collect(engine="streaming")
    .group_by("airport", "day")
    .agg(pl.len().alias("n_dep"))
)
d26 = rk.join(apt, on=["airport", "day"], how="left").with_columns(
    pl.col("dly_total").fill_null(0.0)
)
rec.show(
    d26.with_columns(pl.col("day").dt.strftime("%Y-%m").alias("month"))
    .group_by("month")
    .agg(
        pl.col("n_dep").sum().alias("departures"),
        pl.col("n_dep").filter(pl.col("dly_total") >= EVENT_MIN).sum().alias("on event days"),
        *[
            pl.col("n_dep").filter(pl.col(f"g_{g}") >= EVENT_MIN).sum().alias(g)
            for g in (
                "weather",
                "deicing",
                "aerodrome_cap",
                "atc_staff",
                "atc_cap",
                "strike",
                "special_event",
            )
        ],
    )
    .sort("month"),
    "2026 ranking departures that fall on an event day, by cause",
)
rec.show(
    d26.sort("dly_total", descending=True)
    .head(15)
    .select(
        "airport",
        "day",
        "n_dep",
        pl.col("dly_total").round(0),
        *[
            pl.col(f"g_{g}").round(0)
            for g in ("weather", "deicing", "aerodrome_cap", "atc_staff", "strike", "special_event")
        ],
    ),
    "the fifteen most-delayed airport-days of Jan + Jul 2026",
)

# %%
rec.section("7. en-route ATFM delay by ANSP (ert_dly_ansp): home ANSP and network vs residuals")
ert_raw = pl.concat(
    [
        pl.read_csv(SRC / f"ert_dly_ansp_{y}.csv", infer_schema_length=0, encoding="utf8-lossy")
        for y in (2025, 2026)
    ],
    how="diagonal",
)
ert_causes = {
    c: c.removeprefix("DLY_ERT_").removesuffix("_1")
    for c in ert_raw.columns
    if c.startswith("DLY_ERT_") and c.endswith("_1") and c != "DLY_ERT_1"
}
ert = ert_raw.with_columns(
    pl.col("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
    *[
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0)
        for c in ["FLT_ERT_1", "DLY_ERT_1", *ert_causes]
    ],
)
for group in groups:
    cols = [c for c, code in ert_causes.items() if GROUPS.get(code) == group]
    ert = ert.with_columns((pl.sum_horizontal(cols) if cols else pl.lit(0.0)).alias(f"e_{group}"))
ERT_GROUPS = ("strike", "atc_cap", "atc_staff", "weather")


def per_flight(frame: pl.DataFrame, tag: str) -> pl.DataFrame:
    return frame.select(
        "day",
        (pl.col("DLY_ERT_1") / pl.col("FLT_ERT_1").clip(1)).alias(f"{tag}_all"),
        *[(pl.col(f"e_{g}") / pl.col("FLT_ERT_1").clip(1)).alias(f"{tag}_{g}") for g in ERT_GROUPS],
    )


home = pl.concat(
    [
        per_flight(ert.filter(pl.col("ENTITY_NAME") == ansp), "home").with_columns(
            pl.lit(a).alias("airport")
        )
        for a, ansp in HOME_ANSP.items()
    ]
)
net = per_flight(ert.filter(pl.col("ENTITY_NAME") == NETWORK), "net")
rec.say(
    f"  home ANSPs: {HOME_ANSP}; network row: {NETWORK!r}; "
    f"ANSP-days read: {ert.height:,} ({ert['day'].min()} .. {ert['day'].max()})"
)
ecols = [f"{t}_{g}" for t in ("home", "net") for g in ("all", *ERT_GROUPS)]
de = (
    dm.join(home, on=["airport", "day"], how="left")
    .join(net, on="day", how="left")
    .with_columns(pl.col(c).fill_null(0.0) for c in ecols)
    .with_columns(
        (pl.col(c) - pl.col(c).mean().over("airport", "month")).alias(f"{c}_w") for c in ecols
    )
)
rec.show(
    pl.from_dicts(
        [
            {
                "en-route delay per flight": c,
                "corr with daily taxi (within apt-month)": round(
                    corr(f"{c}_w", "taxi_mean_w", de), 3
                ),
                "corr with daily residual (within apt-month)": round(
                    corr(f"{c}_w", "resid_mean_w", de), 3
                ),
                "days > 0": int((de[c] > 0).sum()),
            }
            for c in ecols
        ]
    ),
    "Jan + Jul 2025 airport-days, body residual of the members' OOF",
)
strike_days = (
    ert.filter((pl.col("ENTITY_TYPE").str.starts_with("ANSP")) & (pl.col("e_strike") >= EVENT_MIN))
    .select("day", "ENTITY_NAME", pl.col("e_strike").round(0))
    .sort("day")
)
rec.show(
    strike_days.filter(pl.col("day").dt.strftime("%Y-%m").is_in([*MONTHS_2025, *MONTHS_2026])),
    f"ANSP-days with >= {EVENT_MIN:.0f} min of industrial-action en-route delay, Jan/Jul",
)
sd = set(strike_days.filter(pl.col("day").dt.year() == 2025)["day"].to_list())
rec.show(
    de.filter(pl.col("month") == "2025-07")
    .with_columns(pl.col("day").is_in(list(sd)).alias("strike_day"))
    .group_by("airport", "strike_day")
    .agg(
        pl.len().alias("days"),
        pl.col("resid_mean").mean().round(1).alias("mean daily residual (s)"),
        pl.col("taxi_mean_w").mean().round(1).alias("taxi vs apt-month (s)"),
    )
    .sort("airport", "strike_day"),
    "July 2025: the members' daily body residual on industrial-action days against the other days",
)
rec.done()
