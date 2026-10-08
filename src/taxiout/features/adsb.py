"""ADS-B: each departure's own ground segment and takeoff, from adsb.lol traces.

The idea of ADS-B features on every departure, and their shape, come from team outgoing-boat's
public solution, where they were worth −16.2 board seconds
(https://github.com/EnioAguiar/prc-taxiout-2026). The code is ours. The traces are pulled by
``scripts/pull_adsb.py`` into
``data/external/adsb/cut/YYYY.MM.DD.parquet`` (adsb.lol, ODbL 1.0: "Contains information from
adsb.lol").

**Events.** Points sorted by aircraft (``icao``) and time. A ground point continues the current
ground segment if the previous point is the same aircraft, also on the ground, and less than
``SEG_GAP`` s earlier. A segment ends in a **takeoff** when its last ground point is followed,
within ``TAKEOFF_GAP`` s, by an airborne point of the same aircraft; the takeoff time is that
airborne point's. Per segment: first ground point, first point moving faster than ``MOVE_KT``,
the ground speed and position of the first point, point count, largest gap, share of MLAT points.

**Matching** (no aircraft identity exists in the challenge data). Candidates are (a) same airport
and same callsign — ``callsign`` from the flight record, else ``flight_no`` — with takeoffs within
``CS_TOL`` s, priority 0; (b) same airport, nearest takeoff before and after within ``TIME_TOL`` s,
priority 1. Assigned greedily by (priority, |Δt|), each departure and each event used once.
Days are processed one at a time with the last two hours of the previous day, so taxis crossing
midnight are seen; an event belongs to the day of its takeoff.

**Features**, NaN where unmatched: ``adsb_taxi = mvt − first_ground``, ``adsb_taxi_move = mvt −
first_move``, ``adsb_gs0``, ``adsb_takeoff_err = takeoff − mvt``, ``adsb_n_ground``,
``adsb_gap_max``, ``adsb_frac_mlat``, ``adsb_lat0``, ``adsb_lon0``, ``adsb_match_pri``. Never
substitute ``adsb_taxi`` for the prediction: the trace starts late by an airport-specific amount
(outgoing-boat: EHAM +6 s, EGLL −1,076 s); the model learns that.

``CONTEXT``: external data and other rows; never a departure's label. **The cache key does not see
the trace files** (registry docstring): rebuild with ``--overwrite`` after the pull changes.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from taxiout import io, paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

CUT = paths.DATA / "external" / "adsb" / "cut"
SEG_GAP = 600.0
TAKEOFF_GAP = 120.0
MOVE_KT = 1.0
CS_TOL = 300.0
TIME_TOL = 90.0
LOOKBACK = 7_200.0

OUTPUTS = (
    "adsb_taxi",
    "adsb_taxi_move",
    "adsb_gs0",
    "adsb_takeoff_err",
    "adsb_n_ground",
    "adsb_gap_max",
    "adsb_frac_mlat",
    "adsb_lat0",
    "adsb_lon0",
    "adsb_match_pri",
)


def events(points: pl.DataFrame) -> pl.DataFrame:
    """Takeoff events with their ground segment. Pure; unit-tested.

    `points`: icao, airport, flight, t (epoch s), lat, lon, ground, gs, src.
    """
    p = points.sort("icao", "t").with_columns(
        same_prev=pl.col("icao") == pl.col("icao").shift(1),
        same_next=pl.col("icao") == pl.col("icao").shift(-1),
        dt_prev=pl.col("t") - pl.col("t").shift(1),
        dt_next=pl.col("t").shift(-1) - pl.col("t"),
        ground_prev=pl.col("ground").shift(1),
        ground_next=pl.col("ground").shift(-1),
        t_next=pl.col("t").shift(-1),
        flight_next=pl.col("flight").shift(-1),
    )
    cont = (
        pl.col("ground")
        & pl.col("same_prev").fill_null(False)
        & pl.col("ground_prev").fill_null(False)
        & (pl.col("dt_prev") < SEG_GAP).fill_null(False)
    )
    is_to = (
        pl.col("ground")
        & pl.col("same_next").fill_null(False)
        & ~pl.col("ground_next").fill_null(True)
        & (pl.col("dt_next") < TAKEOFF_GAP).fill_null(False)
    )
    p = p.with_columns(seg=(pl.col("ground") & ~cont).cum_sum(), is_to=is_to)
    g = p.filter(pl.col("ground"))
    ev = g.group_by("seg", maintain_order=True).agg(
        pl.col("icao").first(),
        pl.col("airport").last(),
        pl.coalesce(
            pl.col("flight_next").filter(pl.col("is_to")).last(),
            pl.col("flight").drop_nulls().last(),
        ).alias("flight"),
        pl.col("t_next").filter(pl.col("is_to")).last().alias("takeoff"),
        pl.col("t").min().alias("first_ground"),
        pl.col("t").filter(pl.col("gs") > MOVE_KT).min().alias("first_move"),
        pl.col("gs").first().alias("gs0"),
        pl.col("lat").first().alias("lat0"),
        pl.col("lon").first().alias("lon0"),
        pl.len().cast(pl.Int32).alias("n_ground"),
        pl.col("t").diff().max().fill_null(0.0).alias("gap_max"),
        pl.col("src").fill_null("").str.contains("mlat").cast(pl.Float64).mean().alias("frac_mlat"),
    )
    return ev.filter(pl.col("takeoff").is_not_null()).drop("seg")


def match(deps: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """One event per departure at most, and one departure per event. Pure; unit-tested.

    `deps`: mvt_id, airport, ts (epoch s), cs (normalised callsign or null).
    `ev`: output of `events`.
    """
    ev = ev.with_row_index("eid").with_columns(pl.col("flight").str.replace_all(" ", ""))
    by_cs = (
        deps.filter(pl.col("cs").is_not_null())
        .join(ev, left_on=["airport", "cs"], right_on=["airport", "flight"], how="inner")
        .filter((pl.col("takeoff") - pl.col("ts")).abs() < CS_TOL)
        .select(
            "mvt_id",
            "eid",
            pl.lit(0).alias("pri"),
            (pl.col("takeoff") - pl.col("ts")).abs().alias("d"),
        )
    )
    evs = ev.select("airport", "takeoff", "eid").sort("takeoff")
    d = deps.select("mvt_id", "airport", "ts").sort("ts")
    by_time = []
    for strategy in ("backward", "forward"):
        j = d.join_asof(
            evs,
            left_on="ts",
            right_on="takeoff",
            by="airport",
            strategy=strategy,
            tolerance=TIME_TOL,
            check_sortedness=False,
        ).filter(pl.col("eid").is_not_null())
        by_time.append(
            j.select(
                "mvt_id",
                "eid",
                pl.lit(1).alias("pri"),
                (pl.col("takeoff") - pl.col("ts")).abs().alias("d"),
            )
        )
    cand = pl.concat([by_cs, *by_time]).sort("pri", "d", "mvt_id", "eid")
    used_m, used_e, keep = set(), set(), []
    for mid, eid, pri in zip(cand["mvt_id"], cand["eid"], cand["pri"], strict=True):
        if mid in used_m or eid in used_e:
            continue
        used_m.add(mid)
        used_e.add(eid)
        keep.append((mid, eid, pri))
    pairs = pl.DataFrame(
        keep, schema={"mvt_id": pl.Int64, "eid": pl.UInt32, "pri": pl.Int32}, orient="row"
    )
    return pairs.join(ev.drop("airport"), on="eid", how="left").drop("eid")


def to_features(deps: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
    """The feature columns for `deps` (mvt_id, ts), null where unmatched. Pure."""
    m = deps.select("mvt_id", "ts").join(matched, on="mvt_id", how="left")
    return m.select(
        "mvt_id",
        (pl.col("ts") - pl.col("first_ground")).alias("adsb_taxi"),
        (pl.col("ts") - pl.col("first_move")).alias("adsb_taxi_move"),
        pl.col("gs0").alias("adsb_gs0"),
        (pl.col("takeoff") - pl.col("ts")).alias("adsb_takeoff_err"),
        pl.col("n_ground").cast(pl.Float64).alias("adsb_n_ground"),
        pl.col("gap_max").alias("adsb_gap_max"),
        pl.col("frac_mlat").alias("adsb_frac_mlat"),
        pl.col("lat0").alias("adsb_lat0"),
        pl.col("lon0").alias("adsb_lon0"),
        pl.col("pri").cast(pl.Float64).alias("adsb_match_pri"),
    )


def _day_points(day: dt.date) -> pl.DataFrame | None:
    """Points of `day` plus the last LOOKBACK seconds of the day before, or None if absent."""
    f = CUT / f"{day:%Y.%m.%d}.parquet"
    if not f.exists():
        return None
    cols = ["icao", "airport", "flight", "t", "lat", "lon", "ground", "gs", "src"]
    frames = [pl.read_parquet(f, columns=cols)]
    prev = CUT / f"{day - dt.timedelta(days=1):%Y.%m.%d}.parquet"
    if prev.exists():
        start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()
        frames.insert(
            0, pl.read_parquet(prev, columns=cols).filter(pl.col("t") >= start - LOOKBACK)
        )
    return pl.concat(frames)


@feature(name="adsb", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def adsb(ctx: FeatureContext) -> pl.DataFrame:
    """Each departure's matched ADS-B ground segment and takeoff."""
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select(
            "mvt_id",
            "airport",
            (pl.col("mvt_time").dt.epoch("ms") / 1000.0).alias("ts"),
            pl.col("mvt_time").dt.date().alias("day"),
            pl.coalesce(pl.col("callsign"), pl.col("flight_no"))
            .str.to_uppercase()
            .str.replace_all(" ", "")
            .alias("cs"),
        )
        .collect(engine="streaming")
    )
    parts = []
    for (day,), d in deps.group_by("day"):
        pts = _day_points(day) if day is not None else None
        if pts is None or pts.height == 0:
            continue
        start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()
        ev = events(pts).filter((pl.col("takeoff") >= start) & (pl.col("takeoff") < start + 86_400))
        if ev.height:
            parts.append(to_features(d, match(d, ev)))
    if parts:
        out = pl.concat(parts)
    else:
        out = deps.select("mvt_id").with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in OUTPUTS]
        )
    out = out.with_columns([pl.col(c).cast(pl.Float64) for c in OUTPUTS])
    assert out["mvt_id"].n_unique() == out.height
    return align(ctx.spine, out, OUTPUTS)
