"""Same-aircraft ground time from the OPDI flight list (EUROCONTROL PRC + OpenSky, v002).

OpenAir (likable-eagle) joined the OPDI flight list to departures by callsign and took the
aircraft's previous leg: ``icao_turn = first_seen − previous last_seen``. On LIRF NM-unmatched
rows a long turn (≥ 2 h: the aircraft sat at the airport) went with a normal taxi (median 1,132 s)
behind a large gate delay, and short turns with ``T ≈ D`` (schedule copies). Their board moved
−0.48 with it as a feature of the gate-delay model (v23) and −1.37 with a long-hold pull (v24)
(https://github.com/ChandanHegde07/OpenAir). The files come from ``scripts/pull_opdi.py``; the
organiser accepted OPDI as open data.

**Matching.** Candidate callsigns, tried in order: the NM callsign; the movement's flight number;
the flight number with a doubled airline prefix removed (``ITYTY672`` → ``ITY672``, ``RYRR8830``
→ ``RYR8830``, ``QFAQFA6`` → ``QFA6``), a form many 2026 rows use. A departure takes the OPDI
flight with that callsign whose ``adep`` is its airport and whose ``first_seen`` — lift-off, not
off-block (OpenAir E46: median MVT − first_seen −26 s at LIRF) — is nearest its take-off, within
``TOL`` seconds.

**Outputs.** ``op_found`` is 1 when a flight matched, else 0. The rest are null where unmatched:
``op_turn_s`` = first_seen − the same icao24's previous ``last_seen``, kept inside ``TURN``;
``op_same_apt`` = 1 when that previous leg ended at this airport; ``op_lag_s`` = first_seen −
take-off (match quality).

``CONTEXT``: external data; never a departure's label. **The cache key does not see the files**:
rebuild with ``--overwrite`` after a pull.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from taxiout import io, paths
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

SRC = paths.DATA / "external" / "opdi"
OUTPUTS = ("op_found", "op_turn_s", "op_same_apt", "op_lag_s")
TOL = 1_800.0  # |first_seen − take-off| allowed for a match, seconds
TURN = (1_200.0, 129_600.0)  # ground times kept: 20 min to 36 h (OpenAir's window)
KEYS = ("k_nm", "k_raw", "k_dedup")
_DOUBLED = r"^([A-Z]{3})[A-Z]{1,3}(\d[0-9A-Z]*)$"


def norm(c: pl.Expr) -> pl.Expr:
    """Upper case, no spaces; empty → null."""
    s = c.cast(pl.Utf8).str.to_uppercase().str.replace_all(" ", "")
    return pl.when(s.str.len_chars() > 0).then(s)


def dedup_prefix(c: pl.Expr) -> pl.Expr:
    """``ITYTY672`` → ``ITY672``: three letters, 1–3 extra letters, then the number; else null."""
    return pl.concat_str([c.str.extract(_DOUBLED, 1), c.str.extract(_DOUBLED, 2)])


def keys(deps: pl.DataFrame) -> pl.DataFrame:
    """Add the three candidate callsigns to ``deps`` (needs ``callsign`` and ``flight_no``)."""
    return deps.with_columns(
        norm(pl.col("callsign")).alias("k_nm"),
        norm(pl.col("flight_no")).alias("k_raw"),
    ).with_columns(dedup_prefix(pl.col("k_raw")).alias("k_dedup"))


def load_flights() -> pl.DataFrame:
    """Every OPDI flight on disk: icao24, cs, adep, ades, fs, ls (UTC), de-duplicated."""
    files = sorted(SRC.glob("flight_list_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no OPDI flight lists in {SRC}; run scripts/pull_opdi.py")
    cols = ["icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]
    return (
        pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed")
        .with_columns(
            norm(pl.col("flt_id")).alias("cs"),
            pl.col("first_seen").str.to_datetime(time_zone="UTC", strict=False).alias("fs"),
            pl.col("last_seen").str.to_datetime(time_zone="UTC", strict=False).alias("ls"),
        )
        .filter(pl.col("icao24").is_not_null() & pl.col("fs").is_not_null())
        .unique(subset=["icao24", "fs"], keep="first")  # month files overlap at their edges
        .select("icao24", "cs", "adep", "ades", "fs", "ls")
        .collect(engine="streaming")
    )


def match(deps: pl.DataFrame, flights: pl.DataFrame, tol: float = TOL) -> pl.DataFrame:
    """mvt_id → icao24, fs of the matched OPDI flight. ``deps`` needs mvt_id, airport, mvt and
    the ``KEYS`` columns. Keys are tried in order; a row matched by one is not retried. Pure."""
    f = flights.filter(pl.col("adep").is_not_null() & pl.col("cs").is_not_null()).sort("fs")
    found = []
    left = deps
    for key in KEYS:
        cand = left.filter(pl.col(key).is_not_null()).sort("mvt")
        if cand.height == 0:
            continue
        j = cand.join_asof(
            f.select("cs", "adep", "fs", "icao24"),
            left_on="mvt",
            right_on="fs",
            by_left=[key, "airport"],
            by_right=["cs", "adep"],
            strategy="nearest",
            tolerance=timedelta(seconds=tol),
        ).filter(pl.col("icao24").is_not_null())
        found.append(j.select("mvt_id", "icao24", "fs"))
        left = left.join(j.select("mvt_id"), on="mvt_id", how="anti")
    if not found:
        return pl.DataFrame(schema={"mvt_id": pl.Int64, "icao24": pl.Utf8, "fs": f["fs"].dtype})
    return pl.concat(found)


def turns(matched: pl.DataFrame, flights: pl.DataFrame) -> pl.DataFrame:
    """For each matched (icao24, fs): the previous leg's last_seen and ades. Pure."""
    legs = (
        flights.sort("icao24", "fs")
        .with_columns(
            pl.col("ls").shift(1).over("icao24").alias("prev_ls"),
            pl.col("ades").shift(1).over("icao24").alias("prev_ades"),
        )
        .select("icao24", "fs", "prev_ls", "prev_ades")
    )
    return matched.join(legs, on=["icao24", "fs"], how="left")


def featurize(deps: pl.DataFrame, flights: pl.DataFrame) -> pl.DataFrame:
    """All outputs for ``deps`` (mvt_id, airport, mvt, callsign, flight_no). Pure; unit-tested."""
    d = keys(deps)
    dtype = d.schema["mvt"]
    flights = flights.with_columns(pl.col("fs").cast(dtype), pl.col("ls").cast(dtype))
    m = turns(match(d, flights), flights)
    turn = (pl.col("fs") - pl.col("prev_ls")).dt.total_seconds().cast(pl.Float64)
    out = d.select("mvt_id", "airport", "mvt").join(m, on="mvt_id", how="left")
    return out.select(
        "mvt_id",
        pl.col("icao24").is_not_null().cast(pl.Int8).alias("op_found"),
        pl.when(turn.is_between(*TURN)).then(turn).alias("op_turn_s"),
        pl.when(pl.col("prev_ades").is_not_null())
        .then((pl.col("prev_ades") == pl.col("airport")).cast(pl.Int8))
        .alias("op_same_apt"),
        (pl.col("fs") - pl.col("mvt")).dt.total_seconds().cast(pl.Float64).alias("op_lag_s"),
    )


@feature(name="opdi", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def opdi(ctx: FeatureContext) -> pl.DataFrame:
    """Same-aircraft ground time before the departure, from the OPDI flight list."""
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", pl.col("mvt_time").alias("mvt"), "callsign", "flight_no")
        .collect(engine="streaming")
    )
    return align(ctx.spine, featurize(deps, load_flights()), OUTPUTS)
