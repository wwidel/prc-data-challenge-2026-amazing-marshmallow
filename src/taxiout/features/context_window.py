"""The airport's state around the takeoff, as windowed *means* rather than counts.

Our traffic blocks count movements (`congestion`, `queue`, `runway_history`) and our taxi-in block
is a static per-stand median (`taxi_in`). Team outgoing-boat's public solution measured three
windowed means that we did not have — together worth −9.1 board seconds to them (their v11), and
the idea is taken from there (https://github.com/EnioAguiar/prc-taxiout-2026). The code is ours.

All keyed on the departure's takeoff time ``t`` (``mvt_time``), from columns the ranking set
carries for every row:

* **neighbours' off-block proxy**: mean of ``mvt_time − aobt_3`` over the *other* departures with
  takeoff in ``[t − W, t)`` (``prev``) and ``(t, t + W]`` (``next``), per airport and per airport ×
  runway, W ∈ {15, 60} min. The proxy is kept only within [0, ``PROXY_MAX``]. Plus the row's own
  proxy minus the 60-minute airport means — how unusual this flight's AOBT is for the moment.
* **live arrival taxi-in**: mean ``taxi_sec`` (clipped to [0, ``TAXI_IN_MAX``]) and count of the
  arrivals landing in ``(t − W, t]``, per airport and per airport × runway.
* **last arrival at the stand**: age and taxi-in of the last arrival whose in-block time is at or
  before ``t`` at the same airport × stand, if within a day. Keyed on the takeoff, so it can be the
  aircraft that took the stand *after* this one left — information the ranking set carries.

``CONTEXT``: other rows' times and the arrivals' own taxi times; never a departure's label.
"""

from __future__ import annotations

import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Window lengths, minutes.
WINDOWS = (15, 60)
#: The neighbours' off-block proxy is kept only inside [0, PROXY_MAX] seconds.
PROXY_MAX = 7_200.0
#: Arrival taxi-in is clipped to [0, TAXI_IN_MAX] seconds.
TAXI_IN_MAX = 3_600.0
#: The last stand arrival is dropped if older than this many seconds.
STAND_MAX_AGE = 86_400.0

_NB = tuple(
    f"cw_nb_{side}_{scope}_{w}"
    for side in ("prev", "next")
    for scope in ("apt", "rw")
    for w in WINDOWS
)
_ARR = tuple(
    f"cw_arr_{stat}_{scope}_{w}"
    for stat in ("tin", "n")
    for scope in ("apt", "rw")
    for w in WINDOWS
)
OUTPUTS = (
    *_NB,
    "cw_own_minus_prev_apt_60",
    "cw_own_minus_next_apt_60",
    *_ARR,
    "cw_stand_last_age",
    "cw_stand_last_tin",
)


def _windowed(events: pl.DataFrame, keys: list[str], suffix: str) -> pl.DataFrame:
    """Window means over one key grouping. `events` holds departures and arrivals together,
    with ``proxy`` set on departures only and ``tin`` on arrivals only. Pure; unit-tested.

    A window depends only on (keys, time), so the rolling results are de-duplicated on those and
    joined back -- `rolling(group_by=...)` does not return rows in input order (it broke an
    order-based version on the airport x runway grouping), so position is never trusted.
    """
    ev = events.filter(pl.col("mvt_time").is_not_null()).sort([*keys, "mvt_time"])
    on = [*keys, "mvt_time"]
    out = events.select("row", *on)
    for w in WINDOWS:
        period = f"{w}m"
        prev = ev.rolling("mvt_time", period=period, closed="left", group_by=keys).agg(
            pl.col("proxy").mean().alias(f"cw_nb_prev_{suffix}_{w}")
        )
        nxt = ev.rolling("mvt_time", period=period, offset="0s", closed="right", group_by=keys).agg(
            pl.col("proxy").mean().alias(f"cw_nb_next_{suffix}_{w}")
        )
        arr = ev.rolling("mvt_time", period=period, closed="right", group_by=keys).agg(
            pl.col("tin").mean().alias(f"cw_arr_tin_{suffix}_{w}"),
            pl.col("tin").count().cast(pl.Int32).alias(f"cw_arr_n_{suffix}_{w}"),
        )
        for r in (prev, nxt, arr):
            out = out.join(r.unique(subset=on), on=on, how="left", maintain_order="left")
    assert out.height == events.height
    return out.drop(*on)


def featurize(movements: pl.DataFrame) -> pl.DataFrame:
    """All outputs for the departures in `movements` (both phases, raw short names). Pure."""
    m = movements.with_row_index("row").with_columns(
        (pl.col("phase") == "DEP").alias("is_dep"),
        pl.when(pl.col("phase") == "DEP")
        .then((pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().cast(pl.Float64))
        .alias("_proxy"),
        pl.when(pl.col("phase") == "ARR")
        .then(pl.col("taxi_sec").cast(pl.Float64).clip(0.0, TAXI_IN_MAX))
        .alias("tin"),
    )
    m = m.with_columns(
        pl.when(pl.col("_proxy").is_between(0.0, PROXY_MAX)).then(pl.col("_proxy")).alias("proxy"),
        pl.col("runway").fill_null("?"),
    )
    ev = m.select("row", "is_dep", "airport", "runway", "mvt_time", "proxy", "tin")
    apt = _windowed(ev, ["airport"], "apt")
    rw = _windowed(ev, ["airport", "runway"], "rw")
    dep = (
        m.filter(pl.col("is_dep"))
        .select("row", "mvt_id", "airport", "stand", "mvt_time", "proxy")
        .join(apt, on="row", how="left")
        .join(rw, on="row", how="left")
        .with_columns(
            (pl.col("proxy") - pl.col("cw_nb_prev_apt_60")).alias("cw_own_minus_prev_apt_60"),
            (pl.col("proxy") - pl.col("cw_nb_next_apt_60")).alias("cw_own_minus_next_apt_60"),
        )
    )
    # last arrival in-block at the same stand, at or before the takeoff
    arrivals = (
        m.filter((pl.col("phase") == "ARR") & pl.col("stand").is_not_null())
        .filter(pl.col("block_time").is_not_null())
        .select("airport", "stand", pl.col("block_time").alias("_inblock"), pl.col("tin"))
        .sort("_inblock")
    )
    with_stand = dep.filter(pl.col("stand").is_not_null()).sort("mvt_time")
    last = (
        with_stand.join_asof(
            arrivals,
            left_on="mvt_time",
            right_on="_inblock",
            by=["airport", "stand"],
            strategy="backward",
            check_sortedness=False,  # both sides are sorted on their time column above
        )
        .with_columns(
            (pl.col("mvt_time") - pl.col("_inblock"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("_age")
        )
        .select(
            "row",
            pl.when(pl.col("_age") <= STAND_MAX_AGE)
            .then(pl.col("_age"))
            .alias("cw_stand_last_age"),
            pl.when(pl.col("_age") <= STAND_MAX_AGE).then(pl.col("tin")).alias("cw_stand_last_tin"),
        )
    )
    return dep.join(last, on="row", how="left").select("mvt_id", *OUTPUTS)


@feature(name="context_window", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def context_window(ctx: FeatureContext) -> pl.DataFrame:
    """Windowed neighbour off-block proxy, live arrival taxi-in and the last stand arrival."""
    movements = (
        io.scan_movements()
        .select(
            "mvt_id",
            "phase",
            "airport",
            "runway",
            "stand",
            "mvt_time",
            "aobt_3",
            "block_time",
            "taxi_sec",
        )
        .collect(engine="streaming")
    )
    return align(ctx.spine, featurize(movements), OUTPUTS)
