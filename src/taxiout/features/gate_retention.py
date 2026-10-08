"""Gate retention at the moment this flight pushed: who was being held, who was already moving.

Our queue blocks (`queue`, `queue2`) count aircraft between an estimated off-block and takeoff.
Team outgoing-boat added counts read at this flight's own push clock (``aobt_3``), including
the departures that were *overdue* — past their EOBT and not yet pushed — and the recent delay to
push (https://github.com/EnioAguiar/prc-taxiout-2026; −0.71 board to them). The code is ours.

At ``t = aobt_3`` of the row, over the other departures ``j`` of the same airport (or runway):

* ``gr_overdue_{apt,rw}``: #{j : eobt_1_j ≤ t < aobt_3_j} — held past their EOBT at this moment.
  Only ``j`` with ``0 ≤ aobt_3 − eobt_1 < 7200`` and ``mvt ≥ aobt_3`` count (their guards).
* ``gr_active_rw``: #{j ≠ row : aobt_3_j ≤ t < mvt_j} on the same runway — pushed, not yet airborne.
* ``gr_delay_15m``: mean ``aobt_3 − sched_time`` of the airport's departures that pushed in
  ``[t − 15 min, t)``.

Null where the row has no ``aobt_3``. ``CONTEXT``: other rows' timestamps, never a label.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

HOLD_MAX = 7_200.0
DELAY_WINDOW = "15m"

OUTPUTS = ("gr_overdue_apt", "gr_overdue_rw", "gr_active_rw", "gr_delay_15m")


def _count_le(sorted_values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """#{v ≤ at} for each `at`, against an ascending array (NaN-free)."""
    return np.searchsorted(sorted_values, at, side="right")


def _overdue(g: pl.DataFrame) -> np.ndarray:
    """#{j valid: eobt_j ≤ t < aobt_j} at each row's t = aobt (NaN where t is null)."""
    t = g["t_aobt"].to_numpy()
    valid = g["valid"].to_numpy()
    e = np.sort(g["t_eobt"].to_numpy()[valid])
    a = np.sort(g["t_aobt"].to_numpy()[valid])
    out = np.full(len(t), np.nan)
    ok = ~np.isnan(t)
    out[ok] = _count_le(e, t[ok]) - _count_le(a, t[ok])
    return out


def _active(g: pl.DataFrame) -> np.ndarray:
    """#{j ≠ row: aobt_j ≤ t < mvt_j} at each row's t = aobt."""
    t = g["t_aobt"].to_numpy()
    has = ~np.isnan(t)
    a = np.sort(t[has])
    m = np.sort(g["t_mvt"].to_numpy()[has])
    out = np.full(len(t), np.nan)
    # the row itself satisfies aobt ≤ t < mvt whenever mvt > aobt: subtract it
    self_in = has & (g["t_mvt"].to_numpy() > t)
    out[has] = _count_le(a, t[has]) - _count_le(m, t[has]) - self_in[has]
    return out


def featurize(deps: pl.DataFrame) -> pl.DataFrame:
    """All outputs. `deps`: mvt_id, airport, runway, mvt_time, aobt_3, eobt_1, sched_time. Pure."""
    secs = {
        "t_aobt": "aobt_3",
        "t_eobt": "eobt_1",
        "t_mvt": "mvt_time",
    }
    d = deps.with_columns(
        *[(pl.col(c).dt.epoch("ms") / 1000.0).fill_null(np.nan).alias(k) for k, c in secs.items()],
        pl.col("runway").fill_null("?"),
    ).with_columns(
        valid=(
            (pl.col("t_aobt") - pl.col("t_eobt")).is_between(0.0, HOLD_MAX, closed="left")
            & (pl.col("t_mvt") >= pl.col("t_aobt"))
        ).fill_null(False)
        & pl.col("t_aobt").is_not_nan()
        & pl.col("t_eobt").is_not_nan()
    )
    parts = []
    for _, g in d.group_by("airport"):
        g = g.with_columns(pl.Series("gr_overdue_apt", _overdue(g)))
        sub = []
        for _, r in g.group_by("runway"):
            sub.append(
                r.with_columns(
                    pl.Series("gr_overdue_rw", _overdue(r)), pl.Series("gr_active_rw", _active(r))
                )
            )
        parts.append(pl.concat(sub))
    out = pl.concat(parts)
    # recent delay to push: rolling mean over the airport's pushes in [t − 15 min, t)
    pushes = (
        deps.filter(pl.col("aobt_3").is_not_null())
        .select(
            "airport",
            pl.col("aobt_3").alias("t"),
            (pl.col("aobt_3") - pl.col("sched_time"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("dl"),
        )
        .sort("airport", "t")
    )
    roll = (
        pushes.rolling("t", period=DELAY_WINDOW, closed="left", group_by="airport")
        .agg(pl.col("dl").mean().alias("gr_delay_15m"))
        .unique(subset=["airport", "t"])
    )
    out = out.join(roll, left_on=["airport", "aobt_3"], right_on=["airport", "t"], how="left")
    return out.select("mvt_id", *OUTPUTS)


@feature(name="gate_retention", kind=FeatureKind.CONTEXT, outputs=OUTPUTS)
def gate_retention(ctx: FeatureContext) -> pl.DataFrame:
    """Overdue and active departures at this flight's push, and the recent delay to push."""
    deps = (
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "airport", "runway", "mvt_time", "aobt_3", "eobt_1", "sched_time")
        .collect(engine="streaming")
    )
    return align(ctx.spine, featurize(deps), OUTPUTS)
