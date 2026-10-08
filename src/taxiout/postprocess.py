"""The last layers of the prediction and the file it is written to.

Everything after the stack: the per airport-month shift, the admissible bounds and the LOBT
window, each a pure function of a frame with a ``prediction`` column; then the competition's file
format (``submitting.parquet`` with ``TAXITIME_SEC_mvt`` filled in). Rows are matched on
``MVT_ID_mvt`` and anything that does not line up exactly is refused: a misaligned file would
score as noise and look like a modelling failure.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from taxiout import io

#: Predictions are clipped into this range before rounding. The lower bound is 0 because a
#: negative taxi-out is not a physical prediction. The upper bound is only a sanity rail: the 2025
#: labels reach 131,167 s, and an earlier 20,000 s ceiling flattened the 27 rows with the largest
#: values.
CLIP = (0.0, 200_000.0)

#: Per-airport prediction ceilings. LEBL and LEMD labels are truncated by the data provider:
#: across all twelve 2025 months no departure or arrival label exceeds ~3,600 s (maxima 3,604
#: and 3,598; 65 and 35 labels sit in 3,300-3,610 and none above). A prediction over the
#: ceiling there cannot be right. Source: scripts/analysis/r03_label_ceilings.py.
AIRPORT_CEILINGS: dict[str, float] = {"LEBL": 3_600.0, "LEMD": 3_600.0}

#: A taxi cannot exceed the delay it was part of by more than this, because the aircraft has to
#: have left the stand after its scheduled time. 2025 violation rate: 0.004% (88 of 2.08M rows).
GAP_SLACK = 1_800.0

#: Half-width of the LOBT window. On every 2025 departure that has a LOBT (2,062,577 of them),
#: ``|block_time − lobt| <= 3606 s``, extremes exactly ±3606 at every airport
#: (scripts/analysis/a112_lobt_window.py). The property was published by team outgoing-boat
#: (https://github.com/EnioAguiar/prc-taxiout-2026), whose diagnostic submission showed that it
#: holds on the 2026 ranking set too. So the taxi-out lies in ``mvt − lobt ± LOBT_WINDOW``.
LOBT_WINDOW = 3_606.0


def apply_gap_bound(frame: pl.DataFrame, slack: float = GAP_SLACK) -> pl.DataFrame:
    """Clip ``prediction`` to ``gap + slack`` on rows with a positive takeoff−schedule gap.

    An admissible bound: ``gap`` uses only movement-side columns the ranking set carries. It
    subsumes the per-airport ceilings for positive-gap rows and, more usefully, caps the handful
    of rows where a mixture predictor puts thousands of seconds on a flight that was barely late
    (16 of the 20 rows steered by the sub-cap hedge's thinnest band violate it). Rows with a
    null or non-positive gap are left alone. Needs a ``gap`` column.
    """
    if "gap" not in frame.columns:
        raise ValueError("apply_gap_bound needs a gap column (takeoff - schedule, seconds)")
    limit = pl.col("gap") + slack
    return frame.with_columns(
        pl.when(pl.col("gap").is_not_null() & (pl.col("gap") > 0) & (pl.col("prediction") > limit))
        .then(limit)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )


def apply_lobt_window(frame: pl.DataFrame, window: float = LOBT_WINDOW) -> pl.DataFrame:
    """Project ``prediction`` onto ``[m_l − window, m_l + window]``, ``m_l = mvt − lobt`` seconds.

    Admissible: since ``block_time`` lies within ``window`` of ``lobt``, the true taxi-out lies in
    that interval, and projecting onto an interval that contains the truth can only shrink each
    row's error (by at least the distance moved). Rows with a null ``m_l`` (no LOBT: every
    NM-unmatched row) are left alone. The lower edge is floored at 0. Needs an ``m_l`` column.
    """
    if "m_l" not in frame.columns:
        raise ValueError("apply_lobt_window needs an m_l column (takeoff - lobt, seconds)")
    lo = pl.max_horizontal(pl.col("m_l") - window, pl.lit(0.0))
    hi = pl.col("m_l") + window
    return frame.with_columns(
        pl.when(pl.col("m_l").is_null())
        .then(pl.col("prediction"))
        .otherwise(pl.col("prediction").clip(lo, hi))
        .alias("prediction")
    )


def apply_cell_shift(frame: pl.DataFrame, shifts: pl.DataFrame) -> pl.DataFrame:
    """Add each airport-month's ``shift`` (seconds) to ``prediction`` on every row of that cell.

    ``shifts`` holds (airport, month ``YYYY-MM``, shift); rows of other airport-months are left
    alone. The shifts come from ``scripts/analysis/a120_cell_shift.py``: the published monthly
    taxi-out (EUROCONTROL's APDF indicator) against our mean, net of the offset the 2025 labels
    measure between the two populations. Relative, so it can sit on any base. Needs ``airport``
    and ``month`` columns.
    """
    for col in ("airport", "month"):
        if col not in frame.columns:
            raise ValueError(f"apply_cell_shift needs an {col} column")
    s = shifts.select("airport", "month", pl.col("shift").cast(pl.Float64).alias("_shift"))
    if s.group_by("airport", "month").len().filter(pl.col("len") > 1).height:
        raise ValueError("apply_cell_shift: an airport-month appears twice in the shifts")
    return (
        frame.join(s, on=["airport", "month"], how="left", maintain_order="left")
        .with_columns((pl.col("prediction") + pl.col("_shift").fill_null(0.0)).alias("prediction"))
        .drop("_shift")
    )


def apply_airport_ceilings(
    frame: pl.DataFrame, ceilings: dict[str, float] | None = None
) -> pl.DataFrame:
    """Clip ``prediction`` down to each airport's ceiling. Needs an ``airport`` column."""
    ceilings = AIRPORT_CEILINGS if ceilings is None else ceilings
    if not ceilings:
        return frame
    limit = pl.lit(None, dtype=pl.Float64)
    for airport, value in ceilings.items():
        limit = pl.when(pl.col("airport") == airport).then(pl.lit(value)).otherwise(limit)
    return frame.with_columns(
        pl.min_horizontal(pl.col("prediction"), limit.fill_null(float("inf"))).alias("prediction")
    )


def template() -> pl.DataFrame:
    """The submission template, with its canonical `mvt_id` alongside the raw column."""
    return io.scan_submitting().collect()


def ranking_departure_ids() -> pl.Series:
    return (
        io.scan_ranking()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id")
        .collect(engine="streaming")["mvt_id"]
    )


def build_frame(predictions: pl.DataFrame) -> pl.DataFrame:
    """Join predictions onto the template by id and return it in the template's row order.

    ``predictions`` needs an ``mvt_id`` column and a ``prediction`` column.
    """
    if "mvt_id" not in predictions.columns or "prediction" not in predictions.columns:
        raise ValueError("predictions must have columns 'mvt_id' and 'prediction'")

    filled = (
        template()
        .join(
            predictions.select("mvt_id", "prediction"),
            on="mvt_id",
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.col("prediction").clip(*CLIP).round(0).cast(pl.Int32).alias("TAXITIME_SEC_mvt")
        )
    )
    return filled.select("MVT_ID_mvt", "TAXITIME_SEC_mvt")


def validate(frame: pl.DataFrame) -> None:
    """Refuse anything the ranking service would reject, and a few things it would not."""
    expected = template()
    problems: list[str] = []

    if frame.columns != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        problems.append(f"columns are {frame.columns}, expected ['MVT_ID_mvt', 'TAXITIME_SEC_mvt']")
    if frame.height != expected.height:
        problems.append(f"{frame.height:,} rows, expected {expected.height:,}")
    if frame["TAXITIME_SEC_mvt"].null_count():
        problems.append(f"{frame['TAXITIME_SEC_mvt'].null_count():,} null predictions")
    if frame["TAXITIME_SEC_mvt"].dtype != pl.Int32:
        problems.append(f"TAXITIME_SEC_mvt is {frame['TAXITIME_SEC_mvt'].dtype}, expected Int32")
    if frame["MVT_ID_mvt"].dtype != expected["MVT_ID_mvt"].dtype:
        problems.append(
            f"MVT_ID_mvt is {frame['MVT_ID_mvt'].dtype}, expected {expected['MVT_ID_mvt'].dtype}"
        )

    submitted, wanted = set(frame["MVT_ID_mvt"].to_list()), set(expected["MVT_ID_mvt"].to_list())
    if submitted - wanted:
        problems.append(f"{len(submitted - wanted):,} ids not in the template")
    if wanted - submitted:
        problems.append(f"{len(wanted - submitted):,} template ids missing")
    if frame["MVT_ID_mvt"].n_unique() != frame.height:
        problems.append("duplicate ids")

    ranking_ids = set(ranking_departure_ids().to_list())
    template_ids = set(template()["mvt_id"].to_list())
    if ranking_ids != template_ids:
        problems.append("template ids do not match the ranking departures — check the raw files")

    if problems:
        raise ValueError("submission is invalid:\n  - " + "\n  - ".join(problems))


def write(predictions: pl.DataFrame, path: Path, *, dry_run: bool = False) -> pl.DataFrame:
    """Build the file frame, validate it, and write it to ``path`` unless ``dry_run``.

    Validation runs before anything touches disk, so a rejected build leaves no file.
    """
    frame = build_frame(predictions)
    validate(frame)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    return frame
