"""Scoring. RMSE in seconds is the competition metric; everything else is diagnostic."""

from __future__ import annotations

import numpy as np
import polars as pl


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2
            )
        )
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)))
    )


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    )


def rmse_by(
    frame: pl.DataFrame, group: str, *, truth: str = "y", pred: str = "yhat"
) -> pl.DataFrame:
    """Per-group RMSE, largest group first. Used for the per-airport breakdown."""
    return (
        frame.group_by(group)
        .agg(
            pl.len().alias("n"),
            ((pl.col(pred) - pl.col(truth)) ** 2).mean().sqrt().alias("rmse"),
        )
        .sort("n", descending=True)
    )


def breakdown(
    frame: pl.DataFrame,
    by: list[str],
    *,
    truth: str = "y",
    pred: str = "yhat",
    trim_above: float | None = None,
) -> pl.DataFrame:
    """Full error breakdown per group: count, RMSE, trimmed RMSE, MAE and bias.

    ``trim_above`` drops rows whose *true* value exceeds it from the trimmed column only, so the
    headline RMSE stays the competition metric while the trimmed one stays readable.
    """
    error = pl.col(pred) - pl.col(truth)
    aggregations = [
        pl.len().alias("n"),
        (error**2).mean().sqrt().alias("rmse"),
        error.abs().mean().alias("mae"),
        error.mean().alias("bias"),
    ]
    if trim_above is not None:
        keep = pl.col(truth) <= trim_above
        aggregations += [
            (error**2).filter(keep).mean().sqrt().alias("rmse_trimmed"),
            (~keep).sum().alias("n_excluded"),
        ]
    return frame.group_by(by).agg(aggregations).sort(by)
