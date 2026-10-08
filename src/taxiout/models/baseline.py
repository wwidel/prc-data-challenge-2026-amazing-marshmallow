"""The constant estimator: the default fallback of the ``subset`` model."""

from __future__ import annotations

import numpy as np
import polars as pl


class ConstantPredictor:
    """Predict one number everywhere — the mean or median of the training target."""

    def __init__(self, stat: str = "mean") -> None:
        if stat not in {"mean", "median"}:
            raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
        self.stat = stat
        self.value_: float | None = None

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> ConstantPredictor:
        self.value_ = float(np.mean(y) if self.stat == "mean" else np.median(y))
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert self.value_ is not None, "fit first"
        return np.full(X.height, self.value_, dtype=np.float64)
