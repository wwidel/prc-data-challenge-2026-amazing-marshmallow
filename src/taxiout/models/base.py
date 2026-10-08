"""The estimator interface every model in this project implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class Predictor(Protocol):
    """Fit on a design matrix, predict taxi-out seconds.

    ``X`` is the assembled matrix: spine columns plus every feature column, so a model may read
    ``airport`` or ``month`` directly as well as the numeric features.
    """

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> Predictor: ...

    def predict(self, X: pl.DataFrame) -> np.ndarray: ...
