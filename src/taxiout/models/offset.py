"""Fit the residual over a per-row baseline column; predict baseline plus residual.

The winners of the two previous PRC Data Challenges normalised the target before modelling —
load fraction in 2024, fuel flow rate in 2025. The additive version suits RMSE: fit
``y − X[column]`` and add the column back at predict. Because the offset is a per-row constant,
minimising squared error on the residual *is* minimising it on the original scale — no
reweighting needed, unlike the multiplicative variants.

In the best submission it serves the Rome gate-delay model (``lg01_lirf_gate_delay``): with
``column: sc_gap_sec`` (takeoff − schedule), the residual is minus the gate delay, so the model
learns the gate delay and predicts taxi-out = gap − gate delay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class OffsetPredictor:
    """Wrap any base spec; shift the target down by a column for fitting, back up for predicting."""

    def __init__(
        self,
        column: str,
        base: dict[str, Any] | None = None,
        clip: list[float] | tuple[float, float] | None = None,
    ) -> None:
        from taxiout import models

        self.column = column
        #: Optional ``[lo, hi]`` applied to the baseline before use, e.g. ``[0, 7200]`` for the
        #: NM clock ``MVT − AOBT_3`` as an anchor (kind-mango v74: an unclipped plan-clock anchor
        #: lost +35.9 s live because the residual then has to undo multi-hour anchors).
        self.clip = (float(clip[0]), float(clip[1])) if clip is not None else None
        self.base_spec = base or {"kind": "constant", "params": {}}
        self._factory = models.factory(self.base_spec["kind"], self.base_spec.get("params"))
        self.base_: Any = None

    def _baseline(self, X: pl.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            raise KeyError(
                f"offset column {self.column!r} is not in the matrix; is the block that "
                "provides it in the experiment's features list?"
            )
        # A null baseline (e.g. a stand with too few arrivals) contributes no shift rather than
        # poisoning the row with NaN.
        col = X[self.column].cast(pl.Float64)
        if self.clip is not None:
            col = col.clip(*self.clip)
        return col.fill_null(0.0).to_numpy()

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> OffsetPredictor:
        shifted = np.asarray(y, dtype=np.float64) - self._baseline(X)
        self.base_ = self._factory().fit(X, shifted)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.base_ is None:
            raise RuntimeError("OffsetPredictor.predict called before fit")
        return np.asarray(self.base_.predict(X), dtype=np.float64) + self._baseline(X)

    def importance(self) -> pl.DataFrame | None:
        from taxiout import models

        return models.feature_importance(self.base_) if self.base_ is not None else None
