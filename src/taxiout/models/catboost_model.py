"""CatBoost, the second model family of the stack."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from taxiout.models.lgbm import CATEGORICAL_PREFIX, feature_columns


class CatBoostPredictor:
    """CatBoost with native categorical handling.

    Categoricals are passed as strings and left to CatBoost's ordered target statistics, which
    is the point of using it — it encodes a 1,900-value stand column by its target behaviour
    rather than by splitting on an arbitrary subset of levels.
    """

    def __init__(
        self,
        iterations: int = 2000,
        learning_rate: float = 0.05,
        depth: int = 8,
        thread_count: int = 4,
        **params: Any,
    ) -> None:
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "loss_function": "RMSE",
            "thread_count": thread_count,
            "random_seed": 0,
            "verbose": False,
            "allow_writing_files": False,
            **params,
        }
        self.model_ = None
        self.features_: list[str] = []
        self.categoricals_: list[str] = []

    def _frame(self, X: pl.DataFrame):
        out = X.select(self.features_).to_pandas()
        for column in self.categoricals_:
            out[column] = out[column].astype("int64").astype(str)
        return out

    def fit(self, X: pl.DataFrame, y: np.ndarray, weight: np.ndarray | None = None):
        from catboost import CatBoostRegressor

        self.features_ = feature_columns(X)
        self.categoricals_ = [c for c in self.features_ if c.startswith(CATEGORICAL_PREFIX)]
        self.model_ = CatBoostRegressor(**self.params, cat_features=self.categoricals_)
        self.model_.fit(self._frame(X), np.asarray(y, dtype=np.float64), sample_weight=weight)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        return np.asarray(self.model_.predict(self._frame(X)), dtype=np.float64)

    def importance(self) -> pl.DataFrame:
        assert self.model_ is not None, "fit first"
        return pl.DataFrame(
            {"feature": self.features_, "importance": self.model_.get_feature_importance()}
        ).sort("importance", descending=True)
