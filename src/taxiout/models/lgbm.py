"""LightGBM, wrapped to the project's estimator interface.

Two things this handles that a bare `LGBMRegressor` does not.

**Feature selection.** The design matrix carries spine columns (`mvt_id`, `month`, `split`, …)
alongside the features. Those must not reach the model — `month` in particular would let it
memorise the fold. Everything in :data:`taxiout.spine.SPINE_COLUMNS` is dropped automatically.

**Honest early stopping.** Stopping on the fold's validation month tunes the number of trees on
the very rows the score is read from, which inflates it. Instead a slice of the *training*
months is held out, taken from the end of the training period so the split is temporal rather
than random. ``stop_slice="days"`` holds out a random set of whole UTC days instead, spread over
every training month; under leave-one-month-out the default slice is Nov-Dec for both seasonal
folds, and whole days keep a day's shared congestion on one side of the cut. It behaves the same
in the twelve-month refit, where no month is held out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from taxiout import spine

#: Columns emitted by feature blocks with this prefix are treated as categorical.
CATEGORICAL_PREFIX = "cat_"

#: Early-stopping slices `LGBMPredictor` can hold out; see the module docstring.
STOP_SLICES = ("last", "days")

DEFAULTS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    # 511, not the 127 this once was. That 127 was chosen to keep the binned dataset inside the
    # development machine's ~3 GB, and it cost 77 s of RMSE: an ablation that changed nothing
    # else gained 76.7 s, while num_leaves and a slower schedule gained 1.7 s and 0.
    # The off-block proxies that dominate feature importance are continuous, so coarse binning
    # blunts exactly the splits that matter.
    "max_bin": 511,
    "verbosity": -1,
}


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Model-visible columns: everything that is not a spine column."""
    return [c for c in frame.columns if c not in set(spine.SPINE_COLUMNS)]


class LGBMPredictor:
    """Gradient-boosted trees over the assembled feature matrix.

    ``num_boost_round`` is the cap. With ``early_stopping_rounds`` set, the actual number of
    trees is chosen on a temporal slice of the training data held out for that purpose, and
    recorded in ``best_iteration_``.
    """

    def __init__(
        self,
        num_boost_round: int = 2000,
        early_stopping_rounds: int | None = 100,
        valid_fraction: float = 0.15,
        seed: int = 0,
        stop_slice: str = "last",
        **params: Any,
    ) -> None:
        if stop_slice not in STOP_SLICES:
            raise ValueError(f"stop_slice must be one of {STOP_SLICES}, got {stop_slice!r}")
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.valid_fraction = valid_fraction
        self.stop_slice = stop_slice
        self.seed = seed
        self.params = {**DEFAULTS, "seed": seed, **params}
        self.booster_ = None
        self.features_: list[str] = []
        self.categoricals_: list[str] = []
        self.best_iteration_: int | None = None

    #: With False (LightGBM's default) ``lgb.train`` round-trips the booster through
    #: ``model_to_string``, which keeps only the first ``best_iteration`` trees. Predictions are
    #: unaffected -- they use ``best_iteration_`` either way -- but a subclass that reads the curve
    #: *past* the global best needs the trees after it.
    _keep_training_booster = False

    def _temporal_split(self, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """The early-stopping split inside the training months.

        ``last``: the last ``valid_fraction`` of rows by month label. Sorting by month keeps it
        chronological even though the matrix arrives in spine order. ``days``: a seeded random
        ``valid_fraction`` of the distinct UTC days in ``mvt_time``, all rows of a chosen day.
        """
        if self.stop_slice == "days":
            day = X["mvt_time"].dt.date().to_physical().to_numpy()
            days = np.unique(day)
            n = max(1, round(len(days) * self.valid_fraction))
            chosen = np.random.default_rng(self.seed).choice(days, size=n, replace=False)
            stop = np.isin(day, chosen)
            return np.flatnonzero(~stop), np.flatnonzero(stop)
        order = np.argsort(X["month"].to_numpy(), kind="stable")
        cut = int(len(order) * (1 - self.valid_fraction))
        return order[:cut], order[cut:]

    def _dataset(self, frame: pl.DataFrame, y, reference=None, weight=None):
        import lightgbm as lgb

        X = frame.select(self.features_).to_pandas()
        for column in self.categoricals_:
            X[column] = X[column].astype("category")
        return lgb.Dataset(
            X,
            label=y,
            weight=weight,
            categorical_feature=self.categoricals_ or "auto",
            reference=reference,
            free_raw_data=True,
        )

    def fit(
        self, X: pl.DataFrame, y: np.ndarray, weight: np.ndarray | None = None
    ) -> LGBMPredictor:
        """Fit the booster.

        ``weight`` scales each row's contribution to the loss. It exists for `RoutedPredictor`,
        which trains a route's model on every row while discounting the ones outside that route.
        """
        import lightgbm as lgb

        self.features_ = feature_columns(X)
        self.categoricals_ = [c for c in self.features_ if c.startswith(CATEGORICAL_PREFIX)]

        y = np.asarray(y, dtype=np.float64)
        callbacks = []
        valid_sets = []

        if self.early_stopping_rounds and self.valid_fraction > 0:
            fit_idx, stop_idx = self._temporal_split(X)
            w_fit = None if weight is None else weight[fit_idx]
            w_stop = None if weight is None else weight[stop_idx]
            train_set = self._dataset(X[fit_idx], y[fit_idx], weight=w_fit)
            valid_sets = [
                self._dataset(X[stop_idx], y[stop_idx], reference=train_set, weight=w_stop)
            ]
            callbacks = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        else:
            train_set = self._dataset(X, y, weight=weight)

        self.booster_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks or None,
            keep_training_booster=self._keep_training_booster,
        )
        self.best_iteration_ = (
            getattr(self.booster_, "best_iteration", None) or self.num_boost_round
        )
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert self.booster_ is not None, "fit first"
        frame = X.select(self.features_).to_pandas()
        for column in self.categoricals_:
            frame[column] = frame[column].astype("category")
        return self.booster_.predict(frame, num_iteration=self.best_iteration_)

    def importance(self, kind: str = "gain") -> pl.DataFrame:
        """Feature importance, largest first — the ablation shortcut."""
        assert self.booster_ is not None, "fit first"
        return pl.DataFrame(
            {"feature": self.features_, "importance": self.booster_.feature_importance(kind)}
        ).sort("importance", descending=True)
