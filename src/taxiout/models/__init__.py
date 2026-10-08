"""Model registry. An experiment config names a ``kind`` and this turns it into an estimator.

Specs nest: a ``routed``, ``subset``, ``offset`` or ``rules`` spec carries another spec in its
params, so "a separate LightGBM for Rome" or "CatBoost on the residual from a column" are
configuration rather than code. The kinds here are the ones the members of the best submission
use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

from taxiout.models.base import Predictor
from taxiout.models.baseline import ConstantPredictor
from taxiout.models.catboost_model import CatBoostPredictor
from taxiout.models.lgbm import LGBMPredictor
from taxiout.models.meta import RoutedPredictor, SubsetPredictor
from taxiout.models.offset import OffsetPredictor
from taxiout.models.rules import RulePredictor

MODELS: dict[str, type] = {
    "constant": ConstantPredictor,
    "lgbm": LGBMPredictor,
    "catboost": CatBoostPredictor,
    "routed": RoutedPredictor,
    "subset": SubsetPredictor,
    "rules": RulePredictor,
    "offset": OffsetPredictor,
}


def feature_importance(model: Any) -> pl.DataFrame | None:
    """Feature importance from any fitted model that can supply it, else None.

    Rough: it shows which features the model leaned on, it does not support an inference.
    """
    if hasattr(model, "importance"):
        return model.importance()
    return None


def factory(kind: str, params: dict[str, Any] | None = None) -> Callable[[], Predictor]:
    """Return a zero-argument constructor, so cross-validation gets a fresh model per fold."""
    if kind not in MODELS:
        raise KeyError(f"unknown model kind {kind!r}; known: {', '.join(sorted(MODELS))}")
    cls = MODELS[kind]
    kwargs = dict(params or {})
    return lambda: cls(**kwargs)


__all__ = [
    "MODELS",
    "CatBoostPredictor",
    "ConstantPredictor",
    "LGBMPredictor",
    "OffsetPredictor",
    "Predictor",
    "RoutedPredictor",
    "RulePredictor",
    "SubsetPredictor",
    "factory",
    "feature_importance",
]
