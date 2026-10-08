"""Estimators built out of other estimators: one model per route, and one fitted on a subset.

Both take a nested model spec — ``{"kind": ..., "params": {...}}`` — so an experiment config can
describe "a separate LightGBM for Rome" without any new code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl


class RoutedPredictor:
    """One model per *route*, where a route is a named group of values — and each model may
    borrow the other routes' rows at a discount.

    One model per airport is all-or-nothing: ten airports, ten models, each blind to the other
    nine. That is the wrong shape when one airport is pathological and the rest are not. Here the
    split is chosen:

    ```yaml
    routes: {rome: [LIRF]}      # everything unlisted falls into the default route
    foreign_weight: 0.2         # a route's model trains on other routes' rows at 0.2x weight
    ```

    ``foreign_weight = 0`` gives one independent model per route: each route sees only its
    own rows. ``foreign_weight = 1`` makes every route identical to a global model, which is
    only useful as a control. Between them, a route keeps the shared structure the other rows
    teach — stand geometry, time of day — while its own rows dominate the fit.

    The motivation is protective as much as corrective. LIRF supplies 8% of departures and 62-80%
    of the squared error, most of it from corrupt labels no model can predict. Isolating it stops
    that distorting a fit shared with the other 92%.
    """

    DEFAULT_ROUTE = "rest"

    def __init__(
        self,
        by: str = "airport",
        routes: dict[str, list[str]] | None = None,
        base: dict[str, Any] | None = None,
        foreign_weight: float = 0.0,
        min_rows: int = 5_000,
        drop_for: dict[str, list[str]] | None = None,
    ) -> None:
        from taxiout import models

        self.by = by
        self.routes = routes or {}
        self.foreign_weight = float(foreign_weight)
        self.min_rows = min_rows
        #: Per-route feature masks: columns (or `prefix*` patterns) hidden from that route's
        #: model. The point is asymmetry — a block like `schedule` earns its keep at Rome and
        #: taxes the other nine (2.4 s of ex-LIRF RMSE in one ablation), so the rest route can
        #: drop it while the Rome route keeps it. The global fallback model sees everything.
        self.drop_for = {k: list(v) for k, v in (drop_for or {}).items()}
        self.base_spec = base or {"kind": "constant", "params": {}}
        self._factory: Callable[[], Any] = models.factory(
            self.base_spec["kind"], self.base_spec.get("params")
        )
        self.models_: dict[str, Any] = {}
        self.global_: Any = None

    def _masked(self, X: pl.DataFrame, route_name: str) -> pl.DataFrame:
        """The matrix as this route's model sees it: listed columns and prefixes removed."""
        patterns = self.drop_for.get(route_name)
        if not patterns:
            return X
        drop = [
            c
            for c in X.columns
            for pat in patterns
            if (pat.endswith("*") and c.startswith(pat[:-1])) or c == pat
        ]
        return X.drop(drop) if drop else X

    def _route_of(self, values: np.ndarray) -> np.ndarray:
        """Map each row's `by` value to its route name."""
        out = np.full(len(values), self.DEFAULT_ROUTE, dtype=object)
        for name, members in self.routes.items():
            out[np.isin(values, list(members))] = name
        return out

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> RoutedPredictor:
        y = np.asarray(y, dtype=np.float64)
        self.global_ = self._factory().fit(X, y)
        self.models_ = {}

        route = self._route_of(X[self.by].to_numpy())
        for name in set(route):
            own = route == name
            if own.sum() < self.min_rows:
                continue
            model = self._factory()
            if self.foreign_weight > 0:
                # Train on everything, with this route's rows carrying full weight and the
                # others discounted. Requires a base that accepts sample weights.
                weight = np.where(own, 1.0, self.foreign_weight)
                try:
                    self.models_[name] = model.fit(self._masked(X, name), y, weight=weight)
                    continue
                except TypeError:
                    pass  # base does not take weights; fall back to own rows only
            self.models_[name] = model.fit(self._masked(X.filter(own), name), y[own])
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert self.global_ is not None, "fit first"
        out = np.asarray(self.global_.predict(X), dtype=np.float64)
        route = self._route_of(X[self.by].to_numpy())
        for name, model in self.models_.items():
            mask = route == name
            if mask.any():
                rows = self._masked(X.filter(mask), name)
                out[mask] = np.asarray(model.predict(rows), dtype=np.float64)
        return out


class SubsetPredictor:
    """Fit ``base`` only on the rows whose ``by`` value is listed; the rest get ``fallback``.

    For a model meant to serve one population — the LIRF gate-delay model (`T = D − G`, OpenAir
    E33–E74) serves only Rome — `routed` would also fit a global model and a model for the other
    nine airports that nothing uses. Here the other rows get a cheap fallback (a constant by
    default), and only the listed rows' predictions mean anything.

    ```yaml
    kind: subset
    params: {by: airport, values: [LIRF], base: {...}}
    ```
    """

    def __init__(
        self,
        by: str = "airport",
        values: list[str] | None = None,
        base: dict[str, Any] | None = None,
        fallback: dict[str, Any] | None = None,
    ) -> None:
        from taxiout import models

        if not values:
            raise ValueError("subset needs at least one value to keep")
        self.by = by
        self.values = list(values)
        self.base_spec = base or {"kind": "constant", "params": {}}
        self.fallback_spec = fallback or {"kind": "constant", "params": {}}
        self._base = models.factory(self.base_spec["kind"], self.base_spec.get("params"))
        self._fallback = models.factory(
            self.fallback_spec["kind"], self.fallback_spec.get("params")
        )
        self.model_: Any = None
        self.fallback_: Any = None

    def _keep(self, X: pl.DataFrame) -> np.ndarray:
        return np.isin(X[self.by].cast(pl.Utf8).to_numpy(), self.values)

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> SubsetPredictor:
        y = np.asarray(y, dtype=np.float64)
        keep = self._keep(X)
        if not keep.any():
            raise ValueError(f"subset: no training row has {self.by} in {self.values}")
        self.model_ = self._base().fit(X.filter(keep), y[keep])
        rest = ~keep
        self.fallback_ = (
            self._fallback().fit(X.filter(rest), y[rest]) if rest.any() else self.model_
        )
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        keep = self._keep(X)
        out = np.zeros(X.height, dtype=np.float64)
        if keep.any():
            out[keep] = np.asarray(self.model_.predict(X.filter(keep)), dtype=np.float64)
        if (~keep).any():
            out[~keep] = np.asarray(self.fallback_.predict(X.filter(~keep)), dtype=np.float64)
        return out

    def importance(self) -> pl.DataFrame | None:
        from taxiout import models

        return models.feature_importance(self.model_) if self.model_ is not None else None
