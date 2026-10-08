"""Deterministic overrides on top of a fitted model.

Some rows are not a regression problem. ``block_time`` is a field an airport system *writes*,
not a quantity it measures, and on a few identifiable populations it writes a value we already
hold — so the target is an exact function of columns the ranking set gives us, and a tree can
only ever memorise points on that function while a rule states it.

The motivating case: unmatched departures at LIRF whose schedule gap exceeds twelve hours. There
are 29 in 2025, of which 18 have ``taxi_sec`` equal to that gap to within six seconds, and the
other 11 sit a day above an ordinary taxi. The best model predicts about 1,000 s for all of
them. The same cell holds 11 rows in the 2026 ranking set, identifiable without any label.

A rule is a *predicate* over feature columns and a *value expression* over the same columns.
Neither may read the target, so this is leakage-free by construction and needs no fold-aware
treatment — the rule fires identically in training, validation and the ranking set.

```yaml
model:
  kind: rules
  params:
    base: {kind: lgbm, params: {max_bin: 511}}
    clip: [300.0, 200000.0]
    rules:
      - name: lirf_unmatched_12h
        when:
          - {column: flt_no_match, op: "==", value: 1}
          - {column: airport, op: in, value: [LIRF]}
          - {column: sc_gap_sec, op: ">", value: 43200}
        predict:
          mix:
            - {weight: 0.62, column: sc_gap_sec}
            - {weight: 0.38, constant: 87540}
```

**Rules are ordered and the first match wins**, so a specific rule is written above the general
one it carves out of. A row matching no rule keeps the base model's prediction.

**A null never matches.** Comparing a null with ``>`` yields null in polars, which would
otherwise propagate into the mask; every predicate is filled with ``False`` before use. That
matters here because ``ob_aobt_sec`` is null on exactly the unmatched rows some rules select on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

#: Comparisons a predicate may use. Kept small and explicit so a typo in a config fails loudly
#: rather than silently matching nothing.
OPERATORS: dict[str, Callable[[pl.Expr, Any], pl.Expr]] = {
    "==": lambda col, v: col == v,
    "!=": lambda col, v: col != v,
    ">": lambda col, v: col > v,
    ">=": lambda col, v: col >= v,
    "<": lambda col, v: col < v,
    "<=": lambda col, v: col <= v,
    "in": lambda col, v: col.is_in(list(v)),
    "not_in": lambda col, v: ~col.is_in(list(v)),
    "is_null": lambda col, v: col.is_null() if _require_bool(v) else col.is_not_null(),
}


def _require_bool(value: Any) -> bool:
    """`is_null` needs an explicit true/false; omitting it used to mean `is_not_null` silently."""
    if not isinstance(value, bool):
        raise ValueError("operator 'is_null' needs an explicit boolean 'value'")
    return value


def _columns_used(rule: dict[str, Any]) -> set[str]:
    """Every column a rule reads, so a typo fails before anything is fitted."""
    used = {clause["column"] for clause in rule["when"]}
    predict = rule["predict"]
    if "column" in predict:
        used.add(predict["column"])
    for part in predict.get("mix", []):
        if "column" in part:
            used.add(part["column"])
    return used


def _predicate(clauses: list[dict[str, Any]]) -> pl.Expr:
    """Conjunction of clauses, with nulls treated as non-matching."""
    if not clauses:
        raise ValueError("a rule needs at least one 'when' clause; an empty one matches every row")
    expr: pl.Expr | None = None
    for clause in clauses:
        column, op = clause["column"], clause.get("op", "==")
        if op not in OPERATORS:
            raise KeyError(f"unknown operator {op!r}; known: {', '.join(sorted(OPERATORS))}")
        term = OPERATORS[op](pl.col(column), clause.get("value"))
        expr = term if expr is None else (expr & term)
    return expr.fill_null(False)  # type: ignore[union-attr]


def _value(spec: dict[str, Any]) -> pl.Expr:
    """The value a matching row takes: a column, a constant, or a weighted mix of both.

    A mix is how a bimodal cell is served under squared loss — the conditional mean minimises it,
    so a cell that is 62% ``gap`` and 38% ``gap + one day`` is predicted as that average rather
    than as either branch.
    """
    if "mix" in spec:
        terms = [
            pl.lit(float(part["weight"]))
            * (
                pl.col(part["column"]).cast(pl.Float64)
                if "column" in part
                else pl.lit(float(part["constant"]))
            )
            for part in spec["mix"]
        ]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total
    if "column" in spec:
        return pl.col(spec["column"]).cast(pl.Float64)
    if "constant" in spec:
        return pl.lit(float(spec["constant"]))
    raise ValueError(f"a 'predict' needs one of column/constant/mix, got {sorted(spec)}")


class RulePredictor:
    """Wrap any base spec; override its predictions where a rule fires."""

    def __init__(
        self,
        base: dict[str, Any] | None = None,
        rules: list[dict[str, Any]] | None = None,
        clip: tuple[float, float] | list[float] | None = None,
    ) -> None:
        from taxiout import models

        self.base_spec = base or {"kind": "constant", "params": {}}
        self.rules = list(rules or [])
        self.clip = tuple(clip) if clip is not None else None
        for rule in self.rules:  # fail at construction, not after an hour of fitting
            _predicate(rule["when"])
            _value(rule["predict"])
        self._factory: Callable[[], Any] = models.factory(
            self.base_spec["kind"], self.base_spec.get("params")
        )
        self.base_: Any = None
        self.fired_: dict[str, int] = {}

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> RulePredictor:
        """Fit the base. The rules need no fitting — they read no label.

        Column names are checked *before* the base is fitted: a typo in a rule should cost a
        second, not an hour of LightGBM followed by a KeyError in `predict`.
        """
        self.check_columns(X)
        self.base_ = self._factory().fit(X, np.asarray(y, dtype=np.float64))
        return self

    def check_columns(self, X: pl.DataFrame) -> None:
        """Raise if any rule reads a column the matrix does not have."""
        available = set(X.columns)
        for index, rule in enumerate(self.rules):
            missing = sorted(_columns_used(rule) - available)
            if missing:
                name = rule.get("name", f"rule_{index}")
                raise KeyError(
                    f"rule {name!r} reads column(s) {missing} which the matrix does not have; "
                    "is the feature block that provides them in the experiment's features list?"
                )

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.base_ is None:
            raise RuntimeError("RulePredictor.predict called before fit")
        out = np.asarray(self.base_.predict(X), dtype=np.float64)

        claimed = np.zeros(X.height, dtype=bool)  # first match wins
        self.fired_ = {}
        for index, rule in enumerate(self.rules):
            name = rule.get("name", f"rule_{index}")
            evaluated = X.select(
                _predicate(rule["when"]).alias("_hit"),
                _value(rule["predict"]).alias("_val"),
            )
            hit = evaluated["_hit"].to_numpy() & ~claimed
            self.fired_[name] = int(hit.sum())
            if hit.any():
                values = evaluated["_val"].to_numpy().astype(np.float64)
                # A rule that fires on a row whose value expression is null is a config error,
                # not something to paper over: leave the base prediction and count it as unfired.
                usable = hit & np.isfinite(values)
                out[usable] = values[usable]
                claimed |= usable
                self.fired_[name] = int(usable.sum())

        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return out

    def importance(self) -> pl.DataFrame | None:
        """The base model's importance. The rules have none — they are not fitted."""
        from taxiout import models

        return models.feature_importance(self.base_) if self.base_ is not None else None
