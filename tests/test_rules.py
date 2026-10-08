"""RulePredictor: deterministic overrides on a fitted base.

These are pure-logic tests — no spine, no raw data — so they always run. The cases that matter
are the ones where a rule could silently do nothing or silently do the wrong thing: nulls,
ordering, and a value expression that does not evaluate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from taxiout import models
from taxiout.models.base import Predictor
from taxiout.models.rules import RulePredictor, _predicate, _value

CONSTANT_BASE = {"kind": "constant", "params": {}}

#: Rule D as the strategy document defines it, and as a027 verified it.
RULE_D = {
    "name": "lirf_unmatched_12h",
    "when": [
        {"column": "flt_no_match", "op": "==", "value": 1},
        {"column": "airport", "op": "in", "value": ["LIRF"]},
        {"column": "sc_gap_sec", "op": ">", "value": 43_200},
    ],
    "predict": {
        "mix": [
            {"weight": 0.62, "column": "sc_gap_sec"},
            {"weight": 0.38, "constant": 87_540},
        ]
    },
}


def frame() -> pl.DataFrame:
    """Six rows spanning every branch a rule can take."""
    return pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4, 5, 6],
            "airport": ["LIRF", "LIRF", "LFPG", "LIRF", "LIRF", "EDDF"],
            "flt_no_match": [1, 1, 1, 0, 1, 0],
            "sc_gap_sec": [50_000.0, 1_000.0, 90_000.0, 60_000.0, None, 2_000.0],
            "ob_aobt_sec": [None, None, None, 20_000.0, None, 900.0],
        }
    )


def fitted(**kwargs) -> RulePredictor:
    model = RulePredictor(base=CONSTANT_BASE, **kwargs)
    X = frame()
    return model.fit(X, np.full(X.height, 1000.0))


def test_rule_predictor_satisfies_the_predictor_protocol():
    assert isinstance(fitted(rules=[RULE_D]), Predictor)


def test_registered_under_its_config_kind():
    assert models.MODELS["rules"] is RulePredictor
    assert isinstance(models.factory("rules", {"base": CONSTANT_BASE})(), RulePredictor)


def test_rule_fires_only_on_rows_matching_every_clause():
    model = fitted(rules=[RULE_D])
    out = model.predict(frame())
    # Row 1 alone satisfies unmatched AND LIRF AND gap > 12 h.
    assert model.fired_["lirf_unmatched_12h"] == 1
    assert out[0] == pytest.approx(0.62 * 50_000 + 0.38 * 87_540)
    # Everything else keeps the base prediction.
    assert out[1:] == pytest.approx(np.full(5, 1000.0))


def test_a_null_never_matches_a_comparison():
    """Row 5 is unmatched at LIRF but its gap is null; polars would give null, not False."""
    model = fitted(rules=[RULE_D])
    out = model.predict(frame())
    assert out[4] == pytest.approx(1000.0)
    mask = frame().select(_predicate(RULE_D["when"]).alias("hit"))["hit"]
    assert mask.null_count() == 0
    assert mask.to_list() == [True, False, False, False, False, False]


def test_first_matching_rule_wins():
    specific = dict(RULE_D, name="specific", predict={"constant": 11.0})
    general = {
        "name": "general",
        "when": [{"column": "airport", "op": "in", "value": ["LIRF"]}],
        "predict": {"constant": 22.0},
    }
    out = fitted(rules=[specific, general]).predict(frame())
    assert out[0] == pytest.approx(11.0)  # claimed by the specific rule
    assert out[1] == pytest.approx(22.0)  # LIRF, but not the specific cell
    assert out[2] == pytest.approx(1000.0)  # LFPG, untouched


def test_rule_order_is_respected_when_reversed():
    specific = dict(RULE_D, name="specific", predict={"constant": 11.0})
    general = {
        "name": "general",
        "when": [{"column": "airport", "op": "in", "value": ["LIRF"]}],
        "predict": {"constant": 22.0},
    }
    model = fitted(rules=[general, specific])
    out = model.predict(frame())
    assert out[0] == pytest.approx(22.0)  # the general rule got there first
    assert model.fired_["specific"] == 0


def test_a_rule_whose_value_is_null_leaves_the_base_prediction():
    """ob_aobt_sec is null on unmatched rows — the rule must not write a NaN."""
    rule = {
        "name": "aobt_identity",
        "when": [{"column": "flt_no_match", "op": "==", "value": 1}],
        "predict": {"column": "ob_aobt_sec"},
    }
    model = fitted(rules=[rule])
    out = model.predict(frame())
    assert np.isfinite(out).all()
    assert out == pytest.approx(np.full(6, 1000.0))
    assert model.fired_["aobt_identity"] == 0


def test_column_and_constant_value_expressions():
    by_column = {
        "name": "identity",
        "when": [{"column": "flt_no_match", "op": "==", "value": 0}],
        "predict": {"column": "ob_aobt_sec"},
    }
    out = fitted(rules=[by_column]).predict(frame())
    assert out[3] == pytest.approx(20_000.0)
    assert out[5] == pytest.approx(900.0)


def test_clip_is_applied_after_the_rules():
    out = fitted(rules=[RULE_D], clip=[300.0, 5_000.0]).predict(frame())
    assert out.max() == pytest.approx(5_000.0)
    assert out.min() >= 300.0


def test_base_passes_through_when_no_rule_is_given():
    out = fitted(rules=[]).predict(frame())
    assert out == pytest.approx(np.full(6, 1000.0))


def test_unknown_operator_raises_at_construction():
    bad = {"when": [{"column": "airport", "op": "≈", "value": "LIRF"}], "predict": {"constant": 1}}
    with pytest.raises(KeyError, match="unknown operator"):
        RulePredictor(base=CONSTANT_BASE, rules=[bad])


def test_empty_predicate_raises_rather_than_matching_everything():
    with pytest.raises(ValueError, match="at least one"):
        RulePredictor(base=CONSTANT_BASE, rules=[{"when": [], "predict": {"constant": 1}}])


def test_malformed_value_expression_raises_at_construction():
    bad = {"when": [{"column": "airport", "op": "==", "value": "LIRF"}], "predict": {"oops": 1}}
    with pytest.raises(ValueError, match="column/constant/mix"):
        RulePredictor(base=CONSTANT_BASE, rules=[bad])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        RulePredictor(base=CONSTANT_BASE, rules=[RULE_D]).predict(frame())


def test_rules_read_no_label_so_two_different_targets_give_identical_overrides():
    """The leakage guarantee: the override depends on columns, never on y."""
    X = frame()
    low = RulePredictor(base=CONSTANT_BASE, rules=[RULE_D]).fit(X, np.full(X.height, 10.0))
    high = RulePredictor(base=CONSTANT_BASE, rules=[RULE_D]).fit(X, np.full(X.height, 99_000.0))
    assert low.predict(X)[0] == pytest.approx(high.predict(X)[0])
    assert low.predict(X)[1] != pytest.approx(high.predict(X)[1])  # the base did move


def test_mix_weights_are_a_conditional_mean():
    """62/38 on a bimodal cell must land between the branches, not on one of them."""
    out = fitted(rules=[RULE_D]).predict(frame())
    gap, offset = 50_000.0, 87_540.0
    assert min(gap, offset) < out[0] < max(gap, offset)


def test_value_expression_builds_for_each_supported_form():
    for spec in ({"column": "sc_gap_sec"}, {"constant": 5.0}, RULE_D["predict"]):
        assert isinstance(_value(spec), pl.Expr)


def test_a_typo_in_a_column_name_fails_before_the_base_is_fitted():
    """Catching this in predict would mean an hour of LightGBM followed by a KeyError."""
    typo = {
        "name": "typo",
        "when": [{"column": "sc_gap_secs", "op": ">", "value": 1}],  # trailing s
        "predict": {"constant": 1.0},
    }
    model = RulePredictor(base=CONSTANT_BASE, rules=[typo])
    X = frame()
    with pytest.raises(KeyError, match="sc_gap_secs"):
        model.fit(X, np.full(X.height, 1000.0))
    assert model.base_ is None  # nothing was fitted


def test_a_typo_in_a_mix_column_is_caught_too():
    bad = dict(RULE_D, predict={"mix": [{"weight": 1.0, "column": "nope"}]})
    with pytest.raises(KeyError, match="nope"):
        RulePredictor(base=CONSTANT_BASE, rules=[bad]).fit(frame(), np.zeros(6))


def test_is_null_requires_an_explicit_boolean():
    """Omitting `value` used to mean is_not_null, which is the opposite of what it reads like."""
    with pytest.raises(ValueError, match="explicit boolean"):
        RulePredictor(
            base=CONSTANT_BASE,
            rules=[
                {"when": [{"column": "ob_aobt_sec", "op": "is_null"}], "predict": {"constant": 1}}
            ],
        )
    model = RulePredictor(
        base=CONSTANT_BASE,
        rules=[
            {
                "name": "missing_aobt",
                "when": [{"column": "ob_aobt_sec", "op": "is_null", "value": True}],
                "predict": {"constant": 7.0},
            }
        ],
    )
    X = frame()
    out = model.fit(X, np.full(X.height, 1000.0)).predict(X)
    assert model.fired_["missing_aobt"] == 4  # rows 1, 2, 3, 5 have a null aobt
    assert out[0] == pytest.approx(7.0)


def test_empty_matrix_predicts_nothing_without_error():
    model = fitted(rules=[RULE_D])
    empty = frame().head(0)
    assert model.predict(empty).shape == (0,)


def test_nests_under_a_real_learner_and_forwards_importance():
    """The config path that matters: rules wrapping lgbm, importance descending to the base."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n = 400
    X = pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "airport": np.where(rng.random(n) < 0.3, "LIRF", "EDDF"),
            "month": np.where(np.arange(n) < n // 2, "2025-01", "2025-07"),
            "flt_no_match": rng.integers(0, 2, n),
            "sc_gap_sec": rng.uniform(0, 60_000, n),
            "ob_aobt_sec": rng.uniform(0, 3_000, n),
        }
    )
    y = 800 + 0.01 * X["sc_gap_sec"].to_numpy() + rng.normal(0, 50, n)
    # min_data_in_leaf defaults to 200; on 400 rows that cannot split and the base would
    # return a constant, which would make the "came from the learner" assertion vacuous.
    spec = {
        "kind": "lgbm",
        "params": {
            "num_boost_round": 50,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "verbosity": -1,
        },
    }
    model = models.factory("rules", {"base": spec, "rules": [RULE_D], "clip": [0.0, 200_000.0]})()
    out = model.fit(X, y).predict(X)

    assert np.isfinite(out).all()
    hit = (
        (X["flt_no_match"] == 1) & (X["airport"] == "LIRF") & (X["sc_gap_sec"] > 43_200)
    ).to_numpy()
    assert model.fired_["lirf_unmatched_12h"] == int(hit.sum()) > 0
    expected = 0.62 * X["sc_gap_sec"].to_numpy()[hit] + 0.38 * 87_540
    assert out[hit] == pytest.approx(expected)
    # Rows no rule touched still came from the learner, not the override.
    assert not np.allclose(out[~hit], out[~hit][0])

    importance = models.feature_importance(model)
    assert importance is not None and importance.height > 0
