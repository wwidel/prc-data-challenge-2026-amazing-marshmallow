"""The ``block_time`` source classifier, decomposed into two stages.

What limits :mod:`taxiout.features.modes5` is not capacity. On the 190,713 held-out July rows the
six-way classifier recalls ``none`` 78% of the time, ``aobt_3`` 49%, and **``eobt_1`` 8.5%** — the
second commonest ``block_time`` source is nearly invisible to it. One softmax over six classes
where ``none`` holds 54% of the rows spends its capacity separating the two big classes. Three
remedies: class weights, balanced subsampling, or a two-stage decomposition. This block is the
third, with the first folded in.

**Stage 1 — copy or not.** A binary classifier on every training row: does ``block_time`` copy any
stamp? The binary question is the one `modes5` already answers reasonably, and a binary
``LGBMClassifier`` reaches leave-one-month-out AUC 0.863 on a near relative of it.

**Stage 2 — which stamp, among copies only.** A five-way classifier fitted **only on copy rows**,
where the imbalance is 71,804 / 11,420 / 2,505 / 1,142 / 96 rather than 103,746 against everything.
Two versions are fitted:

- *calibrated* (unweighted): its conditional ``p(s | copy)`` times stage 1's ``p(copy)`` gives joint
  probabilities that are still probabilities, so the label mixture stays an expectation. These are
  the ``m6_p_*`` columns and everything derived from them.
- *balanced* (class weights ``n / (K · n_c)``, sklearn's "balanced"): its posterior trades
  calibration for recall on the minority stamps, which is exactly what `modes5` lacks. These are
  the ``m6_pb_*`` columns and ``m6_label_copy_b``. They are emitted beside the calibrated ones,
  never mixed into them, because a reweighted posterior is not a probability of the true class.

**Everything else is `modes5`'s,** and deliberately so, because every piece of it was a fix for a
measured defect: the per-stamp match tolerance, signed inter-stamp gaps, the median rather than the
mean for the "none" fallback (+68 to +72 s over-prediction in an earlier version), zero mass on a
null stamp, copy-conditional entropy (see `modes5`), and **internal cross-fitting** — the
training months are split into three groups, each predicted by classifiers fitted on the other two,
so the downstream model never sees sharper-than-honest probabilities on its own training rows.
Classifier capacity is `modes5`'s too, so a change in the result is attributable to the
decomposition and not to more trees.

Those helpers are duplicated from `modes5` rather than imported: a block's cache key hashes only its
own defining module, so an import would let an edit there fail to invalidate this block.

**Why TARGET_DEPENDENT.** The class label is computed from ``block_time``, reconstructed as
``mvt_time − taxi_sec``. Everything is fitted per fold from that fold's training months only;
``tests/test_pipeline.py`` leak-tests it along with every other target-dependent block.

**Cost.** Three classifiers × four fits per fold = twelve, against `modes5`'s four. Stage 2 fits on
the ~46% of rows that copy a stamp. The leak test computes the block twice, so run the suite on a
machine with room for it (we used 31 GB).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Candidate stamps, in priority order. First match wins.
SOURCES = ("aobt_3", "eobt_1", "lobt", "iobt", "sched_time")

#: Per-stamp match tolerance, as in `modes5` -- part of the label definition, not a detail.
TOL = {"aobt_3": 120.0, "eobt_1": 60.0, "lobt": 60.0, "iobt": 60.0, "sched_time": 60.0}

#: Class index of "copies no stamp".
NONE = len(SOURCES)

#: Internal cross-fit groups over the training months, as in `modes5`.
N_INNER = 3

#: `modes5`'s capacity and its determinism settings, which are correctness rather than tuning: a
#: stochastic block has a meaningless cache key and fails the leak test at random.
_COMMON = {
    "n_estimators": 150,
    "num_leaves": 31,
    "learning_rate": 0.15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 8,
    "verbose": -1,
}
STAGE1_PARAMS = {"objective": "binary", **_COMMON}
STAGE2_PARAMS = {"objective": "multiclass", "num_class": len(SOURCES), **_COMMON}

_SHORT = {"aobt_3": "aobt", "eobt_1": "eobt", "lobt": "lobt", "iobt": "iobt", "sched_time": "sched"}
_PROB = {s: f"m6_p_{_SHORT[s]}" for s in SOURCES}
_PROB_B = {s: f"m6_pb_{_SHORT[s]}" for s in SOURCES}

OUTPUTS = (
    *_PROB.values(),
    "m6_p_none",
    "m6_p_copy",
    "m6_top_prob",
    "m6_label_mix",
    "m6_label_copy",
    "m6_entropy",
    *_PROB_B.values(),
    "m6_label_copy_b",
)

#: The ten unordered stamp pairs, for the signed gaps.
PAIRS = tuple((a, b) for i, a in enumerate(SOURCES) for b in SOURCES[i + 1 :])

_GAP = [f"g_{a}_{b}" for a, b in PAIRS]
_TO_TAKEOFF = [f"t_{s}" for s in SOURCES]
#: Numeric covariate order. Fixed, because the classifier is fitted and predicted in separate calls.
NUMERIC = (*_TO_TAKEOFF, *_GAP, "n_null_stamps", "utc_hour", "dow")
#: Categorical covariates, integer-coded from the training rows.
CATEGORICAL = ("airport_code", "operator_code")
FEATURES = (*NUMERIC, *CATEGORICAL)


def covariates(frame: pl.LazyFrame) -> pl.LazyFrame:
    """The classifiers' inputs, identical to `modes5`'s. Reads no label."""
    gaps = [((pl.col(a) - pl.col(b)).dt.total_seconds()).alias(f"g_{a}_{b}") for a, b in PAIRS]
    to_takeoff = [
        ((pl.col("mvt_time") - pl.col(s)).dt.total_seconds()).alias(f"t_{s}") for s in SOURCES
    ]
    return frame.with_columns(
        *gaps,
        *to_takeoff,
        pl.sum_horizontal([pl.col(s).is_null().cast(pl.Int8) for s in SOURCES]).alias(
            "n_null_stamps"
        ),
        pl.col("mvt_time").dt.hour().alias("utc_hour"),
        pl.col("mvt_time").dt.weekday().alias("dow"),
    )


def label_class(frame: pl.DataFrame) -> pl.DataFrame:
    """Which stamp ``block_time`` copies, as a class index; ``NONE`` if none. First match wins."""
    out = frame.with_columns(
        (pl.col("mvt_time") - pl.duration(seconds=pl.col("taxi_sec"))).alias("_block")
    )
    expr = pl.lit(NONE, dtype=pl.Int32)
    for i, s in reversed(list(enumerate(SOURCES))):
        expr = (
            pl.when((pl.col("_block") - pl.col(s)).dt.total_seconds().abs() <= TOL[s])
            .then(pl.lit(i, dtype=pl.Int32))
            .otherwise(expr)
        )
    return out.with_columns(expr.alias("_y"))


def code_maps(train: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Integer codes for the categorical covariates, from the training rows only; unseen -> -1."""
    maps = {}
    for col, out in (("airport", "airport_code"), ("operator", "operator_code")):
        levels = sorted(x for x in train[col].unique().to_list() if x is not None)
        maps[out] = {v: i for i, v in enumerate(levels)}
    return maps


def design(frame: pl.DataFrame, maps: dict[str, dict[str, int]]) -> np.ndarray:
    """The classifiers' matrix, columns in FEATURES order. Pure."""
    coded = frame.with_columns(
        pl.col("airport")
        .replace_strict(maps["airport_code"], default=-1)
        .fill_null(-1)
        .alias("airport_code"),
        pl.col("operator")
        .replace_strict(maps["operator_code"], default=-1)
        .fill_null(-1)
        .alias("operator_code"),
    )
    return np.column_stack([coded[c].to_numpy().astype(np.float64) for c in FEATURES])


def none_label_medians(train: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Per-airport MEDIAN label of the rows that copy no stamp, plus a global floor."""
    none_rows = train.filter(pl.col("_y") == NONE)
    src = none_rows if none_rows.height else train
    global_median = float(src["taxi_sec"].median())
    per_airport = (
        none_rows.group_by("airport")
        .agg(pl.col("taxi_sec").median().alias("_none_label"))
        .with_columns(pl.col("_none_label").fill_null(global_median))
    )
    return per_airport, global_median


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """sklearn's "balanced" weights, ``n / (K · n_c)`` over the classes present. Pure."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    per_class = np.zeros(n_classes)
    per_class[present] = len(y) / (present.sum() * counts[present])
    return per_class[y]


def _fit_stage1(train: pl.DataFrame, maps):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**STAGE1_PARAMS)
    model.fit(
        design(train, maps),
        (train["_y"].to_numpy() != NONE).astype(np.int32),
        categorical_feature=[FEATURES.index(c) for c in CATEGORICAL],
    )
    return model


def _fit_stage2(train: pl.DataFrame, maps, balanced: bool):
    from lightgbm import LGBMClassifier

    copies = train.filter(pl.col("_y") != NONE)
    if copies.height < 2:
        # No copies to learn which-stamp from. Never happens on a real fold, but the converse leak
        # test sets every training label to 999,999 s, so nothing matches a stamp: the block must
        # then degrade to all mass on 'none', as modes5 does, rather than fail.
        return None
    y = copies["_y"].to_numpy()
    model = LGBMClassifier(**STAGE2_PARAMS)
    model.fit(
        design(copies, maps),
        y,
        sample_weight=balanced_weights(y, len(SOURCES)) if balanced else None,
        categorical_feature=[FEATURES.index(c) for c in CATEGORICAL],
    )
    return model


def _predict_stage1(model, frame: pl.DataFrame, maps) -> np.ndarray:
    proba = model.predict_proba(design(frame, maps))
    return (
        proba[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(frame.height)
    )


def _predict_stage2(model, frame: pl.DataFrame, maps) -> np.ndarray:
    """Conditional probabilities, one column per stamp even if a stamp was absent in the fit.

    All zeros when there was nothing to fit: :func:`_live_conditional` then marks the row as having
    no live mass, so it takes ``p_copy = 0`` and the median fallback.
    """
    if model is None:
        return np.zeros((frame.height, len(SOURCES)), dtype=np.float64)
    proba = model.predict_proba(design(frame, maps))
    full = np.zeros((frame.height, len(SOURCES)), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, j]
    return full


def _fit_all(train: pl.DataFrame, maps):
    return (
        _fit_stage1(train, maps),
        _fit_stage2(train, maps, balanced=False),
        _fit_stage2(train, maps, balanced=True),
    )


def _predict_all(models, frame: pl.DataFrame, maps):
    s1, s2, s2b = models
    return (
        _predict_stage1(s1, frame, maps),
        _predict_stage2(s2, frame, maps),
        _predict_stage2(s2b, frame, maps),
    )


def _positions(spine_rows: pl.DataFrame, ids: pl.Series) -> np.ndarray:
    """Row index in ``spine_rows`` of each id, in ``ids`` order."""
    index = spine_rows.select("mvt_id").with_row_index("_pos")
    return (
        pl.DataFrame({"mvt_id": ids})
        .join(index, on="mvt_id", how="left", maintain_order="left")["_pos"]
        .to_numpy()
    )


def probabilities(
    spine_rows: pl.DataFrame, train: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``p(copy)``, calibrated ``p(s | copy)`` and balanced ``p_b(s | copy)`` for every spine row.

    Cross-fitted exactly as `modes5` is: rows in the training months are predicted by classifiers
    that did not see their own month group; everything else by classifiers fitted on all of them.
    """
    maps = code_maps(train)
    n = spine_rows.height
    p_copy = np.zeros(n)
    cond = np.zeros((n, len(SOURCES)))
    cond_b = np.zeros((n, len(SOURCES)))

    def put(frame: pl.DataFrame, preds) -> None:
        pos = _positions(spine_rows, frame["mvt_id"])
        p_copy[pos], cond[pos], cond_b[pos] = preds

    months = sorted(train["month"].unique().to_list())
    groups = {m: i % N_INNER for i, m in enumerate(months)}
    train = train.with_columns(pl.col("month").replace_strict(groups, default=0).alias("_g"))
    for g in range(N_INNER):
        fit_on = train.filter(pl.col("_g") != g)
        held = train.filter(pl.col("_g") == g)
        if fit_on.height == 0 or held.height == 0:
            continue
        put(held, _predict_all(_fit_all(fit_on, maps), held, maps))

    trained_ids = train["mvt_id"].to_list()
    rest = spine_rows.filter(~pl.col("mvt_id").is_in(trained_ids))
    if rest.height:
        put(rest, _predict_all(_fit_all(train, maps), rest, maps))
    return p_copy, cond, cond_b


def _live_conditional(cond: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero the mass on null stamps and renormalise over the live ones. Pure.

    Returns the renormalised conditional and a mask of rows with any live mass at all. A row whose
    every stamp is null -- an unmatched departure -- cannot be a copy of anything we hold.
    """
    c = np.where(np.isnan(t), 0.0, cond)
    total = c.sum(axis=1, keepdims=True)
    ok = total[:, 0] > 1e-12
    out = np.zeros_like(c)
    np.divide(c, total, out=out, where=total > 1e-12)
    return out, ok


def assemble(
    rows: pl.DataFrame,
    p_copy: np.ndarray,
    cond: np.ndarray,
    cond_b: np.ndarray,
    per_airport: pl.DataFrame,
    global_median: float,
) -> pl.DataFrame:
    """Turn the two stages into the emitted columns. Pure; unit-tested.

    ``rows`` must carry ``mvt_id``, ``airport`` and the ``t_<stamp>`` gaps from :func:`covariates`.
    """
    t = np.column_stack([rows[f"t_{s}"].cast(pl.Float64).to_numpy() for s in SOURCES])
    none_label = (
        rows.select("airport")
        .join(per_airport, on="airport", how="left", maintain_order="left")["_none_label"]
        .fill_null(global_median)
        .to_numpy()
        .astype(np.float64)
    )

    c, ok = _live_conditional(cond, t)
    cb, _ = _live_conditional(cond_b, t)
    pc = np.where(ok, np.clip(p_copy, 0.0, 1.0), 0.0)
    joint = pc[:, None] * c
    p_none = 1.0 - pc

    t0 = np.nan_to_num(t, nan=0.0)
    label_copy = np.where(ok, (c * t0).sum(axis=1), np.nan)
    label_copy_b = np.where(ok, (cb * t0).sum(axis=1), np.nan)
    label_mix = (joint * t0).sum(axis=1) + p_none * none_label

    # Copy-conditional entropy, over the stamps only -- see modes5 for why.
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(c > 1e-12, c * np.log(c), 0.0).sum(axis=1)

    def col(name: str, values: np.ndarray) -> pl.Series:
        return pl.Series(name, values, dtype=pl.Float64, nan_to_null=True)

    return pl.DataFrame(
        [
            rows["mvt_id"],
            *[col(_PROB[s], joint[:, i]) for i, s in enumerate(SOURCES)],
            col("m6_p_none", p_none),
            col("m6_p_copy", pc),
            col("m6_top_prob", joint.max(axis=1)),
            col("m6_label_mix", label_mix),
            col("m6_label_copy", label_copy),
            col("m6_entropy", entropy),
            *[col(_PROB_B[s], np.where(ok, cb[:, i], 0.0)) for i, s in enumerate(SOURCES)],
            col("m6_label_copy_b", label_copy_b),
        ]
    )


@feature(name="modes6", kind=FeatureKind.TARGET_DEPENDENT, outputs=OUTPUTS)
def modes6(ctx: FeatureContext) -> pl.DataFrame:
    """Two-stage source probabilities: copy-or-not, then which stamp among copies."""
    raw = covariates(
        io.scan_movements()
        .filter(pl.col("phase") == "DEP")
        .select("mvt_id", "mvt_time", "operator", *SOURCES)
    ).collect(engine="streaming")

    rows = ctx.spine.select("mvt_id", "airport", "month").join(raw, on="mvt_id", how="left")
    train = label_class(
        ctx.training_rows()
        .select("mvt_id", "airport", "month", "taxi_sec")
        .join(raw, on="mvt_id", how="left")
    )
    per_airport, global_median = none_label_medians(train)
    p_copy, cond, cond_b = probabilities(rows, train)
    return align(
        ctx.spine, assemble(rows, p_copy, cond, cond_b, per_airport, global_median), OUTPUTS
    )
