"""A *trained* six-way classifier for which stamp ``block_time`` copies.

The successor to an earlier block (``modes4``, not in this repository), which chased the same
target with per-cell rates and failed for a reason a review pinned down precisely: rates over 406
cells never identify a source confidently — maximum ``p_eobt`` of **0.26**, minimum ``p_none`` of
**0.30**, and 41% of cells holding fewer than 50 rows against a shrinkage constant of 200. The
mixture stays permanently hedged and collapses toward the cell mean. **The concept was intact; the
instrument was not.** Dropping the two-way ``modes`` block cost only **+0.22**, so the bar is low
and the headroom is nearly all unrealised.

Why the headroom is worth another attempt: the label is ``mvt_time − block_time`` and ``block_time``
is the **only** column the ranking set withholds for a departure, so wherever it is a copy of a
stamp we hold, the label is *exactly knowable*. The source distribution over 2,062,623 matched
2025 departures at ``|block_time − stamp| <= 60 s``: none 67.0%, ``aobt_3`` 21.0%, ``eobt_1``
9.8%, ``lobt`` 1.7%, ``sched_time`` 0.5%, ``iobt`` 0.05%. With the source known perfectly, a
CatBoost member's trimmed ex-LIRF goes 220.94 -> **203.36**: −17.58, larger than the entire
feature programme to that date.

**Three changes over ``modes4``, each from the review rather than from taste.**

1. **A trained classifier, not a cell table.** An ``LGBMClassifier`` for the binary
   copy/non-copy question reaches leave-one-month-out AUC 0.863, far more than a cell table over
   the same covariates could.
2. **Signed inter-stamp gaps, not agreement bits.** ``modes4`` reduced ``eobt_1 − aobt_3`` to one
   bit at a 60 s threshold. The magnitude and sign of that gap is almost certainly what decides
   which stamp got recorded, and a boolean discards it. This block feeds all ten signed pairwise
   gaps plus each stamp's gap to takeoff.
3. **Median, not mean, for the "none" fallback.** ``modes4`` used the mean of a right-skewed
   distribution and consequently over-predicted by a consistent **+68 to +72 s** on held-out rows.

**And one the review did not ask for, which matters more than any of them.** A classifier's
in-sample predictions are *sharp* where shrunk cell rates are blunt, so using them directly on the
training rows would hand the downstream model a column that is far more accurate in training than it
can ever be in validation — and the model would learn to trust it accordingly. ``modes`` gets away
with in-sample cell rates because shrinkage keeps them blunt. A classifier does not. So the training
rows here get **internally cross-fitted** predictions: the training months are split into three
groups, each group predicted by a classifier fitted on the other two. Rows outside the training
months get a classifier fitted on all of them. That is four fits per fold and it is not optional.

**Why TARGET_DEPENDENT.** The class label is computed from ``block_time``, reconstructed as
``mvt_time − taxi_sec``. Knowing ``block_time`` is knowing the label, so everything here is fitted
per fold from that fold's training months only. ``tests/test_pipeline.py`` leak-tests every
target-dependent block by corrupting held-out labels and asserting the columns do not move.

**Cost warning for that leak test.** It computes the block twice over the full spine, so for this
block it is **eight classifier fits** and roughly 2.5 GB, so run the suite on a machine with room
for it. It is also why ``CLF_PARAMS`` must be deterministic:
the test compares the two computes column by column.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from taxiout import io
from taxiout.features.registry import FeatureContext, FeatureKind, align, feature

#: Candidate stamps, in priority order. First match wins.
SOURCES = ("aobt_3", "eobt_1", "lobt", "iobt", "sched_time")

#: Per-stamp match tolerance. The A-CDM actual is a coarser record than the plan values, and the
#: coarser tolerance shifts the distribution materially -- aobt_3 39.1% against 21.0% with a
#: uniform 60 s -- so the tolerance is part of the label definition, not a detail.
TOL = {"aobt_3": 120.0, "eobt_1": 60.0, "lobt": 60.0, "iobt": 60.0, "sched_time": 60.0}

#: The six classes, in a fixed order. `none` last so its index is len(SOURCES).
CLASSES = (*SOURCES, "none")

#: Internal cross-fit groups over the training months. Three keeps the cost to four fits per fold.
N_INNER = 3

#: Deliberately small. Six classes means num_class boosters per iteration, so this is 6 x 150 trees
#: per fit and four fits per fold. Grow it only if the block earns its place first.
#:
#: **`random_state`, `deterministic` and `force_row_wise` are not tuning, they are correctness.**
#: `subsample` and `colsample_bytree` make a fit stochastic, and a feature block whose output
#: changes between two builds of the same input has a meaningless cache key -- `store.block_hash`
#: promises that a hash identifies the *output*. It would also break the leak test in
#: `tests/test_pipeline.py`, which computes the block twice and asserts every column is identical:
#: that test would then fail at random and for the wrong reason, which is worse than not having it.
#: `deterministic` needs one of the force_*_wise modes to take effect.
CLF_PARAMS = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
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

OUTPUTS = (
    "m5_p_aobt",
    "m5_p_eobt",
    "m5_p_lobt",
    "m5_p_iobt",
    "m5_p_sched",
    "m5_p_none",
    "m5_p_copy",
    "m5_top_prob",
    "m5_label_mix",
    "m5_label_copy",
    "m5_entropy",
)

_PROB = {
    "aobt_3": "m5_p_aobt",
    "eobt_1": "m5_p_eobt",
    "lobt": "m5_p_lobt",
    "iobt": "m5_p_iobt",
    "sched_time": "m5_p_sched",
}

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
    """The classifier's inputs. Reads no label, so it is safe on every row of both splits.

    `utc_hour` stands in for local hour: `airport` is a covariate too, so the classifier can
    recover the local offset per airport without this module importing the DST logic from
    `features/calendar.py`. A cross-module import would be worse than a slight loss of directness,
    because a block's cache key hashes only its own defining module -- so a change in
    `calendar.py` would silently fail to invalidate this block.
    """
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
    """Which stamp ``block_time`` copies, as a class index. First match wins.

    Deliberately duplicated from `modes4` rather than imported: a block's cache key hashes only its
    own defining module, so importing this from `modes4` would mean an edit there did not
    invalidate the cache here.
    """
    out = frame.with_columns(
        (pl.col("mvt_time") - pl.duration(seconds=pl.col("taxi_sec"))).alias("_block")
    )
    expr = pl.lit(len(SOURCES), dtype=pl.Int32)  # 'none'
    for i, s in reversed(list(enumerate(SOURCES))):
        expr = (
            pl.when((pl.col("_block") - pl.col(s)).dt.total_seconds().abs() <= TOL[s])
            .then(pl.lit(i, dtype=pl.Int32))
            .otherwise(expr)
        )
    return out.with_columns(expr.alias("_y"))


def code_maps(train: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Integer codes for the categorical covariates, built from the training rows only.

    A level unseen in training maps to -1. LightGBM logs "Met negative value in categorical
    features, will convert it to NaN" and does exactly that, so unseen and null levels share the
    NaN branch rather than forming a distinct category of their own. That is the intended
    behaviour here -- both mean "no usable operator" -- but it is worth stating accurately.
    """
    maps = {}
    for col, out in (("airport", "airport_code"), ("operator", "operator_code")):
        levels = sorted(x for x in train[col].unique().to_list() if x is not None)
        maps[out] = {v: i for i, v in enumerate(levels)}
    return maps


def design(frame: pl.DataFrame, maps: dict[str, dict[str, int]]) -> np.ndarray:
    """The classifier's matrix, columns in FEATURES order. Pure."""
    coded = frame.with_columns(
        pl.col("airport")
        .replace_strict(maps["airport_code"], default=-1)
        .fill_null(-1)
        .alias("airport_code"),
        # A null operator is its own category, not a missing value: the _flt record is absent
        # on unmatched rows, which is exactly the population the source question is hardest on.
        pl.col("operator")
        .replace_strict(maps["operator_code"], default=-1)
        .fill_null(-1)
        .alias("operator_code"),
    )
    return np.column_stack([coded[c].to_numpy().astype(np.float64) for c in FEATURES])


def none_label_medians(train: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Per-airport MEDIAN label of the rows that copy no stamp, plus a global fallback.

    Median, not mean: an earlier version used the mean of a right-skewed distribution and
    over-predicted the fallback by a consistent +68 to +72 s on held-out rows. The global floor
    exists because an airport can contain no 'none' rows at all, and a null fallback propagates
    into the mixture.
    """
    none_rows = train.filter(pl.col("_y") == len(SOURCES))
    src = none_rows if none_rows.height else train
    global_median = float(src["taxi_sec"].median())
    per_airport = (
        none_rows.group_by("airport")
        .agg(pl.col("taxi_sec").median().alias("_none_label"))
        .with_columns(pl.col("_none_label").fill_null(global_median))
    )
    return per_airport, global_median


def _fit(train: pl.DataFrame, maps: dict[str, dict[str, int]]):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**CLF_PARAMS)
    model.fit(
        design(train, maps),
        train["_y"].to_numpy(),
        categorical_feature=[FEATURES.index(c) for c in CATEGORICAL],
    )
    return model


def _predict(model, frame: pl.DataFrame, maps: dict[str, dict[str, int]]) -> np.ndarray:
    """Class probabilities, always with one column per class even if a class was absent in a fit."""
    proba = model.predict_proba(design(frame, maps))
    full = np.zeros((frame.height, len(CLASSES)), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, j]
    return full


def probabilities(spine_rows: pl.DataFrame, train: pl.DataFrame) -> np.ndarray:
    """Class probabilities for every spine row, cross-fitted inside the training months.

    Rows in the training months are predicted by a classifier that did not see their own month
    group; everything else by a classifier fitted on all training rows. Without the cross-fit the
    training rows would carry sharp in-sample probabilities and the downstream model would learn to
    trust a column that cannot be that accurate out of fold.
    """
    maps = code_maps(train)
    out = np.zeros((spine_rows.height, len(CLASSES)), dtype=np.float64)
    index = {m: i for i, m in enumerate(spine_rows["mvt_id"].to_list())}

    months = sorted(train["month"].unique().to_list())
    groups = {m: i % N_INNER for i, m in enumerate(months)}
    train = train.with_columns(pl.col("month").replace_strict(groups, default=0).alias("_g"))
    for g in range(N_INNER):
        fit_on = train.filter(pl.col("_g") != g)
        held = train.filter(pl.col("_g") == g)
        if fit_on.height == 0 or held.height == 0:
            continue
        proba = _predict(_fit(fit_on, maps), held, maps)
        for k, mid in enumerate(held["mvt_id"].to_list()):
            out[index[mid]] = proba[k]

    # A list, not a Series: `is_in` against a same-dtype collection is deprecated in Polars.
    trained_ids = train["mvt_id"].to_list()
    rest = spine_rows.filter(~pl.col("mvt_id").is_in(trained_ids))
    if rest.height:
        proba = _predict(_fit(train, maps), rest, maps)
        for k, mid in enumerate(rest["mvt_id"].to_list()):
            out[index[mid]] = proba[k]
    return out


def assemble(
    rows: pl.DataFrame, proba: np.ndarray, per_airport: pl.DataFrame, global_median: float
) -> pl.DataFrame:
    """Turn class probabilities into the emitted columns. Pure; unit-tested."""
    frame = rows.with_columns(
        *[pl.Series(_PROB[s], proba[:, i]) for i, s in enumerate(SOURCES)],
        pl.Series("_p_none_raw", proba[:, len(SOURCES)]),
    ).join(per_airport, on="airport", how="left")
    frame = frame.with_columns(pl.col("_none_label").fill_null(global_median))

    # A null stamp implies no computable label, so it cannot carry probability mass. Zero it and
    # let the removed mass fall to 'none' -- an unmatched row has all four _flt stamps null
    # together, and without this its mass would sit on hypotheses that cannot be evaluated.
    frame = frame.with_columns(
        *[
            pl.when(pl.col(s).is_null()).then(0.0).otherwise(pl.col(_PROB[s])).alias(_PROB[s])
            for s in SOURCES
        ]
    )
    live = pl.sum_horizontal([pl.col(_PROB[s]) for s in SOURCES])
    weighted = pl.sum_horizontal(
        [
            pl.when(pl.col(s).is_null())
            .then(0.0)
            .otherwise(pl.col(_PROB[s]) * (pl.col("mvt_time") - pl.col(s)).dt.total_seconds())
            for s in SOURCES
        ]
    )

    def plogp(column: str) -> pl.Expr:
        p = pl.col(column) / pl.max_horizontal(live, pl.lit(1e-12))
        return pl.when(p > 1e-12).then(p * p.log()).otherwise(0.0)

    return frame.with_columns(
        (1.0 - live).clip(0.0, 1.0).alias("m5_p_none"),
    ).select(
        "mvt_id",
        *(_PROB[s] for s in SOURCES),
        "m5_p_none",
        live.clip(0.0, 1.0).alias("m5_p_copy"),
        pl.max_horizontal([pl.col(_PROB[s]) for s in SOURCES]).alias("m5_top_prob"),
        (weighted + (1.0 - live).clip(0.0, 1.0) * pl.col("_none_label")).alias("m5_label_mix"),
        pl.when(live > 1e-9).then(weighted / live).otherwise(None).alias("m5_label_copy"),
        # Copy-CONDITIONAL entropy, over the five stamps only. Taking it over all six categories
        # inverts the column's meaning: a cell where 98% of rows match no stamp
        # has LOW entropy, so "confident" comes to mean "confidently not a copy", which is the
        # useless case -- on held-out rows the surest quartile then scored RMSE 351.6 against the
        # least sure quartile's 225.9, exactly backwards. `m5_p_none` already carries is-it-a-copy.
        (-pl.sum_horizontal([plogp(_PROB[s]) for s in SOURCES])).alias("m5_entropy"),
    )


@feature(name="modes5", kind=FeatureKind.TARGET_DEPENDENT, outputs=OUTPUTS)
def modes5(ctx: FeatureContext) -> pl.DataFrame:
    """Six-way source probabilities from a trained classifier, and the labels they imply."""
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
    proba = probabilities(rows, train)
    return align(ctx.spine, assemble(rows, proba, per_airport, global_median), OUTPUTS)
