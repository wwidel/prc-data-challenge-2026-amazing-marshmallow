"""Cross-validation: fold definitions and the scoring loop.

The real task is to fit on twelve months of 2025 and predict January and July 2026, so a fold
holds out a whole month and trains on the rest. January and July are the folds that matter —
they are the evaluation months, and after the winter/summer split in the data they behave like
two different problems, so their scores are always reported separately as well as pooled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from taxiout import metrics, models, paths, spine, store
from taxiout.features import registry
from taxiout.features.registry import FeatureContext
from taxiout.models.base import Predictor

#: Validation rows with a true taxi-out above this are corrupt records, not long taxis: 34 rows
#: in all of 2025, carrying 28% of the total squared error. They stay in the headline RMSE
#: because the leaderboard will score them, but a second trimmed figure is reported next to it
#: so model selection is not decided by ten rows.
TRIM_SECONDS = 43_200

#: The airport reported apart. LIRF is 7.8% of the 2026 departures and 44-56% of the squared
#: error, so it dominates any pooled figure and is worth seeing separately.
#:
#: **Do not steer by `rmse_ex_lirf` alone.** That rule was adopted on 2026-09-24 and falsified
#: the same day. It rested on pooled CV appearing anti-correlated with the leaderboard
#: (Spearman -0.20), which was an artifact of computing the correlation over only the four
#: models that had a per-airport breakdown logged -- four models lying within 9 s of each other
#: on the board. That is range restriction. Restore the two baselines and pooled CV is a
#: perfectly good predictor (Spearman +0.657, Pearson +0.988); it simply cannot *resolve*
#: differences of a few seconds.
#:
#: The counterexample: two early submissions differed by 24.81 s on the board, the largest
#: difference this project measured, and `rmse_ex_lirf` scored the better one 0.66 s WORSE.
#: The whole gain is at Rome, from a covariate (`flt_no_match`) whose 2026 rate is verified.
#:
#: Use both numbers together via `predicted_board()`. What does not transfer is a Rome gain
#: bought with model *capacity*: `x1_maxbin_2047` cut its CV Rome score to 933.9 and its implied
#: 2026 Rome score stayed at 1164.
STEERING_AIRPORT = "LIRF"

#: LIRF's share of the 2026 ranking departures (26,899 of 344,841). The board is a single RMSE
#: over all of them, so it decomposes exactly by row share.
LIRF_SHARE_2026 = 0.078


def steering_rows(airport: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The population the steering metric is computed over, as a boolean mask.

    Two exclusions, applied **together**: away from :data:`STEERING_AIRPORT`, and at or below
    :data:`TRIM_SECONDS`. It exists as a function because applying them one at a time -- trimming
    a figure that already excluded Rome, or excluding Rome from an already-trimmed one -- gives a
    different number, and the project has published two different "trimmed ex-LIRF" recipes as a
    result. There is one definition and this is it.
    """
    return (airport != STEERING_AIRPORT) & (truth <= TRIM_SECONDS)


def predicted_board(rmse_ex_lirf: float, rmse_lirf: float) -> float:
    """Combine the two reported halves into a leaderboard estimate.

    Exact by construction: RMSE over a partition is the row-share-weighted quadratic mean. Over
    the four deep models submitted so far this reproduces the board to sd 1.9 s.

    Pass the model's own CV Rome score only when the change driving it is a covariate whose 2026
    rate has been checked. For a capacity or hyperparameter change, pin ``rmse_lirf`` at the
    observed 2026 level (~1156) -- every model submitted so far lands in 1144-1164 regardless of
    scoring 934-1165 in cross-validation.
    """
    share = LIRF_SHARE_2026
    return float(np.sqrt((1.0 - share) * rmse_ex_lirf**2 + share * rmse_lirf**2))


@dataclass(frozen=True)
class Fold:
    name: str
    train_months: tuple[str, ...]
    valid_months: tuple[str, ...]


def leave_month_out(
    valid: tuple[str, ...], pool: tuple[str, ...] = paths.TRAIN_MONTHS
) -> list[Fold]:
    """One fold per month in ``valid``, training on every other month of ``pool``."""
    return [
        Fold(name=m, train_months=tuple(x for x in pool if x != m), valid_months=(m,))
        for m in valid
    ]


def held_out_together(
    groups: tuple[tuple[str, ...], ...], pool: tuple[str, ...] = paths.TRAIN_MONTHS
) -> list[Fold]:
    """One fold per group of months, held out together, trained on every other month of ``pool``."""
    return [
        Fold(
            name="+".join(groups_m),
            train_months=tuple(x for x in pool if x not in groups_m),
            valid_months=tuple(groups_m),
        )
        for groups_m in groups
    ]


#: Named fold sets, referenced by name from an experiment config.
FOLD_SETS: dict[str, list[Fold]] = {
    # The final phase's structure: the organisers rank January + July, February + June, and all
    # four together. Each pair is held out as one fold and trained on the other ten months; the
    # four-month score is the two folds pooled.
    "final_pairs": held_out_together((("2025-01", "2025-07"), ("2025-02", "2025-06"))),
    # The two evaluation months. This is the default and the number we steer by.
    "seasonal": leave_month_out(("2025-01", "2025-07")),
    # Every month — slower, and the right check before trusting a big change.
    "all_months": leave_month_out(paths.TRAIN_MONTHS),
    # Jan and Jul plus two shoulder months. The middle path between `seasonal` and `all_months`:
    # every number steering this project comes from two folds, and validating the whole shipping
    # composition on all twelve costs ~22.7 h of fitting (CatBoost alone is 851 min at 6x its
    # seasonal cost) against ~7.6 h here. It keeps both evaluation months, so the steering signal
    # is retained rather than replaced, and adds April and October to test whether it generalises.
    "quarters": leave_month_out(("2025-01", "2025-04", "2025-07", "2025-10")),
    # One fold, for plumbing checks.
    "quick": leave_month_out(("2025-07",)),
    # A deliberate SIX-MONTH gap between training and validation, which no other fold set has.
    #
    # Every other set is leave-one-month-out, so training includes the months either side of the
    # validation month. The real task trains on twelve months of 2025 and predicts 2026, so our CV
    # asks an easier question than the competition does, and that is a candidate explanation for a
    # transfer ratio of 0.93-1.30 rather than 1.0.
    #
    # The risk it probes is specifically differential: the TARGET_DEPENDENT blocks estimate rates
    # from the training months, so `modes` estimates January's recording-mode rates partly from
    # December and February. If adjacent months flatter target-encoded blocks in particular, then
    # `modes` -- which is in every shipping member -- is over-credited. Run an arm with and without
    # those blocks and compare the differences, not the levels.
    "gap6": [
        Fold(
            name="2025-12-gap6",
            train_months=tuple(f"2025-{m:02d}" for m in range(1, 7)),
            valid_months=("2025-12",),
        )
    ],
    # gap6's control, and it must match gap6 in EVERY respect except adjacency. So it trains on six
    # months too -- the six immediately before December -- not on eleven. The first version used
    # leave-one-month-out here, which gave the control 11 training months against gap6's 6 and
    # confounded the gap with training size: any difference would have had two candidate causes and
    # the experiment would have been unable to separate them.
    "adjacent_dec": [
        Fold(
            name="2025-12-adjacent",
            train_months=tuple(f"2025-{m:02d}" for m in range(6, 12)),
            valid_months=("2025-12",),
        )
    ],
    # Winter only, for de-icing work.
    "winter": leave_month_out(("2025-01", "2025-02", "2025-12")),
    # The ten months `seasonal` does not cover. A member's seasonal OOF plus its `rest10` OOF is its
    # twelve-month OOF, at 10/12 of `all_months`' cost -- each fold trains on the other eleven
    # months in both sets, so the pieces are interchangeable (2026-10-01, the 12-month OOF run).
    "rest10": leave_month_out(
        tuple(m for m in paths.TRAIN_MONTHS if m not in ("2025-01", "2025-07"))
    ),
    # As rest10, minus April and October too, for members already run on `quarters` (`d05` via w09).
    "rest8": leave_month_out(
        tuple(
            m for m in paths.TRAIN_MONTHS if m not in ("2025-01", "2025-04", "2025-07", "2025-10")
        )
    ),
}


def get_folds(name: str) -> list[Fold]:
    if name not in FOLD_SETS:
        raise KeyError(f"unknown fold set {name!r}; known: {', '.join(sorted(FOLD_SETS))}")
    return FOLD_SETS[name]


def build_matrix(
    features: tuple[str, ...],
    months: tuple[str, ...],
    *,
    train_months: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Assemble spine + features for ``months``.

    Cached blocks come straight off disk. Target-dependent blocks are recomputed here against
    ``train_months`` — the only months whose labels the current fold may see.
    """
    blocks = registry.resolve(features)
    cached = tuple(b.name for b in blocks if b.kind.cacheable)
    fold_local = [b for b in blocks if not b.kind.cacheable]

    matrix = store.assemble(cached, months).collect()

    if fold_local:
        if train_months is None:
            raise ValueError(
                f"{[b.name for b in fold_local]} are target-dependent and need train_months"
            )
        full = spine.load()
        mask = full["month"].is_in(list(train_months))
        keep = full["month"].is_in(list(sorted(months)))
        # The blocks below are glued on by POSITION (hstack), not joined on mvt_id. That is only
        # right if the full spine filtered to `months` is row-for-row the assembled matrix --
        # true by construction (both month-major, sorted), and now asserted rather than assumed.
        if not full.filter(keep)["mvt_id"].equals(matrix["mvt_id"]):
            raise AssertionError("spine and assembled matrix disagree on row order")
        for block in fold_local:
            computed = block.compute(FeatureContext(spine=full, train_mask=mask))
            matrix = matrix.hstack(computed.filter(keep))
    return matrix


@dataclass
class FoldResult:
    fold: str
    n_train: int
    n_valid: int
    rmse: float
    rmse_trimmed: float
    n_trimmed: int
    mae: float
    bias: float
    rmse_ex_lirf: float = float("nan")
    rmse_trimmed_ex_lirf: float = float("nan")
    rmse_lirf_only: float = float("nan")
    by_airport: pl.DataFrame = field(default=None, repr=False)
    importance: pl.DataFrame | None = field(default=None, repr=False)
    predictions: pl.DataFrame | None = field(default=None, repr=False)


@dataclass
class CVResult:
    folds: list[FoldResult]
    pooled_rmse: float
    pooled_rmse_trimmed: float
    seconds: float
    #: Pooled RMSE over every airport except STEERING_AIRPORT, **untrimmed**.
    #:
    #: Not the steering metric, despite what this field was called until 2026-09-28.
    #: Untrimmed, two corrupt LFPG rows carried 38.3% of the stack's whole non-LIRF squared
    #: error, which is what moved steering to the trimmed figure below.
    pooled_rmse_ex_lirf: float = float("nan")
    #: **The steering metric.** Pooled RMSE over the rows that are both away from
    #: STEERING_AIRPORT and at or below TRIM_SECONDS -- the two exclusions applied together.
    pooled_rmse_trimmed_ex_lirf: float = float("nan")
    #: Reported beside it, never steered by. See STEERING_AIRPORT.
    pooled_rmse_lirf_only: float = float("nan")

    def importance(self) -> pl.DataFrame | None:
        """Feature importance averaged over folds, largest first.

        Averaging across folds is what makes this readable: a single fold's ranking is noisy,
        and the columns that matter are the ones that matter in both.
        """
        frames = [f.importance for f in self.folds if f.importance is not None]
        if not frames:
            return None
        return (
            pl.concat(frames)
            .group_by("feature")
            .agg(
                pl.col("importance").mean().alias("importance"),
                pl.col("importance").std().alias("importance_sd"),
                pl.len().alias("folds"),
            )
            .sort("importance", descending=True)
        )

    def out_of_fold(self) -> pl.DataFrame | None:
        """Every held-out prediction, stacked. One row per departure per fold it appeared in.

        Under leave-one-month-out every row is held out exactly once, so this is a complete
        out-of-fold prediction set over the validated months.
        """
        frames = [f.predictions for f in self.folds if f.predictions is not None]
        return pl.concat(frames) if frames else None

    def by_airport_month(self) -> pl.DataFrame:
        """Every fold's per-airport breakdown, stacked."""
        return pl.concat([f.by_airport for f in self.folds])

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "fold": [f.fold for f in self.folds],
                "n_valid": [f.n_valid for f in self.folds],
                "rmse": [f.rmse for f in self.folds],
                "rmse_trimmed": [f.rmse_trimmed for f in self.folds],
                "mae": [f.mae for f in self.folds],
                "bias": [f.bias for f in self.folds],
                "rmse_ex_lirf": [f.rmse_ex_lirf for f in self.folds],
                "rmse_trimmed_ex_lirf": [f.rmse_trimmed_ex_lirf for f in self.folds],
                "rmse_lirf_only": [f.rmse_lirf_only for f in self.folds],
            }
        )


#: Additive offset for the log target, so defect labels at or near zero stay finite.
LOG_OFFSET = 1.0

#: Plausible band for a repaired label, in seconds.
REPAIR_BAND = (120.0, 7_200.0)


def repair_training_labels(frame: pl.DataFrame, target: str = "taxi_sec") -> pl.DataFrame:
    """Repair the two provable defect mechanisms and drop the third. Training rows only.

    The generator study identified three ways a recorded ``block_time`` goes wrong
    (``scripts/analysis/fresh_opus/o08_generators.py``, ``fresh_fable/f06_*.py``):

    * **LATE** — the block was stamped at takeoff, so the label is under two minutes. Of the
      2,133 such rows, 57% carry an ``aobt_3`` and 1,180 of those 1,215 implied taxis land in
      the plausible band: the true off-block survived in the flight plan.
    * **SCHED** — the block was filled from the schedule, so the label equals the whole
      departure delay. Identified without touching ``block_time``: the label equals the
      schedule gap, i.e. ``|y - ob_sched_sec| <= 60``.
    * **DAY** — the block carries the scheduled date with the real time of day, so the label
      is about 86,400 s plus a normal taxi. Not repairable; dropped.

    Both repairs use ``ob_aobt_sec`` (= takeoff − ``aobt_3``), a feature already in every
    configuration's ``offblock`` block, so no extra column is needed. Returns the frame with
    ``target`` rewritten and DAY rows removed. Pure; unit-tested.
    """
    for column in ("ob_aobt_sec", "ob_sched_sec"):
        if column not in frame.columns:
            raise ValueError(f"repair_training_labels needs {column}; include the offblock block")
    lo, hi = REPAIR_BAND
    aobt_ok = pl.col("ob_aobt_sec").is_between(lo, hi)
    is_late = pl.col(target) < 120
    # A SCHED defect means the label equals the schedule gap AND disagrees with the physical
    # taxi. Without the second clause the rule fires on every punctual departure — 199,634
    # rows, 9.5% of training, of which 197,471 already had plausible labels — and rewrites
    # them by a median of 176 s, i.e. injects noise instead of repairing anything (rev4 w06).
    is_sched = ((pl.col(target) - pl.col("ob_sched_sec")).abs() <= 60) & (
        (pl.col(target) - pl.col("ob_aobt_sec")).abs() > 300
    )
    is_day = (pl.col(target) - 86_400).is_between(0, 3_600)
    return frame.filter(~is_day).with_columns(
        pl.when((is_late | is_sched) & aobt_ok)
        .then(pl.col("ob_aobt_sec"))
        .otherwise(pl.col(target))
        .alias(target)
    )


#: Share of the training rows held out purely to estimate the log target's smearing factor.
#: In-sample residuals of a 511-leaf booster are far too small, which biases the back-transform
#: low (a held-out factor of 1.053 against an in-sample one near 1.00 — rev4 w05).
SMEAR_HOLDOUT = 0.05


def fit_and_predict(
    model,
    train: pl.DataFrame,
    predict_on: pl.DataFrame,
    *,
    target: str = "taxi_sec",
    train_label_cap: float | None = None,
    target_transform: str | None = None,
    train_relabel: bool = False,
    clip: tuple[float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Fit ``model`` on ``train`` and predict ``predict_on``, applying every label treatment.

    **The single place these treatments are applied.** Cross-validation and both refit paths
    call it, so a model cross-validated with a treatment is necessarily refitted with the same
    one. That invariant was silently broken until 2026-09-27: ``refit_predict.py`` and
    ``submit.predict_experiment`` fitted on raw labels, so a capped model's ranking predictions
    came out bit-identical to the uncapped model's (max difference 9e-13) and every measured
    gain from ``train_label_cap`` was absent from the submission it was supposed to improve.
    """
    if target_transform not in (None, "log"):
        raise ValueError(f"unknown target_transform {target_transform!r}")
    if train_relabel:
        train = repair_training_labels(train, target)
    y = train[target].to_numpy().astype(np.float64)
    if train_label_cap is not None:
        y = np.minimum(y, train_label_cap)

    if target_transform == "log":
        y_fit = np.log(np.clip(y, 0.0, None) + LOG_OFFSET)
        rng = np.random.default_rng(seed)
        held = rng.random(train.height) < SMEAR_HOLDOUT
        fit_rows = train.filter(pl.Series(~held))
        model.fit(fit_rows, y_fit[~held])
        resid = y_fit[held] - np.asarray(
            model.predict(train.filter(pl.Series(held))), dtype=np.float64
        )
        smear = float(np.mean(np.exp(resid)))
        # Refit on every training row before predicting. The holdout is only for estimating the
        # smearing factor; an earlier version predicted with the 95% model, so log-target models
        # trained on 5% fewer rows than the raw-target ones they were compared with.
        model.fit(train, y_fit)
        raw = np.asarray(model.predict(predict_on), dtype=np.float64)
        predicted = np.exp(raw) * smear - LOG_OFFSET
    else:
        model.fit(train, y)
        predicted = np.asarray(model.predict(predict_on), dtype=np.float64)

    if clip is not None:
        predicted = np.clip(predicted, *clip)
    return predicted


def run(
    make_model: callable[[], Predictor],
    features: tuple[str, ...],
    folds: list[Fold],
    *,
    target: str = "taxi_sec",
    clip: tuple[float, float] | None = None,
    train_label_cap: float | None = None,
    target_transform: str | None = None,
    train_relabel: bool = False,
) -> CVResult:
    """Fit and score ``make_model()`` on each fold. Returns per-fold and pooled RMSE.

    The pooled figure weights folds by their row count, which is how the leaderboard will score
    January and July together.
    """
    import time

    started = time.perf_counter()
    results: list[FoldResult] = []
    squared_error = 0.0
    total = 0
    squared_error_trimmed = 0.0
    total_trimmed = 0
    squared_error_ex = 0.0
    total_ex = 0
    squared_error_lirf = 0.0
    total_lirf = 0
    squared_error_ex_trimmed = 0.0
    total_ex_trimmed = 0

    for fold in folds:
        months = tuple(sorted(set(fold.train_months) | set(fold.valid_months)))
        matrix = build_matrix(features, months, train_months=fold.train_months)

        is_train = matrix["month"].is_in(list(fold.train_months))
        train = matrix.filter(is_train)
        valid = matrix.filter(~is_train)

        model = make_model()
        predicted = fit_and_predict(
            model,
            train,
            valid,
            target=target,
            train_label_cap=train_label_cap,
            target_transform=target_transform,
            train_relabel=train_relabel,
            clip=clip,
        )
        # Pull importance out before the model goes out of scope at the end of the fold.
        importance = models.feature_importance(model)

        truth = valid[target].to_numpy().astype(np.float64)
        scored = pl.DataFrame(
            {
                "airport": valid["airport"],
                "month": valid["month"],
                "y": truth,
                "yhat": predicted,
            }
        )
        keep = truth <= TRIM_SECONDS
        at_lirf = (valid["airport"] == STEERING_AIRPORT).to_numpy()
        elsewhere = ~at_lirf
        ex_trimmed = steering_rows(valid["airport"].to_numpy(), truth)
        results.append(
            FoldResult(
                fold=fold.name,
                n_train=train.height,
                n_valid=valid.height,
                rmse=metrics.rmse(truth, predicted),
                rmse_trimmed=metrics.rmse(truth[keep], predicted[keep]),
                n_trimmed=int((~keep).sum()),
                mae=metrics.mae(truth, predicted),
                bias=metrics.bias(truth, predicted),
                rmse_ex_lirf=(
                    metrics.rmse(truth[elsewhere], predicted[elsewhere])
                    if elsewhere.any()
                    else float("nan")
                ),
                rmse_trimmed_ex_lirf=(
                    metrics.rmse(truth[ex_trimmed], predicted[ex_trimmed])
                    if ex_trimmed.any()
                    else float("nan")
                ),
                rmse_lirf_only=(
                    metrics.rmse(truth[at_lirf], predicted[at_lirf])
                    if at_lirf.any()
                    else float("nan")
                ),
                by_airport=metrics.breakdown(scored, ["airport", "month"], trim_above=TRIM_SECONDS),
                importance=importance,
                # Row-level out-of-fold predictions. Without these, "where does this model go
                # wrong" cannot be answered anywhere but the machine that fitted it, and the
                # error correlation a blend decision rests on cannot be computed at all.
                predictions=pl.DataFrame(
                    {
                        "mvt_id": valid["mvt_id"],
                        "fold": pl.Series([fold.name] * valid.height),
                        "airport": valid["airport"],
                        "y": truth,
                        "yhat": predicted,
                    }
                ),
            )
        )
        squared_error += float(np.sum((truth - predicted) ** 2))
        total += truth.size
        squared_error_trimmed += float(np.sum((truth[keep] - predicted[keep]) ** 2))
        total_trimmed += int(keep.sum())
        squared_error_ex += float(np.sum((truth[elsewhere] - predicted[elsewhere]) ** 2))
        total_ex += int(elsewhere.sum())
        squared_error_lirf += float(np.sum((truth[at_lirf] - predicted[at_lirf]) ** 2))
        total_lirf += int(at_lirf.sum())
        squared_error_ex_trimmed += float(np.sum((truth[ex_trimmed] - predicted[ex_trimmed]) ** 2))
        total_ex_trimmed += int(ex_trimmed.sum())

    return CVResult(
        folds=results,
        pooled_rmse=float(np.sqrt(squared_error / total)) if total else float("nan"),
        pooled_rmse_trimmed=(
            float(np.sqrt(squared_error_trimmed / total_trimmed)) if total_trimmed else float("nan")
        ),
        seconds=time.perf_counter() - started,
        pooled_rmse_ex_lirf=(
            float(np.sqrt(squared_error_ex / total_ex)) if total_ex else float("nan")
        ),
        pooled_rmse_lirf_only=(
            float(np.sqrt(squared_error_lirf / total_lirf)) if total_lirf else float("nan")
        ),
        pooled_rmse_trimmed_ex_lirf=(
            float(np.sqrt(squared_error_ex_trimmed / total_ex_trimmed))
            if total_ex_trimmed
            else float("nan")
        ),
    )
