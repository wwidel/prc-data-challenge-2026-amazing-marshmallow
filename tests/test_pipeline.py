"""Invariants of the raw data, the spine, the feature store and the models.

Gating is deliberate and three-tiered. Pure-logic tests always run. Tests needing the raw
competition files carry ``@requires_raw``; tests needing a built spine carry
``@requires_spine``. An earlier version gated the *whole module* on the spine existing, which
meant a machine with no data at all reported a clean run — precisely the reassurance the suite
is supposed to provide and precisely when it cannot.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from taxiout import io, paths, spine, store
from taxiout.features import REGISTRY, registry
from taxiout.models.baseline import ConstantPredictor

requires_raw = pytest.mark.skipif(
    not paths.RANKING.exists(),
    reason="raw competition files absent from data/raw",
)
requires_spine = pytest.mark.skipif(
    not (paths.SPINE / "2025-01.parquet").exists(),
    reason="spine not built; run scripts/build_spine.py",
)

#: Row count of every raw file. A truncated or partial copy fails here rather than three hours
#: into a batch, and naming the individual files is what makes the failure actionable — a total
#: that is 331,548 short says only that something is missing.
EXPECTED_RAW_FILES = {
    "ranking.parquet": 689_534,
    "submitting.parquet": 344_841,
    "training_2025-01-01_2025-02-01.parquet": 307_257,
    "training_2025-02-01_2025-03-01.parquet": 287_411,
    "training_2025-03-01_2025-04-01.parquet": 328_622,
    "training_2025-04-01_2025-05-01.parquet": 350_288,
    "training_2025-05-01_2025-06-01.parquet": 370_105,
    "training_2025-06-01_2025-07-01.parquet": 365_986,
    "training_2025-07-01_2025-08-01.parquet": 381_161,
    "training_2025-08-01_2025-09-01.parquet": 382_047,
    "training_2025-09-01_2025-10-01.parquet": 367_679,
    "training_2025-10-01_2025-11-01.parquet": 371_163,
    "training_2025-11-01_2025-12-01.parquet": 324_530,
    "training_2025-12-01_2026-01-01.parquet": 331_548,
}
EXPECTED_RAW_ROWS = {"training": 4_167_797, "ranking": 689_534, "submitting": 344_841}

CACHEABLE = [n for n, b in REGISTRY.items() if b.kind.cacheable]


@requires_raw
def test_raw_files_are_complete():
    """The data copy landed intact — the check a remote machine most needs before a batch.

    Reports per file rather than in total, because "331,548 rows short" does not tell you which
    `rsync` to re-run and a list of missing filenames does.
    """
    problems = []
    for name, expected in EXPECTED_RAW_FILES.items():
        path = paths.RAW / name
        if not path.exists():
            problems.append(f"{name}: MISSING (expected {expected:,} rows)")
            continue
        got = pl.scan_parquet(path).select(pl.len()).collect().item()
        if got != expected:
            problems.append(f"{name}: {got:,} rows, expected {expected:,} ({got - expected:+,})")

    extra = {p.name for p in paths.RAW.glob("*.parquet")} - set(EXPECTED_RAW_FILES)
    if extra:
        problems.append(f"unexpected files in data/raw: {sorted(extra)}")

    assert not problems, "data/raw is not intact:\n  - " + "\n  - ".join(problems)

    total = io.scan_training().select(pl.len()).collect(engine="streaming").item()
    assert total == EXPECTED_RAW_ROWS["training"]


@requires_spine
def test_spine_covers_every_month():
    assert set(spine.months_available()) == set(paths.ALL_MONTHS)


@requires_spine
def test_spine_rows_are_unique_departures():
    sp = spine.load()
    assert sp["mvt_id"].n_unique() == sp.height
    assert sp["taxi_sec"].null_count() == 344_841, "only ranking departures may lack a target"


@requires_spine
def test_ranking_months_have_no_target_and_training_months_do():
    sp = spine.load()
    rank = sp.filter(pl.col("split") == "rank")
    train = sp.filter(pl.col("split") == "train")
    assert rank["taxi_sec"].null_count() == rank.height
    assert train["taxi_sec"].null_count() == 0
    assert set(rank["month"].unique()) == set(paths.RANK_MONTHS)


@requires_spine
def test_spine_row_order_is_the_declared_sort():
    for month in ("2025-01", "2026-07"):
        frame = pl.read_parquet(paths.SPINE / f"{month}.parquet")
        assert frame.equals(frame.sort(spine.SORT_KEY)), f"{month} is not in canonical order"


@requires_spine
@pytest.mark.parametrize("name", CACHEABLE)
def test_feature_block_heights_match_the_spine(name):
    block = registry.get(name)
    if not store.is_built(block, paths.ALL_MONTHS):
        pytest.skip(f"{name} not built")
    heights = spine.heights()
    for month, expected in heights.items():
        got = (
            pl.scan_parquet(store.block_dir(block) / f"{month}.parquet")
            .select(pl.len())
            .collect()
            .item()
        )
        assert got == expected, f"{name}/{month}: {got} rows against a spine of {expected}"


@requires_spine
def test_assembled_matrix_is_aligned_to_the_raw_data():
    """The load-bearing invariant: horizontal concat must line up with the right movements.

    Recomputes one feature from the raw files for a sample of movements and checks it against
    what came out of the store.
    """
    built = [n for n in CACHEABLE if store.is_built(registry.get(n), paths.ALL_MONTHS)]
    if "offblock" not in built:
        pytest.skip("offblock block not built")

    matrix = store.assemble(["offblock"], ("2025-03",)).collect()
    sample = matrix.select("mvt_id", "ob_aobt_sec").sample(500, seed=0)

    truth = (
        io.scan_training()
        .filter(pl.col("phase") == "DEP")
        .filter(pl.col("mvt_id").is_in(sample["mvt_id"].to_list()))
        .select(
            "mvt_id",
            (pl.col("mvt_time") - pl.col("aobt_3")).dt.total_seconds().alias("expected"),
        )
        .collect()
    )
    merged = sample.join(truth, on="mvt_id", how="inner")
    assert merged.height == sample.height
    delta = (merged["ob_aobt_sec"].cast(pl.Float64) - merged["expected"].cast(pl.Float64)).abs()
    assert delta.fill_null(0).max() < 1e-6


def test_registry_allows_reregistration_from_the_same_module():
    """Editing a feature module under %autoreload re-runs its decorators; that must not raise."""
    from taxiout.features import registry as reg

    name = "_reload_probe"
    try:

        def make():
            @reg.feature(name=name, kind=reg.FeatureKind.ROW_LOCAL, outputs=("_probe",))
            def probe(ctx):  # pragma: no cover - never executed
                raise AssertionError

            return probe

        first = make()
        second = make()
        assert reg.REGISTRY[name] is second
        assert first is not second
    finally:
        reg.REGISTRY.pop(name, None)


def test_registry_rejects_a_name_claimed_by_two_modules():
    from taxiout.features import registry as reg

    name = "_collision_probe"
    try:

        @reg.feature(name=name, kind=reg.FeatureKind.ROW_LOCAL, outputs=("_c",))
        def first(ctx):  # pragma: no cover - never executed
            raise AssertionError

        def impostor(ctx):  # pragma: no cover - never executed
            raise AssertionError

        impostor.__module__ = "some.other.module"
        with pytest.raises(ValueError, match="cannot claim it too"):
            reg.feature(name=name, kind=reg.FeatureKind.ROW_LOCAL, outputs=("_c",))(impostor)
    finally:
        reg.REGISTRY.pop(name, None)


def test_registry_rejects_colliding_output_columns():
    names = list(REGISTRY)
    seen: set[str] = set()
    for name in names:
        outputs = set(REGISTRY[name].outputs)
        assert not (outputs & seen), f"{name} re-emits an existing column"
        seen |= outputs


@requires_spine
def test_target_dependent_blocks_are_never_cached():
    for block in REGISTRY.values():
        if block.kind.cacheable:
            continue
        with pytest.raises(ValueError, match="computed per fold"):
            store.build(block)


def test_constant_predictor_uses_only_training_labels():
    model = ConstantPredictor("median").fit(
        pl.DataFrame({"x": [1, 2, 3]}), pl.Series([10.0, 20.0, 90.0]).to_numpy()
    )
    assert model.predict(pl.DataFrame({"x": [1, 2]})).tolist() == [20.0, 20.0]


@requires_raw
def test_submission_template_matches_the_ranking_departures():
    from taxiout import postprocess

    assert set(postprocess.template()["mvt_id"]) == set(postprocess.ranking_departure_ids())


@requires_raw
def test_submission_validates_and_round_trips():
    from taxiout import postprocess

    built = postprocess.build_frame(
        postprocess.template().select("mvt_id", pl.lit(900.0).alias("prediction"))
    )
    postprocess.validate(built)
    assert built.height == 344_841
    assert built["TAXITIME_SEC_mvt"].dtype == pl.Int32
    assert (built["TAXITIME_SEC_mvt"] == 900).all()


@requires_raw
def test_submission_validation_catches_corruption():
    from taxiout import postprocess

    good = postprocess.build_frame(
        postprocess.template().select("mvt_id", pl.lit(900.0).alias("prediction"))
    )
    with pytest.raises(ValueError, match="rows"):
        postprocess.validate(good.head(100))
    with pytest.raises(ValueError, match="null predictions"):
        postprocess.validate(
            good.with_columns(pl.lit(None, dtype=pl.Int32).alias("TAXITIME_SEC_mvt"))
        )


def _toy_matrix(n: int = 4000, seed: int = 0):
    """A small frame with the spine columns the models expect, plus two features."""
    import numpy as np

    rng = np.random.default_rng(seed)
    airports = rng.choice(["AAAA", "BBBB", "CCCC"], n)
    offsets = {"AAAA": 0.0, "BBBB": 300.0, "CCCC": -200.0}
    signal = rng.normal(0, 100, n)
    frame = pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "airport": airports,
            "month": rng.choice(["2025-01", "2025-02", "2025-03"], n),
            "split": ["train"] * n,
            "mvt_time": [None] * n,
            "taxi_sec": None,
            "cat_airport": pl.Series(
                [{"AAAA": 0, "BBBB": 1, "CCCC": 2}[a] for a in airports], dtype=pl.Int32
            ),
            "feat_signal": pl.Series(signal, dtype=pl.Float32),
        }
    )
    y = 900 + np.array([offsets[a] for a in airports]) + 2.0 * signal + rng.normal(0, 30, n)
    return frame, y


def test_lgbm_fits_and_ignores_spine_columns():
    from taxiout.models.lgbm import LGBMPredictor, feature_columns

    frame, y = _toy_matrix()
    assert set(feature_columns(frame)) == {"cat_airport", "feat_signal"}

    model = LGBMPredictor(num_boost_round=60, early_stopping_rounds=None, valid_fraction=0.0).fit(
        frame, y
    )
    predicted = model.predict(frame)
    assert predicted.shape == (frame.height,)
    # must beat predicting the mean by a wide margin on data this separable
    assert np.sqrt(np.mean((y - predicted) ** 2)) < 0.5 * float(np.std(y))
    assert "month" not in model.features_, "spine columns must never reach the model"


def test_grid_expands_into_one_experiment_per_combination():
    from taxiout.experiment import expand_grid

    raw = {
        "name": "sweep",
        "features": [],
        "model": {"kind": "lgbm", "params": {"learning_rate": 0.05}},
        "grid": {"num_leaves": [63, 127], "min_data_in_leaf": [100, 400]},
    }
    expanded = expand_grid(raw)
    assert len(expanded) == 4
    assert len({e.config_hash() for e in expanded}) == 4, "each variant needs its own identity"
    assert all(e.model_params["learning_rate"] == 0.05 for e in expanded)
    assert {e.model_params["num_leaves"] for e in expanded} == {63, 127}
    assert raw["model"]["params"] == {"learning_rate": 0.05}, "the source config must not mutate"


def test_already_run_consults_the_portable_log(tmp_path, monkeypatch):
    """A run recorded on another machine must count, or a batch redoes everything."""
    import json

    from taxiout import tracking

    log = tmp_path / "runs.jsonl"
    log.write_text(json.dumps({"config_hash": "deadbeef1234", "experiment": "elsewhere"}) + "\n")
    monkeypatch.setattr(tracking, "RESULTS_LOG", log)

    assert tracking._logged_config_hashes() == {"deadbeef1234"}
    # present in the log -> short-circuits before MLflow is touched
    monkeypatch.setattr(tracking, "_mlflow", lambda: pytest.fail("should not reach MLflow"))
    assert tracking.already_run("deadbeef1234") is True


def test_logged_config_hashes_survives_a_corrupt_line(tmp_path, monkeypatch):
    from taxiout import tracking

    log = tmp_path / "runs.jsonl"
    log.write_text('{"config_hash": "aaa"}\nnot json\n{"no_hash": 1}\n{"config_hash": "bbb"}\n')
    monkeypatch.setattr(tracking, "RESULTS_LOG", log)
    assert tracking._logged_config_hashes() == {"aaa", "bbb"}


def test_run_log_carries_breakdowns_that_survive_the_trip():
    """Per-airport and importance must be in runs.jsonl, since mlartifacts/ is gitignored."""
    from taxiout import tracking

    runs = tracking.read_results()
    if runs.is_empty():
        pytest.skip("no runs recorded yet")
    assert "by_airport" in runs.columns, "per-airport breakdown must travel with the run"
    assert "importance" in runs.columns, "feature importance must travel with the run"

    by_airport = tracking.read_by_airport()
    if not by_airport.is_empty():
        assert {"airport", "month", "n", "rmse"} <= set(by_airport.columns)


def test_lgbm_accepts_sample_weights():
    """Weights must reach the booster: RoutedPredictor's borrowing depends on it."""
    from taxiout.models.lgbm import LGBMPredictor

    frame, y = _toy_matrix(n=3000, seed=1)
    heavy = (frame["airport"] == "AAAA").to_numpy().astype(float) * 9 + 1
    plain = LGBMPredictor(num_boost_round=40, early_stopping_rounds=None, valid_fraction=0.0).fit(
        frame, y
    )
    weighted = LGBMPredictor(
        num_boost_round=40, early_stopping_rounds=None, valid_fraction=0.0
    ).fit(frame, y, weight=heavy)
    mask = (frame["airport"] == "AAAA").to_numpy()
    # up-weighting a group should reduce its error relative to the unweighted fit
    err_plain = np.abs(plain.predict(frame)[mask] - y[mask]).mean()
    err_weighted = np.abs(weighted.predict(frame)[mask] - y[mask]).mean()
    assert err_weighted <= err_plain * 1.05


def test_routed_predictor_splits_and_borrows():
    from taxiout.models.meta import RoutedPredictor

    frame, y = _toy_matrix(n=6000, seed=2)
    model = RoutedPredictor(
        by="airport",
        routes={"special": ["AAAA"]},
        base={"kind": "constant", "params": {"stat": "mean"}},
        min_rows=100,
    ).fit(frame, y)
    assert set(model.models_) == {"special", "rest"}

    predicted = model.predict(frame)
    special = (frame["airport"] == "AAAA").to_numpy()
    assert predicted[special][0] == pytest.approx(y[special].mean(), abs=1e-6)
    assert predicted[~special][0] == pytest.approx(y[~special].mean(), abs=1e-6)


def test_routed_foreign_weight_falls_back_when_base_rejects_weights():
    """A base without weight support must still produce a working routed model."""
    from taxiout.models.meta import RoutedPredictor

    frame, y = _toy_matrix(n=4000, seed=3)
    model = RoutedPredictor(
        by="airport",
        routes={"special": ["AAAA"]},
        base={"kind": "constant"},  # ConstantPredictor.fit takes no weight
        foreign_weight=0.3,
        min_rows=100,
    ).fit(frame, y)
    assert set(model.models_) == {"special", "rest"}
    assert np.isfinite(model.predict(frame)).all()


@pytest.mark.parametrize("kind", ["catboost"])
def test_diverse_models_fit_and_beat_the_mean(kind):
    """The second family must fit, respect the spine-column exclusion, and learn something."""
    from taxiout import models

    frame, y = _toy_matrix(n=3000, seed=4)
    params = {
        "catboost": {"iterations": 60, "depth": 4},
    }[kind]
    model = models.factory(kind, params)().fit(frame, y)
    predicted = model.predict(frame)
    assert predicted.shape == (frame.height,)
    assert np.sqrt(np.mean((y - predicted) ** 2)) < float(np.std(y))
    if hasattr(model, "features_"):
        assert "month" not in model.features_, "spine columns must not reach the model"


def _target_dependent_block_names() -> list[str]:
    """Every registered TARGET_DEPENDENT block.

    Deliberately derived from the registry rather than hardcoded: the leak tests below are the
    only thing standing between a new fold-aware block and a beautiful, worthless score, so a
    block must not be able to escape them by not being added to a list: one block was added after
    such a list was written and would have shipped untested.
    """
    from taxiout.features import REGISTRY
    from taxiout.features.registry import FeatureKind

    return sorted(n for n, b in REGISTRY.items() if b.kind is FeatureKind.TARGET_DEPENDENT)


@requires_spine
@pytest.mark.parametrize("block_name", _target_dependent_block_names())
def test_target_dependent_block_cannot_see_heldout_labels(block_name):
    """Every TARGET_DEPENDENT block, against the per-fold path.

    A validation row's `md_p_a` is a rate estimated from *training* rows. If corrupting every
    held-out label moves any output value, the block is reading labels it was not given, and
    every score built on it is a leak. This is CLAUDE.md rule 3 as an assertion.
    """
    import numpy as np

    from taxiout.features import get
    from taxiout.features.registry import FeatureContext

    full = spine.load()
    mask = full["month"].is_in(["2025-02", "2025-03", "2025-04"])
    block = get(block_name)

    clean = block.compute(FeatureContext(spine=full, train_mask=mask))

    corrupted_spine = full.with_columns(
        pl.when(mask)
        .then(pl.col("taxi_sec"))
        .otherwise(pl.lit(999_999, dtype=pl.Int32))
        .alias("taxi_sec")
    )
    corrupted = block.compute(FeatureContext(spine=corrupted_spine, train_mask=mask))

    for column in clean.columns:
        assert np.array_equal(
            clean[column].to_numpy(), corrupted[column].to_numpy(), equal_nan=True
        ), f"{column} moved when held-out labels were corrupted — the block leaks"


@requires_spine
@pytest.mark.parametrize("block_name", _target_dependent_block_names())
def test_target_dependent_block_does_read_its_training_labels(block_name):
    """The converse: corrupting the TRAINING labels must move the rates, or the block reads
    nothing at all and the leak test above is vacuous."""
    import numpy as np

    from taxiout.features import get
    from taxiout.features.registry import FeatureContext

    full = spine.load()
    mask = full["month"].is_in(["2025-02", "2025-03", "2025-04"])
    block = get(block_name)

    clean = block.compute(FeatureContext(spine=full, train_mask=mask))
    corrupted_spine = full.with_columns(
        pl.when(mask)
        .then(pl.lit(999_999, dtype=pl.Int32))
        .otherwise(pl.col("taxi_sec"))
        .alias("taxi_sec")
    )
    corrupted = block.compute(FeatureContext(spine=corrupted_spine, train_mask=mask))
    assert not np.array_equal(
        clean[block.outputs[0]].to_numpy(), corrupted[block.outputs[0]].to_numpy()
    )


def test_train_label_cap_reaches_the_fit_but_not_the_score():
    """cv.run must winsorize what the model learns from and score on raw labels."""
    import numpy as np

    from taxiout import cv

    seen = {}

    class Recorder:
        def fit(self, frame, y):
            seen["max_y"] = float(np.max(y))

        def predict(self, frame):
            return np.zeros(frame.height)

    # pure-logic exercise of the winsorization branch via a tiny fake fold matrix
    import polars as pl

    fake = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "month": ["a", "a", "b", "b"],
            "airport": ["EHAM"] * 4,
            "taxi_sec": [100.0, 99_999.0, 200.0, 50_000.0],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    import taxiout.cv as cvmod

    orig = cvmod.build_matrix
    cvmod.build_matrix = lambda *a, **k: fake
    try:
        fold = cv.Fold(name="t", train_months=("a",), valid_months=("b",))
        res = cv.run(Recorder, ("x",), [fold], train_label_cap=7_200.0)
    finally:
        cvmod.build_matrix = orig
    assert seen["max_y"] == 7_200.0
    # the validation labels stayed raw: the fold RMSE reflects the 50,000 s row
    assert res.folds[0].rmse > 20_000


def test_log_target_back_transform_targets_the_mean_not_the_median():
    """A log fit must come back through Duan smearing, or it under-predicts by construction."""
    import numpy as np
    import polars as pl

    from taxiout import cv as cvmod

    # y is log-normal around a constant: the geometric mean is exp(3) ~ 20, the arithmetic
    # mean is exp(3 + 0.5*0.5^2) ~ 22.7. A constant model must return the arithmetic one.
    rng = np.random.default_rng(0)
    y = np.exp(3.0 + 0.5 * rng.standard_normal(4000))
    fake = pl.DataFrame(
        {
            "mvt_id": range(8000),
            "month": ["a"] * 4000 + ["b"] * 4000,
            "airport": ["EHAM"] * 8000,
            "taxi_sec": np.concatenate([y, y]),
            "x": np.ones(8000),
        }
    )

    class ConstantMean:
        def fit(self, frame, target):
            self._v = float(np.mean(target))

        def predict(self, frame):
            return np.full(frame.height, self._v)

    orig = cvmod.build_matrix
    cvmod.build_matrix = lambda *a, **k: fake
    try:
        fold = cvmod.Fold(name="t", train_months=("a",), valid_months=("b",))
        plain = cvmod.run(ConstantMean, ("x",), [fold])
        logged = cvmod.run(ConstantMean, ("x",), [fold], target_transform="log")
    finally:
        cvmod.build_matrix = orig

    # both should land near the arithmetic mean; the smeared log fit must not sit at the
    # geometric mean (which would be ~2.7 lower and show up as a much worse RMSE)
    assert logged.folds[0].rmse < plain.folds[0].rmse * 1.05
    assert abs(logged.folds[0].bias) < 1.0


def test_repair_training_labels_fixes_two_mechanisms_and_drops_the_third():
    import polars as pl

    from taxiout.cv import repair_training_labels

    frame = pl.DataFrame(
        {
            # ordinary | LATE with a usable aobt | SCHED copy | DAY | LATE, aobt unusable
            # | punctual row whose label happens to equal the schedule gap (must NOT be touched)
            "taxi_sec": [900.0, 48.0, 5_400.0, 87_300.0, 30.0, 700.0],
            "ob_aobt_sec": [880.0, 612.0, 1_100.0, 900.0, 99_999.0, 750.0],
            "ob_sched_sec": [1_500.0, 700.0, 5_400.0, 90_000.0, 800.0, 700.0],
        }
    )
    out = repair_training_labels(frame)
    # the DAY row (87,300 = 86,400 + 900) is gone
    assert out.height == 5
    got = out["taxi_sec"].to_list()
    assert got[0] == 900.0  # ordinary row untouched
    assert got[1] == 612.0  # LATE repaired from the flight plan
    assert got[2] == 1_100.0  # SCHED copy repaired: label == sched gap AND far from aobt
    assert got[3] == 30.0  # LATE with an implausible aobt is left alone
    # the punctual row's label equals its schedule gap but agrees with aobt_3 within 300 s,
    # so it is a real on-time taxi and must survive untouched
    assert got[4] == 700.0


def test_repair_demands_the_offblock_columns():
    import polars as pl
    import pytest

    from taxiout.cv import repair_training_labels

    with pytest.raises(ValueError, match="ob_aobt_sec"):
        repair_training_labels(pl.DataFrame({"taxi_sec": [1.0]}))


def test_every_refit_path_applies_the_label_treatments():
    """Regression guard for a bug we shipped once: a capped model must not refit on raw labels.

    The invariant is structural — cross-validation and the refit path must route through
    cv.fit_and_predict — so this test asserts the treatment actually changes what is fitted.
    """
    import numpy as np
    import polars as pl

    from taxiout import cv as cvmod

    seen = {}

    class Recorder:
        def fit(self, frame, y):
            seen["max_y"] = float(np.max(y))

        def predict(self, frame):
            return np.zeros(frame.height)

    frame = pl.DataFrame({"taxi_sec": [100.0, 90_000.0], "x": [1.0, 2.0]})
    cvmod.fit_and_predict(Recorder(), frame, frame, train_label_cap=7_200.0)
    assert seen["max_y"] == 7_200.0

    # and the refit entry point must call it rather than fitting directly
    source = (paths.ROOT / "scripts" / "refit_predict.py").read_text()
    assert "fit_and_predict" in source
    assert 'model.fit(train, train["taxi_sec"]' not in source
