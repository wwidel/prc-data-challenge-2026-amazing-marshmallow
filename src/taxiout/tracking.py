"""MLflow, kept behind one thin module so it stays replaceable.

Local SQLite store, no server. ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``
to browse. Everything needed to reproduce or
explain a score is logged: the feature blocks *and* their content hashes, the model, the fold
set, per-fold and pooled RMSE, and the per-airport breakdown as an artifact.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taxiout import paths, store

if TYPE_CHECKING:
    import polars as pl

    from taxiout.cv import CVResult
    from taxiout.experiment import Experiment

EXPERIMENT_NAME = "taxi-out"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

#: MLflow 3 retired the filesystem backend. SQLite is the supported local option and still
#: needs no server: one file at the repo root, artifacts beside it. Jobs running side by side
#: should each set MLFLOW_TRACKING_URI to their own SQLite file: concurrent writers to one file can
#: fail with "database is locked", and the out-of-fold file is written inside the MLflow run.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{paths.MLFLOW_DB}"


def _mlflow():
    import mlflow

    paths.MLARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=paths.MLARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def already_run(config_hash: str) -> bool:
    """Whether this exact configuration has a finished run anywhere on record.

    Checks the portable log as well as the local MLflow store. MLflow's SQLite database does not
    travel between machines, so consulting it alone means a second machine re-runs everything the
    first one did, and a rebuilt database loses the lot. `results/runs.jsonl` is plain text that
    can be copied or committed between machines, so it is the record that makes skipping work in a
    batch workflow.
    """
    if config_hash in _logged_config_hashes():
        return True

    mlflow = _mlflow()
    import mlflow as _m

    experiment = _m.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return False
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_hash = '{config_hash}' and attributes.status = 'FINISHED'",
        max_results=1,
    )
    return len(found) > 0


def _logged_config_hashes() -> set[str]:
    """Config hashes present in results/runs.jsonl. Cheap enough to read on every call."""
    if not RESULTS_LOG.exists():
        return set()
    found: set[str] = set()
    for line in RESULTS_LOG.read_text().splitlines():
        try:
            if h := json.loads(line).get("config_hash"):
                found.add(h)
        except json.JSONDecodeError:
            continue
    return found


RESULTS_LOG = paths.ROOT / "results" / "runs.jsonl"


def append_result(
    experiment: Experiment,
    result: CVResult,
    profile_name: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one run to `results/runs.jsonl`.

    MLflow's SQLite store does not travel between machines — two databases cannot be merged.
    This line-per-run file can: it is append-only plain text, so results from two machines
    combine by concatenation (a git `merge=union` attribute does it without conflicts). It is the
    channel a batch machine's results come back through.
    """
    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "machine": socket.gethostname(),
        "profile": profile_name,
        "git_commit": _git_commit(),
        # config_hash covers feature code, params, folds and treatments -- not cv.py or the model
        # wrappers -- so two rows under one config_hash can come from different fitting code
        # (one early config carries four RMSEs). These two make that visible.
        "code_fingerprint": _code_fingerprint(),
        "git_dirty": _git_dirty(),
        "experiment": experiment.name,
        "config_hash": experiment.config_hash(),
        "model_kind": experiment.model_kind,
        "model_params": experiment.model_params,
        # Label/target treatments live outside model_params but change what was fitted, so a
        # reader of this log must see them (they have always been inside config_hash).
        "train_label_cap": experiment.train_label_cap,
        "target_transform": experiment.target_transform,
        "train_relabel": experiment.train_relabel,
        "features": list(experiment.features),
        "fold_set": experiment.folds,
        "rmse_pooled": round(result.pooled_rmse, 3),
        "rmse_pooled_trimmed": round(result.pooled_rmse_trimmed, 3),
        # The steering metric and its counterparts. See cv.STEERING_AIRPORT for why pooled
        # RMSE is not the number to compare models on, and cv.CVResult for why the trimmed
        # figure rather than `rmse_ex_lirf` is the one to steer by.
        "rmse_ex_lirf": round(result.pooled_rmse_ex_lirf, 3),
        "rmse_trimmed_ex_lirf": round(result.pooled_rmse_trimmed_ex_lirf, 3),
        "rmse_lirf_only": round(result.pooled_rmse_lirf_only, 3),
        "folds": {
            f.fold: {
                "rmse": round(f.rmse, 3),
                "rmse_trimmed": round(f.rmse_trimmed, 3),
                "rmse_trimmed_ex_lirf": round(f.rmse_trimmed_ex_lirf, 3),
                "mae": round(f.mae, 3),
                "bias": round(f.bias, 3),
                "n_valid": f.n_valid,
            }
            for f in result.folds
        },
        "seconds": round(result.seconds, 1),
        # The per-airport breakdown and the importance ranking live here, not only in MLflow,
        # because mlartifacts/ is gitignored and does not travel. Without them a run made on
        # the batch machine can only ever be compared on one pooled number.
        "by_airport": _by_airport_records(result),
        "importance": _importance_records(result),
        "n_feature_columns": len(store.feature_columns(experiment.features)),
        **(extra or {}),
    }
    with RESULTS_LOG.open("a") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")


def _json_safe(value):
    """Replace NaN and infinity with null.

    `json.dumps` emits bare `NaN`, which is not valid JSON and which every strict parser -- the
    one in `read_results` included -- rejects. A fold set that trims away every row leaves a NaN
    behind, so this is reachable, not theoretical.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


#: How many features to carry in the log. The full ranking stays in the MLflow artifact.
IMPORTANCE_IN_LOG = 25


def _by_airport_records(result) -> list[dict]:
    """Per airport and fold: count, RMSE, trimmed RMSE, MAE, bias."""
    try:
        frame = result.by_airport_month()
    except Exception:  # noqa: BLE001 - a missing breakdown must not cost a finished run
        return []
    keep = [
        c
        for c in ("airport", "month", "n", "rmse", "rmse_trimmed", "mae", "bias")
        if c in frame.columns
    ]
    rows = frame.select(keep).to_dicts()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 2)
    return rows


def _importance_records(result) -> list[dict]:
    """Top features by mean gain across folds, or [] for a model that cannot report it."""
    try:
        frame = result.importance()
    except Exception:  # noqa: BLE001
        return []
    if frame is None or frame.height == 0:
        return []
    rows = frame.head(IMPORTANCE_IN_LOG).to_dicts()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 1)
    return rows


def _code_fingerprint() -> str:
    """Hash of every source file under src/taxiout, in path order: the code that fitted a run."""
    import hashlib

    h = hashlib.sha256()
    root = paths.ROOT / "src" / "taxiout"
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _git_dirty() -> bool | None:
    """True if tracked files differ from HEAD when the run was logged; None if git is absent."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def _git_commit() -> str:
    """Short commit of the working tree, so a result can be tied to the code that made it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return "unknown"


def read_results() -> pl.DataFrame:
    """Every recorded run, newest first. Empty frame when nothing has run yet.

    ``model_params`` is flattened to a JSON string before the frame is built. Model specs nest
    arbitrarily — a ``rules`` model carries a list of predicate dictionaries, a ``routed`` model
    carries another whole spec — and polars infers one schema for the column across every row, so
    a single new model kind can make the entire log unreadable. That happened: the first three
    ``rules`` runs broke ``read_results`` for all 76. Keeping the column as text costs nothing
    (nothing downstream reads inside it) and cannot break again whatever a model spec contains.
    """
    import json

    import polars as pl

    if not RESULTS_LOG.exists():
        return pl.DataFrame()
    records = []
    for line in RESULTS_LOG.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("model_params"), str):
            record["model_params"] = json.dumps(record.get("model_params"), sort_keys=True)
        records.append(record)
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records, infer_schema_length=None).sort("finished_at", descending=True)


def metrics_for(config_hash: str) -> dict[str, float]:
    """Metrics of the most recent finished run with this config hash, or {} if there is none.

    Used by the submission script so a submission's log row carries the CV score it was chosen
    on, without the caller having to look it up by hand.
    """
    mlflow = _mlflow()
    import mlflow as _m

    experiment = _m.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return {}
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_hash = '{config_hash}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not len(found):
        return {}
    row = found.iloc[0]
    return {
        key.removeprefix("metrics."): float(row[key])
        for key in found.columns
        if key.startswith("metrics.") and row[key] == row[key]
    }


def log(experiment: Experiment, result: CVResult, extra: dict[str, Any] | None = None) -> str:
    """Record one completed experiment. Returns the MLflow run id."""
    mlflow = _mlflow()
    with mlflow.start_run(run_name=experiment.name) as run:
        mlflow.set_tags(
            {
                "config_hash": experiment.config_hash(),
                "fold_set": experiment.folds,
                "model_kind": experiment.model_kind,
                "source": str(experiment.source or ""),
                **experiment.tags,
            }
        )
        mlflow.log_params(experiment.params())
        mlflow.log_metric("rmse_pooled", result.pooled_rmse)
        mlflow.log_metric("rmse_pooled_trimmed", result.pooled_rmse_trimmed)
        mlflow.log_metric("rmse_ex_lirf", result.pooled_rmse_ex_lirf)
        mlflow.log_metric("rmse_trimmed_ex_lirf", result.pooled_rmse_trimmed_ex_lirf)
        mlflow.log_metric("rmse_lirf_only", result.pooled_rmse_lirf_only)
        mlflow.log_metric("runtime_seconds", result.seconds)
        for fold in result.folds:
            mlflow.log_metric(f"rmse_fold_{fold.fold}", fold.rmse)
            mlflow.log_metric(f"rmse_trimmed_fold_{fold.fold}", fold.rmse_trimmed)
            mlflow.log_metric(f"rmse_trimmed_ex_lirf_fold_{fold.fold}", fold.rmse_trimmed_ex_lirf)
            mlflow.log_metric(f"mae_fold_{fold.fold}", fold.mae)
            mlflow.log_metric(f"bias_fold_{fold.fold}", fold.bias)
            for row in fold.by_airport.iter_rows(named=True):
                mlflow.log_metric(f"rmse_{fold.fold}_{row['airport']}", row["rmse"])

        with tempfile.TemporaryDirectory() as tmp:
            _write_artifacts(mlflow, Path(tmp), experiment, result, extra)

        return run.info.run_id


# --- artifacts ------------------------------------------------------------------------------
#: How many features the importance chart shows. The tail is in the CSV regardless.
IMPORTANCE_TOP_N = 25


def _write_artifacts(mlflow, tmp: Path, experiment, result, extra) -> None:
    """Everything worth looking at after a run, as files MLflow can render.

    Tables go out as CSV so they can be diffed and re-read; the two charts exist because a
    ten-by-two error grid and a feature ranking are both read by eye, not by scrolling.
    """
    import polars as pl

    summary = result.summary()
    summary.write_csv(tmp / "fold_summary.csv")
    mlflow.log_artifact(str(tmp / "fold_summary.csv"))

    by_airport_month = result.by_airport_month().with_columns(pl.col(pl.Float64).round(2))
    by_airport_month.write_csv(tmp / "by_airport_month.csv")
    mlflow.log_artifact(str(tmp / "by_airport_month.csv"))

    oof = result.out_of_fold()
    if oof is not None:
        # Written to a stable path as well as to MLflow, because mlartifacts/ is keyed on run id
        # and is awkward to find later. predictions/ is gitignored -- at ~4 MB a run these do not
        # belong in git -- and is fetched between machines on demand.
        stable = paths.PREDICTIONS / f"{experiment.name}__{experiment.config_hash()}.parquet"
        stable.parent.mkdir(parents=True, exist_ok=True)
        oof.write_parquet(stable, compression="zstd")
        oof.write_parquet(tmp / "out_of_fold.parquet", compression="zstd")
        mlflow.log_artifact(str(tmp / "out_of_fold.parquet"))

    importance = result.importance()
    if importance is not None:
        importance.with_columns(pl.col(pl.Float64).round(2)).write_csv(
            tmp / "feature_importance.csv"
        )
        mlflow.log_artifact(str(tmp / "feature_importance.csv"))

    (tmp / "feature_hashes.json").write_text(json.dumps(experiment.feature_hashes(), indent=2))
    mlflow.log_artifact(str(tmp / "feature_hashes.json"))

    if experiment.description:
        (tmp / "description.md").write_text(experiment.description)
        mlflow.log_artifact(str(tmp / "description.md"))

    for key, value in (extra or {}).items():
        mlflow.log_param(key, value)

    try:
        _write_charts(mlflow, tmp, experiment, by_airport_month, importance)
    except Exception as error:  # noqa: BLE001 - a failed chart must not lose a finished run
        mlflow.set_tag("charts_failed", str(error)[:250])


def _write_charts(mlflow, tmp: Path, experiment, by_airport_month, importance) -> None:
    import polars as pl

    from taxiout import viz

    viz.style()
    plt = viz.plt

    # RMSE per airport x month, as a grid — trimmed, so one bad row does not set the scale.
    column = "rmse_trimmed" if "rmse_trimmed" in by_airport_month.columns else "rmse"
    wide = by_airport_month.pivot(on="month", index="airport", values=column).fill_null(
        float("nan")
    )
    order = wide.with_columns(pl.mean_horizontal(pl.exclude("airport")).alias("_m")).sort(
        "_m", descending=True
    )
    months = [c for c in wide.columns if c != "airport"]

    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(months), 0.42 * order.height + 1.8))
    viz.heatmap(
        ax,
        order.select(months).to_numpy().astype(float),
        order["airport"].to_list(),
        months,
        cbar_label=f"{column} (s)",
    )
    viz.titled(
        ax,
        f"{experiment.name} — {column} by airport and fold",
        "Rows ordered worst first. Held-out months only.",
    )
    fig.savefig(tmp / "rmse_by_airport_month.png", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp / "rmse_by_airport_month.png"))

    if importance is None or importance.height == 0:
        return
    top = importance.head(IMPORTANCE_TOP_N).sort("importance")
    fig, ax = plt.subplots(figsize=(8, 0.28 * top.height + 1.6))
    ax.barh(top["feature"].to_list(), top["importance"].to_list(), color=viz.BLUE, height=0.7)
    viz.titled(
        ax,
        f"{experiment.name} — feature importance",
        f"Gain, averaged across folds. Top {min(IMPORTANCE_TOP_N, importance.height)} "
        f"of {importance.height}.",
        xlabel="gain",
    )
    viz.only_x_grid(ax)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.savefig(tmp / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp / "feature_importance.png"))


def read_by_airport() -> pl.DataFrame:
    """Every run's per-airport breakdown, one row per run-airport-fold.

    This is what answers "which airport is consistently mispredicted" without needing the
    machine that fitted the model.
    """
    import polars as pl

    runs = read_results()
    if runs.is_empty() or "by_airport" not in runs.columns:
        return pl.DataFrame()
    return (
        runs.select("experiment", "machine", "fold_set", "rmse_pooled", "by_airport")
        .explode("by_airport")
        .drop_nulls("by_airport")
        .unnest("by_airport")
    )


def read_importance() -> pl.DataFrame:
    """Every run's feature importance, one row per run-feature."""
    import polars as pl

    runs = read_results()
    if runs.is_empty() or "importance" not in runs.columns:
        return pl.DataFrame()
    return (
        runs.select("experiment", "machine", "rmse_pooled", "importance")
        .explode("importance")
        .drop_nulls("importance")
        .unnest("importance")
    )
