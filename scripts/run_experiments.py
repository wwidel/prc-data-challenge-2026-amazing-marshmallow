"""Run experiment configs and log them.

Points at a directory by default, so the normal workflow is: drop a YAML in
``config/experiments/``, run this, and it works through everything not already recorded.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback

from taxiout import paths, store, tracking
from taxiout.experiment import Experiment, load_all


def _print_result(exp: Experiment, result) -> None:
    # The TRIMMED ex-LIRF leads, because that is the metric model selection uses. The untrimmed
    # ex-LIRF is roughly 63 s higher on a seasonal run; both are printed, because the untrimmed
    # one is what feeds cv.predicted_board().
    print(
        f"    trim ex-LIRF {result.pooled_rmse_trimmed_ex_lirf:7.2f} s  <- steer by this"
        f"   | untrim ex-LIRF {result.pooled_rmse_ex_lirf:7.2f}"
        f"   pooled {result.pooled_rmse:8.2f}"
        f"   trimmed {result.pooled_rmse_trimmed:7.2f}"
        f"   LIRF {result.pooled_rmse_lirf_only:8.1f}   {result.seconds:.1f}s"
    )
    for fold in result.folds:
        print(
            f"      {fold.fold:<10} rmse {fold.rmse:8.2f}  trimmed {fold.rmse_trimmed:7.2f}"
            f"  trim ex-LIRF {fold.rmse_trimmed_ex_lirf:7.2f}"
            f"  mae {fold.mae:7.2f}  bias {fold.bias:+7.2f}  n={fold.n_valid:,}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "configs",
        nargs="*",
        default=[str(paths.EXPERIMENTS)],
        help="YAML files or directories (default: config/experiments)",
    )
    ap.add_argument(
        "--force", action="store_true", help="re-run even if this config already has a finished run"
    )
    ap.add_argument("--dry-run", action="store_true", help="show the queue and exit")
    ap.add_argument("--no-track", action="store_true", help="run without writing to MLflow")
    ap.add_argument("--build", action="store_true", help="build any missing feature blocks first")
    ap.add_argument(
        "--folds",
        default=None,
        help="run every config on this fold set instead of its own (e.g. final_pairs)",
    )
    args = ap.parse_args()

    queue: list[Experiment] = []
    for target in args.configs:
        queue.extend(load_all(target))
    if args.folds:
        # A different fold set is a different experiment: it enters the config hash, so the
        # override never skips a run because the same config was scored on other folds.
        queue = [dataclasses.replace(e, folds=args.folds) for e in queue]

    if args.build:
        from taxiout.features import registry as feature_registry

        # Target-dependent blocks are computed per fold inside cv.build_matrix and store.build
        # refuses to cache them, so --build must not hand them over -- a queue containing one
        # would crash before its first experiment.
        needed = sorted(
            {f for e in queue for f in e.features if feature_registry.get(f).kind.cacheable}
        )
        if needed:
            print("building features:")
            for r in store.build_many(needed):
                print(f"  {r.name:<14} {r.hash}  {'built' if r.rebuilt else 'cached'}")

    pending: list[tuple[Experiment, bool]] = []
    for exp in queue:
        done = (not args.force) and (not args.no_track) and tracking.already_run(exp.config_hash())
        pending.append((exp, done))

    print(f"\n{len(queue)} experiment(s):")
    for exp, done in pending:
        mark = "skip" if done else " run"
        print(f"  [{mark}] {exp.name:<28} {exp.config_hash()}  {exp.model_kind}  folds={exp.folds}")
    if args.dry_run:
        return 0

    failures = 0
    for exp, done in pending:
        if done:
            continue
        print(f"\n>> {exp.name}")
        try:
            result = exp.run()
        except Exception:
            failures += 1
            print(f"    FAILED\n{traceback.format_exc()}", file=sys.stderr)
            continue
        _print_result(exp, result)
        if not args.no_track:
            run_id = tracking.log(exp, result)
            tracking.append_result(exp, result)
            print(f"    logged {run_id}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
