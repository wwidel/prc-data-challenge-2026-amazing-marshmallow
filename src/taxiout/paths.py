"""Every filesystem path in the project. Nothing else hardcodes a directory.

Two phases share the code. ``board`` (the default) predicts the leaderboard set,
``ranking.parquet``: January and July 2026. ``final`` (``TAXIOUT_PHASE=final``) predicts the final
phase's set, ``final_ranking.parquet``: January, February, June and July 2026, in which the
organisers jittered takeoff and arrival in-block times. Everything a phase writes about 2026 lives
in its own directories, so the two never mix in one checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: ``board`` or ``final``; read once, at import.
PHASE: str = os.environ.get("TAXIOUT_PHASE", "board")
if PHASE not in ("board", "final"):
    raise ValueError(f"TAXIOUT_PHASE must be 'board' or 'final', got {PHASE!r}")
FINAL: bool = PHASE == "final"
_SUFFIX = "_final" if FINAL else ""

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / f"features{_SUFFIX}"
#: The built prediction file (competition format). Gitignored, like everything under data/.
OUTPUT = DATA / "output"
#: Out-of-fold predictions, one parquet per experiment. Gitignored: ~4 MB each.
PREDICTIONS = DATA / f"predictions{_SUFFIX}"
#: Each config's twelve-month refit, predicting the phase's 2026 rows (``refit_predict.py``).
RANK_PREDS = DATA / f"rank_preds{_SUFFIX}"
#: Where the pulled open data lives (read only).
EXTERNAL = DATA / "external"
#: Where the override generators write their files, and a120 its shifts.
OVERRIDES = EXTERNAL if not FINAL else DATA / "overrides_final"

REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
EXPERIMENTS = CONFIG / "experiments"
MLFLOW_DB = ROOT / "mlflow.db"
MLARTIFACTS = ROOT / "mlartifacts"

SPINE = INTERIM / f"spine{_SUFFIX}"

TRAINING_GLOB = str(RAW / "training_*.parquet")
RANKING = RAW / ("final_ranking.parquet" if FINAL else "ranking.parquet")
SUBMITTING = RAW / ("final_submitting.parquet" if FINAL else "submitting.parquet")

#: Months present in the training files.
TRAIN_MONTHS: tuple[str, ...] = tuple(f"2025-{m:02d}" for m in range(1, 13))
#: Months present in the ranking file (the scored set).
RANK_MONTHS: tuple[str, ...] = (
    ("2026-01", "2026-02", "2026-06", "2026-07") if FINAL else ("2026-01", "2026-07")
)
ALL_MONTHS: tuple[str, ...] = TRAIN_MONTHS + RANK_MONTHS

#: In the final file the organisers moved every takeoff (and arrival in-block) time by an integer
#: number of seconds in [−12, 12], independently per row and per column; nothing else changed
#: (``scripts/analysis/a126_final_randomness.py``). The admissible bounds widen by this much.
TIME_NOISE: float = 12.0 if FINAL else 0.0


def ensure_dirs() -> None:
    """Create the writable data directories. Safe to call repeatedly."""
    for d in (
        INTERIM,
        FEATURES,
        OUTPUT,
        PREDICTIONS,
        RANK_PREDS,
        OVERRIDES,
        REPORTS,
        SPINE,
        MLARTIFACTS,
    ):
        d.mkdir(parents=True, exist_ok=True)


#: Where a stack composition lives: ``v62_members.txt`` lists the members of the best submission.
COMPOSITIONS: Path = CONFIG / "compositions"

#: One line naming the composition the build uses by default (``v62``).
BASE_COMPOSITION: Path = COMPOSITIONS / "BASE"


def composition(version: str | int | None = None) -> list[str]:
    """The member list of a composition; the BASE one if ``version`` is omitted."""
    if not COMPOSITIONS.is_dir():
        raise FileNotFoundError(f"no composition directory at {COMPOSITIONS}")
    files = sorted(COMPOSITIONS.glob("v*_members.txt"))
    if not files:
        raise FileNotFoundError(f"no compositions in {COMPOSITIONS}")
    if version is None and BASE_COMPOSITION.exists():
        version = BASE_COMPOSITION.read_text().strip()
    if version is None:
        chosen = max(files, key=lambda f: int(f.name.split("_")[0].lstrip("v")))
    else:
        want = f"v{str(version).lstrip('v')}_members.txt"
        matches = [f for f in files if f.name == want]
        if not matches:
            raise FileNotFoundError(f"no composition {want}; have {[f.name for f in files]}")
        chosen = matches[0]
    return [line.strip() for line in chosen.read_text().splitlines() if line.strip()]
