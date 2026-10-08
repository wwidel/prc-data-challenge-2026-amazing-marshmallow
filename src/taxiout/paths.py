"""Every filesystem path in the project. Nothing else hardcodes a directory."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
#: The built prediction file (competition format). Gitignored, like everything under data/.
OUTPUT = DATA / "output"
#: Out-of-fold predictions, one parquet per experiment. Gitignored: ~4 MB each.
PREDICTIONS = DATA / "predictions"

REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
EXPERIMENTS = CONFIG / "experiments"
MLFLOW_DB = ROOT / "mlflow.db"
MLARTIFACTS = ROOT / "mlartifacts"

SPINE = INTERIM / "spine"

TRAINING_GLOB = str(RAW / "training_*.parquet")
RANKING = RAW / "ranking.parquet"
SUBMITTING = RAW / "submitting.parquet"

#: Months present in the training files.
TRAIN_MONTHS: tuple[str, ...] = tuple(f"2025-{m:02d}" for m in range(1, 13))
#: Months present in the ranking file (the scored set).
RANK_MONTHS: tuple[str, ...] = ("2026-01", "2026-07")
ALL_MONTHS: tuple[str, ...] = TRAIN_MONTHS + RANK_MONTHS


def ensure_dirs() -> None:
    """Create the writable data directories. Safe to call repeatedly."""
    for d in (INTERIM, FEATURES, OUTPUT, PREDICTIONS, REPORTS, SPINE, MLARTIFACTS):
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
