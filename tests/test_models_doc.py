"""docs/MODELS.md is generated from the configs and the code; it must match them. Pure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_models_doc_is_current():
    spec = importlib.util.spec_from_file_location(
        "describe_models", ROOT / "scripts" / "describe_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "MODELS.md").read_text() == mod.render(), (
        "docs/MODELS.md is stale: run `make models-doc`"
    )
