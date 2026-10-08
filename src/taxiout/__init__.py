"""Taxi-out time prediction for the PRC Data Challenge 2026."""

from __future__ import annotations

import os

# The raw parquet files were written from R and carry an `arrow.r.vctrs` extension type that
# Polars does not know. Loading it as its storage type is correct here and silences the warning
# on every scan. Must be set before the first scan.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "0.1.0"
