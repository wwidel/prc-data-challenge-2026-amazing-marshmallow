"""Environment that must be set before polars is imported anywhere."""

from __future__ import annotations

import os

# The raw parquet files carry an `arrow.r.vctrs` extension type from R. Polars reads this
# setting when it initialises, so it has to be in place before the first import.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
