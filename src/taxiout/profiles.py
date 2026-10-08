"""Machine profiles: the settings that differ between machines but must not change results.

A profile carries **runtime** parameters only — thread counts, verbosity — never anything that
alters a model's output. That line matters: if a profile could change `max_bin` or `seed`, the
same experiment would score differently on two machines while sharing a config hash, and the
results log would silently mix them.

Anything that genuinely should differ per machine *and* changes results — a larger `max_bin`
where there is more memory, say — belongs in its own experiment config, with its own hash.

Selected by ``TAXIOUT_PROFILE``, or ``--profile`` on the runner. Defaults to ``laptop``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from taxiout import paths

#: Parameters a profile is allowed to set, by model kind. Everything else is refused.
RUNTIME_PARAMS: dict[str, set[str]] = {
    "lgbm": {"num_threads", "verbosity", "force_row_wise", "force_col_wise"},
}

DEFAULT_PROFILE = "laptop"


@dataclass(frozen=True)
class Profile:
    name: str
    description: str = ""
    #: model kind -> runtime params
    runtime: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: advisory only; the runner prints them so a log line records the machine
    cores: int | None = None
    memory_gb: float | None = None

    def apply(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Merge this profile's runtime settings *under* the experiment's own params.

        The experiment always wins, so a config that deliberately pins `num_threads` keeps it.
        """
        merged = dict(self.runtime.get(kind, {}))
        merged.update(params)
        return merged

    def apply_nested(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Same, but descending into nested ``base`` specs (and ``members`` lists)."""
        out = self.apply(kind, params)
        if "base" in out and isinstance(out["base"], dict) and "kind" in out["base"]:
            base = out["base"]
            out = {
                **out,
                "base": {
                    **base,
                    "params": self.apply_nested(base["kind"], base.get("params") or {}),
                },
            }
        if "members" in out and isinstance(out["members"], list):
            out = {
                **out,
                "members": [
                    {**m, "params": self.apply_nested(m["kind"], m.get("params") or {})}
                    for m in out["members"]
                ],
            }
        return out


def load(name: str | None = None) -> Profile:
    name = name or os.environ.get("TAXIOUT_PROFILE") or DEFAULT_PROFILE
    path = paths.CONFIG / "profiles" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (paths.CONFIG / "profiles").glob("*.yaml"))
        raise FileNotFoundError(f"no profile {name!r}; available: {available}")
    raw = yaml.safe_load(path.read_text()) or {}

    runtime = raw.get("runtime") or {}
    for kind, params in runtime.items():
        allowed = RUNTIME_PARAMS.get(kind, set())
        refused = set(params) - allowed
        if refused:
            raise ValueError(
                f"profile {name!r} sets {sorted(refused)} for {kind!r}. A profile may only set "
                f"runtime parameters {sorted(allowed)} — anything that changes results belongs "
                "in the experiment config, so that it changes the config hash too."
            )
    return Profile(
        name=raw.get("name", name),
        description=raw.get("description", ""),
        runtime=runtime,
        cores=raw.get("cores"),
        memory_gb=raw.get("memory_gb"),
    )
