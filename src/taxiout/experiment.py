"""Defining, queueing, running and logging experiments.

An experiment is a YAML file naming three things: which feature blocks to use, which model to
fit, and which fold set to score on. Adding a feature to a baseline is one line in a copied
file, which is the point.

The config hash covers the feature blocks *and their resolved content hashes*, so editing a
feature's code produces a different experiment identity — a stale result can never masquerade
as a current one.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from taxiout import cv, models, paths, profiles, store


@dataclass(frozen=True)
class Experiment:
    name: str
    features: tuple[str, ...]
    model_kind: str
    model_params: dict[str, Any] = field(default_factory=dict)
    folds: str = "seasonal"
    description: str = ""
    clip: tuple[float, float] | None = None
    train_label_cap: float | None = None
    target_transform: str | None = None
    train_relabel: bool = False
    tags: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: Path | None = None) -> Experiment:
        model = raw.get("model") or {}
        if "kind" not in model:
            raise ValueError(f"{source or raw.get('name')}: model.kind is required")
        clip = raw.get("clip")
        cap = raw.get("train_label_cap")
        transform = raw.get("target_transform")
        relabel = bool(raw.get("train_relabel", False))
        return cls(
            name=raw["name"],
            features=tuple(raw.get("features") or ()),
            model_kind=model["kind"],
            model_params=dict(model.get("params") or {}),
            folds=raw.get("folds", "seasonal"),
            description=raw.get("description", ""),
            clip=tuple(clip) if clip else None,
            train_label_cap=float(cap) if cap is not None else None,
            target_transform=transform,
            train_relabel=relabel,
            tags={str(k): str(v) for k, v in (raw.get("tags") or {}).items()},
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()), source=path)

    def feature_hashes(self) -> dict[str, str]:
        return store.resolved_hashes(self.features)

    def config_hash(self) -> str:
        """Identity of this experiment, including the code behind its features."""
        payload = json.dumps(
            {
                "features": sorted(self.feature_hashes().items()),
                "model": [self.model_kind, self.model_params],
                "folds": self.folds,
                "clip": list(self.clip) if self.clip else None,
                "train_label_cap": self.train_label_cap,
                "target_transform": self.target_transform,
                "train_relabel": self.train_relabel,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def params(self) -> dict[str, Any]:
        """Flat parameter dict for the tracking backend."""
        return {
            "features": ",".join(self.features) or "<none>",
            "n_features": len(store.feature_columns(self.features)),
            "model_kind": self.model_kind,
            "model_params": json.dumps(self.model_params, sort_keys=True),
            "folds": self.folds,
            "clip": json.dumps(list(self.clip) if self.clip else None),
            "train_relabel": self.train_relabel,
            "target_transform": self.target_transform,
            "train_label_cap": self.train_label_cap,
            "config_hash": self.config_hash(),
        }

    def resolved_params(self, profile: profiles.Profile | None = None) -> dict[str, Any]:
        """Model params with a machine profile's runtime settings merged underneath.

        Deliberately **not** part of `config_hash`: a profile may only set things that do not
        change results, so the same experiment keeps one identity across machines.
        """
        if profile is None:
            return self.model_params
        return profile.apply_nested(self.model_kind, self.model_params)

    def run(self, profile: profiles.Profile | None = None) -> cv.CVResult:
        return cv.run(
            make_model=models.factory(self.model_kind, self.resolved_params(profile)),
            features=self.features,
            folds=cv.get_folds(self.folds),
            clip=self.clip,
            train_label_cap=self.train_label_cap,
            target_transform=self.target_transform,
            train_relabel=self.train_relabel,
        )


def expand_grid(raw: dict[str, Any], source: Path | None = None) -> list[Experiment]:
    """Expand a config's optional ``grid:`` block into one experiment per combination.

    ```yaml
    model:
      kind: lgbm
      params: {learning_rate: 0.05}
    grid:
      num_leaves: [63, 127]
      min_data_in_leaf: [100, 200]
    ```

    gives four experiments, each named with the settings that distinguish it, each with its own
    config hash so the runner skips the ones already scored.
    """
    grid = raw.get("grid") or {}
    if not grid:
        return [Experiment.from_dict(raw, source=source)]

    keys = sorted(grid)
    out: list[Experiment] = []
    for combination in itertools.product(*(grid[k] for k in keys)):
        settings = dict(zip(keys, combination, strict=True))
        variant = copy.deepcopy(raw)
        variant.pop("grid", None)
        # Keys naming top-level experiment fields sweep those fields; everything else is a
        # model parameter. (A grid over `train_label_cap` once landed silently in the LGBM
        # params and swept nothing.)
        top_level = {
            k: v
            for k, v in settings.items()
            if k in ("train_label_cap", "clip", "target_transform", "train_relabel")
        }
        params = {k: v for k, v in settings.items() if k not in top_level}
        variant.update(top_level)
        variant.setdefault("model", {}).setdefault("params", {}).update(params)
        suffix = ",".join(f"{k}={v}" for k, v in settings.items())
        variant["name"] = f"{raw['name']}[{suffix}]"
        out.append(Experiment.from_dict(variant, source=source))
    return out


def load_all(where: str | Path = paths.EXPERIMENTS) -> list[Experiment]:
    """Load every experiment under a directory, or a single file, sorted by filename.

    A config carrying a ``grid:`` block expands into one experiment per combination.
    """
    where = Path(where)
    files = sorted(where.glob("*.yaml")) if where.is_dir() else [where]
    if not files:
        raise FileNotFoundError(f"no experiment configs found at {where}")
    return [
        experiment
        for f in files
        for experiment in expand_grid(yaml.safe_load(f.read_text()), source=f)
    ]
