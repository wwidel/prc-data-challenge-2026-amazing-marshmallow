"""The feature store: content-addressed, month-partitioned, aligned to the spine.

A block's cache key is the hash of its defining module, so editing a feature invalidates that
feature and nothing else. Output is written one parquet per month, in spine row order, which is
what lets :func:`assemble` build an experiment matrix by horizontal concatenation — no joins, no
sorts, no merge keys.

Only leakage-free kinds are cached. A ``TARGET_DEPENDENT`` block reads labels and must be
recomputed inside each fold; :mod:`taxiout.cv` does that, and this module refuses it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass

import polars as pl

from taxiout import paths, spine
from taxiout.features import registry
from taxiout.features.registry import FeatureBlock, FeatureContext

HASH_LENGTH = 12


def block_hash(block: FeatureBlock) -> str:
    return hashlib.sha256(block.source_hash_payload().encode()).hexdigest()[:HASH_LENGTH]


def block_dir(block: FeatureBlock) -> paths.Path:
    return paths.FEATURES / block.name / block_hash(block)


def is_built(block: FeatureBlock, months: tuple[str, ...]) -> bool:
    directory = block_dir(block)
    return all((directory / f"{m}.parquet").exists() for m in months)


@dataclass(frozen=True)
class BuildResult:
    name: str
    hash: str
    rebuilt: bool
    rows: int
    columns: tuple[str, ...]


def build(block: FeatureBlock, *, overwrite: bool = False) -> BuildResult:
    """Compute a block over the whole spine and write it out month by month.

    Always computed over *every* month at once: context features need to see across month
    boundaries, and a partial build would silently truncate their windows.
    """
    if not block.kind.cacheable:
        raise ValueError(
            f"{block.name!r} is {block.kind} and reads labels — it is computed per fold in "
            "taxiout.cv, never cached here"
        )
    months = paths.ALL_MONTHS
    directory = block_dir(block)
    if is_built(block, months) and not overwrite:
        return BuildResult(block.name, block_hash(block), False, 0, block.outputs)

    full_spine = spine.load()
    computed = block.compute(FeatureContext(spine=full_spine))

    directory.mkdir(parents=True, exist_ok=True)
    offset = 0
    for month, height in sorted(spine.heights().items()):
        computed.slice(offset, height).write_parquet(
            directory / f"{month}.parquet", compression="zstd"
        )
        offset += height
    if offset != computed.height:
        raise RuntimeError(f"spine heights sum to {offset:,}, block produced {computed.height:,}")

    (directory / "meta.json").write_text(
        json.dumps(
            {
                "name": block.name,
                "kind": str(block.kind),
                "version": block.version,
                "outputs": list(block.outputs),
                "hash": block_hash(block),
                "rows": computed.height,
                "doc": block.doc,
            },
            indent=2,
        )
    )
    return BuildResult(block.name, block_hash(block), True, computed.height, block.outputs)


def build_many(names: list[str] | tuple[str, ...], *, overwrite: bool = False) -> list[BuildResult]:
    return [build(b, overwrite=overwrite) for b in registry.resolve(names)]


def assemble(
    names: list[str] | tuple[str, ...],
    months: tuple[str, ...],
    *,
    with_spine: bool = True,
) -> pl.LazyFrame:
    """Build the design matrix for ``names`` over ``months``.

    Per month, the spine and each block's file are concatenated horizontally — they are the
    same rows in the same order by construction — and the months are then stacked.
    """
    blocks = registry.resolve(names)
    uncached = [b.name for b in blocks if not b.kind.cacheable]
    if uncached:
        raise ValueError(f"target-dependent blocks cannot be assembled from cache: {uncached}")
    missing = [b.name for b in blocks if not is_built(b, months)]
    if missing:
        raise FileNotFoundError(
            f"feature blocks not built: {missing}; run "
            f"`uv run python scripts/build_features.py {' '.join(missing)}`"
        )

    per_month: list[pl.LazyFrame] = []
    for month in sorted(months):
        parts: list[pl.LazyFrame] = []
        if with_spine:
            parts.append(pl.scan_parquet(paths.SPINE / f"{month}.parquet"))
        parts.extend(pl.scan_parquet(block_dir(b) / f"{month}.parquet") for b in blocks)
        # Equal heights are the store's central invariant, enforced at build time and by
        # tests/test_pipeline.py. Polars currently warns that `horizontal` will start
        # requiring that in a breaking release — which is precisely the behaviour we want,
        # so the transitional warning is suppressed rather than the call weakened.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*how='horizontal'.*", category=DeprecationWarning
            )
            per_month.append(pl.concat(parts, how="horizontal"))
    return pl.concat(per_month, how="vertical")


def feature_columns(names: list[str] | tuple[str, ...]) -> list[str]:
    """The model-visible columns contributed by these blocks, in block order."""
    return [col for block in registry.resolve(names) for col in block.outputs]


def resolved_hashes(names: list[str] | tuple[str, ...]) -> dict[str, str]:
    """``{block name: hash}`` — logged with every experiment so a score is reproducible."""
    return {b.name: block_hash(b) for b in registry.resolve(names)}


def prune(block: FeatureBlock) -> list[str]:
    """Delete cached output of this block under hashes other than the current one."""
    root = paths.FEATURES / block.name
    if not root.exists():
        return []
    keep = block_hash(block)
    removed = []
    for directory in root.iterdir():
        if directory.is_dir() and directory.name != keep:
            shutil.rmtree(directory)
            removed.append(directory.name)
    return removed
