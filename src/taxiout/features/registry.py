"""Declaring features.

A *feature block* is a function that turns the spine into one or more columns, aligned to the
spine row for row. Blocks declare a :class:`FeatureKind`, and that declaration is what keeps
cross-validation honest:

``ROW_LOCAL``
    Depends only on the row's own fields — calendar, stand, runway, aircraft type, the
    ``aobt_3`` proxy. No leakage possible. Computed once, cached, shared by every fold.

``CONTEXT``
    Depends on *other rows* but never on the target — rolling movement counts, taxi-in
    statistics, stand occupancy. Computed over training and ranking together, because arrivals
    in the ranking window are fully observed. Still leakage-free, so still cached once.

``TARGET_DEPENDENT``
    Reads ``taxi_sec``. Group means, target encodings, residual statistics. **Recomputed inside
    every fold from training rows only** and never cached, because a globally-cached version
    would leak the validation months into the training matrix.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class FeatureKind(StrEnum):
    ROW_LOCAL = "row_local"
    CONTEXT = "context"
    TARGET_DEPENDENT = "target_dependent"

    @property
    def cacheable(self) -> bool:
        """Whether a block of this kind may be computed once and reused across folds."""
        return self is not FeatureKind.TARGET_DEPENDENT


@dataclass(frozen=True)
class FeatureContext:
    """What a feature function is given.

    ``spine`` is always the *full* spine, every month, in canonical order — context features
    need to see across month boundaries. ``train_mask`` is supplied only for target-dependent
    blocks: it marks the rows the current fold is allowed to learn from. Reading ``taxi_sec``
    for any row where ``train_mask`` is false is a leak.
    """

    spine: pl.DataFrame
    train_mask: pl.Series | None = None

    def training_rows(self) -> pl.DataFrame:
        """The rows a target-dependent block may read labels from."""
        if self.train_mask is None:
            raise ValueError(
                "training_rows() requires a train_mask; is this block TARGET_DEPENDENT?"
            )
        return self.spine.filter(self.train_mask)


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    kind: FeatureKind
    outputs: tuple[str, ...]
    version: int
    fn: Callable[[FeatureContext], pl.DataFrame]
    doc: str

    def compute(self, ctx: FeatureContext) -> pl.DataFrame:
        """Run the block and enforce its contract."""
        out = self.fn(ctx)
        if out.height != ctx.spine.height:
            raise ValueError(
                f"feature block {self.name!r} returned {out.height:,} rows, "
                f"spine has {ctx.spine.height:,} — blocks must stay aligned to the spine"
            )
        if tuple(out.columns) != self.outputs:
            raise ValueError(
                f"feature block {self.name!r} declared outputs {self.outputs} "
                f"but returned {tuple(out.columns)}"
            )
        return _downcast(out)

    def source_hash_payload(self) -> str:
        """Text whose hash identifies this block's cached output.

        The *whole defining module* is hashed, not just the function, so that a change to a
        helper defined in the same module also invalidates it. The trade-off is that editing any
        block in a module rebuilds every block in that module — keep one family per module.

        **Not covered:** helpers imported from *another* module (stand_runway_lo imports
        ``stand_runway._code``), and external data
        files a block reads (METAR, Gateway, OSM). A change to either leaves the cached block --
        and the config hash -- unchanged. modes5/modes6 duplicate their helpers for this reason.
        Note also that comments and docstrings are part of the hash, so a text-only edit to a
        feature module rebuilds it.
        """
        module = inspect.getmodule(self.fn)
        source = inspect.getsource(module) if module else inspect.getsource(self.fn)
        return f"{self.name}|v{self.version}|{self.kind}|{','.join(self.outputs)}\n{source}"


def _downcast(frame: pl.DataFrame) -> pl.DataFrame:
    """f64 -> f32 and i64 -> i32 where safe. 2.4M rows makes this worth doing."""
    return frame.with_columns(
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
        pl.col(pl.UInt64).cast(pl.UInt32),
    )


REGISTRY: dict[str, FeatureBlock] = {}


def feature(
    *,
    name: str,
    kind: FeatureKind,
    outputs: tuple[str, ...],
    version: int = 1,
) -> Callable[[Callable[[FeatureContext], pl.DataFrame]], FeatureBlock]:
    """Register a feature block.

    ``version`` is a manual escape hatch: bump it to force a rebuild when a block's behaviour
    changes for a reason the module source does not capture, such as new upstream data.

    Re-registering the same name from the same module replaces the entry rather than raising,
    so editing a feature module under ``%autoreload`` works. Two *different* modules claiming
    one name is still an error — that is the collision the check exists for.
    """

    def decorate(fn: Callable[[FeatureContext], pl.DataFrame]) -> FeatureBlock:
        existing = REGISTRY.get(name)
        if existing is not None and existing.fn.__module__ != fn.__module__:
            raise ValueError(
                f"feature block {name!r} is already registered by "
                f"{existing.fn.__module__!r}; {fn.__module__!r} cannot claim it too"
            )
        block = FeatureBlock(
            name=name,
            kind=kind,
            outputs=outputs,
            version=version,
            fn=fn,
            doc=inspect.cleandoc(fn.__doc__ or ""),
        )
        REGISTRY[name] = block
        return block

    return decorate


def get(name: str) -> FeatureBlock:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "<none>"
        raise KeyError(f"unknown feature block {name!r}; registered: {known}")
    return REGISTRY[name]


def resolve(names: list[str] | tuple[str, ...]) -> list[FeatureBlock]:
    """Look up blocks and reject duplicate output column names across them."""
    blocks = [get(n) for n in names]
    seen: dict[str, str] = {}
    for block in blocks:
        for col in block.outputs:
            if col in seen:
                raise ValueError(
                    f"column {col!r} is emitted by both {seen[col]!r} and {block.name!r}"
                )
            seen[col] = block.name
    return blocks


def align(spine: pl.DataFrame, computed: pl.DataFrame, outputs: tuple[str, ...]) -> pl.DataFrame:
    """Join per-``mvt_id`` values back onto the spine, preserving spine row order.

    The standard ending for a block that aggregates over something other than single rows.
    """
    joined = spine.select("mvt_id").join(computed, on="mvt_id", how="left", maintain_order="left")
    return joined.select(outputs)
