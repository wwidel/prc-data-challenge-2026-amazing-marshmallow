"""Write docs/MODELS.md: every model of v62 and of the final submission, in full.

For each config the stack reads (the members of both compositions, the Rome layer r14, the
Rome-unmatched model lg01, and ab07, which the Rome matched-row blend reads), it prints the
model spec with every parameter resolved -- the config's own values and the wrapper defaults
underneath them (`taxiout.models.lgbm.DEFAULTS`, the CatBoost wrapper's fixed settings) -- the
training options, the fold set, and the feature blocks with every column they emit. A catalogue
of those blocks follows.

Everything is read from the code and the configs, nothing is typed by hand, so the document
cannot drift from what runs. Needs no data:

    uv run python scripts/describe_models.py            # writes docs/MODELS.md
    uv run python scripts/describe_models.py --check    # fails if docs/MODELS.md is stale
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from taxiout import cv, paths
from taxiout.experiment import Experiment, load_all
from taxiout.features import REGISTRY
from taxiout.models.catboost_model import CatBoostPredictor
from taxiout.models.lgbm import LGBMPredictor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MODELS.md"
COMPOSITIONS = ROOT / "config" / "compositions"

#: Read from the Makefile's LAYERS and the final phase's EXTRA; checked against it below.
LAYERS = ("r14_rules_plus_unmatched_route", "lg01_lirf_gate_delay")
BLEND_INPUT = "ab07_ab02_adsb_calib"
#: Final-submission members that NNLS gives weight 0 in every vector (docs/SUBMISSIONS.md, v66's
#: row): fitted by the build, but they change no prediction.
ZERO_WEIGHT = ("t05s_t05_snow",)
#: The machine profile the final phase's refits run under (Makefile: REFIT_FLAGS).
FINAL_REFIT_PROFILE = ROOT / "config" / "profiles" / "laptop.yaml"


def members(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def check_makefile() -> None:
    make = (ROOT / "Makefile").read_text()
    for name in (*LAYERS, BLEND_INPUT):
        if name not in make:
            sys.exit(f"{name} is not in the Makefile any more; update describe_models.py")


def fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 10000:
        return f"{v:,}"
    return f"`{v}`" if isinstance(v, str) else str(v)


def param_rows(resolved: dict[str, Any], given: dict[str, Any]) -> list[str]:
    rows = []
    for k in sorted(resolved):
        src = "config" if k in given else "default"
        rows.append(f"| `{k}` | {fmt(resolved[k])} | {src} |")
    return rows


def leaf(kind: str, params: dict[str, Any]) -> list[str]:
    """A LightGBM or CatBoost spec with every parameter the estimator will receive."""
    out = ["| parameter | value | from |", "|---|---:|---|"]
    if kind == "lgbm":
        m = LGBMPredictor(**params)
        wrapper = {
            "num_boost_round": m.num_boost_round,
            "early_stopping_rounds": m.early_stopping_rounds,
            "valid_fraction": m.valid_fraction,
            "stop_slice": m.stop_slice,
        }
        out += param_rows({**wrapper, **m.params}, params)
        if m.early_stopping_rounds:
            if m.stop_slice == "last":
                where = (
                    f"the last {m.valid_fraction:.0%} of the training rows by month, which "
                    "are not used to grow trees"
                )
            else:
                where = f"a seeded random {m.valid_fraction:.0%} of the training days"
            out.append("")
            out.append(
                f"Early stopping: the number of trees is chosen on {where}; "
                "this holds in the twelve-month refit too."
            )
    elif kind == "catboost":
        m = CatBoostPredictor(**params)
        out += param_rows(m.params, params)
        out.append("")
        out.append(
            "No early stopping: CatBoost fits all `iterations`. Columns named `cat_*` are "
            "passed as categorical features (ordered target statistics)."
        )
    else:
        raise ValueError(kind)
    return out


def spec(kind: str, params: dict[str, Any], depth: int = 0) -> list[str]:
    """A model spec, nested specs (`base`) rendered below their wrapper."""
    if kind in ("lgbm", "catboost"):
        label = {"lgbm": "LightGBM", "catboost": "CatBoost"}[kind]
        return [f"**{label}**", "", *leaf(kind, params)]
    own = {k: v for k, v in params.items() if k != "base"}
    lines = [f"**`{kind}`**, with these settings:", "", "```yaml"]
    lines += yaml.safe_dump(own, sort_keys=False, width=100).rstrip().splitlines()
    lines += ["```", ""]
    base = params.get("base") or {"kind": "constant", "params": {}}
    lines.append(f"and this inner model{' (one per route)' if kind == 'routed' else ''}:")
    lines.append("")
    lines += spec(base["kind"], base.get("params") or {}, depth + 1)
    return lines


def anchor(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation other than - and _ dropped."""
    return "".join(ch for ch in heading.lower() if ch.isalnum() or ch in "_-")


def as_markdown(text: str) -> str:
    """A docstring as Markdown: reST's ``code`` becomes `code`."""
    return re.sub(r"``([^`]+)``", r"`\1`", inspect.cleandoc(text or "")).strip()


def _plain(v: Any) -> bool:
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (tuple, list)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    return False


def _yamlable(v: Any) -> Any:
    if isinstance(v, (tuple, list)):
        return [_yamlable(x) for x in v]
    if isinstance(v, dict):
        return {k: _yamlable(x) for k, x in v.items()}
    return v


def module_settings(module: Any) -> dict[str, Any]:
    """The module's upper-case constants of plain values: windows, thresholds, classifier params."""
    return {
        k: _yamlable(v)
        for k, v in vars(module).items()
        if k.isupper() and k != "OUTPUTS" and not k.startswith("_") and _plain(v)
    }


def fold_sets(names: set[str]) -> list[str]:
    lines = [
        "| fold set | folds (held-out months; each trains on the other months of 2025) |",
        "|---|---|",
    ]
    for name in sorted(names):
        folds = "; ".join(
            f"{f.name}: {', '.join(m[-2:] for m in f.valid_months)}" for f in cv.get_folds(name)
        )
        lines.append(f"| `{name}` | {folds} |")
    return lines


def experiment_section(e: Experiment, roles: list[str]) -> list[str]:
    cap = f"{fmt(e.train_label_cap)} s" if e.train_label_cap else "no cap"
    lines = [
        f"### `{e.name}`",
        "",
        f"*{'; '.join(roles)}.* Config: "
        f"[`{e.source.relative_to(ROOT)}`](../{e.source.relative_to(ROOT)})",
        "",
    ]
    if e.description:
        lines += ["> " + ln if ln else ">" for ln in e.description.rstrip().splitlines()]
        lines.append("")
    lines += [
        "| training option | value |",
        "|---|---|",
        f"| fold set | `{e.folds}` |",
        f"| training labels capped at | {cap} |",
        f"| prediction clip | {list(e.clip) if e.clip else 'none'} |",
        f"| target transform | {e.target_transform or 'none'} |",
        f"| relabelled training rows | {'yes' if e.train_relabel else 'no'} |",
        "",
    ]
    lines += spec(e.model_kind, e.model_params)
    n_cols = sum(len(REGISTRY[b].outputs) for b in e.features)
    lines += [
        "",
        f"**Features**: {len(e.features)} blocks, {n_cols} columns "
        "(see the [catalogue](#feature-blocks)).",
        "",
        "| block | kind | columns |",
        "|---|---|---:|",
    ]
    for b in e.features:
        blk = REGISTRY[b]
        lines.append(f"| [`{b}`](#{anchor(b)}) | {blk.kind.value} | {len(blk.outputs)} |")
    lines.append("")
    return lines


def render() -> str:
    check_makefile()
    known = {e.name: e for e in load_all(str(paths.EXPERIMENTS))}
    final = members(COMPOSITIONS / "final_members.txt")
    v62 = members(COMPOSITIONS / "v62_members.txt")
    roles: dict[str, list[str]] = {}
    for n in final:
        roles.setdefault(n, []).append(
            "final submission: fitted, but NNLS weight 0 in every vector; it changes no prediction"
            if n in ZERO_WEIGHT
            else "final submission: stack member"
        )
    for n in v62:
        roles.setdefault(n, []).append("v62: stack member")
    for n in LAYERS:
        roles.setdefault(n, []).append("both builds: Rome layer")
    roles.setdefault(BLEND_INPUT, []).append(
        "final submission: input of the Rome matched-row blend"
    )
    missing = [n for n in roles if n not in known]
    if missing:
        sys.exit(f"no config for {missing}")
    profile = yaml.safe_load(FINAL_REFIT_PROFILE.read_text())

    order = [*final, *LAYERS, BLEND_INPUT, *[n for n in v62 if n not in final]]
    order = list(dict.fromkeys(order))
    used_blocks = sorted({b for n in order for b in known[n].features})

    lines = [
        "# Models",
        "",
        "<!-- Generated by scripts/describe_models.py from config/ and src/; "
        "do not edit by hand. -->",
        "",
        "Every model the two builds read, with every parameter it is fitted with: the config's own",
        "values and, marked *default*, the wrapper's defaults underneath them. How the members are",
        "combined (NNLS stack, Rome layer, overrides, cell shift, bounds) is in the",
        "[README](../README.md#how-it-works).",
        "",
        "- **Final submission**: "
        + ", ".join(f"[`{n}`](#{anchor(n)})" for n in final if n not in ZERO_WEIGHT)
        + "; also fitted, with NNLS weight 0 everywhere: "
        + ", ".join(f"[`{n}`](#{anchor(n)})" for n in final if n in ZERO_WEIGHT)
        + "; Rome layer "
        + ", ".join(f"[`{n}`](#{anchor(n)})" for n in LAYERS)
        + f"; [`{BLEND_INPUT}`](#{anchor(BLEND_INPUT)}) for the Rome matched-row blend.",
        "- **v62**: " + ", ".join(f"[`{n}`](#{anchor(n)})" for n in v62) + "; the same Rome layer.",
        "",
        "Seeds are fixed (LightGBM `seed` 0, CatBoost `random_seed` 0). The final phase's refits",
        f"run under the `{profile['name']}` machine profile, which sets LightGBM's "
        + ", ".join(f"`{k}` = {v}" for k, v in profile["runtime"]["lgbm"].items())
        + " (the defaults); LightGBM's result can change in the last digits with the thread count.",
        "",
        "## Fold sets",
        "",
        *fold_sets({known[n].folds for n in order}),
        "",
        "Cross-validation predicts each held-out month from a model fitted on the other months;",
        "the refit fits all twelve months of 2025 and predicts the 2026 rows. Target-dependent",
        "blocks are recomputed inside every fold from its training rows only.",
        "",
        "## Configs",
        "",
    ]
    for n in order:
        lines += experiment_section(known[n], roles[n])

    lines += [
        "## Feature blocks",
        "",
        "Kinds: `row_local` reads only the row's own fields; `context` reads other rows but never",
        "a label, and is computed once over training and 2026 rows together; `target_dependent`",
        "reads labels and is recomputed inside every fold from its training rows. Columns named",
        "`cat_*` are categorical.",
        "",
    ]
    for b in used_blocks:
        blk = REGISTRY[b]
        module = inspect.getmodule(blk.fn)
        src = Path(inspect.getsourcefile(blk.fn)).relative_to(ROOT)
        users = [n for n in order if b in known[n].features]
        lines += [
            f"### {b}",
            "",
            f"*{blk.kind.value}*, [`{src}`](../{src}); used by {len(users)} of the "
            f"{len(order)} configs above.",
            "",
            as_markdown(module.__doc__ if module and module.__doc__ else blk.doc),
            "",
            "Columns: " + ", ".join(f"`{c}`" for c in blk.outputs) + ".",
            "",
        ]
        settings = module_settings(module)
        if settings:
            lines += ["Settings (the module's constants):", "", "```yaml"]
            lines += (
                yaml.safe_dump(settings, sort_keys=False, width=100, default_flow_style=None)
                .rstrip()
                .splitlines()
            )
            lines += ["```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if docs/MODELS.md is stale")
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit("docs/MODELS.md is stale; run scripts/describe_models.py")
        print("docs/MODELS.md is current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
