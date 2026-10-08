"""Support for analysis scripts.

Analysis scripts are written as ``# %%`` cell scripts: run them top to bottom with
``uv run python scripts/analysis/aNNN_*.py``, or step through them cell by cell in the editor.
Both paths behave identically.

A :class:`Recorder` keeps a transcript so a number quoted in a learnings file can be traced
back to the run that produced it. It writes after every call rather than at the end, so an
interactive session that stops halfway still leaves a usable report behind.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

from taxiout import paths


def interactive() -> bool:
    """True when running inside a notebook kernel rather than as a plain script."""
    return "ipykernel" in sys.modules


class Recorder:
    """Prints to the screen and accumulates a transcript at ``reports/<name>.txt``.

    Figures go to ``reports/figures/<name>__<slug>.png`` and are also displayed inline when
    running interactively.
    """

    def __init__(self, name: str, question: str) -> None:
        paths.ensure_dirs()
        self.name = name
        self.question = question
        self.path = paths.REPORTS / f"{name}.txt"
        self.figure_dir = paths.REPORTS / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = [
            f"{name}  —  {dt.date.today().isoformat()}",
            f"Question: {question}",
            "=" * 78,
        ]
        pl.Config.set_tbl_rows(40)
        pl.Config.set_tbl_width_chars(200)
        self._emit("")

    # -- recording -------------------------------------------------------------------------
    def _emit(self, text: str) -> None:
        print(text)
        self.lines.append(text)
        self.path.write_text("\n".join(self.lines) + "\n")

    def section(self, title: str) -> None:
        self._emit(f"\n== {title} ==")

    def say(self, text: str) -> None:
        self._emit(text)

    def show(self, frame: pl.DataFrame, note: str = "") -> pl.DataFrame:
        """Print a table into the transcript and hand it back, so cells can chain."""
        if note:
            self._emit(note)
        self._emit(str(frame))
        return frame

    def figure(self, fig, slug: str, caption: str = "") -> Path:
        """Save a figure beside the transcript and note where it went."""
        target = self.figure_dir / f"{self.name}__{slug}.png"
        fig.savefig(target)
        self._emit(f"[figure: {target.relative_to(paths.ROOT)}]{'  ' + caption if caption else ''}")
        if interactive():
            import matplotlib.pyplot as plt

            plt.show()
        return target

    def done(self) -> None:
        self._emit(f"\n[transcript: {self.path.relative_to(paths.ROOT)}]")


def find_analyses() -> list[Path]:
    return sorted((paths.ROOT / "scripts" / "analysis").glob("a*.py"))
