"""House chart style.

One validated palette, applied through :func:`style`, so every figure in ``reports/`` reads as
one system. Colours come from the project's reference palette and were checked with the
all-pairs colourblind validator at three categorical slots — past three, facet instead of
adding a fourth hue.

Figures are drawn on an explicit light surface rather than a transparent one, so a saved PNG
stays legible whatever background it is later viewed against.
"""

from __future__ import annotations

import sys

import matplotlib

#: Use a non-interactive backend when this is not running inside a notebook kernel, so the
#: scripts work headless. Must happen before pyplot is imported.
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# --- palette -----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

#: Categorical slots, in fixed order. Never cycle past these; fold the tail into "other"
#: or split into small multiples.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE, ORANGE, AQUA = SERIES

#: For "highlight these, recede the rest".
MUTED_FILL = "#d8d7d2"

#: Single-hue sequential ramp, light to dark.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def style() -> None:
    """Apply the house rcParams. Call once per script, before drawing."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # never dashed
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )


def titled(ax, title: str, subtitle: str = "", *, xlabel: str = "", ylabel: str = "") -> None:
    """Title plus an optional explanatory line underneath, in recessive ink.

    Both are placed as axes text rather than through ``set_title``, so the subtitle can never
    collide with the title however tall the figure is.
    """
    ax.set_title("")
    gap = 0.045 if subtitle else 0.0
    ax.text(
        0.0,
        1.04 + gap,
        title,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color=INK_2,
            va="bottom",
            ha="left",
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def annotate(ax, x, y, text: str, *, dx: float = 6, dy: float = 6, color: str = INK_2) -> None:
    """A selective direct label. Use on the point that carries the story, not on every point."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=color,
        ha="left",
        va="bottom",
    )


def only_y_grid(ax) -> None:
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def only_x_grid(ax) -> None:
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def bar_gap_colors(labels: list[str], highlight: set[str]) -> list[str]:
    """Emphasis colouring: the rows that matter in slot 1, the rest recessive."""
    return [BLUE if label in highlight else MUTED_FILL for label in labels]


# --- heatmaps ----------------------------------------------------------------------------
def _cmap(name: str, colors: tuple[str, ...]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


#: Single hue, light to dark — for magnitude.
SEQUENTIAL_CMAP = _cmap("taxiout_seq", SEQUENTIAL)
#: Two opposed hues with a neutral grey midpoint — for deviation around a baseline.
DIVERGING_CMAP = _cmap("taxiout_div", ("#184f95", "#3987e5", "#f0efec", "#e34948", "#9b2423"))


def heatmap(
    ax,
    values,
    row_labels: list[str],
    col_labels: list[str],
    *,
    diverging: bool = False,
    fmt: str = "{:.0f}",
    annotate_cells: bool = True,
    cbar_label: str = "",
    vmax=None,
):
    """A labelled grid with a colour scale.

    ``diverging=True`` centres the scale on zero with a neutral midpoint, for deviations.
    Cell values are printed by default: a heatmap's colour alone is not readable to a precision,
    and the numbers double as the table view.
    """
    import numpy as np

    grid = np.asarray(values, dtype=float)
    if diverging:
        limit = vmax if vmax is not None else np.nanmax(np.abs(grid))
        image = ax.imshow(grid, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, aspect="auto")
    else:
        image = ax.imshow(grid, cmap=SEQUENTIAL_CMAP, aspect="auto", vmax=vmax)

    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if annotate_cells:
        lo, hi = image.get_clim()
        span = hi - lo or 1.0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell = grid[i, j]
                if not np.isfinite(cell):
                    continue
                intensity = abs(cell - (0 if diverging else lo)) / (span if diverging else span)
                ink = SURFACE if intensity > 0.62 else INK
                ax.text(j, i, fmt.format(cell), ha="center", va="center", fontsize=7, color=ink)

    bar = ax.figure.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7.5, labelcolor=INK_2)
    if cbar_label:
        bar.set_label(cbar_label, fontsize=8, color=INK_2)
    return image


# --- boxplots ----------------------------------------------------------------------------
def box_stats(
    frame,
    keys: list[str],
    value: str,
    *,
    whiskers="tukey",
    min_count: int = 1,
):
    """Five-number summaries per group, ready for :func:`boxplot`.

    Summarising in Polars rather than handing matplotlib millions of raw values keeps this
    usable on the full 2M-row spine.

    ``whiskers="tukey"`` puts the whisker ends at the most extreme observation within
    1.5 x IQR of the quartiles, which costs a second pass. A ``(lo, hi)`` quantile pair —
    ``(0.05, 0.95)`` say — is one pass and often more readable on a long-tailed target.
    """
    import polars as pl

    base = (
        frame.group_by(keys)
        .agg(
            pl.len().alias("n"),
            pl.col(value).quantile(0.25).alias("q1"),
            pl.col(value).median().alias("med"),
            pl.col(value).quantile(0.75).alias("q3"),
            pl.col(value).mean().alias("mean"),
        )
        .filter(pl.col("n") >= min_count)
    )
    if whiskers == "tukey":
        fenced = base.with_columns(
            (pl.col("q1") - 1.5 * (pl.col("q3") - pl.col("q1"))).alias("_lo"),
            (pl.col("q3") + 1.5 * (pl.col("q3") - pl.col("q1"))).alias("_hi"),
        )
        ends = (
            frame.join(fenced.select(*keys, "_lo", "_hi"), on=keys)
            .filter(pl.col(value).is_between(pl.col("_lo"), pl.col("_hi")))
            .group_by(keys)
            .agg(
                pl.col(value).min().alias("whislo"),
                pl.col(value).max().alias("whishi"),
            )
        )
        return fenced.drop("_lo", "_hi").join(ends, on=keys)

    lo, hi = whiskers
    spread = frame.group_by(keys).agg(
        pl.col(value).quantile(lo).alias("whislo"),
        pl.col(value).quantile(hi).alias("whishi"),
    )
    return base.join(spread, on=keys)


def boxplot(
    ax, stats, label_column: str, *, horizontal: bool = True, color: str = BLUE, width=0.62
):
    """Draw precomputed summaries from :func:`box_stats`. Outliers are not drawn."""
    rows = stats.to_dicts()
    payload = [
        {
            "label": str(row[label_column]),
            "med": row["med"],
            "q1": row["q1"],
            "q3": row["q3"],
            "whislo": row["whislo"],
            "whishi": row["whishi"],
            "fliers": [],
        }
        for row in rows
    ]
    drawn = ax.bxp(
        payload,
        vert=not horizontal,
        showfliers=False,
        patch_artist=True,
        widths=width,
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.8, "alpha": 0.55},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
        medianprops={"color": INK, "linewidth": 1.4},
    )
    if horizontal:
        only_x_grid(ax)
    else:
        only_y_grid(ax)
    return drawn
