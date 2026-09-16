"""Chart palette and matplotlib defaults."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def series_color(i: int) -> str:
    if i < len(CATEGORICAL):
        return CATEGORICAL[i]
    return MUTED


def ramp(n: int, lo: int = 3, hi: int = 12) -> list[str]:
    if n <= 1:
        return [SEQUENTIAL[hi]]
    step = (hi - lo) / (n - 1)
    return [SEQUENTIAL[int(round(lo + i * step))] for i in range(n)]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
