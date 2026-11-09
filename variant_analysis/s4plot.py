"""
s4plot.py: figure styling shared by the variant-analysis scripts.

Self-contained (matplotlib only) so the scripts need no external style package.

Conventions set here:
  - three font sizes mapped to role (title/axis, legend/annotation, ticks)
  - top and right spines off, thin spines, outward ticks
  - one colour per cell type, threaded from config.LINEAGE_PALETTE, so a cell type has
    the same colour in every figure
  - vector PDF beside every PNG, both at publication resolution
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                      # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402

BASE, SMALL, TICK = 8, 7, 6
GREY = "#6E6E6E"
LIGHT_GREY = "#BDBDBD"
FOCAL = "#1F4E79"          # focal series / primary rule
COMPARATOR = "#9E9E9E"     # comparator series
ALARM = "#C0392B"          # reserved for excluded / anomalous marks only

# Sequential ramp for ordinal quantities (breadth, stringency). Single hue, so it cannot
# be confused with the categorical lineage palette.
ORDINAL = ["#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"]


def ordinal_cmap():
    """ORDINAL as a continuous colormap, for scatter/heatmap `cmap=` arguments."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("s4_ordinal", ORDINAL)

FIG_W_SINGLE = 3.46   # 88 mm, Genome Biology single column
FIG_W_DOUBLE = 7.09   # 180 mm, double column


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": BASE,
        "axes.titlesize": BASE,
        "axes.labelsize": BASE,
        "legend.fontsize": SMALL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.titlepad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.labelpad": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.color": "#E8E8E8",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def panel_letter(ax, letter: str, dx: float = -0.18, dy: float = 1.06) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=BASE + 2,
            fontweight="bold", va="top", ha="left")


def end_of_line_labels(ax, series: dict, x: float, pad: float = 0.01,
                       fontsize: int = SMALL) -> None:
    """Direct-label series at the right end of their lines instead of a legend box."""
    xmin, xmax = ax.get_xlim()
    for label, (y, colour) in series.items():
        ax.text(x + pad * (xmax - xmin), y, label, color=colour, fontsize=fontsize,
                va="center", ha="left")


def cell_type_colours(cell_types, meta: dict, palette: dict) -> dict:
    """One colour per cell type, keyed by lineage so related types read as related."""
    return {ct: palette.get(meta[ct]["lineage"], GREY) for ct in cell_types}


def verify_no_overlap(fig, verbose: bool = True) -> list:
    """Geometric check that no two visible text boxes collide.

    Run before saving. Leader lines and maths text produce benign hits, so this reports
    rather than raises; the figure scripts print the count into the run log so a problem
    on the server is visible without anyone looking at the image.

    Artists with gid "decoration" are skipped. A deliberate overlay -- a watermark, a
    shading label -- is meant to sit on top of the data, so counting it as a collision
    buries the real hits it was never competing with.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()
             and t.get_gid() != "decoration"]
    hits = [(a.get_text(), b.get_text())
            for i, (a, ba) in enumerate(texts)
            for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    if verbose and hits:
        print(f"  [figure] {len(hits)} overlapping text pair(s): {hits[:5]}")
    return hits


def save(fig, path: Path, formats=("png", "pdf")) -> list[Path]:
    """Save a figure in every requested format and return the paths written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verify_no_overlap(fig)
    written = []
    for ext in formats:
        p = path.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written
