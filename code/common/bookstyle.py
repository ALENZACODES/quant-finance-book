"""Shared figure style for the book.

Classic collegiate palette: deep blue #011F5B and deep red #990000 on white, serif type
(TeX Gyre Pagella, a Palatino-like face that matches the LaTeX body text).
Figures are saved as 300 dpi PNG files, used both by the book and on the web, with metadata removed.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from cycler import cycler

BLUE = "#011F5B"
RED = "#990000"
GREY = "#5F6368"
SKY = "#5B7DB1"     # lighter tint of the blue, for secondary series
ROSE = "#C46A6A"    # lighter tint of the red
PALE = "#DCE3EE"    # fill colour
ROOT = Path(__file__).resolve().parents[2]

_FONT_DIR = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre")
_FAMILY = "serif"
for _f in ("regular", "italic", "bold", "bolditalic"):
    _p = _FONT_DIR / f"texgyrepagella-{_f}.otf"
    if _p.exists():
        fm.fontManager.addfont(str(_p))
        _FAMILY = "TeX Gyre Pagella"

plt.rcParams.update({
    "font.family": _FAMILY,
    "font.size": 9.5,
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=[BLUE, RED, GREY, SKY, ROSE]),
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "lines.linewidth": 1.4,
    "figure.figsize": (5.8, 3.4),
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 3,
    "axes.unicode_minus": False,
})


def save(fig, name, chapter="ch01"):
    """Save a figure as a 300 dpi PNG into figures/<chapter>/ with no metadata. The book includes the
    PNG directly, so no PDF figure is needed to build from the repository."""
    out = ROOT / "figures" / chapter
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    return out / f"{name}.png"
