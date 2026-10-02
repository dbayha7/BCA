"""Shared look for the visualization figures: the slide deck's palette, type and canvas.

Figures are drawn at 16.64 x 7.0 inches, the content area of a 1920 x 1080 slide (1664 x 700 px at 100 dpi), and saved
at 200 dpi for the deck. Text sizes are chosen so that nothing is smaller than 24 px on the slide: 17 pt is the floor.
IBM Plex Sans is used when its files are in runs/viz_fonts (or $BCA_VIZ_FONTS); DejaVu Sans otherwise.
"""

import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "wbcp_viz"

INK = "#14213D"  # headings, axes, the target line
PAPER = "#F7F6F2"  # figure background, same as the deck's content slides
CARD = "#FDFCF9"  # boxes and panels on PAPER
RULE = "#E3E1DA"  # hairlines and grid
SOFT = "#4A5568"  # body text, secondary labels
MUTED = "#8A9199"  # de-emphasized data
BLUE = "#2F6DB5"  # accent 1: WBCP / Q1 / the method working as intended
BLUE_LIGHT = "#A9C4E6"
ORANGE = "#D9822B"  # accent 2: uniform BQ-CP / min(Q1, Q2) / the failure being explained
ORANGE_LIGHT = "#F0C69A"

FIGSIZE = (16.64, 7.0)
BASE, TICK, SMALL = 19, 17, 17  # points; at the deck's scale 17 pt is about 24 px

_fonts_done = False


def setup():
    """Register IBM Plex Sans if available and set matplotlib's defaults to the deck's look."""
    global _fonts_done
    family = "DejaVu Sans"
    if not _fonts_done:
        folder = os.environ.get("BCA_VIZ_FONTS", str(ROOT / "runs" / "viz_fonts"))
        for path in glob.glob(os.path.join(folder, "*.ttf")):
            font_manager.fontManager.addfont(path)
        _fonts_done = True
    if any(f.name == "IBM Plex Sans" for f in font_manager.fontManager.ttflist):
        family = "IBM Plex Sans"
    matplotlib.rcParams.update({
        "font.family": family,
        "font.size": BASE,
        "axes.titlesize": BASE,
        "axes.labelsize": BASE,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": SMALL,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": RULE,
        "legend.frameon": False,
        "lines.linewidth": 3.0,
        "figure.dpi": 100,
    })
    return family


def figure(nrows=1, ncols=1, **kw):
    setup()
    kw.setdefault("figsize", FIGSIZE)
    kw.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kw)


DECK_FIGSIZE = (16.64, 6.4)  # a slide's visual area under a one-line headline: 1664 x 640 px at 100 dpi
DECK_BASE, DECK_TICK = 22, 20  # read from across a room: about 30 px on the slide


def deck_figure(nrows=1, ncols=1, **kw):
    """A single-message chart for the slides: one idea, big type, direct labels, no title (the slide has it)."""
    setup()
    matplotlib.rcParams.update({"font.size": DECK_BASE, "axes.labelsize": DECK_BASE, "xtick.labelsize": DECK_TICK,
                                "ytick.labelsize": DECK_TICK, "legend.fontsize": DECK_TICK, "lines.linewidth": 4.0})
    kw.setdefault("figsize", DECK_FIGSIZE)
    kw.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kw)


def save_deck(fig, key):
    """Write runs/wbcp_viz/deck/<key>.png at 200 dpi (3328 x 1280 for a full-size deck chart) and return its path."""
    folder = OUT / "deck"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.png"
    fig.savefig(path, dpi=200)
    return path


def panel_label(ax, text, x=0.0, y=1.02):
    """A short panel caption above an axis, e.g. 'a  Bootstrap CDFs'. The slide title carries the figure's title."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=BASE, fontweight="semibold",
            color=INK)


def save(fig, key):
    """Write runs/wbcp_viz/<key>.png at 200 dpi (3328 x 1400 for a full-size figure) and return its path."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{key}.png"
    fig.savefig(path, dpi=200)
    return path
