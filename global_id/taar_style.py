"""shared style, palette, gene map and Delta-bar helpers for the paper figures.
single source of truth for colours + row order; imported by fig1..fig3 and paper_plots."""
import ast
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms

# this package is expected to live INSIDE the analysis directory, e.g.
#     apo_holo_analysis/taar_paper_figures/taar_style.py
# so ROOT is simply its parent. Override with the TAAR_ROOT environment variable, or
# hardcode it here, if you move the package somewhere else.
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TAAR_ROOT", PACKAGE_DIR.parent))

GENE_DICT_PATH = os.path.join(ROOT, "reference_data", "hard_coded_gene_dict.txt")
FIG_DIR = os.path.join(ROOT, "output", "paper_figures")

# ---------------------------------------------------------------- palette
STATE_COLORS = {"apo": "#D9B98B", "holo": "#9B5560"}      # sand / dusty burgundy
# holo was slate #3D405B until it was judged too close to the dark end of viridis
# (dE 29 to viridis 0.40); burgundy sits dE 50 from its nearest viridis anchor
GENE_COLORS = {"hTAAR1": "#000000", "mTAAR1": "#000000",
               "mTAAR7f": "#000000", "mTAAR9": "#000000"}  # black: gene names are labels, not data
GENE_ORDER = ["hTAAR1", "mTAAR1", "mTAAR7f", "mTAAR9"]

# volume categories: exact strings as written in pocket_analysis_summary.csv
CATEGORIES = ["Small (<250)", "Medium (250-500)", "Large (500-750)", "Very Large (>750)"]
CAT_POS = [0.12, 0.40, 0.68, 0.92]                        # viridis sample points
CAT_COLORS = {c: mpl.colors.to_hex(plt.get_cmap("viridis")(p))
              for c, p in zip(CATEGORIES, CAT_POS)}
THRESH = [250, 500, 750]                                  # category cut points (A^3)

# ---------------------------------------------------------------- style
def apply_style():
    """ACS/JCIM figure defaults: Arial/Helvetica, >= 8 pt lettering, 0.5 pt lines"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.linewidth": 0.5,
        "axes.titlepad": 3,
    })

def load_gene_map():
    """reads the hard coded PDBID -> short gene name dict from file"""
    text = Path(GENE_DICT_PATH).read_text()
    return ast.literal_eval(text.split("=", 1)[1].strip())

def order_pdbids(pdbids, gene_map):
    """PDB IDs ordered by gene (GENE_ORDER) then PDB ID - same row order as figure 1"""
    return sorted(set(pdbids), key=lambda pdb: (GENE_ORDER.index(gene_map[pdb]), pdb))

def contiguous_blocks(labels):
    """yield (label, start_index, end_index) for runs of equal consecutive labels"""
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            yield labels[start], start, i - 1
            start = i

def gene_labels(ax, fig, genes, pad_pt=34):
    """gene name once per group, left of the y labels, black. matches figure 1 exactly."""
    trans = (transforms.blended_transform_factory(ax.transAxes, ax.transData)
             + transforms.ScaledTranslation(-pad_pt / 72, 0, fig.dpi_scale_trans))
    for gene, start, end in contiguous_blocks(genes):
        ax.text(0, (start + end) / 2, gene, transform=trans, ha="right", va="center",
                fontweight="bold", fontsize=8, color=GENE_COLORS.get(gene, "black"))

def gene_dividers(ax, genes, color="0.85", lw=0.6):
    """faint horizontal rules between gene blocks (light bg version of figure 1's white lines)"""
    for _, _, end in contiguous_blocks(genes):
        if end < len(genes) - 1:
            ax.axhline(end + 0.5, color=color, linewidth=lw)

def state_row_labels(ax, n_row, offset, pad_major=22, fontsize=5.5):
    """minor y ticks marking which sub-bar is apo and which is holo ('a' / 'h', state coloured).
    major ticks stay the PDB IDs."""
    minor, labels = [], []
    for i in range(n_row):
        minor += [i - offset, i + offset]
        labels += ["h", "a"]                     # -offset drawn above (y inverted): holo on top
    ax.set_yticks(minor, minor=True)
    ax.set_yticklabels(labels, minor=True, fontsize=fontsize)
    for tick, lab in zip(ax.yaxis.get_minorticklabels(), labels):
        tick.set_color(STATE_COLORS["holo"] if lab == "h" else STATE_COLORS["apo"])
    ax.tick_params(axis="y", which="minor", length=0, pad=1)
    ax.tick_params(axis="y", which="major", length=0, pad=pad_major)

def diff_barh(ax, values, genes, xmax, direction_labels=True):
    """diverging horizontal bars, apo (sand, negative) | holo (slate, positive). mirrors figure 1."""
    values = np.asarray(values, dtype=float)
    colors = [STATE_COLORS["holo"] if v >= 0 else STATE_COLORS["apo"] for v in values]
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.barh(np.arange(len(values)), values, color=colors, edgecolor="0.3", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.7, zorder=3)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(len(values) - 0.5, -0.5)
    ax.set_xticks([-xmax, 0, xmax])
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", which="both", labelleft=False)   # rows are labelled once, on panel A
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    gene_dividers(ax, genes)
    if direction_labels:
        ax.text(0.02, 1.0, "\u2190 apo", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6, color=STATE_COLORS["apo"])
        ax.text(0.98, 1.0, "holo \u2192", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color=STATE_COLORS["holo"])

def panel_letters(fig, entries, dy_pt=10, dx_pt=14, row_tol=0.03, align_columns=True):
    """place panel letters at a consistent offset from each panel's top-left corner.

    Letters are grouped into rows by the top edge of their axes; every letter in a row
    shares one y (so A/B line up, C/D line up). With align_columns=True, panels that
    start at nearly the same x also share one letter x, so A and C sit in a column even
    when panel A has wide row labels and panel C (an image) does not - the letter is
    pinned to the leftmost panel-start in that column, not to the figure margin.

    entries: list of (letter, axes). Call after the layout is settled.
    Returns the list of Text objects so a caller can remove them (e.g. for a no-letter
    export)."""
    fig.draw_without_rendering()
    dx = dx_pt / 72 / fig.get_size_inches()[0]
    dy = dy_pt / 72 / fig.get_size_inches()[1]

    positions = [(letter, ax, ax.get_position()) for letter, ax in entries]

    # group into rows by top edge
    rows = []
    for item in sorted(positions, key=lambda e: -e[2].y1):
        for row in rows:
            if abs(row[0][2].y1 - item[2].y1) < row_tol:
                row.append(item); break
        else:
            rows.append([item])

    # A row's LEADING letter (leftmost panel) sits at a shared left margin, so the
    # leading letters of every row line up in a column even though panel A is indented
    # past its row labels while panel C (an image) is not. Trailing letters sit at their
    # own panel's left edge.
    leaders = set()
    row_groups = []
    for item in sorted(positions, key=lambda e: -e[2].y1):
        for row in row_groups:
            if abs(row[0][2].y1 - item[2].y1) < row_tol:
                row.append(item); break
        else:
            row_groups.append([item])
    for row in row_groups:
        leaders.add(id(min(row, key=lambda e: e[2].x0)[1]))
    left_margin = max(min(e[2].x0 for e in positions) - dx, 0.002)

    texts = []
    for row in rows:
        top = max(e[2].y1 for e in row) + dy
        for letter, ax, box in row:
            x = left_margin if (align_columns and id(ax) in leaders) else box.x0 - dx
            x = max(x, 0.002)
            texts.append(fig.text(x, top, letter, ha="left", va="bottom",
                                  fontweight="bold", fontsize=11))
    return texts



def nice_max(values, step=10):
    m = np.nanmax(np.abs(np.asarray(values, dtype=float)))
    return float(np.ceil(m / step) * step) if m else float(step)