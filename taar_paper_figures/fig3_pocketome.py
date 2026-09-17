"""figure 3: ALLOSTERIC pocketome per structure (orthosteric pocket excluded).
A size composition | B stability | C Delta n pockets | D Delta median volume (%).
C answers "does the NUMBER of pockets change", D answers "does their SIZE change".
rows = the same gene-ordered PDB IDs as figure 1; apo/holo as paired sub-bars."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import taar_style as ts
import pocketome_metrics as pm
from pipeline import pocket_io

# Constants
POCKET_DATA_DIR = os.path.join(ts.ROOT, "output", "meta_analysis", "across_genes")
OUT_FILE = os.path.join(ts.FIG_DIR, "figure3_allosteric_pocketome.png")

COL_PDB, COL_STATE, COL_REP = "pdb_id", "state", "rep"
COL_CAT = "volume_category"
COL_ORTHO = "is_orthosteric"
#     transient  =  (n_zero_frames >= MIN_ZERO_FRACTION * N_FRAMES)
#                    AND (max_consecutive_zero_frames >= MIN_CONSEC_ZEROS)
# DERIVE_TRANSIENCY = False uses the stored `transient` column (MIN_CONSEC_ZEROS=15) instead.
DERIVE_TRANSIENCY = True
MIN_CONSEC_ZEROS = 15
MIN_ZERO_FRACTION = 0.10
N_FRAMES = 1000
COL_ZERO_FRAMES = "n_zero_frames"
COL_CONSEC_ZEROS = "max_consecutive_zero_frames"
STAB_COL = "transient"

ALLOSTERIC_ONLY = True

# panels: set any to False to drop it. widths are only used for enabled panels.
SHOW_PANEL_LETTERS = True    # set False to export without A/B/C/D
PANELS = {
    "size": True,          # A  size composition
    "stability": True,     # B  stable / transient
    "delta_count": True,   # C  Delta n pockets
    "delta_class": True,   # D  Delta fraction per size class
}
PANEL_WIDTHS = {"size": 1.20, "stability": 0.95, "delta_count": 0.60, "delta_class": 0.85}
PANEL_TITLES = {"size": "Pocket size", "stability": "Stability",
                "delta_count": u"\u0394 n pockets", "delta_class": u"\u0394 size class"}

BAR_H = 0.34
BAR_OFF = 0.19
TITLE_PAD = 10

# Data prep
def load_pockets():
    """pocket table, reduced to allosteric (non-orthosteric) pockets"""
    df = pocket_io.load_table(POCKET_DATA_DIR, "pocket_summary")
    if ALLOSTERIC_ONLY:
        df = df[~df[COL_ORTHO].astype(bool)]
    return df

def _representative_rep_counts(df, group_col, levels, rep_map):
    """breakdown by group_col, read off the representative replicate for each (pdb, state) -
    see pocketome_metrics.representative_replicate()."""
    out = {}
    for (pdb, state), rep in rep_map.items():
        sub = df[(df[COL_PDB] == pdb) & (df[COL_STATE] == state) & (df[COL_REP] == rep)]
        counts = sub[group_col].value_counts()
        out[(pdb, state)] = {lv: float(counts.get(lv, 0.0)) for lv in levels}
    return out

def size_counts(df, rep_map):
    return _representative_rep_counts(df, COL_CAT, ts.CATEGORIES, rep_map)

def transient_mask(df):
    """boolean transiency per pocket - see the note at the top of this file"""
    if not DERIVE_TRANSIENCY:
        return df[STAB_COL].astype(bool)
    return ((df[COL_ZERO_FRAMES] >= MIN_ZERO_FRACTION * N_FRAMES)
            & (df[COL_CONSEC_ZEROS] >= MIN_CONSEC_ZEROS))

def stability_counts(df, rep_map):
    df = df.copy()
    df["_stab"] = np.where(transient_mask(df), "transient", "stable")
    return _representative_rep_counts(df, "_stab", ["stable", "transient"], rep_map)

# Drawing
def _row_frame(ax, n_row, genes):
    ax.set_ylim(n_row - 0.5, -0.5)
    ts.gene_dividers(ax, genes)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.92", linewidth=0.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

def draw_size(ax, size, pdb_order, genes):
    """stacked composition; category = viridis fill, state = bar edge colour"""
    for i, pdb in enumerate(pdb_order):
        for state, off in (("apo", BAR_OFF), ("holo", -BAR_OFF)):
            left = 0.0
            for cat in ts.CATEGORIES:
                width = size.get((pdb, state), {}).get(cat, 0.0)
                ax.barh(i + off, width, left=left, height=BAR_H, color=ts.CAT_COLORS[cat],
                        edgecolor=ts.STATE_COLORS[state], linewidth=0.45)
                left += width
    _row_frame(ax, len(pdb_order), genes)
    ax.tick_params(axis="y", which="both", length=0, labelleft=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.set_xlabel("allosteric pockets (n)")
    ax.tick_params(axis="x", length=2)

def draw_stability(ax, stab, pdb_order, genes):
    """stable = solid state colour, transient = hatched with the state colour as edge"""
    for i, pdb in enumerate(pdb_order):
        for state, off in (("apo", BAR_OFF), ("holo", -BAR_OFF)):
            base = ts.STATE_COLORS[state]
            counts = stab.get((pdb, state), {})
            stable, transient = counts.get("stable", 0.0), counts.get("transient", 0.0)
            ax.barh(i + off, stable, height=BAR_H, color=base, linewidth=0)
            ax.barh(i + off, transient, left=stable, height=BAR_H, facecolor="white",
                    edgecolor=base, hatch="////", linewidth=0.45)
    _row_frame(ax, len(pdb_order), genes)
    ax.tick_params(axis="y", which="both", length=0, labelleft=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    ax.set_xlabel("allosteric pockets (n)")
    ax.tick_params(axis="x", length=2, labelsize=7)

def draw_category_delta(ax, deltas, pdb_order, genes):
    """signed stacked bars: which size classes gained and which lost on ligand binding.
    positive segments stack to the right, negative to the left; each row sums to zero."""
    for i, row in enumerate(deltas):
        left_pos = left_neg = 0.0
        for value, cat in zip(row, ts.CATEGORIES):
            if np.isnan(value) or value == 0:
                continue
            if value > 0:
                ax.barh(i, value, left=left_pos, height=0.6, color=ts.CAT_COLORS[cat],
                        edgecolor="0.35", linewidth=0.3); left_pos += value
            else:
                ax.barh(i, value, left=left_neg, height=0.6, color=ts.CAT_COLORS[cat],
                        edgecolor="0.35", linewidth=0.3); left_neg += value
    ax.axvline(0, color="black", linewidth=0.7, zorder=3)
    _row_frame(ax, len(pdb_order), genes)
    ax.tick_params(axis="y", which="both", length=0, labelleft=False)
    limit = np.nanmax(np.abs(deltas)) if np.isfinite(np.nanmax(np.abs(deltas))) else 0.5
    limit = np.ceil(limit * 10) / 10
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.tick_params(axis="x", length=2)
    ax.set_xlabel(u"\u0394 fraction (holo \u2212 apo)")
    # the sign is easy to read backwards, so the direction is spelled out under the title
    ax.text(0.0, 1.0, u"\u2190 fewer", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6, color="0.35")
    ax.text(1.0, 1.0, u"more \u2192", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6, color="0.35")


# Assembly
def figure3(out_file=None):
    ts.apply_style()
    df = load_pockets()
    gene_map = ts.load_gene_map()
    pdb_order = ts.order_pdbids(df[COL_PDB].unique(), gene_map)
    genes = [gene_map[p] for p in pdb_order]
    n_row = len(pdb_order)

    rep_map = pm.representative_replicate(df, COL_PDB, COL_STATE, COL_REP)
    size, stab = size_counts(df, rep_map), stability_counts(df, rep_map)
    dcount = pm.delta_pocket_count(df, pdb_order, COL_PDB, COL_STATE, COL_REP, rep_map)
    d_deltas = pm.delta_category_fraction(df, pdb_order, COL_PDB, COL_STATE, COL_CAT, COL_REP, rep_map)

    enabled = [key for key in ("size", "stability", "delta_count", "delta_class")
               if PANELS.get(key)]
    if not enabled:
        raise ValueError("PANELS: enable at least one panel")

    fig = plt.figure(figsize=(7.0, 0.9 + n_row * 0.26), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.16)
    gs = fig.add_gridspec(1, len(enabled),
                          width_ratios=[PANEL_WIDTHS[key] for key in enabled])

    axes = {}
    first = None
    for column, key in enumerate(enabled):
        ax = fig.add_subplot(gs[0, column], sharey=first) if first is not None \
            else fig.add_subplot(gs[0, column])
        first = first or ax
        axes[key] = ax

    # the leftmost enabled panel always carries the PDB rows and the gene labels
    leftmost = enabled[0]
    if "size" in axes:
        draw_size(axes["size"], size, pdb_order, genes)
    if "stability" in axes:
        draw_stability(axes["stability"], stab, pdb_order, genes)
    if "delta_count" in axes:
        ts.diff_barh(axes["delta_count"], np.nan_to_num(dcount), genes,
                     ts.nice_max(dcount, 2))
        axes["delta_count"].tick_params(axis="y", which="minor", length=0)
    if "delta_class" in axes:
        draw_category_delta(axes["delta_class"], d_deltas, pdb_order, genes)

    # PDB row labels live on the leftmost panel. Force them ON and set the tick-mark
    # LENGTH to 0 (rather than left=False), so the dash is gone but the label can never
    # be switched off as a side effect of a y-tick tweak elsewhere.
    ax_left = axes[leftmost]
    ax_left.set_yticks(range(n_row))
    ax_left.set_yticklabels(pdb_order, fontsize=7)
    ax_left.tick_params(axis="y", which="both", length=0, labelleft=True, labelsize=7)
    if leftmost in ("size", "stability"):
        ts.state_row_labels(ax_left, n_row, BAR_OFF)
    ts.gene_labels(ax_left, fig, genes, pad_pt=46)

    # left-align every title so its distance from the panel letter is the same in each
    # column (a centred title drifts toward/away from the letter with panel width)
    for key, ax in axes.items():
        ax.set_title(PANEL_TITLES[key], pad=TITLE_PAD, loc="left", x=0.0)
    if SHOW_PANEL_LETTERS:
        ts.panel_letters(fig, [(letter, axes[key]) for letter, key in zip("ABCD", enabled)])

    # one figure-level legend row: axes-level legends would reserve horizontal space
    # inside panel A and blow open the gap to panel B
    handles = []
    if PANELS.get("size") or PANELS.get("delta_class"):
        handles += [Patch(fc=ts.CAT_COLORS[c], ec="0.4", lw=0.3, label=c)
                    for c in ts.CATEGORIES]
    if PANELS.get("stability"):
        handles += [Patch(fc="0.45", label="stable"),
                    Patch(fc="white", ec="0.45", hatch="////", label="transient")]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 6),
                   frameon=False, fontsize=6.5, handlelength=1.1, columnspacing=1.4)

    out_file = out_file or OUT_FILE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure 3 to {out_file}")


if __name__ == "__main__":
    figure3()