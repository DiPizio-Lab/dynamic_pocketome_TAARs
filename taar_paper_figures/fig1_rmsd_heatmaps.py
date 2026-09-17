"""This script visualises the RMSD of MD simulations as a heatmap and a differential plot (apo vs holo).
Useful for the TAAR project because we have simulations of 156 µs of time.
Colourmap is viridis and values are taken from the previously (pipeline) calculated
RMSD values in/.../RMSD_protein_and_name_CA.csv
Median calculated along a column using float(np.median())"""
import time
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import taar_style as ts

# Settings
pd.set_option('expand_frame_repr', False)
pd.options.display.max_colwidth = 500  # long values in columns fully displayed

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TAAR_ROOT", PACKAGE_DIR.parent))
OUT_PATH = os.path.join(ROOT, "output", "meta_analysis", "RMSD_results_median.csv")
FILENAME = "RMSD_protein_and_name_CA.csv"
# .../output/<state>_structures/<state><PDBID>/<rep>/<FILENAME>
GLOB = f"output/*_structures/*/*/{FILENAME}"
METRICS = {
    "protein and name CA in":       "CA",
    "RMSD of binding site in":      "binding_site",
    "side chains of binding site":  "binding_site_sidechains",
    "CA without ICL3":              "CA_wo_ICL3"}

FOLDER_RE = re.compile(r"^(apo|holo)([A-Za-z0-9]{4})$")  # e.g. apo8ITF

GENE_DICT_PATH = os.path.join(ROOT, "reference_data", "hard_coded_gene_dict.txt")
HEATMAP_DIR = os.path.join(ROOT, "output", "meta_analysis", "paper_figures")
GENE_ORDER = ["hTAAR1", "mTAAR1", "mTAAR7f", "mTAAR9"]
# the two column names that get their own panel, plotted left -> right (same as the heatmap)
METRIC_CA = "median_CA_wo_ICL3"
METRIC_BS = "median_binding_site_sidechains"
PANEL_TITLES = ("Cα without ICL3", "Binding site (side chains)")
FIG_H_CONST = 0.7    # vertical padding constant (keeps rows aligned between figures)
COMBINED_W = 7.0     # combined-figure width in inches; saved file stays <= 7 in

# palette comes from taar_style so a global colour change propagates to every figure.
# viridis stays for the continuous heatmap values.
STATE_COLORS = ts.STATE_COLORS
GENE_COLORS = ts.GENE_COLORS


def col_for(df, needle):
    (hit,) = [c for c in df.columns if needle in c]  # expects exactly one
    return hit

def csv_summary_rmsd():
    """taking individual csvs storing RMSD values, calculates median RMSD per experiment,
    returns summarised csv of all experiments"""
    rows = []
    for csv_path in sorted(ROOT.glob(GLOB)):
        match = FOLDER_RE.match(csv_path.parent.parent.name)
        if not match:
            continue
        state, pdbid, rep = match.group(1), match.group(2).upper(), csv_path.parent.name
        df = pd.read_csv(csv_path)
        row = {"experiment": f"{state}_{pdbid}_{rep}","state": state,"pdbid": pdbid, "rep": rep}
        for col_name, short_name in METRICS.items():
            row[f"median_{short_name}"] = float(np.median(df[col_for(df, col_name)]))
        rows.append(row)

    rmsd_df = (pd.DataFrame(rows).sort_values(["pdbid", "state", "rep"]).reset_index(drop=True))
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(rmsd_df)
    rmsd_df.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"Wrote {len(rmsd_df)} experiments to {OUT_PATH}")
    return rmsd_df

# Plotting
# apply_style / load_gene_map / contiguous_blocks / gene labelling live in taar_style (ts.*).
# order_pdbids keeps its df-taking signature here (all call sites pass the summary_csv
# dataframe) but delegates the actual sort to ts.order_pdbids, which takes a plain PDB ID list.
def order_pdbids(summary_csv, gene_map):
    """returns the PDB IDs present in the summary, ordered by gene (GENE_ORDER) then PDB ID"""
    return ts.order_pdbids(summary_csv["pdbid"].unique(), gene_map)

def median_matrix(summary_csv, metric_col, pdb_order):
    """collapses replicates to their median --> median_matrix with rows = PDB IDs, cols = apo, holo"""
    median_matrix = summary_csv.pivot_table(index="pdbid", columns="state", values=metric_col, aggfunc="median")
    median_matrix = median_matrix.reindex(index=pdb_order, columns=["apo", "holo"])
    return median_matrix

def replicate_matrix(summary_csv, metric_col, pdb_order):
    """keeps replicates separate --> matrix with rows = PDB IDs, cols = apo reps then holo reps"""
    df = summary_csv.copy()
    df["state_rep"] = df["state"] + "_" + df["rep"].astype(str)
    matrix = df.pivot_table(index="pdbid", columns="state_rep", values=metric_col)
    apo_cols = sorted(col for col in matrix.columns if col.startswith("apo"))
    holo_cols = sorted(col for col in matrix.columns if col.startswith("holo"))
    matrix = matrix.reindex(index=pdb_order, columns=apo_cols + holo_cols)
    return matrix

def differential_percent(summary_csv, metric_col, pdb_order):
    """per PDB: (holo - apo) / apo * 100, using the median over replicates.
    positive --> holo moves more, negative --> apo moves more """
    matrix = median_matrix(summary_csv, metric_col, pdb_order)
    diff = (matrix["holo"] - matrix["apo"]) / matrix["apo"] * 100.0
    return diff.reindex(pdb_order)

def split_state_rep(col):
    """'apo_2' -> ('apo', '2');  'apo' -> ('apo', None)"""
    text = str(col)
    if "_" in text:
        state, rep = text.split("_", 1)
        return state, rep
    return text, None

def _heat_scale(matrices):
    """shared vmin/vmax and a viridis cmap (missing cells --> light grey)"""
    vmin = min(np.nanmin(m.values) for m in matrices)
    vmax = max(np.nanmax(m.values) for m in matrices)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    return vmin, vmax, cmap

def _draw_heatmap_panel(ax, matrix, genes, vmin, vmax, cmap, value_annot):
    """draw one metric's heatmap onto ax; returns the image for the colourbar"""
    data = matrix.values.astype(float)
    n_row, n_col = data.shape
    states = [split_state_rep(c)[0] for c in matrix.columns]
    reps = [split_state_rep(c)[1] for c in matrix.columns]
    replicate_mode = any(r is not None for r in reps)

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_col))
    ax.set_xticklabels(reps if replicate_mode else states)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for _, _, end in ts.contiguous_blocks(states):      # apo | holo divider
        if end < n_col - 1:
            ax.axvline(end + 0.5, color="white", linewidth=1.2)
    for _, _, end in ts.contiguous_blocks(genes):        # gene-group dividers
        if end < n_row - 1:
            ax.axhline(end + 0.5, color="white", linewidth=0.8)

    if value_annot:
        for i in range(n_row):
            for j in range(n_col):
                value = data[i, j]
                if np.isnan(value):
                    continue
                norm = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if norm < 0.5 else "black", fontsize=8)

    if replicate_mode:                                # apo / holo once beneath the numbers
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for state, start, end in ts.contiguous_blocks(states):
            ax.text((start + end) / 2, -0.05, state, transform=trans,
                    ha="center", va="top", fontsize=9, color=STATE_COLORS.get(state, "black"))
    return im

def _draw_diff_panel(ax, diff_values, genes, xmax):
    """draw one metric's diverging bar panel onto ax; bars coloured by which state moves more
    (sand = apo moves more, mauve = holo moves more). Keeps the 0 reference line."""
    n_row = len(diff_values)
    y = np.arange(n_row)
    colors = [STATE_COLORS["holo"] if v >= 0 else STATE_COLORS["apo"] for v in diff_values]
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.barh(y, diff_values, color=colors, edgecolor="0.3", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.7, zorder=3)   # 0 reference line
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(n_row - 0.5, -0.5)           # row 0 on top, like the heatmap
    ax.set_xticks([-xmax, 0, xmax])
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    for _, _, end in ts.contiguous_blocks(genes):
        if end < n_row - 1:
            ax.axhline(end + 0.5, color="0.85", linewidth=0.6)

def _diff_direction_labels(ax):
    """small colour-coded 'apo <- | -> holo' note tucked just under a differential panel title"""
    ax.text(0.02, 1.0, "← apo ", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6, color=STATE_COLORS["apo"])
    ax.text(0.98, 1.0, "holo →", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=STATE_COLORS["holo"])

def plot_two_metrics(matrix_ca, matrix_bs, gene_map, suptitle, out_file, value_annot=True):
    """separate heatmap figure: both metrics side by side (viridis, shared colour scale)."""
    ts.apply_style()
    pdbids = list(matrix_ca.index)
    genes = [gene_map[pdb] for pdb in pdbids]
    n_col, n_row = matrix_ca.shape[1], len(pdbids)
    vmin, vmax, cmap = _heat_scale((matrix_ca, matrix_bs))

    fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained",
                             figsize=(3.4 + n_col * 0.42, FIG_H_CONST + n_row * 0.24))
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    im = _draw_heatmap_panel(axes[0], matrix_ca, genes, vmin, vmax, cmap, value_annot)
    _draw_heatmap_panel(axes[1], matrix_bs, genes, vmin, vmax, cmap, value_annot)
    axes[0].set_title(PANEL_TITLES[0]); axes[1].set_title(PANEL_TITLES[1])
    axes[0].set_yticks(range(n_row)); axes[0].set_yticklabels(pdbids)
    ts.gene_labels(axes[0], fig, genes)

    cbar = fig.colorbar(im, ax=axes, fraction=0.06, pad=0.05, aspect=16, shrink=0.85)
    # cbar.set_label("median RMSD (Å)", labelpad=10)
    cbar.outline.set_linewidth(0.5); cbar.ax.tick_params(width=0.5)

    # fig.suptitle(suptitle)
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_file}")

def differential_plot(out_file=None):
    """separate differential figure: diverging bars (state-coloured), same order/scale as the heatmap.
    right --> holo moves more, left --> apo moves more."""
    ts.apply_style()
    summary_csv = pd.read_csv(OUT_PATH)
    gene_map = ts.load_gene_map()
    pdb_order = order_pdbids(summary_csv, gene_map)
    genes = [gene_map[pdb] for pdb in pdb_order]
    n_row = len(pdb_order)

    diffs = (differential_percent(summary_csv, METRIC_CA, pdb_order),
             differential_percent(summary_csv, METRIC_BS, pdb_order))
    xmax = np.ceil(max(np.nanmax(np.abs(d.values)) for d in diffs) / 5) * 5

    fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained",
                             figsize=(3.4 + 2 * 0.9, FIG_H_CONST + n_row * 0.24))
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.0)
    for ax, diff, title in zip(axes, diffs, ("Δ Cα (%)", "Δ BS (%)")):
        _draw_diff_panel(ax, diff.values, genes, xmax)
        ax.set_title(title, pad=11)
        _diff_direction_labels(ax)
    axes[0].set_yticks(range(n_row)); axes[0].set_yticklabels(pdb_order)
    ts.gene_labels(axes[0], fig, genes)

    if out_file is None:
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        out_file = os.path.join(HEATMAP_DIR, "RMSD_differential.png")
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved differential plot to {out_file}")

# panels of the combined figure: set any to False to drop it
SHOW_PANEL_LETTERS = True    # set False to export without A/B/C/D
PANELS = {
    "heat_ca": True,      # A  Calpha RMSD heatmap
    "heat_bs": True,      # B  binding-site RMSD heatmap
    "colourbar": True,    #    shared viridis scale (only drawn if a heatmap is on)
    "diff_ca": True,      # C  Delta Calpha (%)
    "diff_bs": True,      # D  Delta binding site (%)
}


def combined_plot(value_annot=False, replicates=True, out_file=None):
    """single stitched figure: two RMSD heatmaps + shared colourbar + two differential panels,
    all sharing the gene-ordered PDB rows. Defaults to the all-replicates heatmap."""
    ts.apply_style()
    summary_csv = pd.read_csv(OUT_PATH)
    gene_map = ts.load_gene_map()
    pdb_order = order_pdbids(summary_csv, gene_map)
    genes = [gene_map[pdb] for pdb in pdb_order]
    n_row = len(pdb_order)

    matrix_fn = replicate_matrix if replicates else median_matrix
    hm_ca = matrix_fn(summary_csv, METRIC_CA, pdb_order)
    hm_bs = matrix_fn(summary_csv, METRIC_BS, pdb_order)
    diff_ca = differential_percent(summary_csv, METRIC_CA, pdb_order)
    diff_bs = differential_percent(summary_csv, METRIC_BS, pdb_order)

    vmin, vmax, cmap = _heat_scale((hm_ca, hm_bs))
    xmax = np.ceil(max(np.nanmax(np.abs(diff_ca.values)),
                       np.nanmax(np.abs(diff_bs.values))) / 5) * 5
    n_hm = hm_ca.shape[1]

    enabled = [key for key in ("heat_ca", "heat_bs", "diff_ca", "diff_bs")
               if PANELS.get(key)]
    if not enabled:
        raise ValueError("PANELS: enable at least one panel")
    show_cbar = PANELS.get("colourbar", True) and any(k.startswith("heat") for k in enabled)

    heat_keys = [key for key in enabled if key.startswith("heat")]
    widths, layout = [], []
    for key in enabled:
        widths.append(n_hm if key.startswith("heat") else n_hm * 0.8)
        layout.append(key)
        if show_cbar and heat_keys and key == heat_keys[-1]:
            widths.append(0.45)              # colourbar sits after the last heatmap
            layout.append("colourbar")

    fig = plt.figure(figsize=(COMBINED_W, FIG_H_CONST + n_row * 0.24), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths)

    axes, first = {}, None
    for column, key in enumerate(layout):
        if key == "colourbar":
            axes[key] = fig.add_subplot(gs[0, column]); continue
        ax = fig.add_subplot(gs[0, column], sharey=first) if first is not None \
            else fig.add_subplot(gs[0, column])
        first = first or ax
        axes[key] = ax

    im = None
    if "heat_ca" in axes:
        im = _draw_heatmap_panel(axes["heat_ca"], hm_ca, genes, vmin, vmax, cmap, value_annot)
        axes["heat_ca"].set_title("Cα w/o ICL3")
    if "heat_bs" in axes:
        im2 = _draw_heatmap_panel(axes["heat_bs"], hm_bs, genes, vmin, vmax, cmap, value_annot)
        im = im or im2
        axes["heat_bs"].set_title("Binding site")
    if "diff_ca" in axes:
        _draw_diff_panel(axes["diff_ca"], diff_ca.values, genes, xmax)
        axes["diff_ca"].set_title("Δ Cα (%)", pad=11)
        _diff_direction_labels(axes["diff_ca"])
    if "diff_bs" in axes:
        _draw_diff_panel(axes["diff_bs"], diff_bs.values, genes, xmax)
        axes["diff_bs"].set_title("Δ BS (%)", pad=11)
        _diff_direction_labels(axes["diff_bs"])

    # only the leftmost enabled panel keeps the PDB row labels
    ax_hca = axes[enabled[0]]
    for key in enabled[1:]:
        axes[key].tick_params(labelleft=False)

    ax_hca.set_yticks(range(n_row)); ax_hca.set_yticklabels(pdb_order)
    ts.gene_labels(ax_hca, fig, genes)

    if show_cbar and im is not None:
        cbar = fig.colorbar(im, cax=axes["colourbar"])
        # cbar.set_label("median RMSD (Å)", labelpad=6)
        cbar.outline.set_linewidth(0.5); cbar.ax.tick_params(width=0.5)

    if SHOW_PANEL_LETTERS:
        # letter the data panels only (not the colourbar), aligned as a grid
        lettered = [k for k in ("heat_ca", "heat_bs", "diff_ca", "diff_bs") if k in axes]
        ts.panel_letters(fig, list(zip("ABCD", (axes[k] for k in lettered))))

    if out_file is None:
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        tag = "replicates" if replicates else "median"
        out_file = os.path.join(HEATMAP_DIR, f"figure1_RMSD_combined_{tag}.png")
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {out_file}")

def heatmap_plots(value_annot=True):
    """uses analysis_results/RMSD_results_median.csv to create heatmaps in viridis colour scale.
    value_annot toggles the per-cell numbers on/off."""
    summary_csv = pd.read_csv(OUT_PATH)
    gene_map = ts.load_gene_map()
    pdb_order = order_pdbids(summary_csv, gene_map)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    # version 1: replicates collapsed to their median -> apo | holo (2 cells per PDB)
    median_ca = median_matrix(summary_csv, METRIC_CA, pdb_order)
    median_bs = median_matrix(summary_csv, METRIC_BS, pdb_order)
    plot_two_metrics(median_ca, median_bs, gene_map, "RMSD - median over replicates",
                     os.path.join(HEATMAP_DIR, "RMSD_heatmap_median.png"), value_annot=value_annot)

    # version 2: replicates side by side -> apo reps | holo reps (6 cells per PDB)
    reps_ca = replicate_matrix(summary_csv, METRIC_CA, pdb_order)
    reps_bs = replicate_matrix(summary_csv, METRIC_BS, pdb_order)
    plot_two_metrics(reps_ca, reps_bs, gene_map, "RMSD - replicates side by side",
                     os.path.join(HEATMAP_DIR, "RMSD_heatmap_replicates.png"), value_annot=value_annot)


if __name__ == '__main__':
    t0 = time.time()
    # csv_summary_rmsd()
    combined_plot()                 # single figure (all replicates) - the one for the paper
    # heatmap_plots(value_annot=False)   # the two heatmaps on their own
    # differential_plot()                # the differential on its own
    t1 = time.time()
    print(f'The plotting took {t1 - t0} seconds')