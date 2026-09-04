"""figure 2: ORTHOSTERIC binding site.
A per-frame volume distribution (apo vs holo) | B Delta volume (%)
C hTAAR1 orthosteric site render | D largest vs smallest pocket (renders + silhouettes).
rows = the same gene-ordered PDB IDs as figure 1."""
import os
import re

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import taar_style as ts

# ---------------------------------------------------------------- constants
VOL_CSV = os.path.join(ts.ROOT, "output", "meta_analysis", "across_genes", "orthosteric_perframe_volumes.csv")
OUT_FILE = os.path.join(ts.FIG_DIR, "figure2_binding_site.png")

# Panels C and D are each ONE finished image.
# Panel D is produced separately by fig2_panel_d_pockets.panel_d(), which composites the
# blender render and the silhouettes; figure 2 just places the result. Keeping them
# decoupled means panel D can be re-rendered without touching this figure.
IMG_SITE = os.path.join(ts.FIG_DIR, "hTAAR1_binding_site_8JLR_nobg.png")
IMG_PANEL_D = os.path.join(ts.FIG_DIR, "panel_d_pocket_silhouettes.png")

CROP_TOL = 0.04          # background-gradient tolerance for autocropping the renders
CROP_MARGIN = 30         # px kept around the detected content

# panels: set any to False to drop it
SHOW_PANEL_LETTERS = True    # set False to export without A/B/C/D
PANELS = {
    "volume": True,     # A  per-frame volume boxplots
    "delta": True,      # B  Delta volume (%)
    "site": True,       # C  hTAAR1 binding-site renders
    "pockets": True,    # D  largest vs smallest pocket, renders + silhouettes
}

VOL_COL = "interpolated_pock_volume"
ID_RE = re.compile(r"^(apo|holo)([A-Za-z0-9]{4})_")
VMAX = 1400
# how the size categories are shown behind the boxes:
#   "lines"  white plot area, coloured dashed line at each threshold (cleanest in print)
#   "bands"  faint tinted bands, controlled by BAND_ALPHA
BAND_STYLE = "bands"
BAND_ALPHA = 0.10
SHOW_STATE_ROW_LABELS = False   # tiny a/h beside each row - redundant with the apo/holo legend
BAR_OFF = 0.19
BOX_H = 0.32
ROW_H = 0.215                 # per-structure row height (in)
ROW_BLOCK_PAD = 0.9          # labels/titles above and below the row block (in)
IMG_BLOCK_H = 2.3            # height of the C/D block (in); total stays <= 9.167 (ACS)
TITLE_PAD = 10               # one pad for every panel -> titles line up
LEGEND_ROW_H = 0.34          # inches reserved under panel A for its category legend

# ---------------------------------------------------------------- data prep
def load_volumes():
    """per-frame orthosteric volumes; ID like 'apo8ITF_2_p19_i3' -> state + pdbid"""
    df = pd.read_csv(VOL_CSV, usecols=["ID", "rep", VOL_COL])
    parsed = df["ID"].str.extract(ID_RE)
    df["state"], df["pdbid"] = parsed[0], parsed[1]
    return df.dropna(subset=["state", "pdbid"])

def volume_dict(df):
    """{(pdbid, state): per-frame volumes pooled over replicates}"""
    return {key: grp[VOL_COL].to_numpy(dtype=float)
            for key, grp in df.groupby(["pdbid", "state"])}

def delta_volume(vol, pdb_order):
    """per PDB: (median holo - median apo) / median apo * 100"""
    out = []
    for pdb in pdb_order:
        apo, holo = vol.get((pdb, "apo")), vol.get((pdb, "holo"))
        if apo is None or holo is None or len(apo) == 0 or len(holo) == 0:
            out.append(np.nan); continue
        med_apo = np.median(apo)
        out.append((np.median(holo) - med_apo) / med_apo * 100.0 if med_apo else np.nan)
    return np.array(out, dtype=float)

# ---------------------------------------------------------------- drawing
def draw_volume_panel(ax, vol, pdb_order, genes):
    """paired horizontal boxplots with viridis size-category bands behind"""
    if BAND_STYLE == "bands":
        for x0, x1, cat in zip([0] + ts.THRESH, ts.THRESH + [VMAX], ts.CATEGORIES):
            ax.axvspan(x0, x1, color=ts.CAT_COLORS[cat], alpha=BAND_ALPHA,
                       linewidth=0, zorder=0)
        for x in ts.THRESH:
            ax.axvline(x, color="0.65", linewidth=0.4, linestyle=(0, (3, 3)), zorder=1)
    else:
        # plot area stays white; each threshold line takes the colour of the category
        # it opens, so the line itself says which class you are entering
        for x, cat in zip(ts.THRESH, ts.CATEGORIES[1:]):
            ax.axvline(x, color=ts.CAT_COLORS[cat], linewidth=0.9,
                       linestyle=(0, (4, 2)), zorder=1, alpha=0.9)
    for i, pdb in enumerate(pdb_order):
        for state, off in (("apo", BAR_OFF), ("holo", -BAR_OFF)):
            data = vol.get((pdb, state))
            if data is None or len(data) == 0:
                continue
            bp = ax.boxplot(data, positions=[i + off], vert=False, widths=BOX_H,
                            patch_artist=True, showfliers=False, zorder=3,
                            medianprops=dict(color="white", linewidth=0.8),
                            whiskerprops=dict(color=ts.STATE_COLORS[state], linewidth=0.5),
                            capprops=dict(color=ts.STATE_COLORS[state], linewidth=0.5),
                            boxprops=dict(linewidth=0))
            for box in bp["boxes"]:
                box.set(facecolor=ts.STATE_COLORS[state])
    ax.set_ylim(len(pdb_order) - 0.5, -0.5)
    ax.set_xlim(0, VMAX)
    ax.set_yticks(range(len(pdb_order)))
    ax.set_yticklabels(pdb_order, fontsize=7)
    ax.tick_params(axis="y", which="major", length=0, pad=3)
    if SHOW_STATE_ROW_LABELS:
        ts.state_row_labels(ax, len(pdb_order), BAR_OFF, fontsize=5.0)
    ts.gene_dividers(ax, genes)
    ax.set_xticks([0] + ts.THRESH + [1000, VMAX])      # ticks on the category cut points
    ax.set_xlabel(u"binding-site volume (\u00c5\u00b3)", loc="left")
    ax.tick_params(axis="x", length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

def _crop_to_mask(rgb, mask, margin=CROP_MARGIN):
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, rgb.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


def load_image(path, tol=CROP_TOL):
    """finished panel image, trimmed to its content.

    A transparent (RGBA) export is composited onto white and cropped to the alpha; an
    opaque render has a soft gradient background rather than a flat colour, so each pixel
    is compared against a heavily blurred copy of itself to find the content."""
    if not path or not os.path.exists(path):
        return None
    image = plt.imread(path)
    if image.shape[-1] == 4 and image[..., 3].min() < 1.0:
        alpha = image[..., 3:4]
        rgb = image[..., :3] * alpha + (1.0 - alpha)
        return _crop_to_mask(rgb, alpha[..., 0] > 0.01)
    rgb = image[..., :3]
    background = ndimage.uniform_filter(rgb, size=(151, 151, 1))
    content = ndimage.binary_closing(
        np.abs(rgb - background).max(axis=-1) > tol, np.ones((9, 9)))
    return _crop_to_mask(rgb, content)


def draw_image_panel(ax, image, placeholder="render\npending"):
    """just the image - no ticks, no spines, no empty coordinate system"""
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("0.97")
        ax.text(0.5, 0.5, placeholder, transform=ax.transAxes, ha="center",
                va="center", fontsize=7.5, color="0.45")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(image is None)
        spine.set_color("0.75"); spine.set_linewidth(0.5)


# ---------------------------------------------------------------- assembly
def figure2(out_file=None):
    ts.apply_style()
    df = load_volumes()
    gene_map = ts.load_gene_map()
    pdb_order = ts.order_pdbids(df["pdbid"].unique(), gene_map)
    genes = [gene_map[p] for p in pdb_order]
    n_row = len(pdb_order)

    category_legend = None
    vol = volume_dict(df)
    dvol = delta_volume(vol, pdb_order)
    xmax = ts.nice_max(dvol[~np.isnan(dvol)], 10)
    row_block = ROW_BLOCK_PAD + n_row * ROW_H

    top_keys = [k for k in ("volume", "delta") if PANELS.get(k)]
    bottom_keys = [k for k in ("site", "pockets") if PANELS.get(k)]
    if not top_keys and not bottom_keys:
        raise ValueError("PANELS: enable at least one panel")

    row_heights, row_specs = [], []
    show_category_legend = PANELS.get("volume", False)
    if top_keys:
        row_heights.append(row_block); row_specs.append(("top", top_keys))
    if show_category_legend:
        row_heights.append(LEGEND_ROW_H); row_specs.append(("legend", []))
    if bottom_keys:
        row_heights.append(IMG_BLOCK_H); row_specs.append(("bottom", bottom_keys))

    fig = plt.figure(figsize=(7.0, sum(row_heights)), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.04, hspace=0.16)
    outer = fig.add_gridspec(len(row_heights), 1, height_ratios=row_heights)

    # the two image panels are sized by their own aspect ratio, so neither ends up
    # padded with white space inside an over-wide box
    images = {"site": load_image(IMG_SITE), "pockets": load_image(IMG_PANEL_D)}
    aspects = {key: (img.shape[1] / img.shape[0]) if img is not None else 1.4
               for key, img in images.items()}

    axes = {}
    legend_ax = None
    for row, (kind, keys) in enumerate(row_specs):
        if kind == "legend":
            legend_ax = fig.add_subplot(outer[row]); legend_ax.axis("off")
            continue
        widths = [aspects[k] if k in aspects else {"volume": 2.6, "delta": 1.0}[k]
                  for k in keys]
        sub = outer[row].subgridspec(1, len(keys), width_ratios=widths)
        for column, key in enumerate(keys):
            share = axes["volume"] if key == "delta" and "volume" in axes else None
            axes[key] = fig.add_subplot(sub[0, column], sharey=share)

    if "volume" in axes:
        draw_volume_panel(axes["volume"], vol, pdb_order, genes)
        ts.gene_labels(axes["volume"], fig, genes, pad_pt=46)
        axes["volume"].set_title("Binding-site volume", pad=TITLE_PAD)
        state_legend = axes["volume"].legend(
            handles=[Patch(fc=ts.STATE_COLORS["apo"], label="apo"),
                     Patch(fc=ts.STATE_COLORS["holo"], label="holo")],
            loc="upper right", frameon=False, fontsize=7, handlelength=1.1)
        axes["volume"].add_artist(state_legend)
        # size categories: line swatches when the thresholds are drawn as lines,
        # filled swatches when they are drawn as bands. The legend describes panel A,
        # so it is anchored under panel A rather than under the whole figure.
        if BAND_STYLE == "lines":
            cat_handles = [Line2D([], [], color=ts.CAT_COLORS[c], linewidth=0.9,
                                  linestyle=(0, (4, 2)), label=c)
                           for c in ts.CATEGORIES[1:]]
            cat_title = "category threshold"
        else:
            cat_handles = [Patch(fc=ts.CAT_COLORS[c], alpha=BAND_ALPHA * 3, ec="0.6",
                                 lw=0.3, label=c) for c in ts.CATEGORIES]
            cat_title = "size category"
        category_legend = (cat_handles, cat_title)
    if "delta" in axes:
        ts.diff_barh(axes["delta"], np.nan_to_num(dvol), genes, xmax)
        axes["delta"].tick_params(axis="x", labelsize=7)
        axes["delta"].set_title(u"\u0394 volume (%)", pad=TITLE_PAD)
        if "volume" in axes:
            # panel A already carries the PDB rows. B shares A's y-axis, so DON'T clear the
            # ticks (that empties A too) - just hide B's dashes and labels on B's side.
            axes["delta"].tick_params(axis="y", which="both", length=0, labelleft=False)
        else:                                         # B is standalone -> it carries rows
            axes["delta"].tick_params(axis="y", which="both", labelleft=True)
            axes["delta"].set_yticks(range(n_row))
            axes["delta"].set_yticklabels(pdb_order, fontsize=7)
            ts.gene_labels(axes["delta"], fig, genes, pad_pt=46)
    if "site" in axes:
        draw_image_panel(axes["site"], images["site"])
    if "pockets" in axes:
        draw_image_panel(axes["pockets"], images["pockets"])

    if category_legend is not None and legend_ax is not None:
        handles, title = category_legend
        legend_ax.legend(handles=handles, title=title, loc="center",
                         bbox_to_anchor=(0.5, 0.5), ncol=len(handles), frameon=False,
                         fontsize=6.5, title_fontsize=6.5, handlelength=1.6,
                         columnspacing=1.4, alignment="left")

    if SHOW_PANEL_LETTERS:
        ts.panel_letters(fig, [(letter, axes[key]) for letter, key
                               in zip("ABCD", [k for k in ("volume", "delta", "site", "pockets")
                                               if k in axes])])

    out_file = out_file or OUT_FILE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure 2 to {out_file}")


if __name__ == "__main__":
    figure2()