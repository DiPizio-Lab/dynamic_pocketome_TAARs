"""figure 2 panel D: largest vs smallest orthosteric pocket, apo vs holo.

Two halves that must agree: the blender render on the left, and on the right the same
four pockets traced as 2D silhouettes projected through the SAME camera. The silhouettes
are what let a reader compare sizes properly - in the 3D view a large pocket occludes a
small one, so the outlines carry the size comparison and the render carries the context.

Encoding:  colour = state (apo sand, holo slate)   dashed = apo, solid = holo
           full-strength colour = large pocket, lightened = small pocket

Inputs (both produced from the SAME camera by the Blender scripts -- not included in this
repo, see paper_figures_README.md):
    pocket_camera_coords.npz   {object_name: (N, 2) pixel coords} + __res__
    <render>.png               ideally the no-background (RGBA) export

Both are expected under BLENDER_DIR (default: blender_renders/ next to this file) -- drop
them there, or point BLENDER_DIR/NPZ/RENDER at wherever you rendered them. Without them,
load_coords() raises a clear error naming exactly what's missing; load_render() degrades
quietly (panel D just renders without the image half) since a render is a nice-to-have,
not something silhouette tracing depends on.

Pocket styles are DERIVED from the object names, so reselecting the representative
pockets needs no edits here. The backbone is excluded by name.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

import taar_style as ts

# ---------------------------------------------------------------- constants
# Not shipped with this repo -- see paper_figures_README.md for how these two files are
# produced (Blender scripts, not included here) and drop them here, or point this
# elsewhere entirely.
BLENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_renders")
NPZ = os.path.join(BLENDER_DIR, "pocket_camera_coords.npz")
# prefer the transparent export; the grey-background one is autocropped as a fallback
RENDER = os.path.join(BLENDER_DIR, "blender_apo_holo_smallest_largest_pockets_nobg.png")
RENDER_FALLBACK = os.path.join(BLENDER_DIR, "blender_apo_holo_smallest_largest_pockets.png")
SHOW_SUBTITLES = False      # small titles over the two halves
RENDER_TITLE = "apo vs holo, largest vs smallest"
SILHOUETTE_TITLE = "silhouettes, common camera"
OUT_FILE = os.path.join(ts.FIG_DIR, "panel_d_pocket_silhouettes.png")

# never trace these - the backbone is geometry, not a pocket
EXCLUDE = ("backbone", "ribbon", "protein", "cartoon")

# The ligand is exported under "ligand__<object>" (full outline) and
# "ligand_com__<object>" (its projected centre of mass). It is never sliced, because it
# is what the cut is meant to reveal. 8JLR binds A77636 (VRK).
SHOW_LIGAND = True
LIGAND_STYLE = "outline"        # "outline" = its silhouette, "marker" = a dot at the COM
LIGAND_COLOR = "#DE7B1E"
LIGAND_LABEL = "ligand (A77636)"
LIGAND_MARKER_SIZE = 22

LIGHT = {"apo": "#EAD8BE", "holo": "#C7A0A6"}   # lightened state colours for "small"
DASH = {"apo": (0, (4, 2)), "holo": "-"}        # dashed = apo, solid = holo

# optional, cosmetic: volume in A^3 keyed by the EXACT npz key (= blender object name)
#   POCKET_VOLUMES = {"holo_ortho_large_8JLQ": 1251, "apo_ortho_small_8IW4": 82}
POCKET_VOLUMES = {}

GRID = 700           # rasterisation grid for the silhouette trace
CLOSE_RADIUS = "auto"  # morphological closing radius in grid cells, or "auto".
# "auto" scales the radius with point density: slicing the pockets in half halves the
# number of vertices, and a radius tuned for the full cloud then fails to bridge the
# gaps, shattering one outline into hundreds of fragments.
FILL_ALPHA = 0.12
MARGIN = 12          # px of frame kept around the pocket bounding box
WSPACE = 0.0         # gap between the render and the silhouettes
CROP_TOL = 0.04      # background tolerance when autocropping an opaque render


# ---------------------------------------------------------------- styles
def pocket_style(key):
    """(state, size, style dict) for an npz key, or None if it is not a pocket"""
    lowered = key.lower()
    if any(token in lowered for token in EXCLUDE):
        return None
    state = "apo" if "apo" in lowered else "holo" if "holo" in lowered else None
    size = "large" if "large" in lowered else "small" if "small" in lowered else None
    if state is None or size is None:
        return None

    is_large = size == "large"
    label = f"{state} \u00b7 {size}"
    tail = key.replace("_ortho", "").replace(f"{state}_", "").replace(f"{size}_", "")
    tail = tail.strip("_")
    if tail:
        label += f"   {tail}"
    volume = POCKET_VOLUMES.get(key)
    if volume is not None:
        label += f"   {volume:.0f} \u00c5\u00b3"

    return state, size, dict(
        label=label,
        color=ts.STATE_COLORS[state] if is_large else LIGHT[state],
        ls=DASH[state],
        lw=1.2 if is_large else 1.0,
        z=(4 if state == "holo" else 2) + (0 if is_large else -1))


# ---------------------------------------------------------------- silhouette
def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x ** 2 + y ** 2 <= radius ** 2


def load_coords(npz=None):
    """{key: (N, 2) camera pixel coords} for every pocket, plus the export resolution.

    Keys are the blender object names, so nothing needs renaming when the pocket
    selection changes; anything failing pocket_style (the backbone) is dropped."""
    npz = npz or NPZ                       # resolved at call time so setting NPZ works
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} not found -- panel D needs pocket_camera_coords.npz, produced by the "
            "Blender export script (not included in this repo, see paper_figures_README.md's "
            f"Blender note). Render it separately and place it at {npz} (or point NPZ/BLENDER_DIR "
            "at wherever you rendered it).")
    data = np.load(npz)
    coords = {key: np.asarray(data[key])[:, :2]
              for key in data.files if key != "__res__" and pocket_style(key)}
    if not coords:
        raise ValueError(f"no pocket keys in {npz}; found {list(data.files)}")
    resolution = tuple(int(v) for v in data["__res__"]) if "__res__" in data.files else None
    return coords, resolution


def load_ligand(npz=None):
    """(outline, centre_of_mass) in the same pixel frame, or (None, None).

    Older exports have no ligand entries, so this degrades quietly rather than raising."""
    npz = npz or NPZ
    data = np.load(npz)
    outline = next((np.asarray(data[k])[:, :2] for k in data.files
                    if k.startswith("ligand__")), None)
    com = next((np.asarray(data[k])[:2] for k in data.files
                if k.startswith("ligand_com__")), None)
    if com is None and outline is not None:
        com = outline.mean(axis=0)
    return outline, com


def auto_close_radius(n_points, grid=GRID):
    """closing radius that keeps a sliced (sparser) cloud connected"""
    return int(np.clip(round(1.4 * grid / max(np.sqrt(n_points), 1.0)), 6, 40))


def silhouette(points, extent, grid=GRID, close_radius=None):
    """rasterise projected vertices, close the gaps, trace the outline back to pixels.

    A point cloud has no edges, so it is binned onto a grid, morphologically closed to
    join neighbouring points into one blob, filled, and contoured."""
    if close_radius is None:
        close_radius = CLOSE_RADIUS
    if close_radius == "auto":
        close_radius = auto_close_radius(len(points), grid)
    x0, x1, y0, y1 = extent
    ix = np.clip(((points[:, 0] - x0) / (x1 - x0) * (grid - 1)).astype(int), 0, grid - 1)
    iy = np.clip(((points[:, 1] - y0) / (y1 - y0) * (grid - 1)).astype(int), 0, grid - 1)
    mask = np.zeros((grid, grid), dtype=bool)
    mask[iy, ix] = True
    mask = ndimage.binary_fill_holes(
        ndimage.binary_closing(mask, structure=_disk(close_radius)))

    contours = []
    for contour in measure.find_contours(mask.astype(float), 0.5):
        cx = contour[:, 1] / (grid - 1) * (x1 - x0) + x0
        cy = contour[:, 0] / (grid - 1) * (y1 - y0) + y0
        contours.append(np.column_stack([cx, cy]))
    return contours


# ---------------------------------------------------------------- render
def _crop_to_mask(rgb, mask, margin=MARGIN):
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, rgb.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


def load_render(path=None, fallback=None):
    """RGBA render composited onto white, or an opaque render autocropped.

    A transparent export needs compositing (matplotlib would otherwise show the alpha
    against the figure background); an opaque one only needs its empty margin trimmed."""
    path = path or RENDER
    fallback = fallback or RENDER_FALLBACK
    if not os.path.exists(path):
        path = fallback
    if not os.path.exists(path):
        return None

    image = plt.imread(path)
    if image.shape[-1] == 4 and image[..., 3].min() < 1.0:
        alpha = image[..., 3:4]
        rgb = image[..., :3] * alpha + (1.0 - alpha)          # composite onto white
        return _crop_to_mask(rgb, alpha[..., 0] > 0.01)

    rgb = image[..., :3]
    background = ndimage.uniform_filter(rgb, size=(151, 151, 1))
    content = ndimage.binary_closing(
        np.abs(rgb - background).max(axis=-1) > CROP_TOL, np.ones((9, 9)))
    return _crop_to_mask(rgb, content)


# ---------------------------------------------------------------- drawing
def draw_render(ax, image, title=RENDER_TITLE):
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("0.97")
        ax.text(0.5, 0.5, "render\npending", transform=ax.transAxes, ha="center",
                va="center", fontsize=6.5, color="0.45")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(image is None)
        spine.set_color("0.75"); spine.set_linewidth(0.5)
    ax.set_title(title, fontsize=7, pad=3)


def silhouette_extent(coords):
    """bounding box of every pocket, padded - shared by the drawing and the layout"""
    points = np.vstack(list(coords.values()))
    return (points[:, 0].min() - MARGIN, points[:, 0].max() + MARGIN,
            points[:, 1].min() - MARGIN, points[:, 1].max() + MARGIN)


def draw_ligand(ax, outline, com):
    """the ligand at its projected position - outline, or a marker at the centre of mass"""
    if LIGAND_STYLE == "outline" and outline is not None and len(outline) > 8:
        x0, x1 = outline[:, 0].min() - 4, outline[:, 0].max() + 4
        y0, y1 = outline[:, 1].min() - 4, outline[:, 1].max() + 4
        for contour in silhouette(outline, (x0, x1, y0, y1), grid=260):
            ax.fill(contour[:, 0], contour[:, 1], color=LIGAND_COLOR, alpha=0.30,
                    linewidth=0, zorder=9)
            ax.plot(contour[:, 0], contour[:, 1], color=LIGAND_COLOR, linewidth=1.0,
                    zorder=10)
        ax.plot([], [], color=LIGAND_COLOR, linewidth=1.0, label=LIGAND_LABEL)
    elif com is not None:
        ax.scatter([com[0]], [com[1]], s=LIGAND_MARKER_SIZE, marker="o",
                   facecolor=LIGAND_COLOR, edgecolor="white", linewidth=0.5, zorder=10,
                   label=LIGAND_LABEL)


def draw_silhouettes(ax, coords, legend=True, ligand=None):
    extent = silhouette_extent(coords)

    for key in sorted(coords, key=lambda k: 0 if "large" in k.lower() else 1):
        style = pocket_style(key)[2]
        for contour in silhouette(coords[key], extent):
            ax.fill(contour[:, 0], contour[:, 1], color=style["color"],
                    alpha=FILL_ALPHA, linewidth=0, zorder=style["z"])
            ax.plot(contour[:, 0], contour[:, 1], color=style["color"],
                    linewidth=style["lw"], linestyle=style["ls"], zorder=style["z"] + 4)
        ax.plot([], [], color=style["color"], linewidth=style["lw"],
                linestyle=style["ls"], label=style["label"])

    if SHOW_LIGAND and ligand is not None:
        draw_ligand(ax, *ligand)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    # camera pixel coordinates are a projection artefact, not data: every outline goes
    # through the same camera, so relative size is the message
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("silhouettes, common camera", fontsize=7, pad=3)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False,
                  fontsize=6, handlelength=2.2, ncol=2, columnspacing=1.2,
                  borderaxespad=0)


def panel_aspects(coords, image):
    """(render, silhouette) width/height ratios.

    Both halves hold fixed-aspect content: imshow and the equal-aspect silhouettes.
    Giving them equal-width boxes leaves the narrower one padded on both sides, which
    is what opened the gap in the middle. Sizing each box to its own content aspect
    removes that padding entirely."""
    x0, x1, y0, y1 = silhouette_extent(coords)
    silhouette_aspect = (x1 - x0) / (y1 - y0)
    render_aspect = (image.shape[1] / image.shape[0]) if image is not None else 1.0
    return render_aspect, silhouette_aspect


def draw_panel_d(fig, gridspec_cell, coords, image=None, ligand=None):
    """render left, silhouettes right, drawn into a parent gridspec cell"""
    inner = gridspec_cell.subgridspec(1, 2, width_ratios=panel_aspects(coords, image),
                                      wspace=WSPACE)
    draw_render(fig.add_subplot(inner[0, 0]), image)
    ax_sil = fig.add_subplot(inner[0, 1])
    draw_silhouettes(ax_sil, coords, ligand=ligand)
    return ax_sil


def panel_d(out_file=None):
    """standalone render of panel D, useful for checking it on its own"""
    ts.apply_style()
    coords, _ = load_coords()
    image = load_render()
    ligand = load_ligand() if SHOW_LIGAND else None
    height = 2.9
    render_aspect, silhouette_aspect = panel_aspects(coords, image)
    width = height * (render_aspect + silhouette_aspect) * 1.05
    fig = plt.figure(figsize=(width, height), layout="constrained")
    draw_panel_d(fig, fig.add_gridspec(1, 1)[0, 0], coords, image, ligand)
    out_file = out_file or OUT_FILE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved panel D to {out_file}")


if __name__ == "__main__":
    panel_d()