"""Figures for the replicate concordance of the reduced example case.
Main figure - three panels in one row, with the bottom of the canvas left free
for the Blender render that becomes panel D:
    A  count collapse   - what each replicate found, per PDB. Circles are local
                          pockets, filled squares the global IDs those pockets
                          map onto, open square the global IDs when all three
                          replicates are pooled. The circles scatter, the squares
                          do not, and the pooled square sits inside the circle
                          range: pooling replicates adds almost no new global
                          pockets even though the raw counts disagree.
    B  occupancy        - core / shell / singleton counted by global ID, by local
                          pocket and by volume.
    C  presence matrix  - which global ID was seen in which replicate, sorted by
                          global ID.
    D  reserved for the render (three replicate surfaces in one shared envelope)
Supplementary figure
    E  Jaccard heatmap  - within-PDB blocks against the between-PDB background
    F  rarefaction      - global IDs recovered from 1, 2, 3 replicates
Colour grammar, one channel per variable: state = hue, taken from taar_style so a
palette change there propagates here. Lightness within that hue carries replicate
number (A, C) and occupancy class (B), always darker = more replicate support.
Genes and states carry no coloured text; labels are black, as elsewhere.
Metrics are not recomputed here - every panel calls the functions in
replicate_analysis_wrapper.
"""

import os
import sys
import time
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from pipeline import pocket_io
import gid_replicate_analysis_wrapper as rep

# config
ROOT = rep.ROOT
# pipeline/global_id_and_comparison.py - subset {state} writes here
STATE_RESULT_DIR_TEMPLATE = os.path.join(ROOT, "global_ID_{state}")
VOXEL_TABLE = "global_pockets_IoU_voxel.csv"
SWEEP_TABLE_TEMPLATE = os.path.join("replicate_analysis", "threshold_sweep_{state}.csv")
SAVING_LOC = os.path.join(ROOT, "replicate_figures")

TAAR_STYLE_PATH = os.path.dirname(os.path.abspath(__file__))
STATES = ["apo", "holo"]
PANELS = {"collapse": True, "occupancy": True, "matrix": True}
SUPPLEMENTARY_PANELS = {"jaccard": True, "rarefaction": True, "spread": True, "sweep": True}

# panel D is the Blender render, dropped into the canvas under the three panels
RESERVE_RENDER_PANEL = True
RENDER_PANEL_HEIGHT_RATIO = 1.30        # only used when AUTO_RENDER_PANEL_HEIGHT is off
RENDER_PANEL_IMAGE = "fig4D_exp_GID_apo_holo_comparison_matching_run_no_singletons.png"
RENDER_PANEL_SEARCH_DIRS = [SAVING_LOC, ROOT, os.path.dirname(os.path.abspath(__file__))]
AUTO_RENDER_PANEL_HEIGHT = True
RENDER_FIT_PASSES = 6
RENDER_FIT_TOLERANCE_INCHES = 0.01

SHOW_PANEL_LETTERS = True
FIGURE_WIDTH = 7.0
MAIN_FIGURE_HEIGHT = 7.2
SUPPLEMENTARY_FIGURE_HEIGHT = 6.4
PANEL_WIDTH_RATIOS = [1.20, 1.05, 0.95]
OUTPUT_FORMATS = ["png", "svg", "pdf"]
DPI = 600

RANDOM_SEED = 42

FALLBACK_STATE_COLORS = {"apo": "#D9B98B", "holo": "#9B5560"}
LIGHT_MIX = 0.44
DARK_MIX = 0.25

NEUTRAL_DARK = "#1A1A1A"
NEUTRAL_GREY = "#57595C"
EMPTY_CELL = "#F2F2F2"

BASE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
TICK_FONT_SIZE = 6
SMALL_FONT_SIZE = 5
LETTER_FONT_SIZE = 9
MARKER_AREA_POINTS = 18
INSIDE_LABEL_MIN_SHARE = 9.0

JACCARD_COLOUR_LIMITS = (0.2, 0.9)

REPLICATE_ORDER = [1, 2, 3]
OCCUPANCY_LABELS = {1: "1 of 3", 2: "2 of 3", 3: "3 of 3"}
SPREAD_GROUP_LABELS = ["orthosteric GID", "other GIDs"]
WEIGHTING_COLUMNS = [("n_global_ids", "by global ID"),
                     ("n_local_pockets", "by local pocket"),
                     ("total_volume", "by volume")]


# palette
def load_state_colors():
    """Pull the state hues from taar_style so a palette change there propagates.
    Falls back to the current hexes if the package is not importable.
    """
    if TAAR_STYLE_PATH and TAAR_STYLE_PATH not in sys.path:
        sys.path.insert(0, TAAR_STYLE_PATH)
    try:
        import taar_style
        return {state: taar_style.STATE_COLORS[state] for state in STATES}
    except (ImportError, KeyError):
        return dict(FALLBACK_STATE_COLORS)


def mix_colours(base_colour, other_colour, fraction):
    base_rgb = np.array(to_rgb(base_colour))
    other_rgb = np.array(to_rgb(other_colour))
    return to_hex((1 - fraction) * base_rgb + fraction * other_rgb)


def build_shade_ramp(base_colour):
    """Three steps of one hue: light = replicate 1 / seen once, dark = 3 / seen thrice."""
    return [mix_colours(base_colour, "#FFFFFF", LIGHT_MIX),
            base_colour,
            mix_colours(base_colour, "#000000", DARK_MIX)]


STATE_COLORS = load_state_colors()
REPLICATE_SHADES = {state: build_shade_ramp(colour) for state, colour in STATE_COLORS.items()}
OCCUPANCY_SHADES = REPLICATE_SHADES


# Data
def load_state_tables():
    """Everything every panel needs, per state, via the wrapper's own functions."""
    state_tables = {}
    for state in STATES:
        state_dir = STATE_RESULT_DIR_TEMPLATE.format(state=state)
        pocket_comparison_df = pocket_io.load_table(state_dir, "pocket_comparison_table")
        occupancy_df = rep.replicate_occupancy(pocket_comparison_df)
        global_id_sets = rep.global_id_sets_per_experiment(pocket_comparison_df)
        voxel_df = pd.read_csv(os.path.join(state_dir, VOXEL_TABLE),
                               usecols=[rep.COLUMN_VOXEL_POCKET, rep.COLUMN_VOXEL_GROUP,
                                        "x", "y", "z"])
        spread_df = rep.gid_spread(voxel_df, pocket_comparison_df)
        orthosteric_global_ids = set()
        if rep.COLUMN_BINDING_SITE in pocket_comparison_df.columns:
            orthosteric_global_ids = set(pocket_comparison_df.loc[
                                             pocket_comparison_df[
                                                 rep.COLUMN_BINDING_SITE] == True, rep.COLUMN_GLOBAL_ID])
        spread_df["is_orthosteric_gid"] = spread_df[rep.COLUMN_GLOBAL_ID].isin(
            orthosteric_global_ids)
        sweep_path = os.path.join(state_dir, SWEEP_TABLE_TEMPLATE.format(state=state))
        sweep_df = pd.read_csv(sweep_path) if os.path.exists(sweep_path) else pd.DataFrame()
        state_tables[state] = {
            "pockets": pocket_comparison_df,
            "counts": rep.counts_per_experiment(pocket_comparison_df),
            "occupancy": occupancy_df,
            "occupancy_summary": rep.occupancy_weighted_summary(pocket_comparison_df, occupancy_df),
            "global_id_sets": global_id_sets,
            "rarefaction": rep.rarefaction(pocket_comparison_df),
            "spread": spread_df,
            "sweep": sweep_df,
        }
    return state_tables


def short_project_label(project_name):
    """holo8JLR -> 8JLR; the state is already carried by the row group and hue."""
    return project_name.replace("apo", "").replace("holo", "")


def apply_axes_style(axes):
    axes.tick_params(labelsize=TICK_FONT_SIZE, length=0, pad=2)
    for spine_name in ("top", "right"):
        axes.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        axes.spines[spine_name].set_color(NEUTRAL_GREY)
        axes.spines[spine_name].set_linewidth(0.6)


def place_panel_letter(axes, letter, x_offset_points=-30, y_offset_points=10):
    if not SHOW_PANEL_LETTERS:
        return
    axes.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                  xytext=(x_offset_points, y_offset_points), textcoords="offset points",
                  fontsize=LETTER_FONT_SIZE, fontweight="bold", color=NEUTRAL_DARK,
                  va="bottom", ha="left")


def place_state_label(axes, state, row_centre, x_offset_points):
    axes.annotate(state, xy=(0, row_centre), xycoords=("axes fraction", "data"),
                  xytext=(x_offset_points, 0), textcoords="offset points", rotation=90,
                  va="center", ha="center", fontsize=LABEL_FONT_SIZE, color=NEUTRAL_DARK)


# Panel A
def draw_collapse_panel(axes, state_tables):
    """Per PDB, what each replicate found. Circles (upper half of the row) are local
    pockets, filled squares (lower half) the distinct global IDs those pockets map
    onto; each replicate sits at its own small vertical offset so two replicates
    reporting the same count stay visible instead of overlapping.
    The open square is the number of distinct global IDs when all three replicates
    are pooled - the union of the three global ID sets. If the replicates were
    finding different pockets, pooling would push it well to the right of the
    filled squares."""
    row_labels, row_positions = [], []
    figure = axes.get_figure()
    marker_diameter_points = np.sqrt(MARKER_AREA_POINTS)
    row_index = 0
    for state in STATES:
        shades = REPLICATE_SHADES[state]
        counts_df = state_tables[state]["counts"]
        pockets_df = state_tables[state]["pockets"]
        for project in sorted(counts_df[rep.COLUMN_PROJECT].unique()):
            project_counts = counts_df[counts_df[rep.COLUMN_PROJECT] == project].sort_values(
                rep.COLUMN_REPLICATE)
            pooled_global = pockets_df.loc[pockets_df[rep.COLUMN_PROJECT] == project,
            rep.COLUMN_GLOBAL_ID].nunique()
            for group_offset, column, marker in ((0.22, "n_local_pockets", "o"),
                                                 (-0.22, "n_global_ids", "s")):
                values = project_counts[column].values
                axes.plot([values.min(), values.max()], [row_index + group_offset] * 2,
                          color=NEUTRAL_GREY, linewidth=0.7, zorder=1)
                # markers only step aside where two replicates report the same count;
                # the step is half a marker, so each stays 50% visible
                positions_sharing_value = {}
                for replicate_position, value in enumerate(values):
                    positions_sharing_value.setdefault(value, []).append(replicate_position)
                for value, replicate_positions in positions_sharing_value.items():
                    n_sharing = len(replicate_positions)
                    for stack_index, replicate_position in enumerate(replicate_positions):
                        offset_points = ((stack_index - (n_sharing - 1) / 2)
                                         * 0.5 * marker_diameter_points)
                        offset_transform = axes.transData + ScaledTranslation(
                            0, offset_points / 72.0, figure.dpi_scale_trans)
                        axes.scatter(value, row_index + group_offset, s=MARKER_AREA_POINTS,
                                     marker=marker, color=shades[replicate_position % len(shades)],
                                     edgecolor=NEUTRAL_DARK, linewidth=0.4, zorder=3,
                                     transform=offset_transform)
            axes.scatter(pooled_global, row_index - 0.22, s=48, marker="s", facecolor="none",
                         edgecolor=NEUTRAL_DARK, linewidth=0.9, zorder=4)
            row_labels.append(short_project_label(project))
            row_positions.append(row_index)
            row_index += 1
        row_index += 0.7

    axes.set_yticks(row_positions)
    axes.set_yticklabels(row_labels, fontsize=TICK_FONT_SIZE)
    axes.invert_yaxis()
    axes.set_xlabel("count per replicate", fontsize=LABEL_FONT_SIZE)
    apply_axes_style(axes)
    for state_index, state in enumerate(STATES):
        place_state_label(axes, state,
                          float(np.mean(row_positions[state_index * 4:(state_index + 1) * 4])),
                          x_offset_points=-30)


# Panel B
def draw_occupancy_panel(axes, state_tables):
    """The same core / shell / singleton split counted three ways. Counting global IDs
    does not represent the pocketome; weighting by local pocket and by volume
    shows how little of it the counts carry. """
    bar_labels, bar_positions = [], []
    position = 0
    for state in STATES:
        shades = OCCUPANCY_SHADES[state]
        summary_df = state_tables[state]["occupancy_summary"].set_index("n_replicates")
        for column_name, weighting_label in WEIGHTING_COLUMNS:
            total = summary_df[column_name].sum()
            left_edge = 0.0
            for occupancy_value in reversed(REPLICATE_ORDER):
                share = 100 * summary_df[column_name].get(occupancy_value, 0.0) / total
                axes.barh(position, share, left=left_edge, height=0.68,
                          color=shades[occupancy_value - 1], edgecolor="white", linewidth=0.5)
                if share >= INSIDE_LABEL_MIN_SHARE:
                    axes.text(left_edge + share / 2, position, "{share:.0f}".format(share=share),
                              ha="center", va="center", fontsize=SMALL_FONT_SIZE,
                              color="white" if occupancy_value == 3 else NEUTRAL_DARK)
                else:
                    axes.text(101.5, position, "{share:.0f}".format(share=share),
                              ha="left", va="center", fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)
                left_edge += share
            bar_labels.append(weighting_label)
            bar_positions.append(position)
            position += 1
        position += 0.7

    axes.set_yticks(bar_positions)
    axes.set_yticklabels(bar_labels, fontsize=TICK_FONT_SIZE)
    axes.invert_yaxis()
    axes.set_xlim(0, 108)
    axes.set_xticks([0, 20, 40, 60, 80, 100])
    axes.set_xlabel("share of total (%)", fontsize=LABEL_FONT_SIZE)
    apply_axes_style(axes)
    for state_index, state in enumerate(STATES):
        place_state_label(axes, state,
                          float(np.mean(bar_positions[state_index * 3:(state_index + 1) * 3])),
                          x_offset_points=-58)


# Panel C
def experiment_columns(pockets_df):
    projects = sorted(pockets_df[rep.COLUMN_PROJECT].unique())
    return projects, [(project, replicate) for project in projects
                      for replicate in sorted(pockets_df.loc[
                                                  pockets_df[rep.COLUMN_PROJECT] == project,
                                                  rep.COLUMN_REPLICATE].unique())]


def draw_presence_matrix(axes, state_tables, state, show_row_labels):
    """Global IDs (rows, in numerical order) against the twelve experiments. A filled
    cell means that global ID was detected in that replicate, in the replicate's
    own shade; grey means it was not. The empty cells are the pocketome's disagreement."""
    pockets_df = state_tables[state]["pockets"]
    shades = REPLICATE_SHADES[state]
    projects, experiments = experiment_columns(pockets_df)
    present = {(project, replicate): set(
        pockets_df.loc[(pockets_df[rep.COLUMN_PROJECT] == project)
                       & (pockets_df[rep.COLUMN_REPLICATE] == replicate), rep.COLUMN_GLOBAL_ID])
        for project, replicate in experiments}
    global_ids = sorted(pockets_df[rep.COLUMN_GLOBAL_ID].unique())

    for row_index, global_id in enumerate(global_ids):
        for column_index, experiment in enumerate(experiments):
            is_present = global_id in present[experiment]
            axes.add_patch(plt.Rectangle(
                (column_index + 0.1, row_index + 0.1), 0.8, 0.8,
                facecolor=shades[column_index % 3] if is_present else EMPTY_CELL,
                edgecolor="none"))

    for project_index in range(1, len(projects)):
        axes.axvline(project_index * 3, color="white", linewidth=1.4)

    axes.set_xlim(0, len(experiments))
    axes.set_ylim(0, len(global_ids))
    axes.invert_yaxis()
    axes.set_xticks([])
    for project_index, project in enumerate(projects):
        axes.annotate(short_project_label(project),
                      xy=(project_index * 3 + 1.5, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -4), textcoords="offset points", rotation=90,
                      ha="center", va="top", fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)
    if show_row_labels:
        axes.set_yticks(np.arange(len(global_ids)) + 0.5)
        axes.set_yticklabels(global_ids, fontsize=SMALL_FONT_SIZE)
        axes.set_ylabel("global ID", fontsize=LABEL_FONT_SIZE)
    else:
        axes.set_yticks([])
    axes.set_title(state, fontsize=LABEL_FONT_SIZE, loc="center", pad=3, color=NEUTRAL_DARK)
    axes.tick_params(labelsize=SMALL_FONT_SIZE, length=0, pad=1)
    for spine in axes.spines.values():
        spine.set_visible(False)


# Panel E
def draw_jaccard_heatmap(axes, state_tables, state, colourbar_axes):
    """Every experiment against every other. The three-by-three blocks on the diagonal
    are replicates of one PDB; everything else is a different receptor. Each
    state gets its own colour bar because each is drawn in its own hue."""
    pockets_df = state_tables[state]["pockets"]
    global_id_sets = state_tables[state]["global_id_sets"]
    experiments = sorted(global_id_sets)
    matrix = np.full((len(experiments), len(experiments)), np.nan)
    for first_index, second_index in itertools.combinations(range(len(experiments)), 2):
        value = rep.jaccard(global_id_sets[experiments[first_index]],
                            global_id_sets[experiments[second_index]])
        matrix[first_index, second_index] = value
        matrix[second_index, first_index] = value

    colour_map = LinearSegmentedColormap.from_list(
        "state_ramp_" + state, ["#FFFFFF", STATE_COLORS[state], NEUTRAL_DARK])
    colour_map.set_bad(EMPTY_CELL)
    image = axes.imshow(matrix, cmap=colour_map, vmin=JACCARD_COLOUR_LIMITS[0],
                        vmax=JACCARD_COLOUR_LIMITS[1], aspect="equal")

    projects, _ = experiment_columns(pockets_df)
    for project_index in range(1, len(projects)):
        axes.axvline(project_index * 3 - 0.5, color="white", linewidth=1.2)
        axes.axhline(project_index * 3 - 0.5, color="white", linewidth=1.2)
    tick_positions = [project_index * 3 + 1 for project_index in range(len(projects))]
    tick_labels = [short_project_label(project) for project in projects]
    axes.set_xticks(tick_positions)
    axes.set_xticklabels(tick_labels, fontsize=TICK_FONT_SIZE)
    axes.set_yticks(tick_positions)
    axes.set_yticklabels(tick_labels, fontsize=TICK_FONT_SIZE)
    axes.set_title(state, fontsize=LABEL_FONT_SIZE, loc="left", x=0.0, pad=4,
                   color=NEUTRAL_DARK)
    axes.tick_params(labelsize=TICK_FONT_SIZE, length=0, pad=2)
    for spine in axes.spines.values():
        spine.set_visible(False)

    colourbar = plt.colorbar(image, cax=colourbar_axes)
    colourbar.set_label("Jaccard, {state}".format(state=state), fontsize=SMALL_FONT_SIZE)
    colourbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, length=2)
    colourbar.outline.set_visible(False)


# Panel F
def draw_rarefaction_panel(axes, state_tables, state):
    """Distinct global IDs recovered from one, two and three replicates, averaged over
    every subset of that size. A flattening curve says three replicates have essentially saturated the pocketome"""
    rarefaction_df = state_tables[state]["rarefaction"]
    shades = REPLICATE_SHADES[state]
    for project in sorted(rarefaction_df["project"].unique()):
        project_df = rarefaction_df[rarefaction_df["project"] == project].sort_values(
            "n_replicates_used")
        axes.errorbar(project_df["n_replicates_used"], project_df["mean_global_ids"],
                      yerr=project_df["sd_global_ids"], color=shades[1], linewidth=1.0,
                      marker="o", markersize=3, markerfacecolor=shades[2],
                      markeredgecolor=NEUTRAL_DARK, markeredgewidth=0.4,
                      elinewidth=0.6, capsize=1.5, zorder=2)
        final_row = project_df.iloc[-1]
        axes.annotate(short_project_label(project),
                      xy=(final_row["n_replicates_used"], final_row["mean_global_ids"]),
                      xytext=(4, 0), textcoords="offset points", va="center",
                      fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)

    axes.set_xticks(REPLICATE_ORDER)
    axes.set_xlim(0.7, 3.9)
    axes.set_xlabel("replicates used", fontsize=LABEL_FONT_SIZE)
    axes.set_ylabel("distinct global IDs", fontsize=LABEL_FONT_SIZE)
    axes.set_title(state, fontsize=LABEL_FONT_SIZE, loc="left", x=0.0, pad=4,
                   color=NEUTRAL_DARK)
    apply_axes_style(axes)


# Panel G
def draw_spread_panel(axes, state_tables, state, random_generator):
    """Silhouette of every local pocket, split by whether its global ID carries the
    orthosteric site. The split is the point: a silhouette measures compactness
    around a centroid, so a global pocket that is one fragmented cavity scores near
    zero however real it is, while compact global pockets score high"""
    spread_df = state_tables[state]["spread"]
    shades = REPLICATE_SHADES[state]
    axes.axvline(0.0, color=NEUTRAL_GREY, linewidth=0.6, zorder=1)

    for group_index, is_orthosteric in enumerate([True, False]):
        group_df = spread_df[spread_df["is_orthosteric_gid"] == is_orthosteric]
        silhouettes = group_df["silhouette"].dropna().values
        jitter = random_generator.uniform(-0.16, 0.16, size=len(silhouettes))
        axes.scatter(silhouettes, group_index + jitter, s=6,
                     color=shades[2] if is_orthosteric else shades[1],
                     edgecolor="none", alpha=0.75, zorder=2)
        median_silhouette = float(np.median(silhouettes))
        axes.plot([median_silhouette, median_silhouette],
                  [group_index - 0.28, group_index + 0.28],
                  color=NEUTRAL_DARK, linewidth=1.2, zorder=3)
        axes.annotate("{median:.2f}".format(median=median_silhouette),
                      xy=(median_silhouette, group_index - 0.32), xytext=(0, 2),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)

    axes.set_yticks([0, 1])
    axes.set_yticklabels(SPREAD_GROUP_LABELS, fontsize=TICK_FONT_SIZE)
    axes.set_ylim(-0.6, 1.6)
    axes.invert_yaxis()
    axes.set_xlim(-0.85, 1.0)
    axes.set_xlabel("silhouette", fontsize=LABEL_FONT_SIZE)
    axes.set_title(state, fontsize=LABEL_FONT_SIZE, loc="left", x=0.0, pad=4,
                   color=NEUTRAL_DARK)
    apply_axes_style(axes)


# Panel H
def draw_sweep_panel(axes, state_tables, state):
    """Percent of local pockets whose occupancy class differs from the reference
    setting, over the parameter grid. Each cell also carries the number of global
    IDs that setting produces, because the two move together: a stricter IoU merges
    less, so more and smaller global pockets, so more pockets change class.
    The reference cell is 0 by construction and is outlined rather than shaded"""
    sweep_df = state_tables[state]["sweep"]
    if sweep_df.empty:
        axes.axis("off")
        axes.text(0.5, 0.5, "threshold_sweep_{state}.csv not found".format(state=state),
                  ha="center", va="center", fontsize=SMALL_FONT_SIZE, color=NEUTRAL_GREY)
        return

    changed_df = sweep_df.pivot_table(index="dilation_radius", columns="iou_thresh",
                                      values="class_changed",
                                      aggfunc=lambda values: 100 * values.mean())
    global_id_df = sweep_df.pivot_table(index="dilation_radius", columns="iou_thresh",
                                        values="n_global_ids_this_setting", aggfunc="first")
    reference_rows = sweep_df.loc[sweep_df["is_reference"]]
    reference_radius = reference_rows["dilation_radius"].iat[0]
    reference_threshold = reference_rows["iou_thresh"].iat[0]

    colour_map = LinearSegmentedColormap.from_list(
        "sweep_ramp_" + state, ["#FFFFFF", STATE_COLORS[state], NEUTRAL_DARK])
    upper_limit = float(np.nanmax(changed_df.values))
    axes.imshow(changed_df.values, cmap=colour_map, vmin=0, vmax=upper_limit, aspect="auto")

    for row_index, dilation_radius in enumerate(changed_df.index):
        for column_index, iou_thresh in enumerate(changed_df.columns):
            percent_changed = changed_df.iloc[row_index, column_index]
            n_global_ids = int(global_id_df.iloc[row_index, column_index])
            is_reference = (dilation_radius == reference_radius
                            and iou_thresh == reference_threshold)
            text_colour = "white" if percent_changed > 0.6 * upper_limit else NEUTRAL_DARK
            axes.text(column_index, row_index - 0.13,
                      "{percent:.1f}%".format(percent=percent_changed),
                      ha="center", va="center", fontsize=SMALL_FONT_SIZE, color=text_colour)
            axes.text(column_index, row_index + 0.20,
                      "{n} GIDs".format(n=n_global_ids),
                      ha="center", va="center", fontsize=SMALL_FONT_SIZE - 0.5,
                      color=text_colour)
            if is_reference:
                axes.add_patch(plt.Rectangle((column_index - 0.5, row_index - 0.5), 1, 1,
                                             facecolor="none", edgecolor=NEUTRAL_DARK,
                                             linewidth=1.4, zorder=3))

    axes.set_xticks(range(len(changed_df.columns)))
    axes.set_xticklabels(changed_df.columns, fontsize=TICK_FONT_SIZE)
    axes.set_yticks(range(len(changed_df.index)))
    axes.set_yticklabels(changed_df.index, fontsize=TICK_FONT_SIZE)
    axes.set_xlabel("IoU threshold", fontsize=LABEL_FONT_SIZE)
    axes.set_ylabel("dilation radius", fontsize=LABEL_FONT_SIZE)
    axes.set_title(state, fontsize=LABEL_FONT_SIZE, loc="left", x=0.0, pad=4,
                   color=NEUTRAL_DARK)
    axes.tick_params(labelsize=TICK_FONT_SIZE, length=0, pad=2)
    for spine in axes.spines.values():
        spine.set_visible(False)


# Legend
def draw_swatch_block(axes, left_x, column_header_rows):
    """One shade ramp per state, three columns, with both meanings of the lightness
    stacked above the columns: occupancy class for B on top, replicate number for
    A and C below. One block, because it is one ramp. Returns the x coordinate just past the block"""
    swatch_width, swatch_height, gap = 0.019, 0.30, 0.004
    for header_row_index, header_row in enumerate(column_header_rows):
        header_y = 1.34 - header_row_index * 0.27
        for column_index, header in enumerate(header_row):
            axes.text(left_x + column_index * (swatch_width + gap) + swatch_width / 2, header_y,
                      header, ha="center", va="top", fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)
    for row_index, state in enumerate(STATES):
        row_y = 0.46 - row_index * 0.38
        axes.text(left_x - 0.006, row_y + swatch_height / 2, state, ha="right", va="center",
                  fontsize=SMALL_FONT_SIZE, color=NEUTRAL_DARK)
        for shade_index, shade in enumerate(REPLICATE_SHADES[state]):
            axes.add_patch(plt.Rectangle(
                (left_x + shade_index * (swatch_width + gap), row_y), swatch_width, swatch_height,
                facecolor=shade, edgecolor="white", linewidth=0.4, transform=axes.transAxes))
    return left_x + len(column_header_rows[0]) * (swatch_width + gap)


def draw_shared_legend(axes):
    """One legend strip for the whole row rather than three axes-level ones, which
    would each reserve horizontal space and blow the inter-panel gaps open.
    Lightness is one ramp with two readings, so it gets one block with two header
    rows: how many replicates hold a global ID (B) over which replicate a mark belongs to (A, C)"""
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")
    draw_swatch_block(axes, 0.055, [["1/3", "2/3", "3/3"], ["r1", "r2", "r3"]])
    marker_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.6,
               markerfacecolor=NEUTRAL_GREY, markeredgecolor=NEUTRAL_DARK,
               markeredgewidth=0.4, label="local pockets, one replicate"),
        Line2D([], [], marker="s", linestyle="none", markersize=3.6,
               markerfacecolor=NEUTRAL_GREY, markeredgecolor=NEUTRAL_DARK,
               markeredgewidth=0.4, label="global IDs, one replicate"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.6,
               markerfacecolor="none", markeredgecolor=NEUTRAL_DARK,
               markeredgewidth=0.9, label="global IDs, three replicates pooled"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.6,
               markerfacecolor=EMPTY_CELL, markeredgecolor="none",
               label="global ID absent from that replicate")]
    axes.legend(handles=marker_handles, fontsize=SMALL_FONT_SIZE, frameon=False,
                loc="upper left", bbox_to_anchor=(0.26, 1.02), ncol=2,
                handletextpad=0.4, labelspacing=0.45, columnspacing=1.0, borderpad=0.0)


# Panel D
def find_render_image():
    """The Blender render for panel D, looked for in the usual places.
    Returns None rather than raising, so the figure still builds (with panel D
    blank) if the render has not been exported yet"""
    if not RENDER_PANEL_IMAGE:
        return None
    if os.path.isabs(RENDER_PANEL_IMAGE) and os.path.isfile(RENDER_PANEL_IMAGE):
        return RENDER_PANEL_IMAGE
    for directory in RENDER_PANEL_SEARCH_DIRS:
        candidate = os.path.join(directory, RENDER_PANEL_IMAGE)
        if os.path.isfile(candidate):
            return candidate
    return None


def draw_render_panel(axes, image_path):
    """The render, filling the reserved row, with no frame and no resampling blur"""
    axes.axis("off")
    if image_path is None:
        return None
    image = plt.imread(image_path)
    axes.imshow(image, aspect="equal", interpolation="antialiased")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return image.shape[1] / image.shape[0]


# Assembly
def save_figure(figure, figure_name):
    os.makedirs(SAVING_LOC, exist_ok=True)
    saved_paths = []
    for output_format in OUTPUT_FORMATS:
        output_path = os.path.join(SAVING_LOC, "{name}.{ext}".format(
            name=figure_name, ext=output_format))
        figure.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        saved_paths.append(output_path)
    plt.close(figure)
    return saved_paths


def compose_main_figure(state_tables, figure_height, render_height_ratio):
    """One layout pass. Returns the figure plus the axes needed to measure it."""
    panel_order = [name for name in ("collapse", "occupancy", "matrix") if PANELS[name]]
    panel_letters = dict(zip(panel_order, ["A", "B", "C", "D"]))
    width_ratios = [PANEL_WIDTH_RATIOS[["collapse", "occupancy", "matrix"].index(name)]
                    for name in panel_order]

    row_heights = [1.0, 0.16]
    if RESERVE_RENDER_PANEL:
        row_heights.append(render_height_ratio)
    figure = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    outer_grid = figure.add_gridspec(len(row_heights), 1, height_ratios=row_heights, hspace=0.30)
    panel_grid = outer_grid[0].subgridspec(1, len(panel_order), width_ratios=width_ratios,
                                           wspace=0.62)

    first_panel_axes = None
    for panel_index, panel_name in enumerate(panel_order):
        if panel_name == "matrix":
            matrix_grid = panel_grid[0, panel_index].subgridspec(1, len(STATES), wspace=0.10)
            for state_index, state in enumerate(STATES):
                axes = figure.add_subplot(matrix_grid[0, state_index])
                draw_presence_matrix(axes, state_tables, state, show_row_labels=state_index == 0)
                if state_index == 0:
                    place_panel_letter(axes, panel_letters[panel_name], x_offset_points=-26)
            first_panel_axes = first_panel_axes or axes
            continue
        axes = figure.add_subplot(panel_grid[0, panel_index])
        first_panel_axes = first_panel_axes or axes
        if panel_name == "collapse":
            draw_collapse_panel(axes, state_tables)
            place_panel_letter(axes, panel_letters[panel_name], x_offset_points=-42)
        else:
            draw_occupancy_panel(axes, state_tables)
            place_panel_letter(axes, panel_letters[panel_name], x_offset_points=-62)

    draw_shared_legend(figure.add_subplot(outer_grid[1]))

    render_axes = None
    image_aspect = None
    if RESERVE_RENDER_PANEL:
        render_axes = figure.add_subplot(outer_grid[2])
        image_aspect = draw_render_panel(render_axes, find_render_image())
        place_panel_letter(render_axes, "D", x_offset_points=-26, y_offset_points=-4)
    return figure, first_panel_axes, render_axes, image_aspect


def build_main_figure(state_tables):
    """A, B, C side by side, a legend strip under them, then the render as panel D"""
    figure_height = MAIN_FIGURE_HEIGHT
    render_height_ratio = RENDER_PANEL_HEIGHT_RATIO
    figure, panel_axes, render_axes, image_aspect = compose_main_figure(
        state_tables, figure_height, render_height_ratio)

    if not (AUTO_RENDER_PANEL_HEIGHT and render_axes is not None and image_aspect):
        if RESERVE_RENDER_PANEL and image_aspect is None:
            print("  panel D: no render found ({name}) - canvas left blank".format(
                name=RENDER_PANEL_IMAGE))
        return figure

    for _ in range(RENDER_FIT_PASSES):
        figure.canvas.draw()
        # original=True is essential: imshow with aspect="equal" shrinks the axes
        # box to the image, so the plain get_position() already matches the image
        # aspect and would report zero error on the first pass. The reserved cell
        # is what needs resizing.
        render_position = render_axes.get_position(original=True)
        render_width_inches = render_position.width * FIGURE_WIDTH
        render_height_inches = render_position.height * figure_height
        panel_height_inches = panel_axes.get_position(original=True).height * figure_height
        needed_height_inches = render_width_inches / image_aspect
        if abs(needed_height_inches - render_height_inches) < RENDER_FIT_TOLERANCE_INCHES:
            break
        render_height_ratio = needed_height_inches / max(panel_height_inches, 1e-6)
        figure_height = figure_height + (needed_height_inches - render_height_inches)
        plt.close(figure)
        figure, panel_axes, render_axes, image_aspect = compose_main_figure(
            state_tables, figure_height, render_height_ratio)

    print("  panel D: {name}".format(name=os.path.basename(find_render_image())))
    print("  panel D fitted to aspect {aspect:.2f} - figure {width:.1f} x {height:.2f} in, "
          "render row ratio {ratio:.2f}".format(
              aspect=image_aspect, width=FIGURE_WIDTH, height=figure_height,
              ratio=render_height_ratio))
    return figure


def build_supplementary_figure(state_tables, random_generator):
    """One row per panel, apo left and holo right, so the Jaccard matrices stay square."""
    row_names = [name for name in ("jaccard", "rarefaction", "spread", "sweep")
                 if SUPPLEMENTARY_PANELS[name]]
    panel_letters = {"jaccard": "E", "rarefaction": "F", "spread": "G", "sweep": "H"}
    row_heights = {"jaccard": 1.25, "rarefaction": 1.0, "spread": 0.75, "sweep": 0.85}
    figure = plt.figure(figsize=(FIGURE_WIDTH, SUPPLEMENTARY_FIGURE_HEIGHT))
    outer_grid = figure.add_gridspec(len(row_names), 1, hspace=0.60,
                                     height_ratios=[row_heights[name] for name in row_names])

    for row_index, row_name in enumerate(row_names):
        # heatmap, its colour bar, then a spacer before the next state
        column_widths, panel_columns = [], []
        for state_index in range(len(STATES)):
            panel_columns.append(len(column_widths))
            column_widths += [1.0, 0.05]
            if state_index < len(STATES) - 1:
                column_widths.append(0.32)
        row_grid = outer_grid[row_index].subgridspec(
            1, len(column_widths), width_ratios=column_widths,
            wspace=0.12 if row_name == "jaccard" else 0.40)
        for state_index, state in enumerate(STATES):
            axes = figure.add_subplot(row_grid[0, panel_columns[state_index]])
            if row_name == "jaccard":
                colourbar_axes = figure.add_subplot(row_grid[0, panel_columns[state_index] + 1])
                draw_jaccard_heatmap(axes, state_tables, state, colourbar_axes)
            elif row_name == "rarefaction":
                draw_rarefaction_panel(axes, state_tables, state)
            elif row_name == "spread":
                draw_spread_panel(axes, state_tables, state, random_generator)
            else:
                draw_sweep_panel(axes, state_tables, state)
            if state_index == 0:
                place_panel_letter(axes, panel_letters[row_name], x_offset_points=-40)
    return figure


def replicate_figures():
    start_time = time.perf_counter()
    plt.rcParams.update({"font.size": BASE_FONT_SIZE, "axes.labelcolor": NEUTRAL_DARK,
                         "text.color": NEUTRAL_DARK, "xtick.color": NEUTRAL_DARK,
                         "ytick.color": NEUTRAL_DARK, "svg.fonttype": "none"})
    random_generator = np.random.default_rng(RANDOM_SEED)
    state_tables = load_state_tables()

    saved_paths = []
    if any(PANELS.values()):
        saved_paths += save_figure(build_main_figure(state_tables), "replicate_concordance_main")
    if any(SUPPLEMENTARY_PANELS.values()):
        saved_paths += save_figure(build_supplementary_figure(state_tables, random_generator),
                                   "replicate_concordance_supplementary")

    elapsed_seconds = time.perf_counter() - start_time
    n_figures = len(saved_paths) // max(len(OUTPUT_FORMATS), 1)
    state_colour_source = "taar_style" if STATE_COLORS != FALLBACK_STATE_COLORS else "local fallback"
    print("replicate figures")
    print("  state hues from {source}".format(source=state_colour_source))
    print("  {n} figure(s) written in {formats}".format(
        n=n_figures, formats=", ".join(OUTPUT_FORMATS)))
    print("  saved to    {loc}".format(loc=SAVING_LOC))
    print("  runtime     {elapsed:.1f} s".format(elapsed=elapsed_seconds))
    return saved_paths


if __name__ == "__main__":
    replicate_figures()