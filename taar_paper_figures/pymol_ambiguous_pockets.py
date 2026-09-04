"""Builds a PyMOL session for the local pockets whose occupancy class changes
somewhere in the threshold sweep - the ones no statistic can settle and that have
to be looked at.

For each ambiguous pocket it puts on screen:
    focus     the pocket itself, in ligand orange
    same_gid  the other members of its REFERENCE global ID, in the state colour
    other_gid the members of the global ID it sits closest to but does not belong
              to (nearest_other_global_id from the spread analysis), light shade
    nearby    every remaining pocket of the same structure within NEIGHBOUR_RADIUS,
              in grey

so the question "is this its own pocket or a member of the neighbouring one" can
be judged by eye. One PyMOL scene per case; the Scenes menu cycles them.

Every object is named pkt_G<gid>_<prj>_<rep>_p<nn>, grouped per global ID and
labelled with its GID at the pocket centroid, so the object panel doubles as a
GID browser and the viewport says which global pocket you are looking at.

Pocket points are written as pseudoatom PDBs, one object per local pocket, taken
straight from global_pockets_IoU_voxel.csv - the same coordinates the clustering
saw, not a re-derivation.

Inputs (per state, from the wrapper and the pipeline):
    replicate_analysis/threshold_sweep_<state>.csv
    replicate_analysis/gid_spread_<state>.csv
    global_pockets_IoU_voxel.csv
    pocket_comparison_table.csv
    optionally a receptor structure per prj, see RECEPTOR_PDB_TEMPLATE

Run from PyCharm or a shell script, then open the written .pml in PyMOL.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # repo root, for `config`
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))  # for `pocket_io`
from config import OUTPUT_DIR, META_ANALYSIS_ROOT
import pocket_io

#config
STATE = "holo"   # "apo" | "holo"

ROOT = META_ANALYSIS_ROOT
# pipeline/global_id_and_comparison.py --subset {state} writes here (was global_ID_{state}_only)
STATE_RESULT_DIR = os.path.join(ROOT, "global_ID_{state}".format(state=STATE))
ANALYSIS_DIR = os.path.join(STATE_RESULT_DIR, "replicate_analysis")

SWEEP_TABLE = os.path.join(ANALYSIS_DIR, "threshold_sweep_{state}.csv".format(state=STATE))
SPREAD_TABLE = os.path.join(ANALYSIS_DIR, "gid_spread_{state}.csv".format(state=STATE))
VOXEL_TABLE = os.path.join(STATE_RESULT_DIR, "global_pockets_IoU_voxel.csv")
POCKET_COMPARISON_TABLE = os.path.join(STATE_RESULT_DIR, "pocket_comparison_table.csv")

# receptor to show behind the pockets; set to None to build the session without it.
# {prj} is already the full holo8PM2 / apo8PM2 name - do not prefix it with {state} again
RECEPTOR_PDB_TEMPLATE = os.path.join(OUTPUT_DIR, "{state}_structures", "{prj}", "1", "aligned_top.pdb")

SAVING_LOC = os.path.join(ANALYSIS_DIR, "pymol_ambiguous_{state}".format(state=STATE))
POCKET_PDB_SUBDIR = "pocket_clouds"
SESSION_SCRIPT = "inspect_ambiguous_pockets_{state}.pml".format(state=STATE)

# a pocket qualifies if its occupancy class differs from the reference setting at
# at least this many of the sweep grid points
MIN_CHANGED_GRID_POINTS = 1
# cap the session; the worst offenders come first
MAX_CASES = 25
# context pockets of the same structure within this centroid distance
NEIGHBOUR_RADIUS = 12.0

# objects share this prefix so a scene can hide them all with one wildcard.
# never disable the GID groups instead - a member of a disabled group stays hidden
# even when the member itself is enabled, which makes every scene blank
OBJECT_PREFIX = "pkt"
POCKET_REPRESENTATION = "spheres"   # "spheres" | "dots" | "nonbonded"
SPHERE_SCALE = 0.25
LABEL_SIZE = 14
RECEPTOR_TRANSPARENCY = 0.75

STATE_COLORS = {"apo": "#D9B98B", "holo": "#9B5560"}
LIGHT_STATE_COLORS = {"apo": "#EAD8BE", "holo": "#C7A0A6"}
FOCUS_COLOR = "#DE7B1E"
NEARBY_COLOR = "#57595C"
RECEPTOR_COLOR = "#B3B3B3"

COLUMN_LOCAL_ID = "Local Pocket ID"
COLUMN_GLOBAL_ID = "Global ID"
COLUMN_PROJECT = "prj"
COLUMN_REPLICATE = "rep"
COLUMN_VOXEL_POCKET = "Pocket ID"
COLUMN_VOXEL_GROUP = "voxel_group_id"
COLUMN_SWEEP_GLOBAL_ID = "global_id_this_setting"


#  io
def load_tables():
    sweep_df = pd.read_csv(SWEEP_TABLE)
    spread_df = pd.read_csv(SPREAD_TABLE)
    voxel_df = pd.read_csv(VOXEL_TABLE, usecols=[COLUMN_VOXEL_POCKET, COLUMN_VOXEL_GROUP,
                                                 "x", "y", "z"])
    pocket_comparison_df = pocket_io.load_table(STATE_RESULT_DIR, "pocket_comparison_table")
    return sweep_df, spread_df, voxel_df, pocket_comparison_df


def pymol_object_name(local_pocket_id, global_id):
    """holo8JLR_2_p09_i3 + GID 4 -> pkt_G04_holo8JLR_2_p09.

    The global ID goes in the name so the object panel sorts and groups by it, and
    so no scene has to explain which pocket belongs where."""
    stem = local_pocket_id.replace(".", "_").rsplit("_i", 1)[0]
    return "{prefix}_G{gid:02d}_{stem}".format(prefix=OBJECT_PREFIX, gid=int(global_id),
                                               stem=stem)


def gid_group_name(global_id):
    return "gid_G{gid:02d}".format(gid=int(global_id))


def hex_to_rgb_list(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    return [round(int(hex_colour[index:index + 2], 16) / 255.0, 4)
            for index in (0, 2, 4)]


# case selection
def ambiguous_pockets(sweep_df, spread_df, pocket_comparison_df):
    """
    One row per pocket that moves occupancy class somewhere in the grid, ordered by
    how unstable it is. Carries the reference class, the competing global ID from
    the spread analysis, and the silhouette, so the session and the worklist csv
    agree.

    If the sweep table carries global_id_this_setting, the set of global IDs the
    pocket takes across the grid comes too - that is the "which GID did it switch
    to" question, which occupancy class alone cannot answer.
    """
    changed_counts = (sweep_df.groupby(COLUMN_LOCAL_ID)["class_changed"].sum()
                      .rename("n_changed_grid_points").reset_index())
    classes_seen = (sweep_df.groupby(COLUMN_LOCAL_ID)["occupancy_class"]
                    .apply(lambda values: sorted(set(int(value) for value in values)))
                    .rename("occupancy_classes_seen").reset_index())
    reference_class = (sweep_df.groupby(COLUMN_LOCAL_ID)["reference_occupancy_class"]
                       .first().rename("reference_occupancy_class").reset_index())

    case_df = changed_counts.merge(classes_seen, on=COLUMN_LOCAL_ID).merge(
        reference_class, on=COLUMN_LOCAL_ID)
    case_df = case_df[case_df["n_changed_grid_points"] >= MIN_CHANGED_GRID_POINTS]

    spread_columns = [COLUMN_LOCAL_ID, COLUMN_GLOBAL_ID, "silhouette",
                      "nearest_other_global_id", "within_distance", "between_distance"]
    case_df = case_df.merge(spread_df[spread_columns], on=COLUMN_LOCAL_ID, how="left")
    case_df = case_df.merge(
        pocket_comparison_df[[COLUMN_LOCAL_ID, COLUMN_PROJECT, COLUMN_REPLICATE,
                              "median_interpolated_volume"]],
        on=COLUMN_LOCAL_ID, how="left")
    case_df["occupancy_classes_seen"] = case_df["occupancy_classes_seen"].apply(
        lambda values: ";".join(str(value) for value in values))

    if COLUMN_SWEEP_GLOBAL_ID in sweep_df.columns:
        grid_global_ids = (sweep_df.groupby(COLUMN_LOCAL_ID)[COLUMN_SWEEP_GLOBAL_ID]
                           .apply(lambda values: ";".join(
                               str(value) for value in sorted(set(int(v) for v in values))))
                           .rename("global_ids_across_grid").reset_index())
        case_df = case_df.merge(grid_global_ids, on=COLUMN_LOCAL_ID, how="left")
    else:
        case_df["global_ids_across_grid"] = "not recorded"

    return case_df.sort_values(["n_changed_grid_points", "silhouette"],
                               ascending=[False, True]).reset_index(drop=True)


def context_for_case(case_row, centroid_df, global_id_by_local_id):
    """
    Split the pockets of the same structure into the three context groups. Same-GID
    and competing-GID members are included whatever their distance, because that is
    the comparison being made; everything else has to be within NEIGHBOUR_RADIUS.
    """
    focus_local_id = case_row[COLUMN_LOCAL_ID]
    focus_centroid = centroid_df.loc[focus_local_id, ["x", "y", "z"]].to_numpy(dtype=float)
    same_project = centroid_df[centroid_df[COLUMN_PROJECT] == case_row[COLUMN_PROJECT]]

    distances = np.linalg.norm(
        same_project[["x", "y", "z"]].to_numpy(dtype=float) - focus_centroid, axis=1)
    distance_by_local_id = dict(zip(same_project.index, distances))

    same_gid, other_gid, nearby = [], [], []
    for local_id in same_project.index:
        if local_id == focus_local_id:
            continue
        global_id = global_id_by_local_id.get(local_id)
        if global_id == case_row[COLUMN_GLOBAL_ID]:
            same_gid.append(local_id)
        elif global_id == case_row["nearest_other_global_id"]:
            other_gid.append(local_id)
        elif distance_by_local_id[local_id] <= NEIGHBOUR_RADIUS:
            nearby.append(local_id)
    return same_gid, other_gid, nearby


# writing
def write_pocket_pdb(object_name, points, output_dir):
    """Point cloud as HETATM pseudoatoms; one file per local pocket."""
    output_path = os.path.join(output_dir, "{name}.pdb".format(name=object_name))
    with open(output_path, "w") as pdb_file:
        for point_index, (x, y, z) in enumerate(points, start=1):
            pdb_file.write(
                "HETATM{serial:>5d}  O   PKT X{residue:>4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n".format(
                    serial=point_index, residue=1, x=x, y=y, z=z))
        pdb_file.write("END\n")
    return output_path


def write_session_script(case_df, context_by_local_id, object_name_by_local_id,
                         global_id_by_local_id, receptor_by_project, output_dir):
    """
    One scene per case. Colours are defined once at the top so the session matches
    the figures, and each scene stores its own message with the numbers that made
    the pocket a case in the first place.

    Scenes hide the pockets with `disable pkt_*` rather than by disabling the GID
    groups: a member of a disabled group stays hidden even when enabled, which
    would leave every scene blank.
    """
    script_path = os.path.join(output_dir, SESSION_SCRIPT)
    pocket_dir = POCKET_PDB_SUBDIR
    lines = [
        "# {state}: {n} pockets whose occupancy class is not stable across the sweep".format(
            state=STATE, n=len(case_df)),
        "# objects are pkt_G<gid>_<prj>_<rep>_p<nn>; the gid_G* groups are for browsing,",
        "# use the Scenes menu to step through the cases",
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, 0",
        "set sphere_scale, {scale}".format(scale=SPHERE_SCALE),
        "set transparency, {value}".format(value=RECEPTOR_TRANSPARENCY),
        "set label_size, {size}".format(size=LABEL_SIZE),
        "set label_color, black",
        "set label_position, (0, 0, 2)",
        "set_color focus_col, {rgb}".format(rgb=hex_to_rgb_list(FOCUS_COLOR)),
        "set_color same_col, {rgb}".format(rgb=hex_to_rgb_list(STATE_COLORS[STATE])),
        "set_color other_col, {rgb}".format(rgb=hex_to_rgb_list(LIGHT_STATE_COLORS[STATE])),
        "set_color nearby_col, {rgb}".format(rgb=hex_to_rgb_list(NEARBY_COLOR)),
        "set_color receptor_col, {rgb}".format(rgb=hex_to_rgb_list(RECEPTOR_COLOR)),
        "",
    ]

    for project, receptor_path in sorted(receptor_by_project.items()):
        lines.append("load {path}, receptor_{prj}".format(path=receptor_path, prj=project))
        lines.append("hide everything, receptor_{prj}".format(prj=project))
        lines.append("show cartoon, receptor_{prj}".format(prj=project))
        lines.append("color receptor_col, receptor_{prj}".format(prj=project))
    lines.append("")

    loaded_local_ids = []
    for _, case_row in case_df.iterrows():
        focus_local_id = case_row[COLUMN_LOCAL_ID]
        same_gid, other_gid, nearby = context_by_local_id[focus_local_id]
        for local_id in [focus_local_id] + same_gid + other_gid + nearby:
            if local_id in loaded_local_ids:
                continue
            object_name = object_name_by_local_id[local_id]
            lines.append("load {dir}/{name}.pdb, {name}".format(
                dir=pocket_dir, name=object_name))
            lines.append("hide everything, {name}".format(name=object_name))
            lines.append("show {rep}, {name}".format(rep=POCKET_REPRESENTATION,
                                                     name=object_name))
            lines.append('label {name} and rank 0, "G{gid}"'.format(
                name=object_name, gid=int(global_id_by_local_id[local_id])))
            loaded_local_ids.append(local_id)
    lines.append("")

    global_ids_present = sorted({int(global_id_by_local_id[local_id])
                                 for local_id in loaded_local_ids})
    for global_id in global_ids_present:
        members = sorted(object_name_by_local_id[local_id] for local_id in loaded_local_ids
                         if int(global_id_by_local_id[local_id]) == global_id)
        lines.append("group {group}, {objects}".format(
            group=gid_group_name(global_id), objects=" ".join(members)))
    lines.append("")

    scene_names = []
    for case_index, case_row in case_df.iterrows():
        focus_local_id = case_row[COLUMN_LOCAL_ID]
        focus_object = object_name_by_local_id[focus_local_id]
        same_gid, other_gid, nearby = context_by_local_id[focus_local_id]
        scene_name = "case{index:02d}_{obj}".format(index=case_index + 1,
                                                    obj=focus_object[len(OBJECT_PREFIX) + 1:])
        scene_names.append(scene_name)

        lines.append("# --- {scene}".format(scene=scene_name))
        lines.append("disable {prefix}_*".format(prefix=OBJECT_PREFIX))
        for members, colour in ((same_gid, "same_col"), (other_gid, "other_col"),
                                (nearby, "nearby_col")):
            for local_id in members:
                object_name = object_name_by_local_id[local_id]
                lines.append("enable {name}".format(name=object_name))
                lines.append("color {colour}, {name}".format(colour=colour, name=object_name))
        lines.append("enable {name}".format(name=focus_object))
        lines.append("color focus_col, {name}".format(name=focus_object))
        lines.append("orient {name}".format(name=focus_object))
        lines.append("zoom {name}, 8".format(name=focus_object))
        message = ("{obj} | reference GID {gid}, GIDs across the grid {grid_gids} | "
                   "occupancy classes seen {classes} at {changed} of the grid points | "
                   "silhouette {silhouette:.2f} | nearest other GID {other} "
                   "({between:.1f} A vs own {within:.1f} A)").format(
            obj=focus_object, gid=int(case_row[COLUMN_GLOBAL_ID]),
            grid_gids=case_row["global_ids_across_grid"],
            classes=case_row["occupancy_classes_seen"],
            changed=int(case_row["n_changed_grid_points"]),
            silhouette=float(case_row["silhouette"]),
            other=int(case_row["nearest_other_global_id"]),
            between=float(case_row["between_distance"]),
            within=float(case_row["within_distance"]))
        lines.append('scene {scene}, store, message="{message}"'.format(
            scene=scene_name, message=message))
        lines.append("")

    lines.append("scene {scene}, recall".format(scene=scene_names[0]))
    with open(script_path, "w") as script_file:
        script_file.write("\n".join(lines) + "\n")
    return script_path


# wrapper
def pymol_ambiguous_pockets():
    start_time = time.perf_counter()
    pocket_dir = os.path.join(SAVING_LOC, POCKET_PDB_SUBDIR)
    os.makedirs(pocket_dir, exist_ok=True)

    sweep_df, spread_df, voxel_df, pocket_comparison_df = load_tables()
    case_df = ambiguous_pockets(sweep_df, spread_df, pocket_comparison_df).head(MAX_CASES)

    centroid_df = voxel_df.groupby(COLUMN_VOXEL_POCKET)[["x", "y", "z"]].mean()
    centroid_df = centroid_df.join(
        pocket_comparison_df.set_index(COLUMN_LOCAL_ID)[[COLUMN_PROJECT, COLUMN_REPLICATE]])
    global_id_by_local_id = dict(zip(pocket_comparison_df[COLUMN_LOCAL_ID],
                                     pocket_comparison_df[COLUMN_GLOBAL_ID]))
    points_by_local_id = {local_id: group[["x", "y", "z"]].values
                          for local_id, group in voxel_df.groupby(COLUMN_VOXEL_POCKET)}

    context_by_local_id = {}
    objects_to_write = set()
    for _, case_row in case_df.iterrows():
        context = context_for_case(case_row, centroid_df, global_id_by_local_id)
        context_by_local_id[case_row[COLUMN_LOCAL_ID]] = context
        objects_to_write.add(case_row[COLUMN_LOCAL_ID])
        for members in context:
            objects_to_write.update(members)

    object_name_by_local_id = {local_id: pymol_object_name(
        local_id, global_id_by_local_id[local_id]) for local_id in objects_to_write}
    for local_id in sorted(objects_to_write):
        write_pocket_pdb(object_name_by_local_id[local_id], points_by_local_id[local_id],
                         pocket_dir)

    receptor_by_project = {}
    if RECEPTOR_PDB_TEMPLATE:
        for project in sorted(case_df[COLUMN_PROJECT].unique()):
            receptor_path = RECEPTOR_PDB_TEMPLATE.format(state=STATE, prj=project)
            if os.path.exists(receptor_path):
                receptor_by_project[project] = receptor_path

    script_path = write_session_script(case_df, context_by_local_id, object_name_by_local_id,
                                       global_id_by_local_id, receptor_by_project, SAVING_LOC)
    worklist_path = os.path.join(SAVING_LOC, "ambiguous_pockets_{state}.csv".format(state=STATE))
    case_df.to_csv(worklist_path, index=False)

    n_cases = len(case_df)
    n_objects = len(objects_to_write)
    n_gid_groups = len({int(global_id_by_local_id[local_id]) for local_id in objects_to_write})
    n_receptors = len(receptor_by_project)
    n_missing_receptors = case_df[COLUMN_PROJECT].nunique() - n_receptors
    grid_ids_recorded = COLUMN_SWEEP_GLOBAL_ID in sweep_df.columns
    elapsed_seconds = time.perf_counter() - start_time

    print("[{state}] {cases} ambiguous pockets, {objects} pocket clouds, "
          "{groups} GID groups".format(state=STATE, cases=n_cases, objects=n_objects,
                                       groups=n_gid_groups))
    print("  receptors   {found} loaded, {missing} not found at RECEPTOR_PDB_TEMPLATE".format(
        found=n_receptors, missing=n_missing_receptors))
    print("  grid GIDs   {status}".format(
        status="recorded" if grid_ids_recorded else
        "missing - add global_id_this_setting to the sweep to see what a pocket switches to"))
    print("  worklist    {path}".format(path=worklist_path))
    print("  open in pymol: {path}".format(path=script_path))
    print("  runtime     {elapsed:.1f} s".format(elapsed=elapsed_seconds))

    return case_df


if __name__ == "__main__":
    pymol_ambiguous_pockets()