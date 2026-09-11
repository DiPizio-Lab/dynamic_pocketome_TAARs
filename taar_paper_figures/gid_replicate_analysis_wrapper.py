"""This wrapper runs the replicate_analysis that is used to argument that the global ID works reliably in my paper
It also is useful to understand the variation in local IDs between replicates"""
"""
replicate_analysis_wrapper.py

Replicate concordance for the reduced example case (4 PDB IDs x 3 replicates,
ONE state). Run from PyCharm or from a shell script on the cluster.

Three blocks, answering three different questions:

    DESCRIPTIVE     - given the global IDs the pipeline assigned, how consistent
                      is the pocketome across replicates? Counts, occupancy
                      spectrum, Jaccard against a permutation null, rarefaction.
                      This describes the simulations, not the identifier.

    GID SPREAD      - do local pockets sharing a global ID actually sit closer to
                      each other than to members of other global IDs? A silhouette
                      on pocket centroids. The clustering works on dilated voxel
                      overlap, not on centroid distance, so this is a partly
                      independent check that the partition is spatially real -
                      and it needs no replicate agreement to make its point.

    THRESHOLD SWEEP - how much of the partition is a consequence of iou_thresh and
                      dilation_radius? Re-clusters the same parsed pockets over a
                      parameter grid and counts how many pockets change occupancy
                      class. The ones that move ARE the ambiguous cases, and are
                      few enough to inspect visually.

Nothing about the ID logic is reimplemented. The sweep calls
voxel_intersection_over_union_global_id directly and uses its return value; the
point cloud is reshaped once (from Step 2.1's already-parsed all_pockets, not
re-parsed from the raw pocket files), and every grid point re-clusters that same
frame.

Outputs land in SAVING_LOC. Both optional blocks have their own RUN_ switch; the
descriptive block needs only pocket_comparison_table.csv, the spread block adds
global_pockets_IoU_voxel.csv, and the sweep needs Step 2.1's all_pockets output
(pipeline/pocket_dataframes.py, default config.META_ANALYSIS_DIR).
"""

import os
import sys
import ast
import time
import itertools

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from config import OUTPUT_DIR, META_ANALYSIS_ROOT, META_ANALYSIS_DIR
import global_id_and_comparison as gid
import pocket_io
from scripts import run_summary
import numpy as np
import pandas as pd

# config
STATE = "apo"   # "apo" | "holo"

ROOT = META_ANALYSIS_ROOT   # global_ID_{state} run directories and this module's output

# pipeline/global_id_and_comparison.py --subset {state} writes here (was global_ID_{state}_only)
STATE_RESULT_DIR = os.path.join(ROOT, "global_ID_{state}".format(state=STATE))
SAVING_LOC = os.path.join(STATE_RESULT_DIR, "replicate_analysis")
THRESHOLD_SWEEP_SCRATCH = os.path.join(SAVING_LOC, "sweep_scratch")

POCKET_COMPARISON_TABLE = os.path.join(STATE_RESULT_DIR, "pocket_comparison_table.csv")
VOXEL_TABLE = os.path.join(STATE_RESULT_DIR, "global_pockets_IoU_voxel.csv")
GLOBAL_ID_MAP_FILE = "local_to_globalVoxelID.txt"

RUN_GID_SPREAD = True
RUN_THRESHOLD_SWEEP = True   # re-clusters once per grid point, not free

# MetaAnalysis parameters - must match the full run these results came from
ISOVALUE = 3.0
MIN_ATOMS = 10
DBSCAN = False
VERBOSE = False
VOXEL_SIZE = 1.0
IOU_THRESH = 0.3
DILATION_RADIUS = 5
MIN_CLUSTER_SIZE = 2

# grid for the sensitivity sweep; the pair above is the reference everything is
# compared against and is always included
SWEEP_IOU_THRESHOLDS = [0.2, 0.3, 0.4]
SWEEP_DILATION_RADII = [3, 4, 5]

# global IDs to drop before the overlap statistics, e.g. [4] to check how much of
# the Jaccard signal rests on the single largest cluster.
EXCLUDE_GLOBAL_IDS = []

N_PERMUTATIONS = 2000
RANDOM_SEED = 42

COLUMN_LOCAL_ID = "Local Pocket ID"
COLUMN_GLOBAL_ID = "Global ID"
COLUMN_PROJECT = "prj"
COLUMN_REPLICATE = "rep"
COLUMN_VOLUME = "median_interpolated_volume"
COLUMN_BINDING_SITE = "is_orthosteric"
COLUMN_VOXEL_GROUP = "voxel_group_id"
COLUMN_VOXEL_POCKET = "Pocket ID"


# io
def load_pocket_comparison_table():
    """The finished global ID result for this state: one row per local pocket."""
    pocket_comparison_df = pocket_io.load_table(STATE_RESULT_DIR, "pocket_comparison_table")
    if EXCLUDE_GLOBAL_IDS:
        pocket_comparison_df = pocket_comparison_df[
            ~pocket_comparison_df[COLUMN_GLOBAL_ID].isin(EXCLUDE_GLOBAL_IDS)]
    return pocket_comparison_df


def read_global_id_map(result_dir):
    """
    Parse local_to_globalVoxelID.txt, same format global_id_wrapper_paper.py inverts:

        1: ['holo8ITF_1_p02_i3', 'holo8ITF_2_p02_i3', ...]

    Returns {local pocket ID: global ID}.
    """
    local_to_global = {}
    with open(os.path.join(result_dir, GLOBAL_ID_MAP_FILE)) as map_file:
        for line in map_file:
            line = line.strip()
            if ":" not in line:
                continue
            global_id_string, member_string = line.split(":", 1)
            for local_id in ast.literal_eval(member_string.strip()):
                local_to_global[local_id] = int(global_id_string.strip())
    return local_to_global


def list_experiments(pocket_comparison_df):
    """Every (project, replicate) pair present, sorted."""
    experiment_df = pocket_comparison_df[[COLUMN_PROJECT, COLUMN_REPLICATE]].drop_duplicates()
    return sorted(experiment_df.itertuples(index=False, name=None))


def pocket_dir_for(project, replicate):
    return POCKET_DIR_TEMPLATE.format(state=STATE, prj=project, rep=replicate)


# descriptive
def counts_per_experiment(pocket_comparison_df):
    """
    Local pockets detected vs distinct global IDs touched, per experiment.
    This is the instability being explained, not evidence on its own.
    """
    return (pocket_comparison_df
            .groupby([COLUMN_PROJECT, COLUMN_REPLICATE])
            .agg(n_local_pockets=(COLUMN_LOCAL_ID, "size"),
                 n_global_ids=(COLUMN_GLOBAL_ID, "nunique"))
            .reset_index())


def replicate_occupancy(pocket_comparison_df):
    """
    For each (project, global ID): in how many replicates does it occur?
    3 = core, 2 = shell, 1 = singleton.
    """
    occupancy_df = (pocket_comparison_df
                    .groupby([COLUMN_PROJECT, COLUMN_GLOBAL_ID])[COLUMN_REPLICATE]
                    .nunique()
                    .rename("n_replicates")
                    .reset_index())
    occupancy_df["occupancy_class"] = occupancy_df["n_replicates"].map(
        {3: "core", 2: "shell", 1: "singleton"})
    return occupancy_df


def occupancy_weighted_summary(pocket_comparison_df, occupancy_df):
    """
    The occupancy spectrum counted three ways. Counting global pockets makes the
    pocketome look unreliable; weighting by local pocket and by volume shows how
    little of it the unreliable part carries. Report all three columns together or
    the number misleads.

    The three counts live on DIFFERENT units and cannot come from one groupby:

      n_global_ids     one per (project, global ID) - a global pocket in one PDB.
                       These are the rows of occupancy_df, which is already at that
                       granularity, so they are counted there.
      n_local_pockets  one per detected pocket. These are the rows of the merged
                       table, several of which can share a global pocket.
      total_volume     summed over those local pockets.

    Previously both counts were taken as .size() on the merged table, which counts
    rows whichever column is named - so n_global_ids silently returned the local
    pocket count and the two were identical by construction. That flattened the
    whole point of the comparison: a global pocket seen in three replicates holds
    at least three local pockets while a singleton holds one, so the core class
    MUST take a larger share by local pocket than by global pocket.
    """
    merged_df = pocket_comparison_df.merge(
        occupancy_df[[COLUMN_PROJECT, COLUMN_GLOBAL_ID, "n_replicates"]],
        on=[COLUMN_PROJECT, COLUMN_GLOBAL_ID])

    global_pocket_counts = (occupancy_df.groupby("n_replicates")
                            .size()
                            .rename("n_global_ids"))
    local_pocket_summary = (merged_df.groupby("n_replicates")
                            .agg(n_local_pockets=(COLUMN_LOCAL_ID, "size"),
                                 total_volume=(COLUMN_VOLUME, "sum"),
                                 median_volume=(COLUMN_VOLUME, "median")))

    summary_df = local_pocket_summary.join(global_pocket_counts, how="outer")
    summary_df["n_global_ids"] = summary_df["n_global_ids"].fillna(0).astype(int)
    summary_df["n_local_pockets"] = summary_df["n_local_pockets"].fillna(0).astype(int)
    summary_df[["total_volume", "median_volume"]] = summary_df[
        ["total_volume", "median_volume"]].fillna(0.0)

    for column_name in ("n_global_ids", "n_local_pockets", "total_volume"):
        summary_df[column_name + "_percent"] = (
            100 * summary_df[column_name] / summary_df[column_name].sum())
    return summary_df.reset_index()[
        ["n_replicates", "n_global_ids", "n_local_pockets", "total_volume", "median_volume",
         "n_global_ids_percent", "n_local_pockets_percent", "total_volume_percent"]]


def global_id_composition(pocket_comparison_df):
    """
    One row per global ID, to tell a genuinely large conserved pocket from a
    single-linkage chain. A conserved orthosteric site should appear once per
    experiment in every experiment; a chained cluster shows several members
    inside the SAME replicate, spread far apart.

    members_per_experiment  - mean local pockets contributed by one experiment
    is_binding_site_fraction - share of members flagged orthosteric (from
                               orthosteric_filter_ligand_based.classify_binding_site)
    """
    composition_rows = []
    for global_id, global_id_df in pocket_comparison_df.groupby(COLUMN_GLOBAL_ID):
        members_per_experiment = global_id_df.groupby(
            [COLUMN_PROJECT, COLUMN_REPLICATE]).size()
        binding_site_fraction = np.nan
        if COLUMN_BINDING_SITE in global_id_df.columns:
            binding_site_flags = global_id_df[COLUMN_BINDING_SITE]
            if binding_site_flags.notna().any():
                binding_site_fraction = float(binding_site_flags.fillna(False).mean())
        composition_rows.append({
            COLUMN_GLOBAL_ID: global_id,
            "n_members": len(global_id_df),
            "n_experiments": int(members_per_experiment.size),
            "mean_members_per_experiment": float(members_per_experiment.mean()),
            "max_members_per_experiment": int(members_per_experiment.max()),
            "median_volume": float(global_id_df[COLUMN_VOLUME].median()),
            "max_volume": float(global_id_df[COLUMN_VOLUME].max()),
            "is_binding_site_fraction": binding_site_fraction,
        })
    return pd.DataFrame(composition_rows).sort_values("n_members", ascending=False)


def global_id_sets_per_experiment(pocket_comparison_df):
    """{(project, replicate): set of global IDs present}."""
    return {experiment: set(experiment_df[COLUMN_GLOBAL_ID])
            for experiment, experiment_df
            in pocket_comparison_df.groupby([COLUMN_PROJECT, COLUMN_REPLICATE])}


def jaccard(first_set, second_set):
    if not first_set or not second_set:
        return 0.0
    return len(first_set & second_set) / len(first_set | second_set)


def pairwise_jaccard(global_id_sets):
    """
    Every pair of experiments, flagged same-PDB or not. Same-PDB pairs are
    replicates of one simulation; different-PDB pairs are different receptors.
    """
    jaccard_rows = []
    for first_experiment, second_experiment in itertools.combinations(sorted(global_id_sets), 2):
        jaccard_rows.append({
            "project_a": first_experiment[0], "replicate_a": first_experiment[1],
            "project_b": second_experiment[0], "replicate_b": second_experiment[1],
            "same_project": first_experiment[0] == second_experiment[0],
            "jaccard": jaccard(global_id_sets[first_experiment],
                               global_id_sets[second_experiment]),
        })
    return pd.DataFrame(jaccard_rows)


def jaccard_permutation_null(global_id_sets, random_generator):
    """
    How high would Jaccard be if the global IDs carried no information at all?

    Each experiment keeps its own number of global IDs but draws WHICH ones at
    random from the pool present in the dataset. With 22 IDs in the pool and
    ~12 per experiment, two random draws already share about a third of their
    members by pure arithmetic, so an observed value is only meaningful once
    that floor is known. Compare the within-project mean against the 95th
    percentile returned here.
    """
    global_id_pool = sorted({global_id
                             for global_id_set in global_id_sets.values()
                             for global_id in global_id_set})
    experiments = sorted(global_id_sets)
    permuted_values = []
    for _ in range(N_PERMUTATIONS):
        permuted_sets = {experiment: set(random_generator.choice(
            global_id_pool, size=len(global_id_sets[experiment]), replace=False))
            for experiment in experiments}
        permuted_values.extend(
            jaccard(permuted_sets[first_experiment], permuted_sets[second_experiment])
            for first_experiment, second_experiment
            in itertools.combinations(experiments, 2))
    return np.array(permuted_values)


def rarefaction(pocket_comparison_df):
    """
    How many distinct global IDs are recovered from 1, 2 and 3 replicates,
    averaged over every subset of that size?

    Only answers "are three replicates enough". If the third replicate still
    adds many new global IDs the pocketome is undersampled; if it adds almost
    none, three replicates already saturate it and the pockets that are missing
    are not being hidden by sampling depth.
    """
    rarefaction_rows = []
    for project, project_df in pocket_comparison_df.groupby(COLUMN_PROJECT):
        replicates = sorted(project_df[COLUMN_REPLICATE].unique())
        for subset_size in range(1, len(replicates) + 1):
            subset_counts = [
                project_df[project_df[COLUMN_REPLICATE].isin(replicate_subset)][
                    COLUMN_GLOBAL_ID].nunique()
                for replicate_subset in itertools.combinations(replicates, subset_size)]
            rarefaction_rows.append({
                "project": project,
                "n_replicates_used": subset_size,
                "mean_global_ids": float(np.mean(subset_counts)),
                "sd_global_ids": (float(np.std(subset_counts, ddof=1))
                                  if len(subset_counts) > 1 else 0.0),
            })
    return pd.DataFrame(rarefaction_rows)


# gid spread
def pocket_centroids(voxel_df):
    """One centroid per local pocket, plus the global ID it was assigned to."""
    centroid_df = (voxel_df.groupby([COLUMN_VOXEL_POCKET, COLUMN_VOXEL_GROUP])[["x", "y", "z"]]
                   .mean()
                   .reset_index()
                   .rename(columns={COLUMN_VOXEL_POCKET: COLUMN_LOCAL_ID,
                                    COLUMN_VOXEL_GROUP: COLUMN_GLOBAL_ID}))
    return centroid_df


def gid_spread(voxel_df, pocket_comparison_df):
    """
    Silhouette of every local pocket against the global ID partition, on centroids.

        within_distance  - mean distance to the other members of its own global ID
        nearest_other    - the global ID whose members it is on average closest to
        between_distance - that mean distance
        silhouette       - (between - within) / max(between, within)

    Positive means the pocket sits closer to its own global pocket than to any
    other, i.e. the partition separates in space. Negative means it would fit some
    other global ID better and is worth looking at. Global IDs with a single member
    get silhouette NaN, since "distance to the rest of my cluster" is undefined.

    The clustering itself operates on dilated voxel overlap rather than on centroid
    distance, so this is not simply the clustering criterion restated.

    Returns one long dataframe, one row per local pocket.
    """
    centroid_df = pocket_centroids(voxel_df)
    volume_by_local_id = dict(zip(pocket_comparison_df[COLUMN_LOCAL_ID],
                                  pocket_comparison_df[COLUMN_VOLUME]))
    binding_site_by_local_id = {}
    if COLUMN_BINDING_SITE in pocket_comparison_df.columns:
        binding_site_by_local_id = dict(zip(pocket_comparison_df[COLUMN_LOCAL_ID],
                                            pocket_comparison_df[COLUMN_BINDING_SITE]))

    local_ids = centroid_df[COLUMN_LOCAL_ID].values
    global_ids = centroid_df[COLUMN_GLOBAL_ID].values
    coordinates = centroid_df[["x", "y", "z"]].values
    distance_matrix = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    unique_global_ids = np.unique(global_ids)

    spread_rows = []
    for pocket_index in range(len(local_ids)):
        own_global_id = global_ids[pocket_index]
        mean_distance_to = {}
        for candidate_global_id in unique_global_ids:
            member_mask = global_ids == candidate_global_id
            if candidate_global_id == own_global_id:
                member_mask = member_mask.copy()
                member_mask[pocket_index] = False
            if not member_mask.any():
                continue
            mean_distance_to[candidate_global_id] = float(
                distance_matrix[pocket_index, member_mask].mean())

        within_distance = mean_distance_to.get(own_global_id, np.nan)
        other_distances = {global_id: distance for global_id, distance
                           in mean_distance_to.items() if global_id != own_global_id}
        nearest_other_global_id = min(other_distances, key=other_distances.get)
        between_distance = other_distances[nearest_other_global_id]
        if np.isnan(within_distance):
            silhouette = np.nan
        else:
            silhouette = ((between_distance - within_distance)
                          / max(between_distance, within_distance))

        local_id = local_ids[pocket_index]
        spread_rows.append({
            COLUMN_LOCAL_ID: local_id,
            COLUMN_GLOBAL_ID: int(own_global_id),
            "within_distance": within_distance,
            "nearest_other_global_id": int(nearest_other_global_id),
            "between_distance": between_distance,
            "silhouette": silhouette,
            "median_volume": volume_by_local_id.get(local_id, np.nan),
            COLUMN_BINDING_SITE: binding_site_by_local_id.get(local_id, np.nan),
        })
    return pd.DataFrame(spread_rows)


# threshold sweep
def point_df_for_sweep():
    """The reshaped point cloud the sweep re-clusters at every grid point, taken straight
    from Step 2.1's already-parsed all_pockets (pipeline/pocket_dataframes.py) instead of
    re-parsing the raw pocket files a second time -- pock_file_parser() already ran once
    to produce that output, so this only filters to STATE and reshapes."""
    all_pockets = pocket_io.load_table(META_ANALYSIS_DIR, 'all_pockets')
    state_pockets = all_pockets[all_pockets['state'] == STATE]
    return gid.pocket_point_dataframe(state_pockets)


def occupancy_class_by_local_id(assignment_df):
    """
    Per local pocket: in how many replicates of its own structure does its global ID
    occur? This is the quantity the sweep asks about, since a pocket changing global
    ID number is meaningless across runs but changing occupancy class is not.
    """
    occupancy = (assignment_df.groupby([COLUMN_PROJECT, COLUMN_GLOBAL_ID])[COLUMN_REPLICATE]
                 .nunique().rename("n_replicates").reset_index())
    merged_df = assignment_df.merge(occupancy, on=[COLUMN_PROJECT, COLUMN_GLOBAL_ID])
    return dict(zip(merged_df[COLUMN_LOCAL_ID], merged_df["n_replicates"]))


def threshold_sweep(pocket_comparison_df):
    """
    Re-cluster the same parsed pockets across the iou_thresh x dilation_radius grid
    and record what each local pocket's occupancy class becomes.

    Global ID numbers are not comparable between grid points, so nothing compares
    them; the comparison is occupancy class, which is label-free. A pocket whose
    class is the same everywhere is robust to the parameter choice; a pocket that
    moves is genuinely ambiguous and is exactly what should be opened in Blender.

    The pockets are parsed and reshaped once; each grid point only redoes the
    voxelisation and the graph, which is the cheap part.

    Returns one long dataframe, one row per local pocket per grid point.
    """
    os.makedirs(THRESHOLD_SWEEP_SCRATCH, exist_ok=True)
    point_df = point_df_for_sweep()

    grid_points = sorted(set(
        [(IOU_THRESH, DILATION_RADIUS)]
        + [(iou_thresh, dilation_radius) for iou_thresh in SWEEP_IOU_THRESHOLDS
           for dilation_radius in SWEEP_DILATION_RADII]))

    reference_class_by_local_id = {}
    sweep_rows = []
    for iou_thresh, dilation_radius in grid_points:
        voxelized_df = gid.voxel_intersection_over_union_global_id(
            THRESHOLD_SWEEP_SCRATCH, point_df=point_df, voxel_size=VOXEL_SIZE,
            iou_thresh=iou_thresh, dilation_radius=dilation_radius,
            min_cluster_size=MIN_CLUSTER_SIZE, make_qc_plot=False)
        assignment_df = (voxelized_df[[COLUMN_VOXEL_POCKET, COLUMN_VOXEL_GROUP,
                                       COLUMN_PROJECT, COLUMN_REPLICATE]]
                         .drop_duplicates()
                         .rename(columns={COLUMN_VOXEL_POCKET: COLUMN_LOCAL_ID,
                                          COLUMN_VOXEL_GROUP: COLUMN_GLOBAL_ID}))
        global_id_by_local_id_this_setting = dict(zip(assignment_df[COLUMN_LOCAL_ID],
                                                      assignment_df[COLUMN_GLOBAL_ID]))
        class_by_local_id = occupancy_class_by_local_id(assignment_df)
        is_reference = (iou_thresh == IOU_THRESH and dilation_radius == DILATION_RADIUS)
        if is_reference:
            reference_class_by_local_id = class_by_local_id

        n_global_ids = assignment_df[COLUMN_GLOBAL_ID].nunique()
        for local_id, occupancy_class in class_by_local_id.items():
            sweep_rows.append({
                "iou_thresh": iou_thresh,
                "dilation_radius": dilation_radius,
                "is_reference": is_reference,
                COLUMN_LOCAL_ID: local_id,
                "n_global_ids_this_setting": n_global_ids,
                "global_id_this_setting": int(global_id_by_local_id_this_setting[local_id]),
                "occupancy_class": int(occupancy_class),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df["reference_occupancy_class"] = sweep_df[COLUMN_LOCAL_ID].map(
        reference_class_by_local_id)
    sweep_df["class_changed"] = (sweep_df["occupancy_class"]
                                 != sweep_df["reference_occupancy_class"])
    return sweep_df



# wrapper
def replicate_analysis():
    start_time = time.perf_counter()
    os.makedirs(SAVING_LOC, exist_ok=True)
    random_generator = np.random.default_rng(RANDOM_SEED)

    pocket_comparison_df = load_pocket_comparison_table()
    counts_df = counts_per_experiment(pocket_comparison_df)
    occupancy_df = replicate_occupancy(pocket_comparison_df)
    occupancy_summary_df = occupancy_weighted_summary(pocket_comparison_df, occupancy_df)
    composition_df = global_id_composition(pocket_comparison_df)
    global_id_sets = global_id_sets_per_experiment(pocket_comparison_df)
    jaccard_df = pairwise_jaccard(global_id_sets)
    permuted_jaccard = jaccard_permutation_null(global_id_sets, random_generator)
    rarefaction_df = rarefaction(pocket_comparison_df)

    result_tables = {
        "counts_per_experiment": counts_df,
        "occupancy": occupancy_df,
        "occupancy_summary": occupancy_summary_df,
        "global_id_composition": composition_df,
        "jaccard_pairs": jaccard_df,
        "rarefaction": rarefaction_df,
    }

    spread_df = pd.DataFrame()
    if RUN_GID_SPREAD:
        voxel_df = pd.read_csv(VOXEL_TABLE, usecols=[COLUMN_VOXEL_POCKET, COLUMN_VOXEL_GROUP,
                                                     "x", "y", "z"])
        spread_df = gid_spread(voxel_df, pocket_comparison_df)
        result_tables["gid_spread"] = spread_df

    sweep_df = pd.DataFrame()
    if RUN_THRESHOLD_SWEEP:
        sweep_df = threshold_sweep(pocket_comparison_df)
        result_tables["threshold_sweep"] = sweep_df

    for table_name, table_df in result_tables.items():
        table_df.to_csv(os.path.join(SAVING_LOC, "{name}_{state}.csv".format(
            name=table_name, state=STATE)), index=False)

    n_local_pockets = pocket_comparison_df[COLUMN_LOCAL_ID].nunique()
    n_global_ids = pocket_comparison_df[COLUMN_GLOBAL_ID].nunique()
    n_core = int((occupancy_df["n_replicates"] == 3).sum())
    n_shell = int((occupancy_df["n_replicates"] == 2).sum())
    n_singleton = int((occupancy_df["n_replicates"] == 1).sum())
    core_summary_row = occupancy_summary_df[occupancy_summary_df["n_replicates"] == 3]
    core_pocket_percent = float(core_summary_row["n_local_pockets_percent"].iat[0])
    core_volume_percent = float(core_summary_row["total_volume_percent"].iat[0])
    within_project_jaccard = jaccard_df.loc[jaccard_df["same_project"], "jaccard"].mean()
    between_project_jaccard = jaccard_df.loc[~jaccard_df["same_project"], "jaccard"].mean()
    null_jaccard_p95 = float(np.percentile(permuted_jaccard, 95))
    largest_global_id_row = composition_df.iloc[0]
    largest_global_id = int(largest_global_id_row[COLUMN_GLOBAL_ID])
    largest_members_per_experiment = float(largest_global_id_row["mean_members_per_experiment"])
    elapsed_seconds = time.perf_counter() - start_time

    print("[{state}] {n_local} local pockets -> {n_global} global IDs".format(
        state=STATE, n_local=n_local_pockets, n_global=n_global_ids))
    print("  occupancy   core {core} | shell {shell} | singleton {singleton}".format(
        core=n_core, shell=n_shell, singleton=n_singleton))
    print("  core holds  {pockets:.0f}% of pockets, {volume:.0f}% of volume".format(
        pockets=core_pocket_percent, volume=core_volume_percent))
    print("  jaccard     within {within:.3f} | between {between:.3f} | null p95 {null:.3f}".format(
        within=within_project_jaccard, between=between_project_jaccard, null=null_jaccard_p95))
    print("  largest ID  G{global_id} with {members:.2f} members per experiment".format(
        global_id=largest_global_id, members=largest_members_per_experiment))
    if RUN_GID_SPREAD:
        scored_spread = spread_df["silhouette"].dropna()
        median_silhouette = float(scored_spread.median())
        n_negative = int((scored_spread < 0).sum())
        median_within = float(spread_df["within_distance"].median(skipna=True))
        median_between = float(spread_df["between_distance"].median(skipna=True))
        print("  gid spread  median silhouette {silhouette:.3f} | "
              "within {within:.1f} A vs nearest other {between:.1f} A | "
              "{negative} of {scored} pockets negative".format(
                  silhouette=median_silhouette, within=median_within,
                  between=median_between, negative=n_negative, scored=len(scored_spread)))
    if RUN_THRESHOLD_SWEEP:
        n_grid_points = sweep_df[["iou_thresh", "dilation_radius"]].drop_duplicates().shape[0]
        changed_per_pocket = sweep_df.groupby(COLUMN_LOCAL_ID)["class_changed"].any()
        n_ambiguous = int(changed_per_pocket.sum())
        n_swept = int(changed_per_pocket.size)
        print("  sweep       {points} grid points | {ambiguous} of {swept} pockets change "
              "occupancy class somewhere".format(points=n_grid_points, ambiguous=n_ambiguous,
                                                 swept=n_swept))
    print("  saved to    {loc}".format(loc=SAVING_LOC))
    print("  runtime     {elapsed:.1f} s".format(elapsed=elapsed_seconds))

    return result_tables



if __name__ == "__main__":
    with run_summary.stage(*run_summary.AGGREGATE_KEY, 'optional_analyses'):
        replicate_analysis()