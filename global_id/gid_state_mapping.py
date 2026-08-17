"""Maps apo global IDs onto holo global IDs for the reduced example case.

Why this is needed: apo and holo were clustered in separate runs, so the numbers
are independent labellings. Global ID 7 in apo and global ID 7 in holo are not
the same pocket, and nothing in the numbers says so - the counter simply starts
at 1 in each run. This module recovers the correspondence geometrically, from
where each global pocket actually sits, and the resulting table doubles as the
demonstration that the identifier is dataset-scoped.

Method: centroid of every global pocket (mean over all dummy-atom points carrying
that voxel_group_id), then plain Euclidean distance between the
apo and holo centroid sets.

BIDIRECTIONAL MATCHING. The earlier version walked the apo -> holo direction only:
one row per apo global pocket, each pointing at its nearest holo one. That silently
hid part of the picture. A holo global pocket that no apo pocket happened to choose
never appeared in the table at all, so it was impossible to tell apart

    "this holo pocket was tested and has no apo counterpart"      - a real finding
    "this holo pocket was never tested"                           - an artefact

and in the reduced example case that affected holo global IDs 6, 7, 15, 18 and 21.
Both directions are now computed and merged into one table with one row per
candidate pair from either direction, so every global pocket in both states is
accounted for. The direction column records which search produced the pair:

    both        each is the other's nearest - the trustworthy, one-to-one set
    apo->holo   the apo pocket chose this holo one, but not the reverse
    holo->apo   the holo pocket chose this apo one, but not the reverse

reciprocal_nearest and matched keep exactly the meanings they had, so anything
already filtering on them (the Blender figure scripts) needs no change.

Non-reciprocal matches and matches beyond MAX_MATCH_DISTANCE are reported rather
than dropped: a holo pocket claimed by two apo pockets, or an apo pocket with
nothing nearby, is a real finding about the two pocketomes.

Both states are already in one coordinate frame (the receptors are superposed
before pocket detection), so raw centroid distance is meaningful.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# Settings
pd.set_option('expand_frame_repr', False)
pd.options.display.max_colwidth = 500  # long values in columns fully displayed

# config
ROOT = OUTPUT_DIR   # global_ID_{state}_only run directories and this module's output both live here
STATE_RESULT_DIR_TEMPLATE = os.path.join(ROOT, "global_ID_{state}_only")
VOXEL_TABLE = "global_pockets_IoU_voxel.csv"
POCKET_COMPARISON_TABLE = "pocket_comparison_table.csv"

SAVING_LOC = os.path.join(ROOT, "apo_holo_global_id_mapping")

REFERENCE_STATE = "apo"
TARGET_STATE = "holo"

# beyond this centroid separation a pair is reported as unmatched instead of
# being forced into the assignment
MAX_MATCH_DISTANCE = 10.0

COLUMN_GLOBAL_ID = "Global ID"
COLUMN_VOXEL_GROUP = "voxel_group_id"
COLUMN_LOCAL_ID = "Local Pocket ID"
COLUMN_VOLUME = "median_interpolated_volume"
COLUMN_BINDING_SITE = "is_binding_site"


# io
def load_state(state):
    """Point cloud (for centroids) and pocket table (for the descriptive columns)."""
    state_dir = STATE_RESULT_DIR_TEMPLATE.format(state=state)
    voxel_df = pd.read_csv(os.path.join(state_dir, VOXEL_TABLE),
                           usecols=[COLUMN_VOXEL_GROUP, "x", "y", "z"])
    pocket_comparison_df = pd.read_csv(os.path.join(state_dir, POCKET_COMPARISON_TABLE))
    return voxel_df, pocket_comparison_df


def global_pocket_centroids(voxel_df, pocket_comparison_df):
    """
    One row per global ID: centroid, spatial spread, and the descriptors that make
    a match interpretable (how many local pockets it holds, its volume, whether it
    carries the orthosteric flag).
    """
    centroid_df = (voxel_df.groupby(COLUMN_VOXEL_GROUP)[["x", "y", "z"]]
                   .mean()
                   .rename(columns={"x": "centroid_x", "y": "centroid_y", "z": "centroid_z"}))
    radius_of_gyration = (voxel_df.groupby(COLUMN_VOXEL_GROUP)[["x", "y", "z"]]
                          .apply(lambda points: float(np.sqrt(
                              ((points.values - points.values.mean(axis=0)) ** 2).sum(axis=1).mean())))
                          .rename("radius_of_gyration"))
    centroid_df = centroid_df.join(radius_of_gyration)

    pocket_summary = pocket_comparison_df.groupby(COLUMN_GLOBAL_ID).agg(
        n_members=(COLUMN_LOCAL_ID, "size"),
        median_volume=(COLUMN_VOLUME, "median"))
    if COLUMN_BINDING_SITE in pocket_comparison_df.columns:
        binding_site_count = (pocket_comparison_df.groupby(COLUMN_GLOBAL_ID)[COLUMN_BINDING_SITE]
                              .apply(lambda flags: int(flags.fillna(False).sum()))
                              .rename("n_binding_site_members"))
        pocket_summary = pocket_summary.join(binding_site_count)

    centroid_df.index.name = COLUMN_GLOBAL_ID
    return centroid_df.join(pocket_summary, how="left").reset_index()


# matching
def centroid_distance_matrix(reference_centroid_df, target_centroid_df):
    reference_points = reference_centroid_df[["centroid_x", "centroid_y", "centroid_z"]].values
    target_points = target_centroid_df[["centroid_x", "centroid_y", "centroid_z"]].values
    return np.linalg.norm(reference_points[:, None, :] - target_points[None, :, :], axis=2)


def describe_pair(reference_row, target_row, centroid_distance,
                  target_is_nearest_for_reference, reference_is_nearest_for_target):
    """One output row, with the descriptors that make the pair interpretable."""
    is_reciprocal = bool(target_is_nearest_for_reference and reference_is_nearest_for_target)
    if is_reciprocal:
        direction = "both"
    elif target_is_nearest_for_reference:
        direction = "{reference}->{target}".format(reference=REFERENCE_STATE, target=TARGET_STATE)
    else:
        direction = "{target}->{reference}".format(reference=REFERENCE_STATE, target=TARGET_STATE)
    return {
        "{state}_global_id".format(state=REFERENCE_STATE): int(reference_row[COLUMN_GLOBAL_ID]),
        "{state}_global_id".format(state=TARGET_STATE): int(target_row[COLUMN_GLOBAL_ID]),
        "centroid_distance": float(centroid_distance),
        "reciprocal_nearest": is_reciprocal,
        "matched": bool(centroid_distance <= MAX_MATCH_DISTANCE),
        "direction": direction,
        "{state}_chose_{other}".format(state=REFERENCE_STATE, other=TARGET_STATE):
            bool(target_is_nearest_for_reference),
        "{state}_chose_{other}".format(state=TARGET_STATE, other=REFERENCE_STATE):
            bool(reference_is_nearest_for_target),
        "{state}_n_members".format(state=REFERENCE_STATE): int(reference_row["n_members"]),
        "{state}_n_members".format(state=TARGET_STATE): int(target_row["n_members"]),
        "{state}_median_volume".format(state=REFERENCE_STATE): float(reference_row["median_volume"]),
        "{state}_median_volume".format(state=TARGET_STATE): float(target_row["median_volume"]),
        "{state}_radius_of_gyration".format(state=REFERENCE_STATE): float(reference_row["radius_of_gyration"]),
        "{state}_radius_of_gyration".format(state=TARGET_STATE): float(target_row["radius_of_gyration"]),
        "{state}_n_binding_site_members".format(state=REFERENCE_STATE):
            int(reference_row.get("n_binding_site_members", 0) or 0),
        "{state}_n_binding_site_members".format(state=TARGET_STATE):
            int(target_row.get("n_binding_site_members", 0) or 0),
    }


def match_global_ids(reference_centroid_df, target_centroid_df):
    """
    Nearest-neighbour matching on centroid distance, run in BOTH directions, with
    three flags because they mean different things:

      reciprocal_nearest - each global pocket is the other's closest. This is the
                           trustworthy set, and it is one-to-one by construction.
      matched            - within MAX_MATCH_DISTANCE. False means this global
                           pocket has no counterpart in the other state.
      direction          - which search produced the pair. A row that is only
                           holo->apo would have been invisible in the previous
                           apo-anchored version.

    Every global pocket in either state appears in at least one row, so "absent
    from the table" is no longer a possible outcome.
    """
    distance_matrix = centroid_distance_matrix(reference_centroid_df, target_centroid_df)
    nearest_target_for_reference = distance_matrix.argmin(axis=1)
    nearest_reference_for_target = distance_matrix.argmin(axis=0)

    # candidate pairs from both searches, de-duplicated on the index pair
    candidate_pairs = {(reference_index, int(nearest_target_for_reference[reference_index]))
                       for reference_index in range(len(reference_centroid_df))}
    candidate_pairs |= {(int(nearest_reference_for_target[target_index]), target_index)
                        for target_index in range(len(target_centroid_df))}

    mapping_rows = []
    for reference_index, target_index in sorted(candidate_pairs):
        mapping_rows.append(describe_pair(
            reference_centroid_df.iloc[reference_index],
            target_centroid_df.iloc[target_index],
            distance_matrix[reference_index, target_index],
            target_is_nearest_for_reference=bool(
                nearest_target_for_reference[reference_index] == target_index),
            reference_is_nearest_for_target=bool(
                nearest_reference_for_target[target_index] == reference_index)))

    mapping_df = pd.DataFrame(mapping_rows).sort_values("centroid_distance").reset_index(drop=True)
    mapping_df["label_agrees"] = (mapping_df["{state}_global_id".format(state=REFERENCE_STATE)]
                                  == mapping_df["{state}_global_id".format(state=TARGET_STATE)])
    return mapping_df


def unmatched_report(mapping_df, reference_centroid_df, target_centroid_df):
    """Global pockets with no partner inside MAX_MATCH_DISTANCE, per state.

    With one direction only this could not be computed for the target state at all.
    """
    matched_rows = mapping_df[mapping_df["matched"]]
    rows = []
    for state, centroid_df in ((REFERENCE_STATE, reference_centroid_df),
                               (TARGET_STATE, target_centroid_df)):
        column = "{state}_global_id".format(state=state)
        partnered = set(matched_rows[column].astype(int))
        for global_id in centroid_df[COLUMN_GLOBAL_ID].astype(int):
            if global_id not in partnered:
                candidates = mapping_df[mapping_df[column] == global_id]
                nearest = (float(candidates["centroid_distance"].min())
                           if len(candidates) else float("nan"))
                rows.append({"state": state, "global_id": global_id,
                             "nearest_partner_distance": nearest})
    return pd.DataFrame(rows)


# driver
def global_id_state_mapping():
    start_time = time.perf_counter()
    os.makedirs(SAVING_LOC, exist_ok=True)

    reference_voxel_df, reference_pocket_df = load_state(REFERENCE_STATE)
    target_voxel_df, target_pocket_df = load_state(TARGET_STATE)
    reference_centroid_df = global_pocket_centroids(reference_voxel_df, reference_pocket_df)
    target_centroid_df = global_pocket_centroids(target_voxel_df, target_pocket_df)
    mapping_df = match_global_ids(reference_centroid_df, target_centroid_df)
    unmatched_df = unmatched_report(mapping_df, reference_centroid_df, target_centroid_df)

    reference_centroid_df.to_csv(os.path.join(
        SAVING_LOC, "global_pocket_centroids_{state}.csv".format(state=REFERENCE_STATE)), index=False)
    target_centroid_df.to_csv(os.path.join(
        SAVING_LOC, "global_pocket_centroids_{state}.csv".format(state=TARGET_STATE)), index=False)
    mapping_df.to_csv(os.path.join(SAVING_LOC, "global_id_state_mapping.csv"), index=False)
    unmatched_df.to_csv(os.path.join(SAVING_LOC, "global_id_state_unmatched.csv"), index=False)

    n_pairs = len(mapping_df)
    n_matched = int(mapping_df["matched"].sum())
    reciprocal_mask = mapping_df["reciprocal_nearest"] & mapping_df["matched"]
    n_reciprocal = int(reciprocal_mask.sum())
    n_label_agrees = int((mapping_df["label_agrees"] & reciprocal_mask).sum())
    median_distance = float(mapping_df.loc[reciprocal_mask, "centroid_distance"].median())
    elapsed_seconds = time.perf_counter() - start_time

    print("{reference} <-> {target} global ID mapping (bidirectional)".format(
        reference=REFERENCE_STATE, target=TARGET_STATE))
    print("  {reference_count} {reference} and {target_count} {target} global pockets".format(
        reference_count=len(reference_centroid_df), reference=REFERENCE_STATE,
        target_count=len(target_centroid_df), target=TARGET_STATE))
    print("  {pairs} candidate pairs from the two searches combined".format(pairs=n_pairs))
    print("  {matched} of those are within {cutoff:.0f} A".format(
        matched=n_matched, cutoff=MAX_MATCH_DISTANCE))
    print("  {reciprocal} are reciprocal nearest neighbours (the one-to-one set)".format(
        reciprocal=n_reciprocal))
    print("  median centroid distance {distance:.2f} A across the reciprocal set".format(
        distance=median_distance))
    for direction, count in mapping_df["direction"].value_counts().items():
        print("    direction {direction:<12} {count}".format(direction=direction, count=count))
    print("  the two runs happened to give the SAME number in {agree} of {reciprocal} "
          "reciprocal pairs".format(agree=n_label_agrees, reciprocal=n_reciprocal))
    if len(unmatched_df):
        print("  global pockets with NO partner within {cutoff:.0f} A:".format(
            cutoff=MAX_MATCH_DISTANCE))
        for state in (REFERENCE_STATE, TARGET_STATE):
            ids = sorted(unmatched_df.loc[unmatched_df["state"] == state, "global_id"].astype(int))
            print("    {state:<5} {ids}".format(state=state, ids=ids if ids else "none"))
    else:
        print("  every global pocket in both states has a partner")
    print("  saved to    {loc}".format(loc=SAVING_LOC))
    print("  runtime     {elapsed:.1f} s".format(elapsed=elapsed_seconds))

    return mapping_df


if __name__ == "__main__":
    global_id_state_mapping()