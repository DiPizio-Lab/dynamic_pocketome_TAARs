"""Solve, for every median-frame pocket, the sphere radius that reproduces its
reported volume - and write the answer to a CSV the Blender scripts read.

WHY THIS EXISTS
---------------
The points in the median-frame PDBs are fpocket alpha-sphere centres, and the last
column is the alpha-sphere radius. That radius must NOT be used to draw the pocket.
An alpha sphere touches four protein atoms, so it reaches from the cavity out to
the atom centres and swallows the surrounding protein: drawn at its own radius, the
example pocket g1_apo8ITF_1_p09 renders at 356 A^3 against a reported 72.87 A^3,
almost five times too large. A single guessed radius is no better - 3.0 A gives
2.7x on that pocket and would give something else entirely on the next one.

So the radius is not guessed at all. mdpocket already reports what the cavity is
worth, and blender_global_id_prep.py writes it into the REMARK line of every file:

    REMARK global 1 apo8ITF rep1 p09 frame 786 vol=72.87 median=72.79

This script inverts the drawing: it searches for the radius r at which spheres of
radius r, centred on that pocket's alpha-sphere centres, merge into a body whose
volume equals the reported one. Every blob in the figure then has the volume the
data says it has, and nothing in the visualisation overstates the pocket.

HOW THE VOLUME IS MEASURED
--------------------------
A voxel grid is laid over the pocket once and the distance from every voxel to its
nearest alpha-sphere centre is computed once. After that, the volume of the union
at any radius r is just the number of voxels whose nearest-centre distance is below
r - so the bisection search costs nothing beyond that single pass. Grid spacing
adapts to keep the cell count bounded, and the achieved volume is written out
alongside the target so any pocket where the search did badly is visible.

WHAT ELSE IT REPORTS
--------------------
  n_atoms vs n_unique   The PDBs repeat every alpha sphere 2-6 times (49 lines,
                        13 distinct coordinates in the example). Duplicates are
                        coincident so they do not change the geometry, but they do
                        multiply the work downstream. Only unique centres are used
                        here, and the Blender scripts should deduplicate too.

  n_components          How many separate pieces the pocket falls into AT THE
                        CALIBRATED RADIUS. Volume-correct and single-bodied are
                        different requirements and can conflict: if a pocket's
                        centres are spread out, the radius that reproduces its
                        volume may leave it in fragments. Those pockets are listed
                        at the end so the conflict is a decision rather than a
                        surprise in the render.

Run this once per run directory (apo_only and holo_only). Output goes to
pocket_radius_calibration.csv inside each run, next to blender_median_frames.
"""

import os
import sys
import re
import glob
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import META_ANALYSIS_ROOT

# config
ROOT = META_ANALYSIS_ROOT
# blender_global_id_prep.py writes one combined apo+holo run's median frames now
# (was two separate apo-only/holo-only directories)
RUN_DIRECTORIES = [os.path.join(ROOT, "global_ID_combined")]
MEDIAN_SUBDIR = "blender_median_frames"
OUTPUT_NAME = "pocket_radius_calibration.csv"

# which reported volume the radius is solved against.
#   "median" - median_interpolated_volume, the number the tables and the text use
#   "frame"  - the volume of the single frame actually drawn
# blender_global_id_prep.py picks the frame closest to the median, so these agree
# closely except for transient pockets, where the interpolated median can sit away
# from any real frame.
CALIBRATION_TARGET = "median"

# A pocket whose alpha-sphere centres are spread out can be volume-correct and
# still fall into pieces. The radius that just closes it is exactly half the
# largest edge of the point cloud's minimum spanning tree, so it needs no search -
# but raising every sphere to it also inflates the pocket, and for a badly split
# cloud that inflation can be larger than the pocket itself. So the floor is
# applied only while it stays cheap:
#
#   "floor"   raise the radius to the connectivity floor whenever the pocket is
#             fragmented, but only if the resulting volume overshoot stays under
#             MAX_VOLUME_OVERSHOOT. Above that the calibrated radius is kept and
#             needs_bridge is set, and the Blender scripts close the gap by
#             inserting a thin neck instead - which costs almost no volume.
#   "none"    never raise; report the fragments and leave them.
CONNECTIVITY_POLICY = "floor"
MAX_VOLUME_OVERSHOOT = 0.25          # 25% over the reported volume

RADIUS_SEARCH_RANGE = (0.05, 6.00)   # A
BISECTION_STEPS = 30                 # ~1e-8 A precision, costs nothing
GRID_CELL_BUDGET = 1_200_000         # caps memory and time per pocket
FINEST_GRID_STEP = 0.12              # A
POINT_CHUNK = 128                    # centres per vectorised block

# a pocket whose achieved volume misses the target by more than this is flagged
VOLUME_TOLERANCE_FRACTION = 0.02

FILENAME_PATTERN = re.compile(r"^g(\d+)_((apo|holo)[^_]+)_(\d+)_(p\d+)$")


def read_median_frame_pdb(pdb_path):
    """Unique alpha-sphere centres, their fpocket radii, and the reported volumes.

    The median-frame PDBs are not standard-column PDB - the line is 71 characters
    and the trailing occupancy/radius fields do not sit where a PDB reader expects.
    Coordinates happen to survive a fixed-column read, the radius does not, so
    every field is taken from a whitespace split instead.
    """
    coordinates = []
    alpha_radii = []
    frame_volume = np.nan
    median_volume = np.nan

    for line in open(pdb_path):
        if line.startswith("REMARK"):
            frame_match = re.search(r"vol=([-\d.]+)", line)
            median_match = re.search(r"median=([-\d.]+)", line)
            if frame_match:
                frame_volume = float(frame_match.group(1))
            if median_match:
                median_volume = float(median_match.group(1))
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        try:
            coordinates.append((float(fields[-5]), float(fields[-4]), float(fields[-3])))
            alpha_radii.append(float(fields[-1]))
        except (ValueError, IndexError):
            continue

    if not coordinates:
        return None

    all_coordinates = np.asarray(coordinates, dtype=float)
    unique_coordinates, first_occurrence = np.unique(all_coordinates, axis=0, return_index=True)
    return {
        "n_atoms": len(all_coordinates),
        "unique_coordinates": unique_coordinates,
        "alpha_radii": np.asarray(alpha_radii, dtype=float)[first_occurrence],
        "frame_volume": frame_volume,
        "median_volume": median_volume,
    }


def nearest_centre_distance_grid(centres, maximum_radius):
    """Distance from every voxel to its nearest centre, plus the voxel volume.

    Computed once per pocket. Every later volume question - at any radius - becomes
    a count over this array, which is what makes the bisection free.
    """
    lower = centres.min(axis=0) - maximum_radius
    upper = centres.max(axis=0) + maximum_radius
    span = upper - lower

    step = FINEST_GRID_STEP
    estimated_cells = np.prod(span / step + 1)
    if estimated_cells > GRID_CELL_BUDGET:
        step = float((np.prod(span) / GRID_CELL_BUDGET) ** (1.0 / 3.0))

    axes = [np.arange(lower[axis], upper[axis] + step, step) for axis in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_points = np.stack(mesh, axis=-1).reshape(-1, 3)

    nearest = np.full(len(grid_points), np.inf)
    for start in range(0, len(centres), POINT_CHUNK):
        block = centres[start:start + POINT_CHUNK]
        distances = np.linalg.norm(grid_points[:, None, :] - block[None, :, :], axis=2)
        nearest = np.minimum(nearest, distances.min(axis=1))
    return nearest, step ** 3


def solve_radius_for_volume(centres, target_volume):
    """Radius at which the union of equal spheres on these centres has that volume.

    Volume increases monotonically with radius, so plain bisection is exact and
    needs no starting guess.
    """
    if len(centres) == 1:
        radius = float((3.0 * target_volume / (4.0 * np.pi)) ** (1.0 / 3.0))
        return radius, target_volume

    lower_radius, upper_radius = RADIUS_SEARCH_RANGE
    nearest, voxel_volume = nearest_centre_distance_grid(centres, upper_radius)

    def union_volume(radius):
        return float((nearest <= radius).sum() * voxel_volume)

    for _ in range(BISECTION_STEPS):
        middle_radius = 0.5 * (lower_radius + upper_radius)
        if union_volume(middle_radius) < target_volume:
            lower_radius = middle_radius
        else:
            upper_radius = middle_radius
    calibrated_radius = 0.5 * (lower_radius + upper_radius)
    return calibrated_radius, union_volume(calibrated_radius)


def connectivity_radius(centres):
    """Smallest equal radius at which these centres form a single body.

    Spheres of radius r merge when their centres are closer than 2r, so the cloud
    is connected exactly when 2r reaches the largest edge of its minimum spanning
    tree. That bottleneck edge is found with Prim's algorithm, which makes this
    exact and search-free.
    """
    n_centres = len(centres)
    if n_centres < 2:
        return 0.0
    distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    inside = [0]
    cheapest_edge = distances[0].copy()
    cheapest_edge[0] = np.inf
    largest_edge = 0.0
    for _ in range(n_centres - 1):
        next_index = int(np.argmin(cheapest_edge))
        largest_edge = max(largest_edge, float(cheapest_edge[next_index]))
        inside.append(next_index)
        cheapest_edge = np.minimum(cheapest_edge, distances[next_index])
        cheapest_edge[inside] = np.inf
    return largest_edge / 2.0


def union_volume_at(centres, radius):
    """Volume of the union of equal spheres, measured the same way as the search."""
    if len(centres) == 1:
        return float(4.0 / 3.0 * np.pi * radius ** 3)
    nearest, voxel_volume = nearest_centre_distance_grid(centres, radius)
    return float((nearest <= radius).sum() * voxel_volume)


def count_components(centres, radius):
    """Separate pieces the pocket falls into when drawn at this radius.

    Two spheres of equal radius merge when their centres are closer than twice it.
    """
    n_centres = len(centres)
    if n_centres < 2:
        return 1
    distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    parent = list(range(n_centres))

    def find_root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in np.argwhere(distances < 2.0 * radius):
        root_first, root_second = find_root(int(first)), find_root(int(second))
        if root_first != root_second:
            parent[root_first] = root_second
    return len({find_root(index) for index in range(n_centres)})


def calibrate_run(run_directory):
    median_directory = os.path.join(run_directory, MEDIAN_SUBDIR)
    pdb_paths = sorted(glob.glob(os.path.join(median_directory, "g*_*.pdb")))
    print("\n{divider}\n{directory}\n{divider}".format(divider="=" * 78, directory=run_directory))
    if not pdb_paths:
        print("  no median-frame PDBs found - nothing to calibrate")
        return None

    calibration_rows = []
    start_time = time.perf_counter()
    for position, pdb_path in enumerate(pdb_paths, start=1):
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        name_match = FILENAME_PATTERN.match(stem)
        pocket = read_median_frame_pdb(pdb_path)
        if pocket is None or name_match is None:
            print("  skipped (unreadable or unparsable name): {stem}".format(stem=stem))
            continue

        target_volume = (pocket["median_volume"] if CALIBRATION_TARGET == "median"
                         else pocket["frame_volume"])
        if not np.isfinite(target_volume) or target_volume <= 0:
            target_volume = (pocket["frame_volume"] if CALIBRATION_TARGET == "median"
                             else pocket["median_volume"])
        if not np.isfinite(target_volume) or target_volume <= 0:
            print("  skipped (no usable volume in REMARK): {stem}".format(stem=stem))
            continue

        centres = pocket["unique_coordinates"]
        calibrated_radius, achieved_volume = solve_radius_for_volume(centres, target_volume)
        extent = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))

        # connectivity floor: only worth paying for while it stays cheap
        floor_radius = connectivity_radius(centres)
        final_radius = calibrated_radius
        needs_bridge = False
        if CONNECTIVITY_POLICY == "floor" and floor_radius > calibrated_radius:
            volume_at_floor = union_volume_at(centres, floor_radius)
            overshoot = volume_at_floor / target_volume - 1.0
            if overshoot <= MAX_VOLUME_OVERSHOOT:
                final_radius = floor_radius
                achieved_volume = volume_at_floor
            else:
                needs_bridge = True
        elif floor_radius > calibrated_radius:
            needs_bridge = True

        calibration_rows.append({
            "pdb_file": os.path.basename(pdb_path),
            "state": name_match.group(3),
            "global_id": int(name_match.group(1)),
            "prj": name_match.group(2),
            "rep": int(name_match.group(4)),
            "pocket": name_match.group(5),
            "n_atoms": pocket["n_atoms"],
            "n_unique_centres": len(centres),
            "frame_volume": pocket["frame_volume"],
            "median_volume": pocket["median_volume"],
            "target_volume": target_volume,
            "calibrated_radius": calibrated_radius,
            "connectivity_radius": floor_radius,
            "radius": final_radius,          # the one the Blender scripts use
            "achieved_volume": achieved_volume,
            "volume_error_fraction": abs(achieved_volume - target_volume) / target_volume,
            "volume_overshoot_fraction": achieved_volume / target_volume - 1.0,
            "needs_bridge": needs_bridge,
            "n_components": count_components(centres, final_radius),
            "alpha_radius_median": float(np.median(pocket["alpha_radii"])),
            "centre_extent": extent,
        })
        if position % 50 == 0:
            print("  {done}/{total} pockets".format(done=position, total=len(pdb_paths)))

    calibration_df = pd.DataFrame(calibration_rows)
    output_path = os.path.join(run_directory, OUTPUT_NAME)
    calibration_df.to_csv(output_path, index=False)

    radii = calibration_df["radius"]
    duplication = calibration_df["n_atoms"] / calibration_df["n_unique_centres"]
    fragmented = calibration_df[calibration_df["n_components"] > 1]
    inaccurate = calibration_df[calibration_df["volume_error_fraction"] > VOLUME_TOLERANCE_FRACTION]
    floored = calibration_df[calibration_df["radius"] > calibration_df["calibrated_radius"] + 1e-9]
    to_bridge = calibration_df[calibration_df["needs_bridge"]]

    print("  {n} pockets calibrated in {seconds:.1f} s".format(
        n=len(calibration_df), seconds=time.perf_counter() - start_time))
    print("  radius used         min {low:.2f}  median {mid:.2f}  max {high:.2f} A".format(
        low=radii.min(), mid=radii.median(), high=radii.max()))
    print("  alpha-sphere radius median {alpha:.2f} A  "
          "- {ratio:.1f}x larger than the calibrated one, which is why it must not be used".format(
              alpha=calibration_df["alpha_radius_median"].median(),
              ratio=calibration_df["alpha_radius_median"].median() / radii.median()))
    print("  duplicate alpha spheres per file: median {mid:.1f}x, max {high:.1f}x".format(
        mid=duplication.median(), high=duplication.max()))
    print("  volume reproduced within {tolerance:.0%} for {ok} of {total} pockets".format(
        tolerance=VOLUME_TOLERANCE_FRACTION,
        ok=len(calibration_df) - len(inaccurate), total=len(calibration_df)))
    if len(floored):
        print("  {n} pockets raised to their connectivity floor "
              "(single body, volume over by a known amount):".format(n=len(floored)))
        for _, row in floored.nlargest(min(10, len(floored)), "volume_overshoot_fraction").iterrows():
            print("    {file:<34} r {calibrated:.2f} -> {final:.2f} A, "
                  "volume +{overshoot:.0%}".format(
                      file=row["pdb_file"], calibrated=row["calibrated_radius"],
                      final=row["radius"], overshoot=row["volume_overshoot_fraction"]))
    if len(to_bridge):
        print("  {n} pockets left at their calibrated radius because flooring would "
              "have overshot by more than {cap:.0%} - the Blender scripts bridge these "
              "with a thin neck instead:".format(n=len(to_bridge), cap=MAX_VOLUME_OVERSHOOT))
        for _, row in to_bridge.iterrows():
            print("    {file:<34} r {calibrated:.2f} A, floor would be {floor:.2f} A, "
                  "extent {extent:.1f} A".format(
                      file=row["pdb_file"], calibrated=row["calibrated_radius"],
                      floor=row["connectivity_radius"], extent=row["centre_extent"]))
    if len(fragmented) and not len(to_bridge):
        print("  {n} pockets still fall into more than one piece".format(n=len(fragmented)))
    if not len(floored) and not len(to_bridge):
        print("  every pocket is a single body at its calibrated radius")
    print("  written to {path}".format(path=output_path))
    return calibration_df


def main():
    print(__doc__)
    print("calibration target: {target}".format(target=CALIBRATION_TARGET))
    per_run = [calibrate_run(directory) for directory in RUN_DIRECTORIES]
    combined = pd.concat([df for df in per_run if df is not None], ignore_index=True)
    if combined.empty:
        return
    print("\n{divider}\nCOMBINED\n{divider}".format(divider="=" * 78))
    print("{n} pockets across {runs} runs".format(n=len(combined), runs=len(RUN_DIRECTORIES)))
    print("calibrated radius: min {low:.2f}  median {mid:.2f}  max {high:.2f} A".format(
        low=combined["radius"].min(),
        mid=combined["radius"].median(),
        high=combined["radius"].max()))
    print("\nradius by state:")
    for state, group in combined.groupby("state"):
        print("  {state:<5} median {mid:.2f} A over {n} pockets".format(
            state=state, mid=group["radius"].median(), n=len(group)))
    print("\nNext: the Blender scripts read pocket_radius_calibration.csv and give")
    print("every point its pocket's calibrated radius, so pockets and the envelopes")
    print("that wrap them share one surface and neither overstates the cavity.")


if __name__ == "__main__":
    main()