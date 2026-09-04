"""
Transiency classification -- single source of truth for the `transient` column
used everywhere in the pipeline.

Definition: a pocket is transient if interpolated_pock_volume == 0 for at least
MIN_ZERO_FRACTION of the simulation's frames in total, AND for at least
MIN_CONSECUTIVE_ZERO_FRAMES of them consecutively. This tells real transiency
(long, sustained closures) apart from noise (a handful of scattered zero frames).
Both conditions are needed: consecutive alone flags anything with one long-enough
closed stretch regardless of how much of the trajectory that is, and the fraction
alone flags anything scattered across enough individual frames regardless of
whether any single closure is meaningful.

At NS_PER_FRAME (1 ns/frame here), MIN_CONSECUTIVE_ZERO_FRAMES=50 is a run of
>=50 ns; MIN_ZERO_FRACTION=0.10 is >=10% of the simulation, i.e. >=100 frames /
100 ns for a standard 1000-frame/1000 ns run -- but expressed as a fraction so it
scales correctly if a trajectory is a different length.
"""
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`

NS_PER_FRAME = 1.0
MIN_ZERO_FRACTION = 0.10
MIN_CONSECUTIVE_ZERO_FRAMES = 50


def find_max_consecutive_zeros(volume_series: np.ndarray) -> int:
    """
    Find the longest consecutive sequence of zero volumes.

    Parameters:
    -----------
    volume_series : np.ndarray
        Array of volume values for a single pocket across all frames

    Returns:
    --------
    int : Maximum number of consecutive frames at volume == 0
    """
    zeros = (volume_series == 0.0).astype(int)

    if zeros.sum() == 0:
        return 0

    # Find where zeros start and end
    diff = np.diff(np.concatenate(([0], zeros, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return 0

    max_consecutive = np.max(ends - starts)
    return int(max_consecutive)


def _per_frame_volumes(group: pd.DataFrame, volume_col: str) -> np.ndarray:
    """One volume value per snapshot, sorted by snapshot. A pocket has multiple rows
    per frame (one per alpha sphere/dummy atom, all sharing the same volume), so this
    must dedupe to a single value per frame before counting frames -- otherwise frame
    counts (and anything compared against an ns-based threshold) are inflated by the
    number of alpha spheres per frame."""
    return group.groupby('snapshot')[volume_col].first().sort_index().values


def add_max_consecutive_zeros_to_df(all_data_df: pd.DataFrame,
                                    volume_col: str = 'interpolated_pock_volume') -> pd.DataFrame:
    """
    Add 'max_consecutive_zero_frames' column to dataframe.

    For each pocket (ID), finds the longest stretch of consecutive frames
    where volume == 0. This helps identify true transient vs. noise.

    Parameters:
    -----------
    all_data_df : pd.DataFrame
        Full data dataframe with 'ID' and volume_col columns
    volume_col : str
        Name of volume column to analyze

    Returns:
    --------
    pd.DataFrame : DataFrame with new 'max_consecutive_zero_frames' column
    """

    print(f"\nCalculating max consecutive zero frames per pocket...")
    print(f"Processing {all_data_df['ID'].nunique():,} unique pockets")

    max_zero_dict = {}

    # Group by pocket ID and find max consecutive zeros
    for pocket_id, group in all_data_df.groupby('ID'):
        volumes = _per_frame_volumes(group, volume_col)
        max_zero_dict[pocket_id] = find_max_consecutive_zeros(volumes)

    # Add to dataframe
    all_data_df = all_data_df.copy()
    all_data_df['max_consecutive_zero_frames'] = all_data_df['ID'].map(max_zero_dict)

    return all_data_df


def classify_transient(all_data_df: pd.DataFrame, volume_col: str = 'interpolated_pock_volume',
                       min_zero_fraction: float = MIN_ZERO_FRACTION,
                       min_consecutive_zero_frames: float = MIN_CONSECUTIVE_ZERO_FRAMES
                       ) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    Classify every pocket (ID) as transient or stable.

    A pocket is transient if volume_col == 0 for at least min_zero_fraction of its
    frames in total, AND for at least min_consecutive_zero_frames of them
    consecutively.

    Adds 'n_zero_frames', 'max_consecutive_zero_frames' and 'transient' columns.

    Returns
    -------
    (all_data_df, transient_dict) : the input df with the three columns added, and
    a {ID: bool} dict for the plotting functions in visualizations.py.
    """
    n_zero_dict = {}
    max_consecutive_dict = {}
    transient_dict = {}

    for pocket_id, group in all_data_df.groupby('ID'):
        volumes = _per_frame_volumes(group, volume_col)
        n_zero = int((volumes == 0.0).sum())
        max_consecutive = find_max_consecutive_zeros(volumes)

        n_zero_dict[pocket_id] = n_zero
        max_consecutive_dict[pocket_id] = max_consecutive
        transient_dict[pocket_id] = (n_zero >= min_zero_fraction * len(volumes)
                                     and max_consecutive >= min_consecutive_zero_frames)

    all_data_df = all_data_df.copy()
    all_data_df['n_zero_frames'] = all_data_df['ID'].map(n_zero_dict)
    all_data_df['max_consecutive_zero_frames'] = all_data_df['ID'].map(max_consecutive_dict)
    all_data_df['transient'] = all_data_df['ID'].map(transient_dict)

    return all_data_df, transient_dict


def add_max_consecutive_zeros_chunked(data_path: str,
                                      output_path: str,
                                      volume_col: str = 'interpolated_pock_volume',
                                      chunksize: int = 100000,
                                      min_zero_fraction: float = MIN_ZERO_FRACTION,
                                      min_consecutive_zero_frames: float = MIN_CONSECUTIVE_ZERO_FRAMES) -> None:
    """
    Add n_zero_frames, max_consecutive_zero_frames and transient columns by reading
    a large file in chunks -- same definition as classify_transient(), for files too
    big to load in memory at once.

    Parameters:
    -----------
    data_path : str
        Path to input CSV file
    output_path : str
        Path to save output CSV with new columns
    volume_col : str
        Name of volume column
    chunksize : int
        Number of rows per chunk

    Returns:
    --------
    None (saves to disk)
    """
    print(f"\nReading data in chunks from: {data_path}")
    print(f"Chunk size: {chunksize:,} rows")

    # First pass: read entire file to group by ID and collect per-frame volumes
    print("\n[PASS 1] Collecting per-frame volumes for each pocket...")

    all_ids_seen: Dict[str, Dict[object, float]] = {}

    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Processed chunk {i}...")

        for pocket_id, group in chunk.groupby('ID'):
            frame_volumes = all_ids_seen.setdefault(pocket_id, {})
            # dedupe to one value per snapshot (multiple alpha spheres share a frame's volume)
            for snapshot, volume in zip(group['snapshot'], group[volume_col]):
                frame_volumes.setdefault(snapshot, volume)

    # Calculate zero-frame stats and transiency for each pocket
    print("\n[PASS 1] Calculating consecutive/total zero frames...")

    n_zero_dict = {}
    max_zero_dict = {}
    transient_dict = {}
    for pocket_id, frame_volumes in all_ids_seen.items():
        volumes = np.array([frame_volumes[s] for s in sorted(frame_volumes)])
        n_zero = int((volumes == 0.0).sum())
        max_consecutive = find_max_consecutive_zeros(volumes)
        n_zero_dict[pocket_id] = n_zero
        max_zero_dict[pocket_id] = max_consecutive
        transient_dict[pocket_id] = (n_zero >= min_zero_fraction * len(volumes)
                                     and max_consecutive >= min_consecutive_zero_frames)

    # Second pass: read file again and add the new columns
    print("\n[PASS 2] Writing file with new columns...")

    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Writing chunk {i}...")

        chunk['n_zero_frames'] = chunk['ID'].map(n_zero_dict)
        chunk['max_consecutive_zero_frames'] = chunk['ID'].map(max_zero_dict)
        chunk['transient'] = chunk['ID'].map(transient_dict)

        if first_chunk:
            chunk.to_csv(output_path, mode='w', index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_path, mode='a', header=False, index=False)

    print(f"\nSaved to: {output_path}")


def summarize_consecutive_zeros(df: pd.DataFrame) -> None:
    """
    Print summary statistics about max consecutive zero frames.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'max_consecutive_zero_frames' column
    """

    if 'max_consecutive_zero_frames' not in df.columns:
        print("ERROR: 'max_consecutive_zero_frames' column not found")
        return

    # Get unique pocket values
    pocket_summary = df.groupby('ID')['max_consecutive_zero_frames'].first()

    print("MAX CONSECUTIVE ZERO FRAMES SUMMARY")
    print(f"\nTotal pockets analyzed: {len(pocket_summary):,}")
    print(f"Pockets with NO zero frames: {(pocket_summary == 0).sum():,}")
    print(f"\nStatistics for pockets with zero frames:")

    non_zero = pocket_summary[pocket_summary > 0]
    if len(non_zero) > 0:
        print(f"  Count: {len(non_zero):,}")
        print(f"  Mean: {non_zero.mean():.2f} frames")
        print(f"  Median: {non_zero.median():.0f} frames")
        print(f"  Min: {non_zero.min():.0f} frames")
        print(f"  Max: {non_zero.max():.0f} frames")
        print(f"  Std: {non_zero.std():.2f} frames")

    # Distribution
    print(f"\nDistribution:")
    print(f"  0 frames: {(pocket_summary == 0).sum():,} pockets")
    print(f"  1-3 frames (noise): {((pocket_summary > 0) & (pocket_summary <= 3)).sum():,} pockets")
    print(f"  4-10 frames: {((pocket_summary > 3) & (pocket_summary <= 10)).sum():,} pockets")
    print(f"  11-50 frames: {((pocket_summary > 10) & (pocket_summary <= 50)).sum():,} pockets")
    print(f"  >{MIN_CONSECUTIVE_ZERO_FRAMES:.0f} frames (meets the consecutive half of the transient definition, "
          f"see classify_transient): {(pocket_summary > MIN_CONSECUTIVE_ZERO_FRAMES).sum():,} pockets")


# =============================================================================

if __name__ == '__main__':
    # For large files that need chunk reading
    from config import META_ANALYSIS_DIR
    cluster_loc = META_ANALYSIS_DIR
    data_path = os.path.join(cluster_loc, 'summary_df_3d_coords.csv')
    output_path = os.path.join(cluster_loc, 'summary_df_3d_coords_cont_zero.csv')
    add_max_consecutive_zeros_chunked(data_path, output_path, chunksize=100000)