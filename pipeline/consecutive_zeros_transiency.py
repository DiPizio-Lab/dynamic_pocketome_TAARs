"""
Calculate maximum consecutive frames at zero volume for each pocket.

This helps differentiate between:
- True transient pockets (many consecutive frames at 0)
- Noise/artifacts (scattered 1-3 frames at 0)
"""
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`


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
        volumes = group[volume_col].sort_values(
            key=lambda x: group.loc[x.index, 'snapshot']
        ).values
        max_consecutive = find_max_consecutive_zeros(volumes)
        max_zero_dict[pocket_id] = max_consecutive

    # Add to dataframe
    all_data_df = all_data_df.copy()
    all_data_df['max_consecutive_zero_frames'] = all_data_df['ID'].map(max_zero_dict)

    return all_data_df


def add_max_consecutive_zeros_chunked(data_path: str,
                                      output_path: str,
                                      volume_col: str = 'interpolated_pock_volume',
                                      chunksize: int = 100000) -> None:
    """
    Add max_consecutive_zero_frames column by reading large file in chunks.

    Parameters:
    -----------
    data_path : str
        Path to input CSV file
    output_path : str
        Path to save output CSV with new column
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

    # First pass: read entire file to group by ID and calculate max zeros
    print("\n[PASS 1] Calculating max consecutive zeros for each pocket...")

    max_zero_dict = {}
    all_ids_seen = {}

    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Processed chunk {i}...")

        for pocket_id, group in chunk.groupby('ID'):
            if pocket_id not in all_ids_seen:
                all_ids_seen[pocket_id] = []

            # Store volume data sorted by snapshot
            sorted_group = group.sort_values('snapshot')
            all_ids_seen[pocket_id].extend(sorted_group[volume_col].tolist())

    # Calculate max consecutive zeros for each pocket
    print("\n[PASS 1] Calculating consecutive zeros...")
    for pocket_id, volumes in all_ids_seen.items():
        max_zero_dict[pocket_id] = find_max_consecutive_zeros(np.array(volumes))

    # Second pass: read file again and add the new column
    print("\n[PASS 2] Writing file with new column...")

    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Writing chunk {i}...")

        chunk['max_consecutive_zero_frames'] = chunk['ID'].map(max_zero_dict)

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
    print(f"  >50 frames (true transient): {(pocket_summary > 50).sum():,} pockets")


# =============================================================================

if __name__ == '__main__':
    # For large files that need chunk reading
    from config import META_ANALYSIS_DIR
    cluster_loc = META_ANALYSIS_DIR
    data_path = os.path.join(cluster_loc, 'summary_df_3d_coords.csv')
    output_path = os.path.join(cluster_loc, 'summary_df_3d_coords_cont_zero.csv')
    add_max_consecutive_zeros_chunked(data_path, output_path, chunksize=100000)