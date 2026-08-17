"""
Map max_consecutive_zero_frames from 60GB file to pocket_analysis_summary.csv
and create true_transient column.

Deduplicates by ID (one value per pocket) and adds true_transient flag for >=50 frames.
"""
import pandas as pd
import os


def create_max_zeros_dict_from_large_file(large_file_path: str,
                                          chunksize: int = 100000) -> dict:
    """
    Create dictionary mapping ID -> max_consecutive_zero_frames from large file.

    Since large file has thousands of rows per pocket (one per frame + alpha sphere),
    keeps only first occurrence of each ID (all values are identical per pocket).

    Parameters:
    -----------
    large_file_path : str
        Path to 60GB file with max_consecutive_zero_frames column
    chunksize : int
        Chunk size for reading large file

    Returns:
    --------
    dict : Mapping of ID -> max_consecutive_zero_frames
    """

    print(f"\nCreating dictionary from: {large_file_path}")
    print(f"  (Reading only ID and max_consecutive_zero_frames columns)")

    max_zero_dict = {}

    for i, chunk in enumerate(pd.read_csv(large_file_path,
                                          usecols=['ID', 'max_consecutive_zero_frames'],
                                          chunksize=chunksize,
                                          on_bad_lines='skip')):
        if i % 10 == 0:
            print(f"  Processed chunk {i}...")

        # Keep only first occurrence of each ID (all values same per pocket)
        for idx, row in chunk.iterrows():
            if row['ID'] not in max_zero_dict:
                max_zero_dict[row['ID']] = int(row['max_consecutive_zero_frames'])

    print(f"\nCreated dictionary with {len(max_zero_dict):,} unique pockets")
    return max_zero_dict


def add_transient_columns_to_summary(summary_path: str,
                                      max_zero_dict: dict,
                                      output_path: str = None,
                                      id_column: str = 'ID') -> pd.DataFrame:
    """
    Add max_consecutive_zero_frames and true_transient columns to summary.

    Parameters:
    -----------
    summary_path : str
        Path to pocket_analysis_summary.csv
    max_zero_dict : dict
        Dictionary mapping ID -> max_consecutive_zero_frames
    output_path : str, optional
        Path to save updated summary. If None, overwrites summary_path
    id_column : str
        Name of ID column in summary (usually 'ID' or 'Local ID')

    Returns:
    --------
    pd.DataFrame : Updated summary dataframe
    """

    print(f"\nLoading summary from: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    print(f"  Loaded {len(summary_df):,} pockets")

    print(f"\nMapping max_consecutive_zero_frames using '{id_column}' column...")
    summary_df['max_consecutive_zero_frames'] = summary_df[id_column].map(max_zero_dict)

    # Check for missing mappings
    missing = summary_df['max_consecutive_zero_frames'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} pockets not found in dictionary")
    else:
        print(f"  ✓ All pockets successfully mapped")

    print(f"\nCreating true_transient column (>= 50 consecutive frames at zero)...")
    summary_df['true_transient'] = summary_df['max_consecutive_zero_frames'] >= 50

    n_true_transient = summary_df['true_transient'].sum()
    print(f"  Found {n_true_transient:,} true transient pockets")

    # Save updated summary
    if output_path is None:
        output_path = summary_path

    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return summary_df


# =============================================================================

if __name__ == '__main__':
    cluster_loc = '/mnt/mmlab_shared/RienaeckerC/apo_holo_analysis/meta_analysis/across_genes'

    large_file_path = os.path.join(cluster_loc, 'summary_df_3d_coords_cont_zero.csv')
    summary_path = os.path.join(cluster_loc, 'pocket_analysis_summary.csv')

    max_zero_dict = create_max_zeros_dict_from_large_file(large_file_path, chunksize=100000)

    print(max_zero_dict)
    summary_df = add_transient_columns_to_summary(
        summary_path=summary_path,
        max_zero_dict=max_zero_dict,
        output_path=os.path.join(cluster_loc, 'pocket_analysis_summary_cont_zero.csv'),
        id_column='ID')
    print(summary_df)