#!/usr/bin/env python3
"""
QUICK FIX: Generate the missing plots_all_pockets plots.
Uses pocket_analysis_summary.csv - takes ~1 minute, NOT 2.3 hours!

Usage:
    python quick_all_pockets_plots.py /path/to/across_genes
"""

import os
import sys
import pandas as pd
import visualizations as vis


def replot_summary_for_all_pockets(saving_loc):
    print("=" * 70)
    print("QUICK FIX: Generating plots_all_pockets")
    print("=" * 70)

    # Load the summary CSV (should already exist from your 2.3h run)
    summary_path = os.path.join(saving_loc, 'pocket_analysis_summary.csv')

    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found!")
        print("Looking for alternative files...")
        # Try alternative locations
        for alt in ['summary_df.csv', 'first_frame_summary_df.csv']:
            alt_path = os.path.join(saving_loc, alt)
            if os.path.exists(alt_path):
                print(f"Found: {alt_path}")
                summary_path = alt_path
                break
        else:
            print("No summary file found. Cannot continue.")
            return

    print(f"\nLoading: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df):,} pockets")
    print(f"Columns: {list(summary_df.columns)}")

    # Create output directory
    output_dir = os.path.join(saving_loc, 'plots_all_pockets')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Build transient_dict
    print("\nBuilding transient dictionary...")
    if 'stability' in summary_df.columns:
        transient_dict = {row['ID']: row['stability'] == 'Transient'
                          for _, row in summary_df.iterrows()}
    elif 'transient' in summary_df.columns:
        transient_dict = {row['ID']: bool(row['transient'])
                          for _, row in summary_df.iterrows()}
    else:
        print("ERROR: No stability/transient column found!")
        return

    # Ensure required columns
    if 'volume_category' not in summary_df.columns:
        print("Adding volume_category...")
        summary_df['volume_category'] = summary_df['interpolated_pock_volume'].apply(vis.categorize_volume)

    if 'state' not in summary_df.columns:
        print("Adding state...")
        summary_df['state'] = summary_df['ID'].str.extract(r'^(apo|holo)')[0]

    if 'pdb_id' not in summary_df.columns:
        print("Adding pdb_id...")
        summary_df['pdb_id'] = summary_df['ID'].str.extract(r'^(?:apo|holo)([A-Z0-9]+)')[0]

    if 'rep' not in summary_df.columns and 'state_pdbid_rep' in summary_df.columns:
        summary_df['rep'] = summary_df['state_pdbid_rep'].str.extract(r'_(\d+)$')[0]

    # Generate ALL 8 plots
    print("\n" + "=" * 70)
    print("GENERATING 8 PLOTS...")
    print("=" * 70)

    # 1 & 2: Hierarchical plots
    print("\n[1/4] Hierarchical stability...")
    try:
        vis.plot_hierarchical_stability(
            pocket_df=summary_df, transient_dict=transient_dict,
            saving_loc=output_dir, out_prefix='hierarchical_stability', n_cols=4)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[2/4] Hierarchical volume...")
    try:
        vis.plot_hierarchical_volume_categories(
            pocket_df=summary_df, saving_loc=output_dir,
            out_prefix='hierarchical_volume', n_cols=4)
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3 & 4: Aggregated plots
    print("\n[3/4] Aggregated by structure (stability + volume)...")
    try:
        vis.plot_aggregated_by_structure(
            pocket_df=summary_df, transient_dict=transient_dict,
            saving_loc=output_dir, out_prefix='aggregated_median', aggregation='median')
    except Exception as e:
        print(f"  ERROR: {e}")

    # 5-8: Overall distributions
    print("\n[4/4] Overall distributions (4 plots)...")
    try:
        vis.plot_overall_distributions(
            pocket_df=summary_df, transient_dict=transient_dict,
            saving_loc=output_dir, out_prefix='overall')
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")



replot_summary_for_all_pockets(saving_loc='/mnt/cpm_crienaecker/Z/RienaeckerC/apo_holo_analysis/meta_analysis/across_genes')