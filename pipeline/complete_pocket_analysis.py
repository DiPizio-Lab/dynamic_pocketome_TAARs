"""
Complete pocket analysis with all visualizations.

This script performs:
1. Largest pocket identification
2. Orthosteric pocket detection (ligand-based)
3. Transient/stable categorization
4. Visualization plots from the merged visualizations module
"""
import sys
import time
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `config`

import visualizations as vis
from orthosteric_filter_ligand_based import add_orthosteric_column, DEFAULT_HOLO_BASE
from consecutive_zeros_transiency import add_max_consecutive_zeros_to_df, summarize_consecutive_zeros


def transient_stable_categorisation(descriptor_df, volume_col='interpolated_pock_volume', threshold=0.1):
    """
    Categorize pockets as transient or stable based on volume data.
    A pocket is transient if interpolated_pock_volume is zero in >= 10% of frames.
    Accounts for multiple alpha spheres per frame.
    """
    def is_transient(group):
        # Get one value per frame (first occurrence since all spheres have same volume)
        unique_frames = group.groupby('snapshot')[volume_col].first()
        n_frames = len(unique_frames)
        n_zeros = (unique_frames == 0.0).sum()
        return n_zeros > threshold * n_frames

    transient_status = descriptor_df.groupby('ID').apply(is_transient)
    descriptor_df = descriptor_df.copy()
    descriptor_df['transient'] = descriptor_df['ID'].map(transient_status)
    transient_dict = transient_status.to_dict()

    return descriptor_df, transient_dict


def run_complete_pocket_analysis(saving_loc: str, all_data_df: pd.DataFrame = None, data_path: str = None,
                                 holo_base: str = DEFAULT_HOLO_BASE,distance_threshold: float = 5.0,
                                 volume_col: str = 'interpolated_pock_volume', bw_file_loc: str = None,
                                 pdb_file_loc: str = None, export_pdfs: bool = True, n_cols: int = 4,
                                 greyscale: bool = False, run_3d_heatmap: bool = True,
                                 run_apo_holo_comparison: bool = True, run_violin_plots: bool = True,
                                 run_consecutive_zeros: bool = True):
    """
    Run complete pocket analysis with all visualizations.
    Parameters
    ----------
    saving_loc : str
        Directory to save output files
    all_data_df : pd.DataFrame, optional
        Pre-loaded data. If None, loads from data_path
    data_path : str, optional
        Path to summary_df_3d_coords.csv
    holo_base : str
        Path to holo structure directory for ligand extraction
    distance_threshold : float
        Distance threshold for orthosteric classification (Angstrom)
    volume_col : str
        Column name for pocket volume
    bw_file_loc : str, optional
        Path to BW numbering files for backbone visualization
    pdb_file_loc : str, optional
        Path to PDB files for backbone visualization
    export_pdfs : bool
        Whether to export individual PDFs
    n_cols : int
        Number of columns for hierarchical subplot grids
    greyscale : bool
        Use greyscale color scheme
    run_3d_heatmap : bool
        Generate 3D heatmap with slider
    run_apo_holo_comparison : bool
        Generate apo/holo bar charts
    run_violin_plots : bool
        Generate binding site violin plots
     run_consecutive_zeros : bool
        Calculate max consecutive zero frames (set False to skip)

    Returns
    -------
    dict with all results and dataframes
    """
    if all_data_df is None:
        if data_path is None:
            data_path = os.path.join(saving_loc, 'summary_df_3d_coords.csv')
        print(f"\n[1/9] Loading data from {data_path}...")
        all_data_df = pd.read_csv(data_path)
    else:
        print(f"\n[1/9] Using provided dataframe...")
        all_data_df = all_data_df.copy()

    print(f"  Unique pocket IDs: {all_data_df['ID'].nunique():,}")

    print(f"\n[2/9] Identifying largest pocket per experiment...")

    all_data_df['state_pdbid_rep'] = all_data_df['ID'].str.extract(r'^([a-z]+[A-Z0-9]+_\d+)')[0]

    pocket_volumes = all_data_df.groupby('ID')['pock_volume'].first().reset_index()
    pocket_volumes['state_pdbid_rep'] = pocket_volumes['ID'].str.extract(r'^([a-z]+[A-Z0-9]+_\d+)')[0]

    largest_ids = pocket_volumes.loc[pocket_volumes.groupby('state_pdbid_rep')['pock_volume'].idxmax(), 'ID']
    all_data_df['is_largest_pocket'] = all_data_df['ID'].isin(largest_ids)
    n_experiments = all_data_df['state_pdbid_rep'].nunique()
    print(f"  Found {len(largest_ids)} largest pockets across {n_experiments} experiments")


    print(f"\n[3/9] Identifying orthosteric pockets (distance threshold: {distance_threshold} Å)...")

    all_data_df = add_orthosteric_column(all_data_df, holo_base=holo_base, distance_threshold=distance_threshold,
                                         id_column='ID', coord_columns=['x', 'y', 'z'], verbose=True)

    print(f"\n[4/9] Checking overlap between largest and orthosteric pockets...")

    pocket_info = all_data_df.groupby('ID').agg({'is_largest_pocket': 'first', 'is_orthosteric': 'first',
                                                 'state_pdbid_rep': 'first'}).reset_index()

    pocket_info['is_largest_and_orthosteric'] = (pocket_info['is_largest_pocket'] & pocket_info['is_orthosteric'])

    largest_ortho_map = pocket_info.set_index('ID')['is_largest_and_orthosteric'].to_dict()
    all_data_df['is_largest_and_orthosteric'] = all_data_df['ID'].map(largest_ortho_map)

    n_largest_pockets = pocket_info['is_largest_pocket'].sum()
    n_orthosteric = pocket_info['is_orthosteric'].sum()
    n_both = pocket_info['is_largest_and_orthosteric'].sum()

    print(f"  Largest pockets: {n_largest_pockets}")
    print(f"  Orthosteric pockets: {n_orthosteric}")
    print(f"  Largest AND orthosteric: {n_both} ({100 * n_both / n_largest_pockets:.1f}% of largest)")

    print(f"\n[5/9] Categorising pockets as transient/stable...")

    all_data_df, transient_dict = transient_stable_categorisation(all_data_df, volume_col=volume_col)

    n_transient = sum(transient_dict.values())
    n_stable = len(transient_dict) - n_transient
    print(f"  Transient pockets: {n_transient}")
    print(f"  Stable pockets: {n_stable}")

    print(f"\n[6/9] Calculating max consecutive zero frames...")

    if run_consecutive_zeros:
        all_data_df = add_max_consecutive_zeros_to_df(all_data_df, volume_col=volume_col)
        print(f"  Added 'max_consecutive_zero_frames' column")
    else:
        print(f"  Skipped (run_consecutive_zeros=False)")


    print(f"\n[7/9] Creating filtered dataframes...")

    cols_to_keep = ['ID', 'gene', 'pocket_number', 'x', 'y', 'z', 'snapshot', 'pock_volume', 'pock_asa',
                    'pock_pol_asa', 'pock_apol_asa', 'pock_asa22', 'pock_pol_asa22', 'pock_apol_asa22', 'nb_AS',
                    'mean_as_ray', 'mean_as_solv_acc', 'apol_as_prop', 'mean_loc_hyd_dens', 'hydrophobicity_score',
                    'volume_score', 'polarity_score', 'charge_score', 'prop_polar_atm', 'as_density', 'as_max_dst',
                    'convex_hull_volume', 'nb_abpa', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                    'interpolated_pock_volume', 'Local ID', 'Global ID', 'clean_id',
                    'is_largest_pocket', 'is_orthosteric', 'is_largest_and_orthosteric', 'transient',
                    'state_pdbid_rep']

    cols_available = [c for c in cols_to_keep if c in all_data_df.columns]

    largest_pocket_df = all_data_df[all_data_df['is_largest_pocket']][cols_available].reset_index(drop=True)
    orthosteric_pocket_df = all_data_df[all_data_df['is_orthosteric']][cols_available].reset_index(drop=True)

    print(f"  Largest pocket dataframe: {len(largest_pocket_df):,} rows")
    print(f"  Orthosteric pocket dataframe: {len(orthosteric_pocket_df):,} rows")

    print(f"\n[8/9] Creating summary dataframe...")

    first_frame_idx = all_data_df.groupby('ID')['snapshot'].idxmin()
    summary_df = all_data_df.loc[first_frame_idx].copy()
    summary_df['stability'] = summary_df['transient'].map({True: 'Transient', False: 'Stable'})
    summary_df['volume_category'] = summary_df[volume_col].apply(vis.categorize_volume)
    summary_df['state'] = summary_df['ID'].str.extract(r'^(apo|holo)')
    summary_df['pdb_id'] = summary_df['ID'].str.extract(r'^(?:apo|holo)([A-Z0-9]+)')[0]

    summary_df.to_csv(os.path.join(saving_loc, 'pocket_analysis_summary.csv'), index=False)
    print(f"  Saved: pocket_analysis_summary.csv ({len(summary_df)} pockets)")


    print(f"\n[9/9] Generating ALL visualizations...")

    # --- A. LARGEST POCKET PLOTS ---
    print("\n--- A. Largest Pocket Plots ---")
    largest_plot_dir = os.path.join(saving_loc, 'plots_largest')
    os.makedirs(largest_plot_dir, exist_ok=True)

    if bw_file_loc and pdb_file_loc:
        try:
            vis.plot_largest_pockets(
                saving_loc=saving_loc,
                pocket_df=largest_pocket_df,
                pdb_file=os.path.join(pdb_file_loc, 'aligned_top.pdb'),
                bw_csv_file=os.path.join(bw_file_loc, 'taar9_mouse.csv'),
                out_html='largest_pocket_3d.html',
                backbone=True,
                export_pdfs=export_pdfs,
                pdf_output_dir=os.path.join(saving_loc, 'pdfs_largest'),
                use_cdn=True,
                title='Largest Pocket Analysis')
        except Exception as e:
            print(f"WARNING: Could not create largest pocket 3D plot: {e}")

    vis.create_all_distribution_plots(
        pocket_df=largest_pocket_df,
        transient_dict=transient_dict,
        saving_loc=largest_plot_dir,
        volume_col=volume_col,
        n_cols=n_cols,
        greyscale=greyscale)

    # --- B. ORTHOSTERIC POCKET PLOTS ---
    print("\n--- B. Orthosteric Pocket Plots ---")
    ortho_plot_dir = os.path.join(saving_loc, 'plots_orthosteric')
    os.makedirs(ortho_plot_dir, exist_ok=True)

    if len(orthosteric_pocket_df) > 0:
        if bw_file_loc and pdb_file_loc:
            try:
                vis.plot_largest_pockets(
                    saving_loc=saving_loc,
                    pocket_df=orthosteric_pocket_df,
                    pdb_file=os.path.join(pdb_file_loc, 'aligned_top.pdb'),
                    bw_csv_file=os.path.join(bw_file_loc, 'taar9_mouse.csv'),
                    out_html='orthosteric_pocket_3d.html',
                    backbone=True,
                    export_pdfs=export_pdfs,
                    pdf_output_dir=os.path.join(saving_loc, 'pdfs_orthosteric'),
                    use_cdn=True,
                    title='Orthosteric Pocket Analysis')
            except Exception as e:
                print(f"WARNING: Could not create orthosteric pocket 3D plot: {e}")

        vis.create_all_distribution_plots(
            pocket_df=orthosteric_pocket_df,
            transient_dict=transient_dict,
            saving_loc=ortho_plot_dir,
            volume_col=volume_col,
            n_cols=n_cols,
            greyscale=greyscale)
    else:
        print("WARNING: No orthosteric pockets found, skipping plots")

        # --- C. ALL POCKETS DISTRIBUTION PLOTS ---
        print("\n--- C. All Pockets Distribution Plots ---")
        all_pockets_dir = os.path.join(saving_loc, 'plots_all_pockets')
        os.makedirs(all_pockets_dir, exist_ok=True)

        vis.create_all_distribution_plots(
            pocket_df=summary_df,  # Use summary_df which contains ALL unique pockets
            transient_dict=transient_dict,
            saving_loc=all_pockets_dir,
            volume_col=volume_col,
            n_cols=n_cols,
            greyscale=greyscale)

    # --- D. COMPARISON PLOTS ---
    print("\n--- D. Comparison Plots (Largest vs Orthosteric) ---")
    create_comparison_plots(summary_df, saving_loc, greyscale=greyscale)

    # --- E. 3D HEATMAP (from visualizations.py) ---
    if run_3d_heatmap:
        print("\n--- E. 3D Heatmap with Slider ---")
        first_frame_csv = os.path.join(saving_loc, 'first_frame_summary_df.csv')
        if os.path.exists(first_frame_csv):
            try:
                vis.plot_3d_heatmap(saving_loc=saving_loc, df=first_frame_csv)
            except Exception as e:
                print(f"  WARNING: Could not create 3D heatmap: {e}")

            # Also create side-by-side apo/holo 3D comparison
            try:
                vis.plot_3d_apo_holo_comparison(saving_loc=saving_loc, df=first_frame_csv)
            except Exception as e:
                print(f"WARNING: Could not create 3D apo/holo comparison: {e}")
        else:
            print(f"Skipping 3D heatmap (file not found: {first_frame_csv})")

    # --- F. APO/HOLO NUMBER OF POCKETS BAR CHARTS ---
    if run_apo_holo_comparison:
        print("\n--- F. Apo vs Holo Pocket Count Plots ---")
        first_frame_csv = os.path.join(saving_loc, 'first_frame_summary_df.csv')
        if os.path.exists(first_frame_csv):
            try:
                df_first = pd.read_csv(first_frame_csv)
                vis.vis_apo_holo_number_pockets(
                    df_first, figsize=(14, 6), version='median',
                    save_path=os.path.join(saving_loc, 'apo_holo_pockets_median.png'))
                vis.vis_apo_holo_number_pockets(
                    df_first, figsize=(14, 6), version='replicates',
                    save_path=os.path.join(saving_loc, 'apo_holo_pockets_replicates.png'))
            except Exception as e:
                print(f"WARNING: Could not create apo/holo comparison: {e}")
        else:
            print(f"Skipping apo/holo comparison (file not found: {first_frame_csv})")

    # --- G. BINDING SITE VIOLIN PLOTS ---
        if run_violin_plots:
            print("\n--- G. Orthosteric Site Violin Plots ---")
            try:
                vis.plot_binding_site_volume_violin(
                    pocket_comparison_df=summary_df,  # Has is_orthosteric and ID columns
                    summary_df=all_data_df,  # Full data with all frames
                    saving_loc=saving_loc,
                    summary_csv_path=None,  # Don't need to load - we have all_data_df
                    create_individual_plots=True)
            except Exception as e:
                print(f"  WARNING: Could not create violin plots: {e}")
                import traceback
                traceback.print_exc()

    return {'all_data_df': all_data_df, 'largest_pocket_df': largest_pocket_df,
            'orthosteric_pocket_df': orthosteric_pocket_df, 'summary_df': summary_df, 'transient_dict': transient_dict}


def create_comparison_plots(summary_df, saving_loc, greyscale=False):
    """
    Create comparison plots between largest and orthosteric pockets.
    """
    import plotly.graph_objects as go

    stability_colors = vis.get_color_dict('stability', greyscale)
    volume_colors = vis.get_color_dict('volume', greyscale)

    comparison_dir = os.path.join(saving_loc, 'plots_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Venn-style comparison (bar chart)
    largest_only = ((summary_df['is_largest_pocket']) & (~summary_df['is_orthosteric'])).sum()
    ortho_only = ((~summary_df['is_largest_pocket']) & (summary_df['is_orthosteric'])).sum()
    both = summary_df['is_largest_and_orthosteric'].sum()
    neither = ((~summary_df['is_largest_pocket']) & (~summary_df['is_orthosteric'])).sum()

    fig1 = go.Figure(data=[
        go.Bar(x=['Largest Only', 'Orthosteric Only', 'Both', 'Neither'],
               y=[largest_only, ortho_only, both, neither],
               marker_color=vis.get_viridis_colors(4, greyscale=greyscale),
               text=[largest_only, ortho_only, both, neither], textposition='auto' )])

    fig1.update_layout(title=dict(text='Pocket Classification Overlap', x=0.5, font=dict(size=18)),
                       xaxis_title='Category', yaxis_title='Number of Pockets', template='plotly_white',
                        height=500, width=700)

    fig1.write_html(os.path.join(comparison_dir, 'pocket_overlap.html'), include_plotlyjs='cdn')
    print(f"  Saved: plots_comparison/pocket_overlap.html")

    # Stability comparison
    largest_stability = summary_df[summary_df['is_largest_pocket']]['stability'].value_counts()
    ortho_stability = summary_df[summary_df['is_orthosteric']]['stability'].value_counts()

    fig2 = go.Figure()
    x_labels = ['Largest Pockets', 'Orthosteric Pockets']

    for stability in vis.STABILITY_ORDER:
        fig2.add_trace(go.Bar(name=stability, x=x_labels, y=[largest_stability.get(stability, 0),
                                                             ortho_stability.get(stability, 0)],
                              marker_color=stability_colors[stability], text=[largest_stability.get(stability, 0),
                                                                              ortho_stability.get(stability, 0)],
                              textposition='auto'))

    fig2.update_layout(barmode='stack', title=dict(text='Stability: Largest vs Orthosteric Pockets', x=0.5,
                                                   font=dict(size=18)),
                       xaxis_title='Pocket Type', yaxis_title='Number of Pockets', legend_title='Stability',
                       template='plotly_white', height=500, width=600 )

    fig2.write_html(os.path.join(comparison_dir, 'stability_comparison.html'), include_plotlyjs='cdn')
    print(f"  Saved: plots_comparison/stability_comparison.html")

    #Volume category comparison
    largest_volume = summary_df[summary_df['is_largest_pocket']]['volume_category'].value_counts()
    ortho_volume = summary_df[summary_df['is_orthosteric']]['volume_category'].value_counts()

    fig3 = go.Figure()

    for cat in vis.VOLUME_CATEGORY_ORDER:
        fig3.add_trace(go.Bar(name=cat, x=x_labels, y=[largest_volume.get(cat, 0), ortho_volume.get(cat, 0)],
                              marker_color=volume_colors[cat], text=[largest_volume.get(cat, 0),
                                                                     ortho_volume.get(cat, 0)],
                              textposition='auto'))

    fig3.update_layout(barmode='stack',title=dict(text='Volume Categories: Largest vs Orthosteric Pockets',
                                                  x=0.5, font=dict(size=18)),
                        xaxis_title='Pocket Type', yaxis_title='Number of Pockets', legend_title='Volume Category',
                        template='plotly_white', height=500, width=600)

    fig3.write_html(os.path.join(comparison_dir, 'volume_comparison.html'), include_plotlyjs='cdn')
    print(f"  Saved: plots_comparison/volume_comparison.html")

    # Is the largest pocket orthosteric?
    experiment_summary = summary_df[summary_df['is_largest_pocket']].copy()
    experiment_summary['largest_is_ortho'] = experiment_summary['is_orthosteric'].map({True: 'Yes', False: 'No'})
    by_state = experiment_summary.groupby(['state', 'largest_is_ortho']).size().unstack(fill_value=0)

    fig4 = go.Figure()

    colors_yn = {'Yes': vis.get_viridis_colors(2, greyscale=greyscale)[0],
                 'No': vis.get_viridis_colors(2, greyscale=greyscale)[1] }

    for answer in ['Yes', 'No']:
        if answer in by_state.columns:
            fig4.add_trace(go.Bar(
                name=f'Largest is Orthosteric: {answer}',
                x=by_state.index,
                y=by_state[answer],
                marker_color=colors_yn[answer],
                text=by_state[answer],
                textposition='auto'))

    fig4.update_layout(
        barmode='group',
        title=dict(text='Is the Largest Pocket also Orthosteric?', x=0.5, font=dict(size=18)),
        xaxis_title='State',
        yaxis_title='Number of Experiments',
        template='plotly_white',
        height=500, width=600)

    fig4.write_html(os.path.join(comparison_dir, 'largest_is_orthosteric.html'), include_plotlyjs='cdn')
    print(f"  Saved: plots_comparison/largest_is_orthosteric.html")

    comparison_summary = pd.DataFrame({
        'Category': ['Largest only', 'Orthosteric only', 'Both (largest & orthosteric)', 'Neither'],
        'Count': [largest_only, ortho_only, both, neither],
        'Percentage': [100 * largest_only / len(summary_df), 100 * ortho_only / len(summary_df),
                       100 * both / len(summary_df), 100 * neither / len(summary_df)]})
    comparison_summary.to_csv(os.path.join(comparison_dir, 'comparison_summary.csv'), index=False)
    print(f"  Saved: plots_comparison/comparison_summary.csv")


# =============================================================================

if __name__ == '__main__':
    from config import META_ANALYSIS_DIR, HOLO_BASE_DIR
    t0 = time.time()
    saving_loc_cluster = META_ANALYSIS_DIR
    holo_base = HOLO_BASE_DIR

    bw_file_loc = os.path.join(os.getcwd(), 'TAARs_numbered')
    pdb_file_loc = os.path.join(os.getcwd(), 'holo_structures', 'results', 'holo8ITF', '1')

    results = run_complete_pocket_analysis(
        saving_loc=saving_loc_cluster,
        data_path=os.path.join(saving_loc_cluster, 'summary_df_3d_coords.csv'),
        holo_base=holo_base,
        distance_threshold=5.0,
        volume_col='interpolated_pock_volume',
        bw_file_loc=bw_file_loc,
        pdb_file_loc=pdb_file_loc,
        export_pdfs=False,
        n_cols=4,
        greyscale=False,
        run_3d_heatmap=False,
        run_apo_holo_comparison=False,
        run_violin_plots=False,
        run_consecutive_zeros=True)
    t1 = time.time()
    print(f'The analysis took: {(t1-t0)/60} minutes.')