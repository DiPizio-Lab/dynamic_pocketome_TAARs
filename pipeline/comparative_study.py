import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `scripts` and `config`
# ## Comparative Study of the pocketome
# 
# ### 1) Analysis of similarities and differences of local pockets
# - within PDB structure - why do we get different numbers of pockets in the same PDB? really biologically different and sampled or just because of definition of a pocket
# - within genes - influence of the co-crystallized molecules and the experimental setting
# - between genes - how similar are human and murine TAAR1? --> pairwise comparison
# 
# ### 2) Influence of the Ligand in the orthosteric site
# - volume
# - stability
# - residue flexibility --> how to measure this? (RMSD between 1st frame and highest deviation)
# 

# Imports and Settings
import warnings
import os
import itertools
from collections import Counter
import re
import time
import pandas as pd
# import polars as pl
import upsetplot
from matplotlib import pyplot as plt
import gc
import ast
import plotly.graph_objects as go
import plotly.express as px
from scripts import config as conf
from scripts import basic_analysis
from scripts import pocket_analysis
import meta_analysis_class as meta
from scripts import quality_checks
from scripts import logging as log
import visualizations as vis_en
import orthosteric_filter_ligand_based as filter_ortho

# Settings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# BioPython Deprecation Warning:  The Bio.Application modules and modules relying on it have been deprecated.
pd.set_option('display.max_columns', None)

# ## Helpers and Plotters

def within_pdb_plot(df, pdb_id, cluster_col='Global ID', saving_loc=conf.folder_meta_analysis):
    """Colour global IDs and plot three reps as different markers with replicate selector"""

    df['project_replicate'] = df['ID'].str.extract(r'^([A-Za-z0-9]+_\d)')
    df[['PDB ID', 'replicate']] = df['project_replicate'].str.rsplit('_', expand=True)
    df['pocket_number'] = df['ID'].str.extract(r'p(\d+)')  # Only if 'p' + digits exists
    df['hover_text'] = "ID: " + df['ID'] + "<br>Group: " + df[cluster_col].astype(str)

    unique_replicates = sorted(df['replicate'].unique())
    markers = ['circle', 'square', 'cross', 'diamond']
    symbol_map = {r: markers[i % len(markers)] for i, r in enumerate(unique_replicates)}

    unique_globals = sorted(df[cluster_col].dropna().unique(), key=lambda x: int(x))
    color_sequence = px.colors.qualitative.Dark24
    color_map = {gid: color_sequence[i % len(color_sequence)] for i, gid in enumerate(unique_globals)}

    x_col, y_col, z_col = 'x', 'y', 'z'

    data_traces = []
    visibility_map = {rep: [] for rep in unique_replicates}

    for gid in unique_globals:
        subset = df[df[cluster_col] == gid]
        for rep in unique_replicates:
            sub = subset[subset['replicate'] == rep]
            visible = True  # default visibility
            trace = go.Scatter3d(
                x=sub[x_col], y=sub[y_col], z=sub[z_col],
                mode='markers',
                name=f'Global {gid} – Rep {rep} - PDB {pdb_id}',
                text=sub['hover_text'], hoverinfo='text',
                marker=dict(size=4, color=color_map[gid], symbol=symbol_map[rep]),
                legendgroup=str(gid), showlegend=True,
                visible=visible)
            visibility_map[rep].append(len(data_traces))
            data_traces.append(trace)

    fig = go.Figure(data=data_traces)

    # Create buttons for each replicate and one for all reps
    buttons = []
    for rep in unique_replicates:
        visibility = [False] * len(data_traces)
        for idx in visibility_map[rep]:
            visibility[idx] = True
        buttons.append(dict(
            label=f'Replicate {rep}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'3D Pocket Clustering – Replicate {rep} – PDB {pdb_id}'}]))
    buttons.insert(0, dict(
        label='All Replicates',
        method='update',
        args=[{'visible': [True] * len(data_traces)},
              {'title': f'3D Pocket Clustering – All Replicates – PDB {pdb_id}'}]))
    fig.update_layout(
        title=f'3D Pocket Clustering by Global ID of {pdb_id}',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        template='simple_white',
        legend=dict(x=1.02, y=1),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.0,
            xanchor='left',
            y=1.15,
            yanchor='top')])

    # fig.show()
    fig.write_html(os.path.join(saving_loc, f'{pdb_id}_within_pdb_global_id_plot.html'))


def annotate_uniqueness(df, prj_col='prj', pocket_id_col='Global ID', id_col='ID',
                        gene_col='gene'):
    """Assumes the input DataFrame already contains a column named `gene`.
    Adds these columns (memory-conscious, small intermediates):
      - project_replicate, PDB ID, replicate (extracted from id_col)
      - n_replicates_within_prj (int)
      - unique_in_one_rep (bool)
      - genes_list (semicolon-separated sorted distinct gene names for each pocket)
      - n_distinct_genes (int)
      - unique_to_one_gene (bool)

    Returns the DataFrame reference with added columns. """
    working = df

    working[id_col] = working[id_col].astype(str)
    rep_counts = (working.loc[working[pocket_id_col].notna() & working['rep'].notna(), [prj_col, pocket_id_col, 'rep']]
        .drop_duplicates().groupby([prj_col, pocket_id_col], observed=True)['rep'].nunique()
        .rename('n_replicates_within_prj').reset_index())

    working = working.merge(rep_counts, on=[prj_col, pocket_id_col], how='left')
    working['n_replicates_within_prj'] = working['n_replicates_within_prj'].fillna(0).astype(int)
    working['unique_in_one_rep'] = working['n_replicates_within_prj'] == 1

    del rep_counts
    gc.collect()

    gene_df = working.loc[working[pocket_id_col].notna() & working[gene_col].notna(), [pocket_id_col, gene_col]].drop_duplicates()
    if not gene_df.empty:
        agg = (gene_df.groupby(pocket_id_col, observed=True)[gene_col].agg(lambda s: ';'.join(sorted(map(str, pd.unique(s)))))
            .rename('genes_list').reset_index() )
        agg['n_distinct_genes'] = agg['genes_list'].str.count(';').fillna(0).astype(int) + (agg['genes_list'].notna().astype(int))
        agg['unique_to_one_gene'] = agg['n_distinct_genes'] == 1
    else:
        agg = pd.DataFrame(columns=[pocket_id_col, 'genes_list', 'n_distinct_genes', 'unique_to_one_gene'])

    working = working.merge(agg, on=pocket_id_col, how='left')

    working['genes_list'] = working['genes_list'].where(working['genes_list'].notna(), pd.NA)
    working['n_distinct_genes'] = working['n_distinct_genes'].fillna(0).astype(int)
    working['unique_to_one_gene'] = working['unique_to_one_gene'].fillna(False).astype(bool)

    del gene_df, agg
    gc.collect()

    return working


def make_pocket_summary(df, local_id_col='Local ID', global_id_col='Global ID',
                       volume_col='interpolated_pock_volume',
                       keep_first_cols=('prj', 'rep', 'isovalue', 'gene'), sort_within_local=None):
    """ Collapse frame-level df into one row per Local Pocket ID (keeps Global ID).
    - median_interpolated_volume: median of volume_col per Local ID
    - x,y,z: coordinates from the first frame observed for that Local ID
    - keep_first_cols: tuple of other columns to keep from the first frame (prj, rep, isovalue, gene)
    - sort_within_local: optional column name to sort frames before selecting the 'first' (e.g., 'snapshot')
    Returns a new compact DataFrame."""

    cols_needed = [local_id_col, global_id_col, volume_col] + list(keep_first_cols)
    existing_cols = [c for c in cols_needed if c in df.columns]
    small = df[existing_cols].copy()

    if sort_within_local is not None and sort_within_local in df.columns:
        small = small.sort_values([local_id_col, sort_within_local])

    small = small.drop_duplicates()
    agg_dict = { volume_col: 'median'}
    for c in keep_first_cols:
        if c in small.columns:
            agg_dict[c] = 'first'

    if global_id_col in small.columns:
        agg_dict[global_id_col] = 'first'
    grouped = small.groupby(local_id_col, observed=True).agg(agg_dict)

    grouped = grouped.rename(columns={volume_col: 'median_interpolated_volume', global_id_col: 'Global ID'})

    grouped = grouped.reset_index().rename(columns={local_id_col: 'Local Pocket ID'})

    columns_order = ['Local Pocket ID', 'Global ID', 'prj', 'rep', 'isovalue', 'gene', 'median_interpolated_volume']
    columns_present = [c for c in columns_order if c in grouped.columns]
    grouped = grouped[columns_present]

    return grouped


def upset_plot_pockets_shared(df, saving_loc=conf.folder_meta_analysis):
    """Trying to visualize which global pockets are unique or shared in the four genes / all PDB IDs;
    packaged used: upsetplot (https://upsetplot.readthedocs.io/en/stable/formats.html#Use-DataFrame-as-input:)"""
    # data has to be in following format:
    # booleans for categories, i.e., per global or local ID gene membership as boolean columns (mTAAR1 = True)
    # or PDB ID1 = True
    # df = pd.read_csv(os.path.join(saving_loc, 'pockets_dens', 'pocket_comparison_table.csv'))
    df['genes_list'] = df['genes_list'].fillna('').astype(str).str.strip()
    upset_df = upsetplot.from_memberships(df['genes_list'].str.split(';'), data=df)
    upsetplot.plot(upset_df, show_counts=True)
    plt.savefig(os.path.join(saving_loc, 'upsetplot_gene_comparison.png'))


def barchart_pocket_count(df, saving_loc=conf.folder_meta_analysis):
    """"""
    # df = pd.read_csv(os.path.join(saving_loc, 'pockets_dens', 'pocket_comparison_table.csv'))
    all_genes = sorted({g for genes in df["genes_list"] for g in genes.split(";")})

    combos = []
    for r in range(1, len(all_genes) + 1):
        for c in itertools.combinations(all_genes, r):
            combos.append("-".join(sorted(c)))

    normalized_keys = ["-".join(sorted(set(v.split(";")))) for v in df["genes_list"]]
    counts = Counter(normalized_keys)

    combo_counts = {c: counts.get(c, 0) for c in combos}

    # Sort: singletons first, then by size, then alphabetically
    sorted_keys = sorted(combo_counts.keys(), key=lambda k: (k.count("-"), k))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "font.size": 10})

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sorted_keys, [combo_counts[k] for k in sorted_keys], color="midnightblue", edgecolor="black",
                  linewidth=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Gene Combinations", fontsize=12)
    ax.set_ylabel("Number of Shared Pockets", fontsize=12)
    ax.set_title("Frequency of Gene Combinations", fontsize=13)
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(saving_loc, 'bar_chart_pocket_comparison.png'))


def plot_3d_heatmap(saving_loc=conf.folder_meta_analysis, df='first_frame_summary_df.csv',
                    out_html='3D_coordinates_frequency_colour_coded.html', cluster_col='Global ID', genes_of_interest=None,
                    colorscale='Viridis', marker_size=4, title='3D pockets colored by n_structures (first frame)', 
                    highlighted_pocket=None):
    """ Plot a 3D, interactive scatter of pocket coordinates colored by the number of distinct PDB structures in which
    each global pocket appears, with dropdown controls to filter by PDB or by gene.

    This function reads a precomputed CSV of first-frame pocket summaries, prepares hover text and per-global-pocket
    annotations (list of PDBs and genes), computes how many distinct PDB structures contain each global pocket, builds
    one Plotly 3D scatter trace per PDB and finally writes an interactive HTML file and a CSV for reuse.

    Args:
        df (pandas.DataFrame or str): Either a DataFrame already containing the required columns or a path-like object.
         The expected columns after reading are: 'ID', 'pocket_number', 'Global ID', 'x', 'y', 'z', 'gene'
        out_html (str, optional): Filename (or relative path) to write the interactive HTML output.
            Default: 'all_pdbs_3d.html'
        cluster_col (str, optional): Column used to identify the clustered/global pocket. Default: 'Global ID'.
        genes_of_interest (list of str or None, optional): List of gene names to create dedicated gene-filter buttons.
            If None, a default list is used: ['hTAAR1', 'mTAAR1', 'mTAAR7f', 'mTAAR9'].
        colorscale (str or list, optional): Plotly colorscale name or sequence used to map the number of structures to
            colors. Default: 'Viridis'.
        marker_size (int, optional): Marker size for points in the 3D scatter. Default: 4.
        title (str, optional): Plot title used in the Plotly figure. Default: '3D pockets colored by
            n_structures (first frame)'.

    Returns: None

    Side effects:
    - Writes a CSV with prepared plotting data to: os.path.join(saving_loc,
    'for_plot_3D_coordinates_frequency_colour_coded.csv').

    Notes and behavior details:
    - The CSV is expected to provide coordinates from the "first frame" of an MD simulation.
    - Coordinates 'x', 'y', 'z' are coerced to numeric; non-numeric values become NaN.
    - A per-global-pocket  comma-separated list of PDB IDs and genes is constructed and included in each point's hover text.
    - The color of each marker encodes 'n_structures', the number of distinct PDB IDs in which the same
    Global ID is observed.
    - One Plotly trace is created per unique PDB ID and all replicates for that PDB are included in the same trace.
    Dropdown menus allow toggling visibility to: * show all PDBs * show only a single PDB * show all genes * show only
    a single gene from genes_of_interest - The function currently builds visibility maps and per-PDB / per-gene dropdowns;
    the mapping from PDBs to traces and genes to traces is derived from the input CSV.

    Example: >>> plot_3d_heatmap(df=None, out_html='pockets_3d.html', cluster_col='Global ID',
    genes_of_interest=['mTAAR9'], colorscale='Viridis', marker_size=3, title='Pockets (first frame)') """

    if df is str:
        df = pd.read_csv(os.path.join(saving_loc, df))

    # Convert numeric columns used to numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df[cluster_col] = df[cluster_col].astype(str)

    # df['project_replicate'] = df['ID'].str.extract(r'^(apo|holo)_[A-Z0-9]+_rep$', expand=False)
    # df[['PDB ID', 'replicate']] = df['project_replicate'].str.rsplit('_', expand=True)
    df[['PDB ID', 'replicate', 'pocket']] = df['ID'].str.extract(r'^((?:apo|holo)[A-Z0-9]+)_(\d)_([a-z]\d+)(?:_.*)?$')
    df['primary_genes'] = df['gene'].fillna('').astype(str)
    df['state'] = df['PDB ID'].str.extract(r'^(apo|holo)')

    symbol_map = {"apo": "circle", "holo": "square"}

    pdb_per_global = (df.groupby('Global ID')['PDB ID'].unique()
                      .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['pdb_list'] = df['Global ID'].map(pdb_per_global)
    gene_per_global = (df.groupby('Global ID')['gene'].unique()
                      .apply(lambda arr: ', '.join(sorted([p for p in arr if p and p != 'nan']))))
    df['gene_list'] = df['Global ID'].map(gene_per_global)

    df['hover_text'] = ("ID: " + df['ID'].astype(str)
                        + "<br>Global: " + df[cluster_col].astype(str)
                        + "<br>PDB: " + df['PDB ID'].astype(str)
                        + "<br>in genes: " + df['gene_list'].astype(str)
                        + "<br>in PDB structures: " + df['pdb_list'].astype(str))

    # compute number of distinct PDBs per Global ID (n_structures)
    counts = (df.dropna(subset=[cluster_col, 'PDB ID']).groupby(cluster_col)['PDB ID'].nunique()
              .rename('n_structures').reset_index())
    # merge counts to df (so every row has n_structures)
    df = df.merge(counts, on=cluster_col, how='left')
    df['n_structures'] = pd.to_numeric(df['n_structures'].fillna(0), downcast='integer')

    unique_pdbs = sorted(df['PDB ID'].dropna().unique())
    # genes of interest: default the four requested genes
    if genes_of_interest is None:
        genes_of_interest = ['hTAAR1', 'mTAAR1', 'mTAAR7f', 'mTAAR9']
    # build mapping: PDB -> trace indices and gene -> trace indices
    data_traces = []
    pdb_to_traceidx = {}
    gene_to_traceidxs = {g: [] for g in genes_of_interest}
    idx = 0
    # color scale limits
    cmin = int(df['n_structures'].min())
    cmax = int(df['n_structures'].max())

    # Create one trace per PDB containing all points (all replicates included)
    for pdb in unique_pdbs:
        sub = df[df['PDB ID'] == pdb]
        if sub.empty:
            continue
        trace = go.Scatter3d(x=sub['x'], y=sub['y'],z=sub['z'], mode='markers', name=f'PDB {pdb}',
            text=sub['hover_text'],hoverinfo='text', marker=dict(size=marker_size,color=sub['n_structures'],
                                                    colorscale=colorscale, cmin=cmin, cmax=cmax,
                                                    colorbar=dict(title='Structures') if idx == 0 else None,
                                                    symbol=[symbol_map.get(s, "circle") for s in sub['state']]),
            customdata=sub[['ID', 'primary_genes']].values, showlegend=True, visible=True)
        data_traces.append(trace)
        pdb_to_traceidx[pdb] = [idx]
        # if this PDB contains any of the genes of interest, register mapping
        present_genes = {g for s in sub['primary_genes'].dropna().astype(str) for g in s.split(';') if g}
        for g in genes_of_interest:
            if g in present_genes:
                gene_to_traceidxs[g].append(idx)
        idx += 1

    highlight_df = pd.DataFrame()
    # Add highlighted pockets as a separate trace
    if highlighted_pocket is not None:
        highlight_df = df[df['ID'].isin(highlighted_pocket)]
        if not highlight_df.empty:
            highlight_trace = go.Scatter3d(
                x=highlight_df['x'],
                y=highlight_df['y'],
                z=highlight_df['z'],
                mode='markers',
                name='Highlighted Pockets',
                text=highlight_df['hover_text'],
                hoverinfo='text',
                marker=dict(size=marker_size + 2, color='darkred', symbol='diamond'),
                customdata=highlight_df[['ID', 'primary_genes']].values,
                showlegend=True,
                visible=True)
            data_traces.append(highlight_trace)

    total_traces = len(data_traces)

    vis_all = [True] * total_traces
    pdb_visibility_map = {}
    for pdb, idx_list in pdb_to_traceidx.items():
        vis = [False] * total_traces
        for i in idx_list:
            vis[i] = True
        pdb_visibility_map[pdb] = vis
    gene_visibility_map = {}
    for g, idx_list in gene_to_traceidxs.items():
        vis = [False] * total_traces
        for i in idx_list:
            vis[i] = True
        gene_visibility_map[g] = vis
    if not highlight_df.empty:
        total_traces += 1
        vis_all.append(True)
        for vis_map in pdb_visibility_map.values():
            vis_map.append(True)
        for vis_map in gene_visibility_map.values():
            vis_map.append(True)

    # build PDB buttons (All + each PDB)
    pdb_buttons = []
    pdb_buttons.append(dict(label='All PDBs', method='update',
                            args=[{'visible': vis_all}, {'title': f'{title} (All PDBs)'}]))
    for pdb in unique_pdbs:
        pdb_buttons.append(dict(label=str(pdb), method='update',
            args=[{'visible': pdb_visibility_map.get(pdb, [False] * total_traces)},
                  {'title': f'{title} (PDB {pdb})'}]))

    # build Gene buttons (All Genes + each gene of interest)
    gene_buttons = []
    gene_buttons.append(dict(label='All Genes',method='update',
        args=[{'visible': vis_all},{'title': f'{title} (All Genes)'}]))
    for g in genes_of_interest:
        gene_buttons.append(dict(label=g, method='update',
            args=[{'visible': gene_visibility_map.get(g, [False] * total_traces)},
                  {'title': f'{title} (Gene {g})'}]))

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        template='simple_white',
        legend=dict(x=1.02, y=1),
        updatemenus=[
            dict(buttons=pdb_buttons, direction='down',
                showactive=True,
                x=0.0, xanchor='left',
                y=1.18, yanchor='top',
                bgcolor='white'),
            dict(buttons=gene_buttons,
                direction='down',
                showactive=True,
                x=0.28, xanchor='left',
                y=1.18, yanchor='top',
                bgcolor='white')])
    fig.update_layout(legend=dict(x=1.12, y=0.95, traceorder='normal', bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black', borderwidth=0.5), margin=dict(l=60, r=220, t=120, b=60))

    # write html
    fig.write_html(os.path.join(saving_loc, out_html))
    # fig.show()
    df.to_csv(os.path.join(saving_loc, 'for_plot_3D_coordinates_frequency_colour_coded.csv'))


def transient_stable_categorisation(descriptor_df):
    """
    Categorize pockets as transient or stable based on volume data.
    A pocket is transient if interpolated_pock_volume is zero in >= 10% of frames.

    Parameters:
    -----------
    descriptor_df : pd.DataFrame
        DataFrame containing pocket descriptors with columns 'ID' and 'interpolated_pock_volume'

    Returns:
    --------
    descriptor_df : pd.DataFrame
        Original DataFrame with added 'transient' column (boolean)
    transient_dict : dict
        Dictionary mapping pocket ID to transient status (boolean)
    """

    # Calculate transient status for each pocket ID
    transient_status = (
        descriptor_df.groupby('ID')['interpolated_pock_volume']
        .apply(lambda x: (x == 0.0).sum() > 0.1 * len(x))
    )

    # Map the transient status back to the dataframe
    descriptor_df['transient'] = descriptor_df['ID'].map(transient_status)

    # Create dictionary
    transient_dict = transient_status.to_dict()

    return descriptor_df, transient_dict


def run_analysis(prj_ls, saving_loc=conf.folder_meta_analysis):
    """"""
    # Instantiate Meta Analysis Class
    meta_obj = meta.MetaAnalysis(pocket_dirs=prj_ls, saving_loc=saving_loc,
                                 isovalue=3.0, verbose=False, dbscan=False, min_atoms=10)

    # get the necessary dataframes from the parser
    result_dict, pockets_to_drop = meta_obj.pock_file_parser()
    intpol_descr_df = meta_obj.interpolation_volume(df=result_dict['descriptor_df'],
                                                    col_to_interpolate=['pock_volume'], limit=2,
                                                    limit_area='inside', method='linear')
    summary_df = pd.merge(result_dict['dummy_atom_df'], intpol_descr_df,
                             on=['ID', 'pocket_number', 'prj', 'rep', 'isovalue'],
                             how='outer')

    # Assign global cluster
    voxelized_df = meta_obj.voxel_intersection_over_union_global_id(summary_df=summary_df)  # info about the local and global clusters with all intermediate results, also saves a txt file used as mapping dict
    # read in the mapping between local and global ID and append global ID column in summary df
    mapping_global_to_local = {}
    with open(os.path.join(saving_loc, 'local_to_globalVoxelID.txt')) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key_str, val_str = line.split(':', 1)
                key = key_str.strip()
                val_list = ast.literal_eval(val_str.strip())
                mapping_global_to_local[key] = val_list
    inverted_mapping_dict = {}
    for key, values in mapping_global_to_local.items():
        for val in values:
            inverted_mapping_dict[val] = key
    summary_df['Local ID'] = summary_df['ID'].str.replace('_i3.0$', '', regex=True)
    summary_df['Global ID'] = summary_df['Local ID'].map(inverted_mapping_dict)

    for pdb_id, pdb_df in summary_df.groupby('prj'):
        within_pdb_plot(pdb_df, pdb_id=pdb_id, cluster_col='Global ID', saving_loc=saving_loc)

    short_gene_names = {}
    summary_df['clean_id'] = summary_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)
    pdb_id_ls = summary_df['clean_id'].unique().tolist()
    #TODO: deprecated because RCSB-API package is down
    # mapping_dict = meta_obj._get_pdb_gene_dict(id_list=pdb_id_ls, pdb_ids=True)
    #
    # for prj_id, gene_name in mapping_dict.items():
    #     gene_name = gene_name.replace('Traceamineareceptor', 'TAAR')
    #     if gene_name.endswith('F'):
    #         gene_name = gene_name[:-1] + 'f'
    #     short_gene_names[prj_id] = gene_name

    short_gene_names = {
        '8ITF': 'mTAAR9', '8IW4': 'mTAAR9', '8IW7': 'mTAAR9', '8IWM': 'mTAAR7f',
        '8JLJ': 'mTAAR1', '8JLK': 'mTAAR1', '8JLN': 'mTAAR9', '8JLO': 'hTAAR1',
        '8JLP': 'hTAAR1', '8JLQ': 'hTAAR1', '8JLR': 'hTAAR1', '8JSO': 'hTAAR1',
        '8PM2': 'mTAAR7f', '8W87': 'hTAAR1', '8W88': 'hTAAR1', '8W89': 'hTAAR1',
        '8W8A': 'hTAAR1', '8WC3': 'mTAAR1', '8WC4': 'mTAAR1', '8WC5': 'mTAAR1',
        '8WC6': 'mTAAR1', '8WC7': 'hTAAR1', '8WC8': 'hTAAR1', '8WC9': 'mTAAR9',
        '8WCB': 'mTAAR1', '8WCC': 'hTAAR1'}

    summary_df['gene'] = summary_df['clean_id'].map(short_gene_names)
    summary_df.to_csv(os.path.join(saving_loc, 'summary_df_3d_coords.csv'), index=False)
    small_df = summary_df[['prj', 'rep', 'ID','Global ID', 'gene', 'clean_id']]
    pocket_summary = make_pocket_summary(summary_df)
    annotated_df = annotate_uniqueness(small_df)

    annotated_df = annotated_df[['prj', 'rep','Global ID', 'gene','n_replicates_within_prj', 'unique_in_one_rep',
                                 'genes_list', 'n_distinct_genes', 'unique_to_one_gene']]

    summary_df_upd = pocket_summary.merge(annotated_df, on=['prj', 'rep', 'Global ID', 'gene'],
                                      how='outer')

    # filter orthosteric binding site pockets and merge the boolean to the df
    maestro_bind_site_str = {
        'hTAAR1': {'72', '83', '100', '103', '104', '107', '108', '111', '112', '150', '151', '154', '158', '179',
                   '182', '183', '184', '185', '186', '193', '194', '195', '197', '198', '264', '267', '268', '271',
                   '286', '289', '290', '291', '293', '294', '297'},
        'mTAAR1': {'75', '102', '103', '106', '107', '153', '183', '185', '193', '194', '196', '197', '261', '264',
                   '265', '268', '286', '287', '288', '290', '291'},
        'mTAAR7f': {'127', '128', '131', '132', '213', '217', '286', '289', '308', '312', '315', '316'},
        'mTAAR9': {'81', '99', '102', '103', '106', '107', '109', '112', '113', '114', '115', '116', '117', '118',
                   '119', '120', '153', '163', '168', '182', '183', '184', '185', '191', '192', '193', '194', '197',
                   '202', '261', '264', '265', '271', '274', '275', '278', '287', '290', '291', '293', '297', '298',
                   '300', '301', '304'}}

    merged_residue_df = result_dict['res_to_pock_df']
    merged_residue_df['residue_id'] = merged_residue_df['residue_id'].astype(int)
    merged_residue_df['clean_id'] = merged_residue_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)
    merged_residue_df['gene'] = merged_residue_df['clean_id'].map(short_gene_names)

    pocket_df_ls = []
    for gene, gene_df in merged_residue_df.groupby('gene'):
        gene_df_ortho = meta_obj.filter_ortho_site(gene_df, maestro_bind_site_str, min_overlap=0.8)
        pocket_df_ls.append(gene_df_ortho)

    ortho_pockets = pd.concat(pocket_df_ls)

    bind_cols = [col for col in ortho_pockets.columns if col.startswith('is_binding_site')]
    # mask_cols = ortho_pockets.filter(like="is_binding_site").columns
    ortho_pockets = ortho_pockets.loc[ortho_pockets[bind_cols].any(axis=1)]
    ortho_pockets['Local Pocket ID'] = (ortho_pockets['prj'] + '_' + ortho_pockets['rep'] +
                                        '_p' + ortho_pockets['pocket_number'])
    ortho_pockets['is_binding_site_any'] = ortho_pockets[bind_cols].any(axis=1)

    ortho_dict = dict(zip(ortho_pockets['Local Pocket ID'], ortho_pockets['is_binding_site_any']))
    ortho_res_dict = dict(zip(ortho_pockets['Local Pocket ID'], ortho_pockets['residues']))
    summary_df_upd['is_binding_site'] = summary_df_upd['Local Pocket ID'].map(ortho_dict)
    summary_df_upd['pocket_residues'] = summary_df_upd['Local Pocket ID'].map(ortho_res_dict)
    # drop duplicated rows of the same local pocket ID
    summary_df_upd = summary_df_upd.drop_duplicates(['Local Pocket ID'])
    summary_df_upd.to_csv(os.path.join(saving_loc, 'pocket_comparison_table.csv'),
                          index=False)

    unique_to_gene = summary_df_upd.loc[summary_df_upd['unique_to_one_gene'] == True]
    unique_to_gene.to_csv(os.path.join(saving_loc, 'unique_to_gene_comparison.csv'),
                          index=False)
    log.log(f'unique to a gene \n {unique_to_gene} \n {unique_to_gene.shape} \n')
    log.log(f'number of local IDs {unique_to_gene['Local Pocket ID'].nunique()}'
            f'number of global IDs {unique_to_gene['Global ID'].nunique()}')
    print('unique to a gene')
    print(unique_to_gene)
    print(unique_to_gene.shape)
    print(f'number of local IDs {unique_to_gene['Local Pocket ID'].nunique()}')
    print(f'number of global IDs {unique_to_gene['Global ID'].nunique()}')
    print('\n')
    unique_to_structure = summary_df_upd.loc[summary_df_upd['unique_in_one_rep'] == True]
    unique_to_structure.to_csv(os.path.join(saving_loc, 'unique_to_structure_comparison.csv'),
                          index=False)
    log.log(f'unique to structure \n {unique_to_structure} \n {unique_to_structure.shape} \n')
    log.log(f'number of local IDs {unique_to_structure['Local Pocket ID'].nunique()} \n '
            f'number of global IDs {unique_to_structure['Global ID'].nunique()}')
    print('unique to a PDB structure')
    print(unique_to_structure)
    print(unique_to_structure.shape)
    print(f'number of local IDs {unique_to_structure['Local Pocket ID'].nunique()}')
    print(f'number of global IDs {unique_to_structure['Global ID'].nunique()}')
    print('\n')
    shared_in_all = summary_df_upd.loc[summary_df_upd['n_distinct_genes'] == 4]
    shared_in_all.to_csv(os.path.join(saving_loc, 'shared_in_all_gene_comparison.csv'),
            index=False)
    log.log(f'present in all 4 genes {shared_in_all} \n {shared_in_all.shape} \n')
    log.log(f'number of local IDs {shared_in_all['Local Pocket ID'].nunique()} \n '
            f'number of global IDs {shared_in_all['Global ID'].nunique()}')
    print('present in all 4 genes')
    print(shared_in_all)
    print(shared_in_all.shape)
    print(f'number of local IDs {shared_in_all['Local Pocket ID'].nunique()}')
    print(f'number of global IDs {shared_in_all['Global ID'].nunique()}')
    # Plots
    # make sure that we have a first frame only dataframe:
    # with large datasets, reading the files fails because of memory
    #  --> read only every 1st frame of the file to plot the heatmap
    df_firstframe = pd.read_csv(os.path.join(saving_loc, 'summary_df_3d_coords.csv'),
                              skiprows=lambda i: i > 0 and (i-1) % 1000 != 0)
    df_firstframe = df_firstframe.loc[:, ['ID', 'pocket_number', 'Global ID', 'x', 'y', 'z', 'gene']]
    df_firstframe = df_firstframe.reset_index(drop=True)
    df_firstframe.to_csv(os.path.join(os.path.join(saving_loc, 'first_frame_summary_df.csv')))
    # upset_plot_pockets_shared(summary_df_upd, saving_loc)
    # barchart_pocket_count(summary_df_upd, saving_loc)
    # plot_3d_heatmap(df=df_firstframe, saving_loc=saving_loc)
    try:
        upset_plot_pockets_shared(summary_df_upd, saving_loc)
    except Exception as e:
        print(f"Error in upset_plot_pockets_shared: {e}")

    try:
        barchart_pocket_count(summary_df_upd, saving_loc)
    except Exception as e:
        print(f"Error in barchart_pocket_count: {e}")

    try:
        plot_3d_heatmap(df=df_firstframe, saving_loc=saving_loc)
    except Exception as e:
        print(f"Error in plot_3d_heatmap: {e}")


def largest_pocket_stability_analysis():
    """launches several functions and methods designed to filter for the largest pocket and calculate the percentage
    of transient pockets per given category """
    from config import META_ANALYSIS_DIR
    saving_loc_cluster = META_ANALYSIS_DIR
    saving_loc_local = META_ANALYSIS_DIR

    first_frame_df = pd.read_csv(os.path.join(saving_loc_cluster, 'first_frame_summary_df.csv'))
    all_data_df = pd.read_csv(os.path.join(saving_loc_cluster, 'summary_df_3d_coords.csv'),
                              # nrows=500000
                              )

    # Task 1: find the largest pocket (including the coordinates of all alpha_spheres that make up the largest pocket)
    # 1.1: separate to frames and find max. vol per state_PDBID_rep_isovalue
    # Find which UNIQUE pocket ID has the largest volume
    all_data_df['state_pdbid_rep'] = all_data_df['ID'].str.extract(r'^([a-z]+[A-Z0-9]+_\d+)')[0]

    pocket_volumes = all_data_df.groupby('ID')['pock_volume'].first().reset_index()
    pocket_volumes['state_pdbid_rep'] = pocket_volumes['ID'].str.extract(r'^([a-z]+[A-Z0-9]+_\d+)')[0]
    largest_ids = pocket_volumes.loc[pocket_volumes.groupby('state_pdbid_rep')['pock_volume'].idxmax(), 'ID']
    all_data_df['is_largest_pocket'] = all_data_df['ID'].isin(largest_ids)
    # Filter to only largest pockets and select columns, i.e., descriptors and identifiers
    cols_to_keep = ['ID', 'gene', 'pocket_number', 'x', 'y', 'z', 'snapshot', 'pock_volume', 'pock_asa',
                    'pock_pol_asa', 'pock_apol_asa', 'pock_asa22', 'pock_pol_asa22', 'pock_apol_asa22', 'nb_AS',
                    'mean_as_ray', 'mean_as_solv_acc', 'apol_as_prop', 'mean_loc_hyd_dens', 'hydrophobicity_score',
                    'volume_score', 'polarity_score', 'charge_score', 'prop_polar_atm', 'as_density', 'as_max_dst',
                    'convex_hull_volume', 'nb_abpa', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                    'interpolated_pock_volume', 'Local ID', 'Global ID', 'clean_id', 'is_largest_pocket']

    largest_pocket_df = all_data_df[all_data_df['is_largest_pocket']].reset_index(drop=True)[cols_to_keep]
    # 1.2 plot the largest pocket to see if it is always in the same location --> one plot with all 152 largest
    # pockets plotted on top of each other as well as volume & hydrophobicity over time
    bw_file_loc = os.path.join(os.getcwd(), 'TAARs_numbered')
    pdb_file_loc = os.path.join(os.getcwd(), 'holo_structures', 'results', 'holo8ITF', '1')
    # for the backbone in the plots 8ITF is used (we just need the rough outline for orientation, not exact measurements)

    vis_en.plot_largest_pockets(
        saving_loc=saving_loc_cluster,
        pocket_df=largest_pocket_df,
        pdb_file=os.path.join(pdb_file_loc, 'aligned_top.pdb'),
        bw_csv_file=os.path.join(bw_file_loc, 'taar9_mouse.csv'),
        backbone=True,
        export_pdfs=True,
        use_cdn=True)  # reduces file size

    result_df, transient_dict = transient_stable_categorisation(all_data_df)
    vis_en.plot_volume_category_distribution(all_data_df, saving_loc_cluster)
    vis_en.plot_transient_stable_distribution(all_data_df, transient_dict, saving_loc_cluster)
    vis_en.plot_combined_volume_stability(all_data_df, transient_dict, saving_loc_cluster)
    vis_en.create_all_distribution_plots(all_data_df, transient_dict, saving_loc=saving_loc_cluster,
                                         volume_col='interpolated_pock_volume', n_cols=4, greyscale=False)


# IV Main
def main():
    t0 = time.time()
    largest_pocket_stability_analysis()
    # prj_ls = sys.argv[1:]
    # run_analysis(prj_ls)
    # upset_plot_pockets_shared()
    # barchart_pocket_count()
    t1 = time.time()
    t_min = (t1 - t0) / 60
    t_hr = t_min / 60
    print('time:  ', t1 - t0, '  sec')
    print('time:  ', t_min, '  min')
    print('time:  ', t_hr, '  hours')


if __name__ == '__main__':
    main()

