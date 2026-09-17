"""
Step 2.2: assigns Global IDs on a chosen subset of Step 2.1's pockets, and compares apo vs holo.
A pocket's Global ID is only meaningful within the run that produced it - two pockets can
only share a Global ID if they were voxel-clustered together, so Global IDs from two different
subsets are NOT comparable (see match_states() below for the one case where you deliberately
want to reconcile two separate runs). This script therefore always clusters ONE chosen subset
of pockets by state (apo/holo/both-together), gene, and/or PDB ID.

Reads all_pockets, pocket_summary (pipeline/pocket_io.py, either .csv or .parquet -
pocket_dataframes.py's output), from config.META_ANALYSIS_DIR by default.

Writes, under saving_loc, as both .csv and .parquet:
    pocket_comparison_table   one row per Local Pocket ID in the subset: Global ID, gene/state/
                              PDB uniqueness annotations, and the is_orthosteric / transient /
                              volume_category columns inherited from Step 2.1
    apo_holo_pocketome_summary.csv   per-PDB-ID apo/holo comparison (if requested)
plus diagnostic/QC output: global_pockets_IoU_voxel.csv, local_to_globalVoxelID.txt,
pocket_clusters_qc.html, per-PDB within_pdb_global_id_plot.html, upset/bar-chart gene-overlap
plots, unique_to_gene/unique_to_structure/shared_in_all_gene comparison csvs.
"""
import gc
import itertools
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import upsetplot
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as conf
from scripts import logging as logger
from pipeline import pocket_io
from taar_paper_figures import taar_style as ts
from taar_paper_figures import pocketome_metrics as pm
from taar_paper_figures import fig2_binding_site as fig2

pd.set_option('display.max_columns', None)

VOXEL_SIZE = 1.0          # Angstroem per voxel
IOU_THRESH = 0.3          # require >= this IoU to link two pockets into the same global one
DILATION_RADIUS = 5       # pad each pocket by this many voxels to absorb small shifts
MIN_CLUSTER_SIZE = 2      # connected components smaller than this are singleton/noise (group 0)
# defaults tuned against hTAAR1, mTAAR1, mTAAR7f and mTAAR9 overlap
# (cf. Rienaecker et al.: "Dynamic Pocketome of Trace Amine-Associated Receptors", 2026)

COL_PDB, COL_STATE, COL_REP, COL_CAT, COL_ORTHO = 'pdb_id', 'state', 'rep', 'volume_category', 'is_orthosteric'
VOL_CATEGORIES = {'Small (<250)': 'small', 'Medium (250-500)': 'medium', 'Large (500-750)': 'large',
                  'Very Large (>750)': 'very_large'}

def select_subset(all_pockets, states=None, genes=None, pdb_ids=None):
    """Filters Step 2.1's all_pockets (or pocket_summary) down to the pockets that should be
    clustered together in one Global ID run. states/genes/pdb_ids are optional lists;
    None/empty means "no filter on this axis"."""
    df = all_pockets
    if states:
        df = df[df['state'].isin(states)]
    if genes:
        df = df[df['gene'].isin(genes)]
    if pdb_ids:
        df = df[df['pdb_id'].isin(pdb_ids)]
    return df.copy()


def pocket_point_dataframe(df):
    """One row per unique (Pocket ID, x, y, z) alpha-sphere position - the deduplicated point
    cloud that defines each pocket's spatial extent, independent of how many frames it was
    seen in (all_pockets repeats every point once per frame).
    """
    flat = df[['ID', 'x', 'y', 'z', 'prj', 'rep']].drop_duplicates(subset=['ID', 'x', 'y', 'z']).copy()
    flat = flat.rename(columns={'ID': 'Pocket ID'})
    return flat


def _dilate_voxels(voxel_set, radius):
    """Dilates a set of integer voxel coordinates by radius (Euclidean distance transform)."""
    pts = np.array(list(voxel_set))
    mins = pts.min(axis=0) - radius
    maxs = pts.max(axis=0) + radius
    shape = (maxs - mins + 1).astype(int)
    mask = np.zeros(shape, dtype=bool)
    for v in voxel_set:
        mask[tuple((v - mins).astype(int))] = True
    dilated = distance_transform_edt(~mask) <= radius
    return {tuple(coord + mins) for coord in np.argwhere(dilated)}


def plot_pocket_clusters_qc(df, saving_loc, cluster_col='voxel_group_id'):
    """QC 3D scatter of pocket points, colored by cluster_col. df needs ['Pocket ID','x','y','z',
    cluster_col]; PDB ID and hover text are derived from Pocket ID."""
    df = df.copy()
    df['PDB ID'] = df['Pocket ID'].str.extract(r'^((?:apo|holo)[A-Za-z0-9]+)_\d', expand=False)
    df['hover_text'] = "ID: " + df['Pocket ID'] + "<br>Group: " + df[cluster_col].astype(str)

    fig1 = go.Figure()
    for pdb_id in df['PDB ID'].unique():
        subset = df[df['PDB ID'] == pdb_id]
        fig1.add_trace(go.Scatter3d(
            x=subset['x'], y=subset['y'], z=subset['z'], mode='markers',
            marker=dict(size=4, color=subset[cluster_col], colorscale='Viridis',
                       colorbar=dict(title=cluster_col), opacity=0.8),
            name=pdb_id, text=subset['hover_text'], hoverinfo='text'))
    fig1.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                       title=f'3D Pocket Clustering by {cluster_col}', template='simple_white',
                       margin=dict(l=0, r=0, b=0, t=40), legend=dict(x=1, y=1))
    fig1.write_html(os.path.join(saving_loc, 'pocket_clusters_qc.html'))

    unique_clusters = sorted(df[cluster_col].dropna().unique(), key=lambda x: int(x))
    color_sequence = px.colors.qualitative.Dark24
    color_map = {c: color_sequence[i % len(color_sequence)] for i, c in enumerate(unique_clusters)}
    data_traces = []
    for cluster in unique_clusters:
        subset = df[df[cluster_col] == cluster]
        for pdb in subset['PDB ID'].unique():
            sub = subset[subset['PDB ID'] == pdb]
            data_traces.append(go.Scatter3d(
                x=sub['x'], y=sub['y'], z=sub['z'], mode='markers', name=f'Grp {cluster} - {pdb}',
                text=sub['hover_text'], hoverinfo='text',
                marker=dict(size=4, color=color_map[cluster]), legendgroup=str(cluster), showlegend=True))
    fig2_ = go.Figure(data=data_traces)
    fig2_.update_layout(title=f'3D Pocket Clustering by {cluster_col}',
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                        template='simple_white', legend=dict(x=1.02, y=1))
    fig2_.write_html(os.path.join(saving_loc, f'detailed_pocket_clusters_{cluster_col}_qc.html'))


def voxel_intersection_over_union_global_id(saving_loc, summary_df=None, point_df=None,
                                            voxel_size=VOXEL_SIZE, iou_thresh=IOU_THRESH,
                                            dilation_radius=DILATION_RADIUS,
                                            min_cluster_size=MIN_CLUSTER_SIZE, make_qc_plot=True):
    """
    Global ID clustering. Concept: voxelize every pocket's point cloud at voxel_size resolution
    (optionally dilated to absorb small shifts between replicates), compute pairwise
    Intersection-over-Union, build a graph (nodes = local pockets, edge = IoU >= iou_thresh),
    and take connected components as Global IDs. Components smaller than min_cluster_size are
    reassigned to their nearest labelled neighbour (1-NN on centroid) rather than left as noise.
    Pass either summary_df (all_pockets or a subset of it - pocket_point_dataframe() is run
    for you) or an already-reshaped point_df (pocket_point_dataframe() output).
    Returns the point-level dataframe with an added 'voxel_group_id' column (== Global ID).
    Also writes global_pockets_IoU_voxel.csv, local_to_globalVoxelID.txt,
    reduced_local_to_globalVoxelID.csv, and (if make_qc_plot) the QC scatter plots.
    """
    if point_df is None:
        if summary_df is None:
            raise ValueError('Pass either summary_df or point_df')
        point_df = pocket_point_dataframe(summary_df)
    df = point_df.copy()

    pocket_voxels, pocket_bounds = {}, {}
    for pid, grp in df.groupby('Pocket ID'):
        xyz = np.floor(grp[['x', 'y', 'z']].values / voxel_size).astype(int)
        vox = set(map(tuple, xyz))
        if dilation_radius > 0:
            vox = _dilate_voxels(vox, dilation_radius)
        pocket_voxels[pid] = vox
        mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
        pocket_bounds[pid] = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

    graph = nx.Graph()
    graph.add_nodes_from(pocket_voxels)
    for p1, p2 in combinations(pocket_voxels, 2):
        inter = pocket_voxels[p1] & pocket_voxels[p2]
        if not inter:
            continue
        iou = len(inter) / len(pocket_voxels[p1] | pocket_voxels[p2])
        if iou >= iou_thresh:
            graph.add_edge(p1, p2)

    mapping, group_id = {}, 1
    for comp in nx.connected_components(graph):
        if len(comp) >= min_cluster_size:
            for pid in comp:
                mapping[pid] = group_id
            group_id += 1
        else:
            for pid in comp:
                mapping[pid] = 0  # noise/singleton, reassigned below

    centroids = df.groupby('Pocket ID')[['x', 'y', 'z']].mean().rename(
        columns={'x': 'cx', 'y': 'cy', 'z': 'cz'})
    centroids['group'] = centroids.index.map(mapping)
    labeled_mask, noise_mask = centroids['group'] > 0, centroids['group'] == 0
    if noise_mask.any() and labeled_mask.any():
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(centroids.loc[labeled_mask, ['cx', 'cy', 'cz']].values, centroids.loc[labeled_mask, 'group'].values)
        centroids.loc[noise_mask, 'group'] = knn.predict(centroids.loc[noise_mask, ['cx', 'cy', 'cz']].values)
        for pid, group in centroids['group'].items():
            mapping[pid] = int(group)

    df['voxel_group_id'] = df['Pocket ID'].map(mapping)
    if make_qc_plot:
        plot_pocket_clusters_qc(df, saving_loc, cluster_col='voxel_group_id')
    pocket_io.save_table(df, saving_loc, 'global_pockets_IoU_voxel', formats=('csv',))

    # Kept isovalue-inclusive (== 'Pocket ID', not stripped) so it matches Local ID in
    # pocket_dataframes.py exactly
    df['Pocket ID local'] = df['Pocket ID']
    df['prj'] = df['Pocket ID local'].str.extract(r'^([^_]+)')
    df['rep'] = df['Pocket ID local'].str.extract(r'_(\d)_')
    df['pocket_number local'] = df['Pocket ID local'].str.extract(r'_p(\d+)')
    df['Pocket ID global'] = df['voxel_group_id']
    reduced_df = df[['Pocket ID local', 'Pocket ID global', 'prj', 'rep', 'pocket_number local']].drop_duplicates()
    reduced_df.to_csv(os.path.join(saving_loc, 'reduced_local_to_globalVoxelID.csv'), index=False)

    voxel_to_pockets = df.groupby('voxel_group_id')['Pocket ID local'].apply(set).to_dict()
    with open(os.path.join(saving_loc, 'local_to_globalVoxelID.txt'), 'w') as f:
        for voxel_id, pockets in voxel_to_pockets.items():
            f.write(f"{voxel_id}: {sorted(pockets)}\n")

    return df


def make_pocket_summary(df, local_id_col='Local ID', global_id_col='Global ID',
                        volume_col='interpolated_pock_volume',
                        keep_first_cols=('prj', 'rep', 'gene', 'state', 'pdb_id',
                                         'is_orthosteric', 'is_largest_pocket', 'transient',
                                         'volume_category')):
    """Collapses the (already Global-ID-tagged) subset into one row per Local Pocket ID."""
    cols_needed = [local_id_col, global_id_col, volume_col] + list(keep_first_cols)
    small_df = df[[c for c in cols_needed if c in df.columns]].drop_duplicates()

    agg_dict = {volume_col: 'median'}
    for col in keep_first_cols:
        if col in small_df.columns:
            agg_dict[col] = 'first'
    if global_id_col in small_df.columns:
        agg_dict[global_id_col] = 'first'

    grouped = small_df.groupby(local_id_col, observed=True).agg(agg_dict)
    grouped = grouped.rename(columns={volume_col: 'median_interpolated_volume', global_id_col: 'Global ID'})
    loc_to_glob = grouped.reset_index().rename(columns={local_id_col: 'Local Pocket ID'})
    return loc_to_glob


def annotate_uniqueness(df, prj_col='prj', pocket_id_col='Global ID', id_col='ID', gene_col='gene'):
    """Adds n_replicates_within_prj, unique_in_one_rep, genes_list, n_distinct_genes,
    unique_to_one_gene. How widely each Global ID is shared across replicates/genes."""
    working_df = df
    working_df[id_col] = working_df[id_col].astype(str)

    rep_counts = (working_df.loc[working_df[pocket_id_col].notna() & working_df['rep'].notna(),
                              [prj_col, pocket_id_col, 'rep']]
                 .drop_duplicates().groupby([prj_col, pocket_id_col], observed=True)['rep']
                 .nunique().rename('n_replicates_within_prj').reset_index())
    working_df = working_df.merge(rep_counts, on=[prj_col, pocket_id_col], how='left')
    working_df['n_replicates_within_prj'] = working_df['n_replicates_within_prj'].fillna(0).astype(int)
    working_df['unique_in_one_rep'] = working_df['n_replicates_within_prj'] == 1

    gene_df = working_df.loc[working_df[pocket_id_col].notna() & working_df[gene_col].notna(),
                          [pocket_id_col, gene_col]].drop_duplicates()
    if not gene_df.empty:
        agg = (gene_df.groupby(pocket_id_col, observed=True)[gene_col]
              .agg(lambda s: ';'.join(sorted(map(str, pd.unique(s))))).rename('genes_list').reset_index())
        agg['n_distinct_genes'] = agg['genes_list'].str.count(';').fillna(0).astype(int) + 1
        agg['unique_to_one_gene'] = agg['n_distinct_genes'] == 1
    else:
        agg = pd.DataFrame(columns=[pocket_id_col, 'genes_list', 'n_distinct_genes', 'unique_to_one_gene'])

    working_df = working_df.merge(agg, on=pocket_id_col, how='left')
    working_df['genes_list'] = working_df['genes_list'].where(working_df['genes_list'].notna(), pd.NA)
    working_df['n_distinct_genes'] = working_df['n_distinct_genes'].fillna(0).astype(int)
    working_df['unique_to_one_gene'] = working_df['unique_to_one_gene'].fillna(False).astype(bool)

    return working_df


def within_pdb_plot(df, pdb_id, saving_loc, cluster_col='Global ID'):
    """Per-PDB 3D scatter of pockets colored by Global ID, one marker shape per replicate."""
    df = df.copy()
    df['replicate'] = df['ID'].str.extract(r'_(\d+)_p\d', expand=False)
    df['hover_text'] = "ID: " + df['ID'] + "<br>Group: " + df[cluster_col].astype(str)

    unique_replicates = sorted(df['replicate'].dropna().unique())
    markers = ['circle', 'square', 'cross', 'diamond']
    symbol_map = {r: markers[i % len(markers)] for i, r in enumerate(unique_replicates)}
    unique_globals = sorted(df[cluster_col].dropna().unique(), key=lambda x: int(x))
    color_sequence = px.colors.qualitative.Dark24
    color_map = {gid: color_sequence[i % len(color_sequence)] for i, gid in enumerate(unique_globals)}

    data_traces, visibility_map = [], {rep: [] for rep in unique_replicates}
    for gid in unique_globals:
        subset = df[df[cluster_col] == gid]
        for rep in unique_replicates:
            sub = subset[subset['replicate'] == rep]
            visibility_map[rep].append(len(data_traces))
            data_traces.append(go.Scatter3d(
                x=sub['x'], y=sub['y'], z=sub['z'], mode='markers',
                name=f'Global {gid} - Rep {rep} - PDB {pdb_id}', text=sub['hover_text'], hoverinfo='text',
                marker=dict(size=4, color=color_map[gid], symbol=symbol_map[rep]),
                legendgroup=str(gid), showlegend=True, visible=True))

    buttons = [dict(label='All Replicates', method='update',
                    args=[{'visible': [True] * len(data_traces)},
                          {'title': f'3D Pocket Clustering - All Replicates - PDB {pdb_id}'}])]
    for rep in unique_replicates:
        visibility = [False] * len(data_traces)
        for idx in visibility_map[rep]:
            visibility[idx] = True
        buttons.append(dict(label=f'Replicate {rep}', method='update',
                            args=[{'visible': visibility}, {'title': f'3D Pocket Clustering - Replicate {rep} - PDB {pdb_id}'}]))

    fig = go.Figure(data=data_traces)
    fig.update_layout(title=f'3D Pocket Clustering by Global ID of {pdb_id}',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      template='simple_white', legend=dict(x=1.02, y=1),
                      updatemenus=[dict(buttons=buttons, direction='down', showactive=True,
                                       x=0.0, xanchor='left', y=1.15, yanchor='top')])
    fig.write_html(os.path.join(saving_loc, f'{pdb_id}_within_pdb_global_id_plot.html'))


def upset_plot_pockets_shared(df, saving_loc):
    """Which global pockets are unique or shared across genes (upsetplot)."""
    df = df.copy()
    df['genes_list'] = df['genes_list'].fillna('').astype(str).str.strip()
    memberships = df['genes_list'].str.split(';')
    n_genes = len({g for genes in memberships for g in genes})
    if n_genes < 2:
        print('  upset_plot_pockets_shared skipped: needs >=2 distinct genes in the subset '
              f'to compare (found {n_genes})')
        return
    upset_df = upsetplot.from_memberships(memberships, data=df)
    upsetplot.plot(upset_df, show_counts=True)
    plt.savefig(os.path.join(saving_loc, 'upsetplot_gene_comparison.png'))
    plt.close()


def barchart_pocket_count(df, saving_loc):
    """Frequency of every gene-combination a global pocket is shared across."""
    all_genes = sorted({g for genes in df["genes_list"] for g in genes.split(";")})
    combos = ["-".join(sorted(c)) for r in range(1, len(all_genes) + 1) for c in itertools.combinations(all_genes, r)]
    normalized_keys = ["-".join(sorted(set(v.split(";")))) for v in df["genes_list"]]
    counts = Counter(normalized_keys)
    combo_counts = {c: counts.get(c, 0) for c in combos}
    sorted_keys = sorted(combo_counts.keys(), key=lambda k: (k.count("-"), k))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sorted_keys, [combo_counts[k] for k in sorted_keys], color="midnightblue",
                  edgecolor="black", linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Gene Combinations")
    ax.set_ylabel("Number of Shared Pockets")
    ax.set_title("Frequency of Gene Combinations")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(saving_loc, 'bar_chart_pocket_comparison.png'))
    plt.close()


def plot_3d_heatmap(df, saving_loc, out_html='3D_coordinates_frequency_colour_coded.html',
                    cluster_col='Global ID', genes_of_interest=None, colorscale='Viridis',
                    marker_size=4, title='3D pockets colored by n_structures (first frame)'):
    """Interactive 3D scatter of first-frame pocket coordinates, colored by how many distinct
    PDB structures share each Global ID, with PDB/gene filter dropdowns."""
    df = df.copy()
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df[cluster_col] = df[cluster_col].astype(str)
    df['pdb_id'] = df['ID'].str.extract(r'^((?:apo|holo)[A-Za-z0-9]+)', expand=False)
    df['state'] = df['pdb_id'].str.extract(r'^(apo|holo)', expand=False)

    pdb_per_global = df.groupby(cluster_col)['pdb_id'].unique().apply(lambda a: ', '.join(sorted(set(a))))
    gene_per_global = df.groupby(cluster_col)['gene'].unique().apply(
        lambda a: ', '.join(sorted({str(x) for x in a if pd.notna(x)})))
    df['pdb_list'] = df[cluster_col].map(pdb_per_global)
    df['gene_list'] = df[cluster_col].map(gene_per_global)
    df['hover_text'] = ("ID: " + df['ID'].astype(str) + "<br>Global: " + df[cluster_col].astype(str)
                        + "<br>in genes: " + df['gene_list'].astype(str)
                        + "<br>in PDB structures: " + df['pdb_list'].astype(str))

    counts = df.dropna(subset=[cluster_col, 'pdb_id']).groupby(cluster_col)['pdb_id'].nunique().rename('n_structures')
    df = df.merge(counts, on=cluster_col, how='left')
    df['n_structures'] = df['n_structures'].fillna(0)

    if genes_of_interest is None:
        genes_of_interest = ['hTAAR1', 'mTAAR1', 'mTAAR7f', 'mTAAR9']
    symbol_map = {"apo": "circle", "holo": "square"}
    unique_pdbs = sorted(df['pdb_id'].dropna().unique())
    data_traces, pdb_to_idx, gene_to_idx = [], {}, {g: [] for g in genes_of_interest}
    cmin, cmax = int(df['n_structures'].min()), int(df['n_structures'].max())
    for idx, pdb in enumerate(unique_pdbs):
        sub = df[df['pdb_id'] == pdb]
        data_traces.append(go.Scatter3d(
            x=sub['x'], y=sub['y'], z=sub['z'], mode='markers', name=f'PDB {pdb}',
            text=sub['hover_text'], hoverinfo='text',
            marker=dict(size=marker_size, color=sub['n_structures'], colorscale=colorscale, cmin=cmin, cmax=cmax,
                       colorbar=dict(title='Structures') if idx == 0 else None,
                       symbol=[symbol_map.get(s, 'circle') for s in sub['state']])))
        pdb_to_idx[pdb] = [idx]
        for gene in genes_of_interest:
            if gene in set(sub['gene'].dropna()):
                gene_to_idx[gene].append(idx)

    total = len(data_traces)
    vis_all = [True] * total
    pdb_buttons = [dict(label='All PDBs', method='update', args=[{'visible': vis_all}, {'title': f'{title} (All PDBs)'}])]
    for pdb, idxs in pdb_to_idx.items():
        vis = [False] * total
        for i in idxs:
            vis[i] = True
        pdb_buttons.append(dict(label=str(pdb), method='update', args=[{'visible': vis}, {'title': f'{title} (PDB {pdb})'}]))
    gene_buttons = [dict(label='All Genes', method='update', args=[{'visible': vis_all}, {'title': f'{title} (All Genes)'}])]
    for gene, idxs in gene_to_idx.items():
        vis = [False] * total
        for i in idxs:
            vis[i] = True
        gene_buttons.append(dict(label=gene, method='update', args=[{'visible': vis}, {'title': f'{title} (Gene {gene})'}]))

    fig = go.Figure(data=data_traces)
    fig.update_layout(title=title, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      template='simple_white',
                      updatemenus=[dict(buttons=pdb_buttons, direction='down', showactive=True,
                                       x=0.0, xanchor='left', y=1.18, yanchor='top', bgcolor='white'),
                                  dict(buttons=gene_buttons, direction='down', showactive=True,
                                       x=0.28, xanchor='left', y=1.18, yanchor='top', bgcolor='white')],
                      legend=dict(x=1.12, y=0.95, bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=0.5),
                      margin=dict(l=60, r=220, t=120, b=60))
    fig.write_html(os.path.join(saving_loc, out_html))
    df.to_csv(os.path.join(saving_loc, 'for_plot_3D_coordinates_frequency_colour_coded.csv'), index=False)


def assign_global_ids(subset_df, saving_loc, formats=('csv', 'parquet'), make_plots=True, verbose=True):
    """Runs Global ID clustering on subset_df (already the chosen subset - see select_subset())
    and writes pocket_comparison_table (+ uniqueness/overlap csvs and QC/overview plots).
    Returns pocket_comparison_table as a dataframe."""
    os.makedirs(saving_loc, exist_ok=True)

    point_df = pocket_point_dataframe(subset_df)
    voxelized_df = voxel_intersection_over_union_global_id(saving_loc, point_df=point_df,
                                                            make_qc_plot=make_plots)
    id_to_global = voxelized_df.drop_duplicates('Pocket ID local').set_index('Pocket ID local')['voxel_group_id']

    subset_df['Global ID'] = subset_df['Local ID'].map(id_to_global)

    if make_plots:
        for pdb_id, pdb_df in subset_df.groupby('prj'):
            try:
                within_pdb_plot(pdb_df, pdb_id=pdb_id, saving_loc=saving_loc)
            except Exception as e:
                print(f"  WARNING: within_pdb_plot failed for {pdb_id}: {e}")

    pocket_summary = make_pocket_summary(subset_df)
    small_df = subset_df[['prj', 'rep', 'ID', 'Global ID', 'gene']].copy()
    annotated = annotate_uniqueness(small_df)[['prj', 'rep', 'Global ID', 'gene', 'n_replicates_within_prj',
                                               'unique_in_one_rep', 'genes_list', 'n_distinct_genes',
                                               'unique_to_one_gene']]
    pocket_comparison = pocket_summary.merge(annotated, on=['prj', 'rep', 'Global ID', 'gene'], how='outer')
    pocket_comparison = pocket_comparison.drop_duplicates(['Local Pocket ID'])
    pocket_io.save_table(pocket_comparison, saving_loc, 'pocket_comparison_table', formats=formats)

    for name, mask in [('unique_to_gene_comparison', pocket_comparison['unique_to_one_gene'] == True),
                       ('unique_to_structure_comparison', pocket_comparison['unique_in_one_rep'] == True),
                       ('shared_in_all_gene_comparison', pocket_comparison['n_distinct_genes']
                        == pocket_comparison['n_distinct_genes'].max())]:
        pocket_comparison.loc[mask].to_csv(os.path.join(saving_loc, f'{name}.csv'), index=False)

    if make_plots:
        first_frame = subset_df.sort_values('snapshot').drop_duplicates('ID')[
            ['ID', 'Global ID', 'x', 'y', 'z', 'gene']]
        for fn, args in ((upset_plot_pockets_shared, (pocket_comparison, saving_loc)),
                         (barchart_pocket_count, (pocket_comparison, saving_loc)),
                         (plot_3d_heatmap, (first_frame, saving_loc))):
            try:
                fn(*args)
            except Exception as e:
                print(f"  WARNING: {fn.__name__} failed: {e}")

    message = (f'{subset_df["Local ID"].nunique()} local pockets -> '
              f'{pocket_comparison["Global ID"].nunique()} global IDs')
    if verbose:
        print(message)
    else:
        logger.log(message)
    return pocket_comparison


# apo vs holo state mapping
def global_pocket_centroids(pocket_comparison_df):
    """One row per Global ID: centroid + spread of its member pockets' alpha-sphere clouds is
    NOT available from pocket_comparison_table alone (that's one row per Local Pocket ID, not
    per point) - callers needing point clouds should use global_pockets_IoU_voxel.csv instead;
    this uses the Local Pocket IDs' own first-frame x,y,z as a lightweight proxy centroid."""
    grouped = pocket_comparison_df.groupby('Global ID').agg(
        n_members=('Local Pocket ID', 'size'),
        median_volume=('median_interpolated_volume', 'median'))
    if 'is_orthosteric' in pocket_comparison_df.columns:
        grouped['n_orthosteric_members'] = pocket_comparison_df.groupby('Global ID')['is_orthosteric'].apply(
            lambda s: int(s.fillna(False).sum()))
    return grouped.reset_index()

def centroids_from_voxels(voxel_csv, comparison_df):
    voxel_df = pd.read_csv(voxel_csv, usecols=['voxel_group_id', 'x', 'y', 'z'])
    centroid_df = voxel_df.groupby('voxel_group_id')[['x', 'y', 'z']].mean().rename(
        columns={'x': 'centroid_x', 'y': 'centroid_y', 'z': 'centroid_z'})
    centroid_df.index.name = 'Global ID'
    centroid_df = centroid_df.join(global_pocket_centroids(comparison_df).set_index('Global ID'), how='left').reset_index()

    return centroid_df


def match_states(reference_table, target_table, saving_loc, reference_voxel_csv, target_voxel_csv,
                 reference_state='apo', target_state='holo', max_match_distance=10.0):
    """Reconciles two SEPARATELY clustered runs (e.g. apo-only and holo-only) by nearest-
    centroid matching, run in both directions - for when you deliberately clustered a state
    subset alone rather than with --subset combined (which shares numbering already and needs
    no reconciliation). reference_table/target_table: pocket_comparison_table dataframes;
    reference_voxel_csv/target_voxel_csv: paths to the matching global_pockets_IoU_voxel.csv
    (for true point-cloud centroids).

    Method: centroid of every global pocket (mean over all its alpha-sphere points), then
    Euclidean distance between the two centroid sets. Both directions are computed and merged
    into one table (direction: 'both' | '<ref>-><target>' | '<target>-><ref>') so a global pocket
    with no reciprocal partner is still reported, not dropped.
    NOTE: This is used in the article as a proof of concept rather than a recommendation to be used in general.
    """
    os.makedirs(saving_loc, exist_ok=True)

    ref_c = centroids_from_voxels(reference_voxel_csv, reference_table)
    tgt_c = centroids_from_voxels(target_voxel_csv, target_table)

    dist = np.linalg.norm(ref_c[['centroid_x', 'centroid_y', 'centroid_z']].values[:, None, :]
                          - tgt_c[['centroid_x', 'centroid_y', 'centroid_z']].values[None, :, :], axis=2)
    nearest_for_ref = dist.argmin(axis=1)
    nearest_for_tgt = dist.argmin(axis=0)
    pairs = {(i, int(nearest_for_ref[i])) for i in range(len(ref_c))}
    pairs |= {(int(nearest_for_tgt[j]), j) for j in range(len(tgt_c))}

    rows = []
    for i, j in sorted(pairs):
        reciprocal = bool(nearest_for_ref[i] == j and nearest_for_tgt[j] == i)
        direction = 'both' if reciprocal else (f'{reference_state}->{target_state}'
                                                if nearest_for_ref[i] == j else f'{target_state}->{reference_state}')
        rows.append({f'{reference_state}_global_id': int(ref_c.iloc[i]['Global ID']),
                    f'{target_state}_global_id': int(tgt_c.iloc[j]['Global ID']),
                    'centroid_distance': float(dist[i, j]), 'reciprocal_nearest': reciprocal,
                    'matched': bool(dist[i, j] <= max_match_distance), 'direction': direction})
    mapping_df = pd.DataFrame(rows).sort_values('centroid_distance').reset_index(drop=True)
    mapping_df.to_csv(os.path.join(saving_loc, 'global_id_state_mapping.csv'), index=False)

    n_reciprocal = int((mapping_df['reciprocal_nearest'] & mapping_df['matched']).sum())
    print(f"  {reference_state} <-> {target_state}: {len(ref_c)}/{len(tgt_c)} global pockets, "
          f"{n_reciprocal} reciprocal matches within {max_match_distance:.0f} A")
    return mapping_df


# apo vs holo pocketome comparison
def allosteric_pocketome_table(df, pdb_order):
    """Per PDB: delta_pocket_count, delta_median_volume, JS distance, delta size-class
    fractions. All computed by taar_paper_figures/pocketome_metrics.py."""
    allosteric = df[~df[COL_ORTHO].astype(bool)]
    delta_count = pm.delta_pocket_count(allosteric, pdb_order, COL_PDB, COL_STATE, COL_REP)
    delta_vol = pm.delta_median_volume(allosteric, pdb_order, pdb_col=COL_PDB, state_col=COL_STATE)
    js_med, js_lo, js_hi = pm.js_per_structure(allosteric, pdb_order, pdb_col=COL_PDB, state_col=COL_STATE,
                                               rep_col=COL_REP, cat_col=COL_CAT)
    delta_sizeclass = pm.delta_category_fraction(allosteric, pdb_order, COL_PDB, COL_STATE, COL_CAT)
    out = pd.DataFrame({'pdb_id': pdb_order, 'delta_pocket_count': delta_count, 'delta_median_volume_pct': delta_vol,
                        'js_distance_median': js_med, 'js_distance_min': js_lo, 'js_distance_max': js_hi})
    for cat, col in zip(pm.CATEGORIES, delta_sizeclass.T):
        out[f'delta_fraction_{VOL_CATEGORIES[cat]}'] = col
    return out


def orthosteric_site_table(pdb_order, source_loc):
    """Per-PDB median orthosteric-site volume shift, reusing fig2_binding_site's volume_dict/
    delta_volume. The same functions that produce figure 2 panel B. source_loc: Step 2.1's
    saving_loc (default config.META_ANALYSIS_DIR), where orthosteric_perframe_volumes.csv
    lives"""

    df = pd.read_csv(os.path.join(source_loc, 'orthosteric_perframe_volumes.csv'), usecols=['ID', 'rep', fig2.VOL_COL])
    parsed = df['ID'].str.extract(fig2.ID_RE)
    df['state'], df['pdbid'] = parsed[0], parsed[1]
    df = df.dropna(subset=['state', 'pdbid'])
    vol = fig2.volume_dict(df)
    delta_vol = fig2.delta_volume(vol, pdb_order)
    return pd.DataFrame({'pdb_id': pdb_order, 'delta_orthosteric_volume_pct': delta_vol})


def run_apo_holo_comparison(pocket_summary_subset, source_loc, saving_loc=None, out=None):
    """Combines the allosteric pocketome metrics with the orthosteric site volume shift into
    one per-PDB-ID apo_holo_pocketome_summary.csv. Requires both apo and holo pockets to be
    present in pocket_summary_subset.

    source_loc: Step 2.1's saving_loc, where orthosteric_perframe_volumes.csv lives.
    saving_loc: where to write apo_holo_pocketome_summary.csv (default: source_loc)."""
    saving_loc = saving_loc or source_loc
    gene_map = ts.load_gene_map()
    pdb_order = ts.order_pdbids(pocket_summary_subset[COL_PDB].unique(), gene_map)
    allo = allosteric_pocketome_table(pocket_summary_subset, pdb_order)
    ortho = orthosteric_site_table(pdb_order, source_loc)
    combined = allo.merge(ortho, on='pdb_id')
    combined['gene'] = combined['pdb_id'].map(gene_map)
    out_path = out or os.path.join(saving_loc, 'apo_holo_pocketome_summary.csv')
    combined.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(combined)} PDB IDs)")
    return combined


def run_global_id_states(all_pockets, pocket_summary, states, source_loc, gid_root,
                         genes=None, pdb_ids=None, formats=('csv', 'parquet'), make_plots=True,
                         verbose=True, reconcile_apo_holo=False, match_max_distance=10.0):
    """Runs Global ID clustering for one or more state subsets in a single call.

    states: None, or a list made up of any of 'apo', 'holo', 'both' (any combination/order,
    duplicates ignored). This is NOT select_subset()'s `states` filter (which picks real values
    off the 'state' column and can't express "apo and holo together" vs. "apo and holo as two
    separate runs" as distinct options) - it's a list of *runs* to do, each written to its own
    subdirectory of gid_root:
        'apo'  -> apo pockets only,  clustered alone   -> gid_root/global_ID_apo/
        'holo' -> holo pockets only, clustered alone   -> gid_root/global_ID_holo/
        'both' -> apo+holo together, one shared Global ID numbering -> gid_root/global_ID_combined/
                  (also runs run_apo_holo_comparison() automatically)
    None means ['both'] aka a single combined run.

    reconcile_apo_holo: if True AND both 'apo' and 'holo' are requested, also runs match_states()
    afterwards to reconcile their independent numberings by nearest pocket centroid, writing
    gid_root/apo_holo_global_id_mapping/global_id_state_mapping.csv. Off by default;  only turn this
    on if you specifically need the cross-run correspondence (their Global IDs are NOT comparable
    without it - a pocket's Global ID is only meaningful within the run that produced it).

    source_loc: Step 2.1's saving_loc (where orthosteric_perframe_volumes.csv lives) - needed
    only for the 'both' run's apo/holo comparison.
    genes/pdb_ids: same optional filters as select_subset(), applied on top of each state filter.

    """
    valid_runs = {'apo', 'holo', 'both'}
    requested = list(dict.fromkeys(states)) if states else ['both']
    unknown = set(requested) - valid_runs
    if unknown:
        raise ValueError(f"states must be a list made up of {sorted(valid_runs)}, got unknown "
                         f"values: {sorted(unknown)}")

    loc_by_run = {'apo': os.path.join(gid_root, 'global_ID_apo'),
                 'holo': os.path.join(gid_root, 'global_ID_holo'),
                 'both': os.path.join(gid_root, 'global_ID_combined')}
    state_filter_by_run = {'apo': ['apo'], 'holo': ['holo'], 'both': None}

    results = {}
    for run in requested:
        subset = select_subset(all_pockets, states=state_filter_by_run[run], genes=genes, pdb_ids=pdb_ids)
        results[run] = assign_global_ids(subset, loc_by_run[run], formats=formats, make_plots=make_plots,
                                         verbose=verbose)
        if run == 'both':
            summary_subset = select_subset(pocket_summary, states=None, genes=genes, pdb_ids=pdb_ids)
            results['apo_holo_summary'] = run_apo_holo_comparison(summary_subset, source_loc,
                                                                   saving_loc=loc_by_run['both'])

    if reconcile_apo_holo and 'apo' in results and 'holo' in results:
        mapping_loc = os.path.join(gid_root, 'apo_holo_global_id_mapping')
        results['mapping'] = match_states(
            results['apo'], results['holo'], mapping_loc,
            os.path.join(loc_by_run['apo'], 'global_pockets_IoU_voxel.csv'),
            os.path.join(loc_by_run['holo'], 'global_pockets_IoU_voxel.csv'),
            max_match_distance=match_max_distance)

    return results


# default run
def main(states=None, genes=None, pdb_ids=None, saving_loc=None, compare_apo_holo=True,
        formats=('csv', 'parquet'), make_plots=True, verbose=True, isovalue=None):
    """Clusters one subset of all_pockets into Global IDs and, if both states end up in it,
    compares apo vs holo. states/genes/pdb_ids are the same filters as select_subset(). The
    default (all None) is the `combined` subset: every state/gene/PDB clustered together, one
    shared Global ID numbering.

    Call with different arguments for a different subset, e.g. main(states=['apo']) or
    main(genes=['hTAAR1']) from Python or the notebook. For apo/holo
    clustered SEPARATELY and reconciled afterwards, call select_subset() / assign_global_ids()
    per state and match_states() yourself instead (see the notebook for that recipe).

    isovalue picks which isovalue's all_pockets/pocket_summary to read because Global ID clustering never compares
    pockets across isovalues, so each isovalue gets a fully separate result. Leaving
    isovalue=None runs main() once per conf.ISOVALUES entry.
    """
    isovalues = conf.require_float_isovalues(conf.ISOVALUES)
    if isovalue is None and len(isovalues) > 1:
        return {iso: main(states=states, genes=genes, pdb_ids=pdb_ids, saving_loc=saving_loc,
                          compare_apo_holo=compare_apo_holo, formats=formats, make_plots=make_plots,
                          verbose=verbose, isovalue=iso)
                for iso in isovalues}
    if isovalue is None:
        isovalue = isovalues[0]

    input_loc = conf.isovalue_meta_analysis_dir(isovalue, isovalues)
    if saving_loc is None:
        label = '_'.join(states) if states else 'combined'
        saving_loc = os.path.join(conf.isovalue_meta_analysis_root(isovalue, isovalues), f'global_ID_{label}')

    all_pockets = pocket_io.load_table(input_loc, 'all_pockets')
    subset = select_subset(all_pockets, states=states, genes=genes, pdb_ids=pdb_ids)
    pocket_comparison = assign_global_ids(subset, saving_loc, formats=formats, make_plots=make_plots,
                                          verbose=verbose)

    if compare_apo_holo:
        if subset['state'].nunique() < 2:
            print('compare_apo_holo needs both states in the subset - skipping (states=' f'{states})')
        else:
            pocket_summary = pocket_io.load_table(input_loc, 'pocket_summary')
            summary_subset = select_subset(pocket_summary, states=states, genes=genes, pdb_ids=pdb_ids)
            run_apo_holo_comparison(summary_subset, input_loc, saving_loc=saving_loc)

    return pocket_comparison


if __name__ == '__main__':
    main()
