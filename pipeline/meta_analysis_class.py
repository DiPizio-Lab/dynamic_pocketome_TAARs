import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import ast
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
import networkx as nx
from itertools import combinations
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import KNeighborsClassifier
from scripts import logging as logger
# from scripts import pdb_api
pio.renderers.default = 'notebook_connected'
# TODO: summarize the info of 3D identification --> how many pockets are always found?
#  are they at the same position? do they have the same size
# TODO: remove relicts but document them: Relict — unreachable:
#  add_pocket_identifier_clustering_approaches — the older centroid/kmeans route, superseded by the voxel-IoU one
#  _reassignment_global_clusters — only called by the above. Its rule ("a local cluster may not be subdivided into 2+ global clusters") is not enforced in your current path
#  summary_statistics and _get_pdb_gene_dict — the latter is your own deprecated PDB-API code; gene mapping now comes from SHORT_GENE_NAMES in the wrapper
#  _global_clustering: the kmeans and agglomerative branches
#  _within_pdb_clustering: the centroids=True branch (wrapper always passes False)

class MetaAnalysis:
    """Class containing methods that can create merged dataframes of the dummy atoms of different
    pocket files (pock_file_parser), interpolates numeric values of a dataframe (interpolation_volume),
    can plot the descriptor over time, e.g., pocket volume"""
    def __init__(self, pocket_dirs, isovalue, dbscan=False, saving_loc='.', min_atoms=3, verbose=False):
        self.pocket_dirs = pocket_dirs
        self.isovalue = isovalue
        self.dbscan = dbscan
        self.saving_loc = saving_loc
        self.verbose = verbose
        self.min_atoms = min_atoms

    def _normalize_pocket_filenames(self):
        """Because filenames including a float are saved by ATClus and mdpocket using a dot as separator we're
        running into parsing problems. Therefore, this function is called first to make sure that filenames
        do not include a dot anywhere but the file extension. If self.isovalue is a float, the dot of the float
        will be replaced with an underscore,
        Example: mdpout_dens_iso_3.5-out-01.pdb --> mdpout_dens_iso_3_5-out-01.pdb"""
        if isinstance(self.isovalue, float) and not self.isovalue.is_integer():
            iso_str = str(self.isovalue)
            iso_dot = f"iso_{iso_str}"
            iso_underscore = f"iso_{iso_str.replace('.', '_')}"

            for pocket_path in self.pocket_dirs:
                if os.path.isdir(pocket_path):
                    for filename in os.listdir(pocket_path):
                        if iso_dot in filename and iso_underscore not in filename:
                            new_filename = filename.replace(iso_dot, iso_underscore)
                            old_path = os.path.join(pocket_path, filename)
                            new_path = os.path.join(pocket_path, new_filename)
                            os.rename(old_path, new_path)
                            if self.verbose:
                                print(f"Renamed: {filename} → {new_filename}")

    def _parse_filename(self, filename):
        """Parses filename to extract isovalue, pocket number, and type."""
        parsed = {}
        if self.dbscan:
            if filename.startswith("coords_pocket"):
                parts = filename.replace('.pdb', '').replace('.txt', '').split('_')
                if 'descriptors' in filename:
                    parsed['type'] = 'descriptor'
                elif 'atoms' in filename:
                    parsed['type'] = 'residue'
                else:
                    parsed['type'] = 'dummy_atom'
                parsed['pocket_no'] = parts[2]
                parsed['isovalue'] = float(parts[3].replace('_', '.'))
            else:
                parsed['type'] = 'unknown'

        elif not self.dbscan:
            if filename.endswith('.xtc') or filename.endswith('.dcd'):
                parsed['type'] = 'trajectory'
            elif filename.endswith('_descriptors.txt'):  # mdpout_dens_iso_3-out-01_descriptors.txt
                parsed['type'] = 'descriptor'
            elif filename.endswith('mdpocket_atoms.pdb'):
                parsed['type'] = 'residue'
            elif filename.endswith('mdpocket.pdb'):
                parsed['type'] = 'trajectory_pocket'
            elif filename.endswith('.pdb') and '-out-' in filename:
                parsed['type'] = 'dummy_atom'
            else:
                parsed['type'] = 'unknown'

            match = re.search(r'iso_(\d+_\d+|\d+)-out-(\d+)', filename)
            if match:
                iso_str = match.group(1).replace('_', '.')
                pocket_no = match.group(2)
                try:
                    parsed['isovalue'] = float(iso_str)
                    parsed['pocket_no'] = pocket_no
                except ValueError:
                    parsed['isovalue'] = 'unknown'
                    parsed['pocket_no'] = 'unknown'
            else:
                parsed['isovalue'] = 'unknown'
                parsed['pocket_no'] = 'unknown'

        return parsed

    def _descriptor_extraction(self, file_path, prj, rep, file):
        df_descriptors = pd.read_csv(file_path, sep='\s+')
        parsed = self._parse_filename(file)
        iso_value = parsed['isovalue']
        if iso_value.is_integer():  # handles 3.0 vs 3 case for unique IDs
            iso_value = int(iso_value)

        pocket_num = parsed['pocket_no']
        unique_id = f"{prj}_{rep}_p{pocket_num}_i{iso_value}"

        df_descriptors[['prj', 'rep', 'pocket_file', 'isovalue', 'pocket_number']] = [
            prj, rep, file, iso_value, pocket_num]
        df_descriptors.insert(0, 'ID', unique_id)

        return df_descriptors

    @staticmethod
    def _extract_amino_acids_of_pocket(file_path, prj, rep):
        name = Path(file_path).name
        match = re.search(r'iso_(\d+_\d+|\d+)-out-(\d+)', name)
        if match:
            iso_str = match.group(1).replace('_', '.')
            pocket_no = match.group(2)
            isovalue = float(iso_str)
        else:
            isovalue = None
            pocket_no = None
        lines = []
        with open(file_path, 'r') as in_f:
                for line in in_f.readlines()[1:]:
                    lines.append(line.rstrip('\n'))
                    if 'ENDMDL' in line:
                        break
        columns = ['atom', 'atom_no', 'atom_name', 'residue_name', 'chain', 'residue_id', 'x', 'y', 'z',
                   'n1', 'n2', 'n3', 'n4']
        data_split = [line.split() for line in lines if 'ENDMDL' not in line]
        df = pd.DataFrame(data_split, columns=columns)
        df[['prj', 'rep', 'isovalue', 'pocket_number']] = [prj, rep, isovalue, pocket_no]
        df['aa_no'] = df['residue_name'] + df['residue_id'].astype(str)
        unique_residues = set(df['aa_no'])
        filtered = df[df['aa_no'].isin(unique_residues)]

        return filtered


    #TODO: Deprecated because the RCSB-API package is down --> try again in the future
    """
    @staticmethod
    def _get_pdb_gene_dict(id_list: list[str], pdb_ids: bool = True, organism_list: list[str] = None) -> dict[
        str, str]:
        Takes either an PDB ID list or an organism list to build a dictionary.
        To discriminate between the 2 versions, the pdb_ids flag is used.
        Input:
            id_list: list of PDB IDs, e.g., ['8JLO', '8WCC']
            pdb_ids: boolean to tell the function if you are providing PDB IDs or gene names
            organism_list: list of organism names, e.g., ['Homo sapiens', 'Mus musculus']
        Output: dictionary mapping gene names and PDB IDs
        pdb_id_to_gene = {}
        if pdb_ids:
            # Use detailed PDB->Gene logic here (from the corrected version above)
            for pdb_id in id_list:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                entry_data = requests.get(url).json()
                entity_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
                best_label = None

                for entity_id in entity_ids:
                    entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
                    entity_data = requests.get(entity_url).json()
                    gene_names = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("gene_name", [])
                    label = gene_names[0] if gene_names else None
                    if not label:
                        lineage = entity_data.get("rcsb_genomic_lineage", [])
                        for item in lineage:
                            if item.get("depth") == 2:
                                label = item.get("name", "")
                                break
                    if not label:
                        continue
                    is_receptor = "receptor" in label.lower()
                    clean = label.replace("-", "").replace("associated", "a").replace(" ", "")
                    clean = clean.replace("Traceaminea", "TAa").replace("Traceamine", "TA")
                    clean = clean[0].upper() + clean[1:]
                    prefix = ""
                    lineage = entity_data.get("rcsb_genomic_lineage", [])
                    for item in lineage:
                        if item.get("depth") == 0:
                            org = item.get("name", "")
                            if "Homo sapiens" in org:
                                prefix = "h"
                            elif "Mus musculus" in org:
                                prefix = "m"
                            elif org:
                                prefix = org[0].lower()
                            break

                    full_label = f"{prefix}{clean}"
                    if is_receptor or best_label is None:
                        best_label = full_label
                pdb_id_to_gene[pdb_id] = best_label if best_label else None
        else:
            if organism_list is None or len(organism_list) != len(id_list):
                raise ValueError("Must provide organism_list when pdb_ids=False.")

            pdbq = pdb_api.PDBQuery()
            bulk_results = pdbq.bulk_gene_queries(id_list, organism_list, id_list)

            for gene_label, group in bulk_results.items():
                for pdb_id in group.get("pdb_ids", []):
                    pdb_id_to_gene[pdb_id] = gene_label

        return pdb_id_to_gene
    """

    def pock_file_parser(self):
        self._normalize_pocket_filenames()  # essential step to handle float isovalues
        result_dict = {}
        pocket_dummy_atom_dfs = []
        pock_dummies_traj_paths = []
        res_to_pock_dfs = []
        descriptor_dfs = []
        exclusion_list = []
        pockets_to_drop = []
        if not isinstance(self.isovalue, float):
            isovalue = round(np.float64(self.isovalue), 3)
        elif self.isovalue.is_integer():
            isovalue = int(self.isovalue)
        else:
            isovalue = self.isovalue

        for pocket_path in self.pocket_dirs:
            if os.path.isdir(pocket_path):
                path_parts = os.path.normpath(pocket_path).split(os.sep)
                prj = path_parts[-3] if len(path_parts) >= 3 else "unknown_prj"
                rep = path_parts[-2] if len(path_parts) >= 2 else "unknown_rep"

                # Filter files that contain the isovalue after 'iso_'
                iso_str = str(isovalue).replace('.', '_')
                files = [f for f in os.listdir(pocket_path) if f"iso_{iso_str}-out-" in f]
                # TODO: does not work with DBSCAN like files!!!
                if self.verbose:
                    print(f'Reading in pocket files of {prj}_{rep}')

                pocket_dict = {}
                for file in files:
                    parsed = self._parse_filename(file)
                    if parsed['type'] == 'dummy_atom' and parsed['isovalue'] == isovalue:
                        pocket_dict[parsed['pocket_no']] = os.path.join(pocket_path, file)

                for pock_no, file_path in pocket_dict.items():
                    df = pd.read_csv(file_path, sep='\s+', header=None)
                    columns = ['atom', 'atom_no', 'chain', 'molecule_name', 'pocket_number',
                               'x', 'y', 'z', 'n1', 'n2']
                    df.columns = columns
                    df[['prj', 'rep', 'isovalue']] = [prj, rep, isovalue]
                    df['isovalue'] = df['isovalue'].astype(float)
                    df['pocket_number'] = pock_no
                    df['pocket_file_path'] = file_path
                    unique_id = f"{prj}_{rep}_p{pock_no}_i{isovalue}"
                    df.insert(0, 'ID', unique_id)

                    top_path = os.path.join(pocket_path, 'aligned_top.pdb')
                    traj_path = None
                    for ext in ['xtc', 'dcd']:
                        test_path = os.path.join(pocket_path, f'aligned_traj.{ext}')
                        if os.path.exists(test_path):
                            traj_path = test_path
                            break
                    df[['topology loc', 'trajectory loc']] = [top_path, traj_path]

                    if df.shape[0] > self.min_atoms:
                        pocket_dummy_atom_dfs.append(df)
                    else:
                        pockets_to_drop.append(unique_id)

                    exclusion_list.append(file_path)

                for file in files:
                    file_abs = os.path.join(pocket_path, file)
                    if file_abs in exclusion_list:
                        continue
                    file_path = os.path.join(pocket_path, file)
                    parsed = self._parse_filename(file)

                    if parsed['type'] == 'trajectory_pocket':
                        pock_dummies_traj_paths.append(file_path)
                        exclusion_list.append(file_abs)
                    elif parsed['type'] == 'residue':
                        curr_df = self._extract_amino_acids_of_pocket(file_path, prj, rep)
                        curr_df['file'] = file_path
                        res_to_pock_dfs.append(curr_df)
                        exclusion_list.append(file_abs)
                    elif parsed['type'] == 'descriptor':
                        df_descriptors = self._descriptor_extraction(file_path, prj, rep, file)
                        descriptor_dfs.append(df_descriptors)
                        exclusion_list.append(file_abs)

        merged_pocket_dummy_atom_df = pd.concat(pocket_dummy_atom_dfs, ignore_index=True)
        merged_descriptor_df = pd.concat(descriptor_dfs, ignore_index=True)
        merged_residue_df = pd.concat(res_to_pock_dfs, ignore_index=True)
        merged_descriptor_df['isovalue'] = merged_descriptor_df['isovalue'].astype(float).round(3)
        merged_descriptor_df = merged_descriptor_df[~merged_descriptor_df['ID'].isin(pockets_to_drop)]
        merged_descriptor_df = merged_descriptor_df[merged_descriptor_df['isovalue'] == isovalue]

        result_dict.update({'dummy_atom_df': merged_pocket_dummy_atom_df, 'descriptor_df': merged_descriptor_df,
            'res_to_pock_df': merged_residue_df, 'pock_dummies_traj_paths': pock_dummies_traj_paths})

        if self.verbose:
            print(f'Saving the excluded pockets to {os.path.join(self.saving_loc, f"projects_with_less_than_{self.min_atoms}_atoms.txt")}')
        with open(os.path.join(self.saving_loc, f"projects_with_less_than_{self.min_atoms}_atoms.txt"), "w") as output:
            output.write(str(pockets_to_drop))

        return result_dict, pockets_to_drop

    def interpolation_volume(self, df, col_to_interpolate, limit=2, limit_area='inside', method='linear'):
        """Interpolates a given column based on the pandas interpolate function:
                https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
        Input: list of column names to interpolate
            choices to make: method of interpolation (check pandas docs for details)
                             limit = Maximum number of consecutive NaNs to fill. Must be greater than 0
                              limit_area = ‘inside’: Only fill NaNs surrounded by valid values (interpolate).
                                            ‘outside’: Only fill NaNs outside valid values (extrapolate).
        Returns the interpolated column(s)"""
        if self.verbose:
            print(f'Interpolation of the {col_to_interpolate} column')
        else:
            logger.log(f'Interpolation of the {col_to_interpolate} column')

        for col in col_to_interpolate:
            new_col_name = 'interpolated_' + col
            df[new_col_name] = df[col_to_interpolate].replace(0, np.NaN)
            df[new_col_name] = df[new_col_name].interpolate(limit=limit, limit_area=limit_area, method=method)
            df[new_col_name] = df[new_col_name].fillna(0)

        return df

    @staticmethod
    def _data_prep_for_clustering(df_single_pock):
        """"""
        descriptor_dict = {}
        # calculate median along frames of certain descriptors (volume, hydrophobicity, charge, asa, residues of pocket
        descriptor_dict['median_interpolated_vol'] = df_single_pock['interpolated_pock_volume'].median()
        descriptor_dict['median_hydrophob_score'] = df_single_pock['hydrophobicity_score'].median()
        descriptor_dict['median_pock_asa'] = df_single_pock['pock_asa'].median()
        descriptor_dict['median_charge_score'] = df_single_pock['charge_score'].median()
        # residues with at least one frame of 1 to be included??
        residue_cols = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        aa_occurence = df_single_pock[residue_cols].sum()
        aa_boolean = aa_occurence.map(lambda x: 1 if x > 0 else 0)
        descriptor_dict['residues'] = aa_boolean.to_dict()

        # calculate centroid of the pocket
        centroid_coords = df_single_pock[['x', 'y', 'z']].mean()
        stan_dev_coords = df_single_pock[['x', 'y', 'z']].std()
        number_alpha_spheres = df_single_pock[['x', 'y', 'z']].drop_duplicates().shape[0]
        descriptor_dict['number_alpha_spheres'] = number_alpha_spheres
        descriptor_dict['centroid_coords'] = centroid_coords.to_dict()
        descriptor_dict['stan_dev_coords'] = stan_dev_coords.to_dict()
        df_pock_nodups = df_single_pock.drop_duplicates(subset=['x', 'y', 'z'])
        df_pock_nodups = df_pock_nodups[['x', 'y', 'z']]
        coordinate_list = df_pock_nodups.values.tolist()
        descriptor_dict['coordinates'] = coordinate_list

        return descriptor_dict


    def _global_clustering(self, feature_df, feature_cols, no_prj_reps, algorithm='hdbscan', largest_pocket=500):
        """Turns out that HDBSCAN used on set of x,y, z coordinates works best, results also in larger global clusters
        which is a tradeoff we want to do"""
        if algorithm == 'hdbscan':
            # try HDBSCAN because it works better with clusters of different densities and hyperparameter eps is not needed
            hdbscan = HDBSCAN(min_cluster_size=no_prj_reps, max_cluster_size=largest_pocket)
            # idea: set max_cluster size as largest occurring local pocket?
            hdbscan.fit(feature_df[feature_cols])
            labels = hdbscan.labels_
            feature_df['cluster_label'] = labels
            noise_mask = labels == -1
            labeled_mask = labels != -1
            if np.any(noise_mask) and np.any(labeled_mask):
                # 1: Train k-NN on labeled points
                x_labeled = feature_df.loc[labeled_mask, feature_cols].values
                y_labeled = labels[labeled_mask]
                x_noise = feature_df.loc[noise_mask, feature_cols].values

                knn = KNeighborsClassifier(n_neighbors=5)  # 5 is default
                knn.fit(x_labeled, y_labeled)
                reassigned_labels = knn.predict(x_noise)

                final_labels = labels.copy()
                final_labels[noise_mask] = reassigned_labels
            else:
                final_labels = labels
            feature_df['knearest_cluster_label'] = final_labels
        elif algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=20)  # because max. clusters = 20
            kmeans.fit(feature_df[feature_cols])
            labels = kmeans.labels_
            feature_df['cluster_label'] = labels
        elif algorithm == 'agglomerative':
            agglomerative = AgglomerativeClustering(linkage='ward', distance_threshold=2, n_clusters=None)
            agglomerative.fit(feature_df[feature_cols])
            labels = agglomerative.labels_
            feature_df['cluster_label'] = labels
        else:
            raise ValueError(f'Algorithm {algorithm} not recognized')

        return feature_df

    def _plot_pocket_clusters_qc(self, df: pd.DataFrame, centroids: bool = False, cluster_col: str = "voxel_group_id"):
        """
        QC 3D scatter of pocket points or centroids, colored by any cluster column.

        Parameters
        ----------
        df
            Must contain columns ['Pocket ID','x','y','z','centroid_coords.x','centroid_coords.y','centroid_coords.z',
            'PDB ID','hover_text', and the cluster_col you choose].
        centroids
            If True, plot the 'centroid_coords.*' columns; else plot raw x,y,z.
        cluster_col
            Name of the DataFrame column to use for coloring (e.g. 'voxel_group_id').
        """

        df = df.copy()
        df['project_replicate'] = df['Pocket ID'].str.extract(r'^((?:apo|holo)[A-Z0-9]+_\d)')
        df[['PDB ID', 'replicate']] = df['project_replicate'].str.split('_', expand=True)
        df['pocket_number'] = df['Pocket ID'].str.extract(r'p(\d+)')
        df['hover_text'] = "ID: " + df['Pocket ID'] + "<br>Group: " + df[cluster_col].astype(str)

        unique_projects = df['PDB ID'].unique()
        # markers = ['circle', 'square', 'cross', 'diamond', 'triangle-up']
        # symbol_map = {p: markers[i % len(markers)] for i, p in enumerate(unique_projects)}

        if centroids:
            x_col, y_col, z_col = 'centroid_coords.x', 'centroid_coords.y', 'centroid_coords.z'
        else:
            x_col, y_col, z_col = 'x', 'y', 'z'

        fig1 = go.Figure()

        for pdb_id in df['PDB ID'].unique():
            subset = df[df['PDB ID'] == pdb_id]
            fig1.add_trace(go.Scatter3d(x=subset[x_col], y=subset[y_col], z=subset[z_col],
                mode='markers', marker=dict(size=4,
                    color=subset[cluster_col],  # cluster_col should be numeric or categorical
                    colorscale='Viridis', colorbar=dict(title=cluster_col), opacity=0.8),
                name=pdb_id, text=subset['hover_text'], hoverinfo='text'))
        # fig1 = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=cluster_col,
        #                      # symbol='PDB ID',
        #                      # symbol_map=symbol_map,
        #                      hover_name='hover_text',
        #                      title=f'3D Pocket Clustering by {cluster_col}', template='simple_white')
        fig1.update_traces(marker=dict(size=5))
        fig1.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                           title=f'3D Pocket Clustering by {cluster_col}', template='simple_white',
                           margin=dict(l=0, r=0, b=0, t=40), legend=dict(x=1, y=1))
        # fig1.show()
        fig1.write_html(os.path.join(self.saving_loc, 'pocket_clusters_qc.html'))

        unique_clusters = sorted(df[cluster_col].dropna().unique(), key=lambda x: int(x))
        color_sequence = px.colors.qualitative.Dark24
        color_map = {c: color_sequence[i % len(color_sequence)] for i, c in enumerate(unique_clusters)}

        data_traces = []
        for cluster in unique_clusters:
            subset = df[df[cluster_col] == cluster]
            for pdb in subset['PDB ID'].unique():
                sub = subset[subset['PDB ID'] == pdb]
                trace = go.Scatter3d(x=sub[x_col], y=sub[y_col], z=sub[z_col], mode='markers',
                                     name=f'Grp {cluster} – {pdb}', text=sub['hover_text'], hoverinfo='text',
                                     marker=dict(size=4, color=color_map[cluster],
                                                 # symbol=symbol_map[pdb]
                                                 ),
                                     legendgroup=str(cluster), showlegend=True)
                data_traces.append(trace)

        fig2 = go.Figure(data=data_traces)
        fig2.update_layout(title=f'3D Pocket Clustering by {cluster_col}',
                           scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                           template='simple_white', legend=dict(x=1.02, y=1))
        # fig2.show()
        fig2.write_html(os.path.join(self.saving_loc,
                                     f'detailed_pocket_clusters_{cluster_col}_qc.html'))


    @staticmethod
    def _to_padded_tuples(coord_list, target_len):
        """Converts a list of [x, y, z] coordinates into a flat list of scalars,
        padding with np.nan to reach target_len * 3."""

        if not isinstance(coord_list, list):
            return [np.nan] * (target_len * 3)
        as_tuples = [tuple(coord) for coord in coord_list if isinstance(coord, (list, tuple)) and len(coord) == 3]
        # Flatten to [x1, y1, z1, x2, y2, z2, ...]
        flat = [val for coord in as_tuples for val in coord]
        # Pad with NaNs to reach target_len * 3
        padded = flat + [np.nan] * ((target_len * 3) - len(flat))

        return padded[:target_len * 3]

    @staticmethod
    def _melt_coordinates(df, coord_col='coordinates'):
        """"""
        rows = []
        for idx, row in df.iterrows():
            pocket_id = row['Pocket ID']
            coords = row[coord_col]
            # Make sure it's a valid list of 3D coords
            if isinstance(coords, list):
                for i, point in enumerate(coords):
                    if isinstance(point, list) and len(point) == 3:
                        x, y, z = point
                        rows.append({'Pocket ID': pocket_id, 'point_idx': i,
                                     'x': x,'y': y,'z': z})

        return pd.DataFrame(rows)

    @staticmethod
    def _reassignment_global_clusters(clustered_df):
        """Concept: rule based reassignment --> A local cluster may not be subdivided into 2 or more global clusters"""
        # get individual pockets
        # check final cluster label
        # if there is more than one final cluster label, flag the pocket to re-assign
        # re-assignment based on lower cluster number, i.e., use the lower number as the future cluster
        local_to_global = {}
        for id_p, pock_df in clustered_df.groupby('Pocket ID'):
            global_labels = pock_df['knearest_cluster_label'].to_list()
            local_to_global[id_p] = [set(global_labels)]
        mapping_df = pd.DataFrame.from_dict(local_to_global, orient='index', columns=['global_cluster_label']).reset_index()
        mapping_df = mapping_df.rename(columns={'index': 'local_pocket_ID'})
        mapping_df[['PDB ID', 'replicate', 'local_pocket_number', 'isovalue']] = mapping_df['local_pocket_ID'].str.split('_',
                                                                                                             expand=True)
        mapping_df['flagged_reassignment'] = mapping_df['global_cluster_label'].apply(lambda x: len(x) > 1)
        mapping_df['new_global'] = mapping_df['global_cluster_label'].apply(min)

        return mapping_df

    @staticmethod
    def _dilate_voxels(voxel_set, radius):
        """helper to dilate a set of integer voxels by radius"""
        pts = np.array(list(voxel_set))
        mins = pts.min(axis=0) - radius
        maxs = pts.max(axis=0) + radius
        shape = (maxs - mins + 1).astype(int)
        mask = np.zeros(shape, dtype=bool)
        for v in voxel_set:
            idx = tuple((v - mins).astype(int))
            mask[idx] = True
        # distance transform: distance_to_background <= radius survives
        dist = distance_transform_edt(~mask)
        dilated = dist <= radius
        new_set = set()
        for coord in np.argwhere(dilated):
            new_set.add(tuple(coord + mins))

        return new_set

    # reshape pt. 1
    def _pocket_feature_dataframe(self, pdb_pocket_df):
        """
        One row per local pocket: descriptor medians, amino-acid occupancy, centroid,
        and the deduplicated coordinate list. Split out of _within_pdb_clustering so
        the reshape can be used without running any clustering.

        Returns (feature_df, max_no_points, unique_projects).
        """
        global_pockets_dict = {}
        for id_p, df_pock in pdb_pocket_df.groupby('ID'):
            feature_dict = self._data_prep_for_clustering(df_pock)
            global_pockets_dict[id_p] = feature_dict

        feature_df = pd.json_normalize([{'Pocket ID': k, **v} for k, v in global_pockets_dict.items()])
        feature_df['project_replicate'] = feature_df['Pocket ID'].str.extract(
            r'^((?:apo|holo)[A-Z0-9]+_\d)', expand=False)
        unique_projects = len(feature_df['project_replicate'].unique())
        max_no_points = feature_df['number_alpha_spheres'].max()
        feature_df[['prj', 'rep']] = feature_df['project_replicate'].str.rsplit('_', expand=True)
        return feature_df, max_no_points, unique_projects

    # reshape pt. 2
    def pocket_point_dataframe(self, pdb_pocket_df):
        """
        Long format consumed by the voxelisation: one row per (Pocket ID, x, y, z),
        plus prj / rep / pocket / isovalue. No clustering, no file written.

        This is the actual prerequisite of voxel_intersection_over_union_global_id.
        """
        feature_df, max_no_points, _ = self._pocket_feature_dataframe(pdb_pocket_df)

        col_names = [f'datapoint_{i}_{axis}' for i in range(max_no_points) for axis in ('x', 'y', 'z')]
        feature_df['coordinates'] = feature_df['coordinates'].astype(str)
        feature_df['coordinates'] = feature_df['coordinates'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        coords_expanded = feature_df['coordinates'].apply(
            lambda coords: self._to_padded_tuples(coords, target_len=max_no_points))
        coord_df = pd.DataFrame(coords_expanded.tolist(), columns=col_names)
        feature_df = pd.concat([feature_df, coord_df], axis=1)

        flat_df = self._melt_coordinates(feature_df)
        flat_df[['prj', 'rep', 'pocket', 'isovalue']] = flat_df['Pocket ID'].str.extract(
            r'(\w+)_([0-9]+)_p([0-9]+)_i([0-9.]+)')
        return flat_df

    # clustering > never used in the paper
    def _within_pdb_clustering(self, pdb_pocket_df, centroids=False, algorithm='hdbscan',
                               point_df=None):
        """groups the provided dataframe by 'prj' and then performs global clustering on the replicates
        if this PDB ID returns dataframe with added column and saves the data in the
        meta_analysis folder as 'detailed_pocket_clusters_pdb_unique.csv'

        NOTE: the cluster labels produced here are NOT read by
        voxel_intersection_over_union_global_id - that method needs only the reshape,
        which now lives in pocket_point_dataframe(). These labels are kept for the QC
        csv and plot.

        point_df: an already-reshaped frame from pocket_point_dataframe(), to avoid
        redoing the reshape when clustering the same pockets more than once.
        """
        clustered_df_list = []
        if centroids:
            feature_df, max_no_points, unique_projects = self._pocket_feature_dataframe(pdb_pocket_df)
            for pdb_id, one_pdb_df in feature_df.groupby('prj'):
                clustered_df = self._global_clustering(
                    one_pdb_df,
                    feature_cols=['centroid_coords.x', 'centroid_coords.y', 'centroid_coords.z'],
                    algorithm=algorithm, no_prj_reps=unique_projects,
                    largest_pocket=2 * max_no_points)
                clustered_df_list.append(clustered_df)
        else:
            flat_df = self.pocket_point_dataframe(pdb_pocket_df) if point_df is None else point_df
            largest_pocket = 2 * int(flat_df.groupby('Pocket ID').size().max())
            for pdb_id, one_pdb_df in flat_df.groupby('prj'):
                clustered_one_pdb = self._global_clustering(one_pdb_df, feature_cols=['x', 'y', 'z'],
                                                            algorithm=algorithm,
                                                            no_prj_reps=5,  # refers to min_clusters
                                                            largest_pocket=largest_pocket)
                clustered_one_pdb = clustered_one_pdb.rename(
                    columns={'cluster_label': 'cluster_label_PDB'})
                clustered_df_list.append(clustered_one_pdb)

        clustered_df = pd.concat(clustered_df_list)
        clustered_df['cluster_label_PDB_unique'] = (clustered_df['prj'] + '_'
                                                    + clustered_df['cluster_label_PDB'].astype(str))
        clustered_df.to_csv(os.path.join(self.saving_loc, 'detailed_pocket_clusters_pdb_unique.csv'),
                            index=False)
        return clustered_df

    def voxel_intersection_over_union_global_id(self, summary_df, voxel_size=1.0, iou_thresh=0.3, dilation_radius=5,
                                                min_cluster_size=2,  point_df=None, run_within_pdb_clustering=True):
        """
        Input df should contain dummy atom and descriptor info

        Concept: clustering within one PDB structure first, then
        create grid of 1 A resolution --> voxelize the pockets
        compute pairwise Intersection over Union (IoU)
        build graph (nodes = pockets, edge = if IoU greater than threshold)
        extract connected components
        # -----------------------------------
        # PARAMETERS — tune these four! the default settings worked well for hTAAR1, mTAAR1, mTAAR7f and mTAAR9 overlap
        # -----------------------------------
        voxel_size       = 1.0     # Å per voxel
        iou_thresh       = 0.3     # require x % IoU to link two pockets
        dilation_radius  = 5       # pad each pocket by x voxels to absorb small shifts
        min_cluster_size = 2       # drop any connected component smaller than this
        # -----------------------------------
        """
        if point_df is not None:
            df = point_df.copy()
        elif run_within_pdb_clustering:
            df = self._within_pdb_clustering(pdb_pocket_df=summary_df, centroids=False,
                                             algorithm='hdbscan')
        else:
            df = self.pocket_point_dataframe(summary_df)

        # Voxelization anchored at (0,0,0) — negative indices OK in sets
        origin = np.array([0.0, 0.0, 0.0])
        pocket_voxels = {}
        pocket_bounds = {}
        for pid, grp in df.groupby('Pocket ID'):
            # shift & voxelize
            pts = grp[['x', 'y', 'z']].values - origin
            ijk = np.floor(pts / voxel_size).astype(int)
            vox = set(map(tuple, ijk))
            # optional dilation
            if dilation_radius > 0:
                vox = self._dilate_voxels(vox, dilation_radius)
            pocket_voxels[pid] = vox
            # store AABB in voxel coords
            mins = ijk.min(axis=0)
            maxs = ijk.max(axis=0)
            pocket_bounds[pid] = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

        # Build IoU‐graph
        G = nx.Graph()
        G.add_nodes_from(pocket_voxels)

        for p1, p2 in combinations(pocket_voxels, 2):
            # quick axis-aligned bounding box reject
            x1min, x1max, y1min, y1max, z1min, z1max = pocket_bounds[p1]
            x2min, x2max, y2min, y2max, z2min, z2max = pocket_bounds[p2]
            if (x1max < x2min or x2max < x1min or
                    y1max < y2min or y2max < y1min or
                    z1max < z2min or z2max < z1min):
                continue
            v1 = pocket_voxels[p1]
            v2 = pocket_voxels[p2]
            inter = v1 & v2
            if not inter:
                continue
            union = v1 | v2
            iou = len(inter) / len(union)
            if iou >= iou_thresh:
                G.add_edge(p1, p2)

        # Connected components → group IDs, drop tiny ones
        mapping = {}
        group_id = 1
        for comp in nx.connected_components(G):
            if len(comp) >= min_cluster_size:
                for pid in comp:
                    mapping[pid] = group_id
                group_id += 1
            else:
                # label singleton/noise as 0
                for pid in comp:
                    mapping[pid] = 0

        # Re-assign the 0 clusters to their nearest neighbours (KNeighboursClassifier scikit-learn)
        centroids = (df.groupby('Pocket ID')[['x', 'y', 'z']].mean().rename(columns={'x': 'cx', 'y': 'cy', 'z': 'cz'}))
        centroids['group'] = centroids.index.map(mapping)

        labeled_mask = centroids['group'] > 0  # labelled clusters
        noise_mask = centroids['group'] == 0  # noise / un-assigned

        if noise_mask.any() and labeled_mask.any():
            # train 1-NN on the labeled centroids
            X_train = centroids.loc[labeled_mask, ['cx', 'cy', 'cz']].values
            y_train = centroids.loc[labeled_mask, 'group'].values
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            X_noise = centroids.loc[noise_mask, ['cx', 'cy', 'cz']].values
            new_labels = knn.predict(X_noise)
            centroids.loc[noise_mask, 'group'] = new_labels
            for pid, grp in centroids['group'].items():
                mapping[pid] = int(grp)

        # Map back as additional column
        df['voxel_group_id'] = df['Pocket ID'].map(mapping)
        self._plot_pocket_clusters_qc(df, centroids=False, cluster_col='voxel_group_id')
        df.to_csv(os.path.join(self.saving_loc, 'global_pockets_IoU_voxel.csv'), index=False)

        # save dictionary of old ID and new, global ID
        df['Pocket ID local'] = df['Pocket ID'].str.replace('_i3.0$', '', regex=True)
        voxel_to_pockets = (df.groupby('voxel_group_id')['Pocket ID local'].apply(set).to_dict())

        df['prj'] = df['Pocket ID local'].str.extract(r'^([^_]+)')
        df['rep'] = df['Pocket ID local'].str.extract(r'_(\d)_')
        df['pocket_number local'] = df['Pocket ID local'].str.extract(r'_p(\d+)')

        df['Pocket ID global'] = df['voxel_group_id']
        reduced_df = df[['Pocket ID local', 'Pocket ID global', 'prj', 'rep', 'pocket_number local']].drop_duplicates()
        reduced_df.to_csv(os.path.join(self.saving_loc, 'reduced_local_to_globalVoxelID.csv'), index=False)

        with open(os.path.join(self.saving_loc, 'local_to_globalVoxelID.txt'), 'w') as f:
            for voxel_id, pockets in voxel_to_pockets.items():
                f.write(f"{voxel_id}: {sorted(pockets)}\n")  # sorted for consistency

        if self.verbose:
            print(f'Global voxel based IoU for the purpose of a global pocket identifier resulted in the following data. \n'
                  f'Data saved under {os.path.join(self.saving_loc, 'global_pockets_IoU_voxel.csv')}')
            logger.log('Global voxel based IoU for the purpose of a global pocket identifier resulted in the following: \n'
                        '\n')
            print(df)
            print(f'Number of global Clusters found {len(set(df['voxel_group_id'].to_list()))}')
            logger.log(f'Number of Clusters found {len(set(df['voxel_group_id'].to_list()))}')
        else:
            logger.log(f'Global voxel based IoU for the purpose of a global pocket identifier resulted in the following data. '
                       '\n'
                       f'Data saved under {os.path.join(self.saving_loc, 'global_pockets_IoU_voxel.csv')}')
            logger.log(f'Number of Clusters found {len(set(df['voxel_group_id'].to_list()))}')

        return df

    def _deprecated_add_pocket_identifier_clustering_approaches(self, all_pock_coords_descriptors_df, centroids=True,
                                                                algorithm='hdbscan'):
        """The global ID is assigned based on a global clustering of the x, y, z coordinates of the alpha spheres'
        centroid as defined by the mdpocket algorithm
        The global clustering may result in a loss of detail compared to the local clustering but taken together the
        cross-referenced info should make a comparison of specific locations, e.g., the intracellular G-protein
        binding site possible"""
        # deprecated because the voxel IoU works well

        # Concept: global clustering based on the x, y, z coordinates of the alpha spheres composing the pocket
        # Clustering options include: KMeans, Agglomerative, HDBSCAN and ATClus
        # Clustering can also be performed using centroids of the pockets,
        # i.e., the mean of the coordinates for an individual pocket
        global_pockets_dict = {}
        for id_p, df_pock in all_pock_coords_descriptors_df.groupby('ID'):
            feature_dict = self._data_prep_for_clustering(df_pock)
            global_pockets_dict[id_p] = feature_dict

        # build dataframe of nested dict
        feature_df = pd.json_normalize([{'Pocket ID': k, **v} for k, v in global_pockets_dict.items()])
        feature_df['project_replicate'] = feature_df['Pocket ID'].str.extract(r'^((?:apo|holo)[A-Z0-9]+_\d)')
        unique_projects = len(feature_df['project_replicate'].unique())
        max_no_points = feature_df['number_alpha_spheres'].max()  # = no of cols needed, then split the array

        if centroids:
            clustered_df = self._global_clustering(feature_df, feature_cols=['centroid_coords.x',
                                                    'centroid_coords.y', 'centroid_coords.z'],
                                                    algorithm=algorithm, no_prj_reps=unique_projects,
                                                   largest_pocket=2*max_no_points)
            self._plot_pocket_clusters_qc(clustered_df)
        else:
            feature_df['coordinates'] = feature_df['coordinates'].astype(str)
            col_names = [f'datapoint_{i}_{axis}' for i in range(max_no_points) for axis in ('x', 'y', 'z')]
            feature_df['coordinates'] = feature_df['coordinates'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            coords_expanded = feature_df['coordinates'].apply(
                lambda coords: self._to_padded_tuples(coords, target_len=max_no_points))
            coord_df = pd.DataFrame(coords_expanded.tolist(), columns=col_names)
            feature_df = pd.concat([feature_df, coord_df], axis=1)
            flat_df = self._melt_coordinates(feature_df)
            clustered_df = self._global_clustering(flat_df, feature_cols=['x', 'y', 'z'],
                                                          algorithm=algorithm, no_prj_reps=unique_projects,
                                                   largest_pocket=2*max_no_points)
            mapping_df = self._reassignment_global_clusters(clustered_df)
            mapping_df.to_csv(os.path.join(self.saving_loc, 'local_to_global.csv'), index=False)
            clustered_df['new_global'] = clustered_df['Pocket ID'].map(dict(zip(mapping_df['local_pocket_ID'].to_list(),
                                                                                mapping_df['new_global'].to_list())))

            self._plot_pocket_clusters_qc(clustered_df, centroids=False)

        # save outcome in the saving_loc (default = meta_analysis)
        clustered_df.to_csv(os.path.join(self.saving_loc, f'global_clustering_{algorithm}.csv'), index=False)

        if self.verbose:
            print(f'Global clustering for the purpose of a global pocket identifier resulted in the following data. '
                  f'Parameters used: centroids={centroids}, algorithm={algorithm} \n'
                  f'Data saved under {os.path.join(self.saving_loc, 'global_clustering.csv')}')
            logger.log('Global clustering for the purpose of a global pocket identifier resulted in the following: \n'
                        f'Parameters used: centroids={centroids}, algorithm={algorithm} \n')
            print(clustered_df)
            print(f'Number of Clusters found {clustered_df['cluster_label'].max() +1 }')
            logger.log(f'Number of Clusters found {clustered_df['cluster_label'].max() + 1}')
        else:
            logger.log(f'Global clustering for the purpose of a global pocket identifier resulted in the following data. '
                       f'Parameters used: centroids={centroids}, algorithm={algorithm} \n'
                       f'Data saved under {os.path.join(self.saving_loc, 'global_clustering.csv')}')
            logger.log(f'Number of Clusters found {clustered_df['cluster_label'].max() +1 }')

    def filter_ortho_site(self, pockets_df_per_gene, binding_site_res_dict, col_name='', min_overlap=0.8,
                          filename='pocket_labels_ortho_def.csv'):
        """Given a dictionary of residues defining the binding site per gene, this function selects
           pockets that have at least the selected overlap with the given residues for their specific gene.

           Args:
               pockets_df_per_gene: DataFrame containing pocket information with 'gene' column
               binding_site_res_dict: Dictionary mapping gene names to sets of residue IDs
                                     e.g., {'hTAAR1': {'72', '83', ...}, 'mTAAR1': {...}}
               col_name: Optional suffix for the is_binding_site column
               min_overlap: Minimum overlap ratio required (default 0.8)
               filename: Output filename for the results

           Returns:
               DataFrame with added label "is_binding_site"""""

        pocket_labels = []
        for (pock_no, prj, rep), df_pocket in pockets_df_per_gene.groupby(['pocket_number', 'prj', 'rep']):
            df_pocket = df_pocket.reset_index(drop=True)
            res_ids = df_pocket['residue_id'].astype(str).tolist()
            if 'residue_name' in df_pocket.columns:
                unique_pairs = (df_pocket[['residue_name', 'residue_id']].drop_duplicates().sort_values('residue_id'))
                aa_list = [f"{row.residue_name}{int(row.residue_id)}" for _, row in unique_pairs.iterrows()]
            else:
                aa_list = sorted(set(res_ids), key=lambda x: int(x))
            gene = df_pocket['gene'].iloc[0]
            if gene == 'mTAAR7F':
                gene = 'mTAAR7f'

            if gene not in binding_site_res_dict.keys():
                print(f"Warning: Gene {gene} not found in binding_site_res_dict. Skipping pocket.")
                continue

            ortho_site_res = {str(r) for r in binding_site_res_dict[gene]}

            # calculate overlap
            pocket_residues = set(res_ids)
            isovalue = df_pocket['isovalue'][0]
            overlap = pocket_residues & ortho_site_res
            overlap_ratio = len(overlap) / len(pocket_residues) if pocket_residues else 0
            is_binding = overlap_ratio >= min_overlap
            pocket_labels.append({'pocket_number': pock_no, 'prj': prj, 'rep': rep, 'isovalue': isovalue,
                                  'total_residues': len(pocket_residues), 'residues': aa_list,
                                  'binding_residues': sorted(overlap, key=int), 'overlap_count': len(overlap),
                                  'overlap_ratio': round(overlap_ratio, 2), f'is_binding_site_{col_name}': is_binding})

        pocket_df = pd.DataFrame(pocket_labels)
        pocket_df.to_csv(os.path.join(self.saving_loc, filename), index=False)
        return pocket_df


    def summary_statistics(self, pocket_summary_df, col_names):
        """creates a new dataframe with summarising statistics on the columns specified in the given col_name list
           and also adds the number of pockets per prj, rep, isovalue"""
        # need: prj, rep, gene, isovalue, #pockets, max. volume, min. volume,
        # max. hydrophobicity, min hydrophobicity, max, polarity min polarity
        if self.verbose:
            print(f'Analysing the descriptors {col_names} ')
        pocket_dict = {}
        for id_g, df in pocket_summary_df.groupby(['prj', 'rep', 'isovalue', 'pocket_number']):
            for id_x, single_pock in df.groupby(['x', 'y', 'z']):
                val_ls = []
                col_ls = []
                for col in col_names:
                    max_value_pocket = single_pock[col].max()
                    min_valu_pocket = single_pock[col].min()
                    min_nonzero_value = single_pock[col][single_pock[col] != 0].min()
                    std_value_pocket = single_pock[col].std()
                    pocket_desc_data = [max_value_pocket, min_valu_pocket, min_nonzero_value, std_value_pocket]
                    cols = [f'max. {col}', f'min. {col}', f'min. non-zero {col}', f'std.{col}']
                    val_ls.extend(pocket_desc_data)
                    col_ls.extend(cols)
                value_dict = dict(zip(col_ls, val_ls))
                key_id = '_'.join(map(str, id_g))
                if key_id not in pocket_dict.keys():
                    pocket_dict[key_id] = value_dict

        stat_df = pd.DataFrame.from_dict(pocket_dict, orient='index')
        stat_df.reset_index(inplace=True)
        stat_df = stat_df.rename(columns={'index': 'prj_rep_isovalue_pocketno'})

        stat_df[['prj', 'rep', 'isovalue', 'pocketno']] = stat_df['prj_rep_isovalue_pocketno'].str.split('_',
                                                                                                         expand=True)
        # add no of pockets overall
        structure = pocket_summary_df.groupby(['prj', 'rep', 'isovalue'])['pocket_number'].nunique().reset_index()
        structure.rename(columns={'pocket_number': 'pockets in this group'}, inplace=True)
        stat_df['isovalue'] = stat_df['isovalue'].astype(float)
        stat_df = stat_df.merge(structure, on=['prj', 'rep', 'isovalue'], how='left')
        pdb_ids = set(stat_df['prj'].to_list())
        pdb_to_gene = self._get_pdb_gene_dict(pdb_ids=True, id_list=list(pdb_ids))

        stat_df['gene_name'] = stat_df['prj'].map(pdb_to_gene)

        # save stat_df
        stat_df.to_csv(os.path.join(self.saving_loc, 'summary_statistics.csv'), index=False)
        return stat_df


