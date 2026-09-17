"""
Step 2.1: turns Step 1's per-replicate pockets/ output into two per-pocket dataframes.

For every replicate under output/{apo,holo}_structures/<state><PDBID>/<rep>/pockets/ (Step 1
output), this:
  1. Parses the mdpocket/ATClus dummy-atom PDBs, descriptor files and residue files into one
     merged per-(pocket, frame, alpha sphere) table - PocketFileParser.pock_file_parser()
  2. Interpolates pock_volume across short gaps - PocketFileParser.interpolation_volume()
  3. Classifies each pocket as orthosteric/binding-site - orthosteric_filter_ligand_based.classify_binding_site()
  4. Classifies each pocket as transient/stable - consecutive_zeros_transiency.classify_transient()
  5. Flags the largest pocket per (state, PDB ID, replicate), and is_largest_and_orthosteric
  6. Derives volume_category from the per-pocket trajectory-median volume (open frames only)

Writes, under saving_loc (default config.META_ANALYSIS_DIR), as BOTH .csv and .parquet
(pipeline/pocket_io.py - callers read either through pocket_io.load_table):
    all_pockets       one row per (pocket, frame, alpha sphere) - every pocket, every
                      experiment, every frame, every alpha-sphere coordinate
    pocket_summary    one row per pocket (its first frame) - the same columns, collapsed
Also writes orthosteric_perframe_volumes.csv (csv only, small: all_pockets filtered to
is_orthosteric), a convenience file taar_paper_figures/fig2_binding_site.py reads directly.
"""
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as conf
from scripts import gene_selections
from scripts import logging as logger
from pipeline import pocket_io
from pipeline import visualizations as vis
from pipeline.orthosteric_filter_ligand_based import classify_binding_site
from pipeline.consecutive_zeros_transiency import classify_transient

ISOVALUE = conf.ISOVALUES[0]
MIN_ATOMS = 10
DBSCAN = False

DESCRIPTOR_COLS = [
    'pock_asa', 'pock_pol_asa', 'pock_apol_asa', 'pock_asa22', 'pock_pol_asa22', 'pock_apol_asa22',
    'nb_AS', 'mean_as_ray', 'mean_as_solv_acc', 'apol_as_prop', 'mean_loc_hyd_dens',
    'hydrophobicity_score', 'volume_score', 'polarity_score', 'charge_score', 'prop_polar_atm',
    'as_density', 'as_max_dst', 'convex_hull_volume', 'nb_abpa']  # mdpocket descriptor columns
RESIDUE_COLS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

KEEP_COLS = (['ID', 'Local ID', 'prj', 'rep', 'state', 'pdb_id', 'gene', 'pocket_number',
             'x', 'y', 'z', 'snapshot', 'interpolated_pock_volume']
             + DESCRIPTOR_COLS + RESIDUE_COLS
             + ['is_largest_pocket', 'is_orthosteric', 'is_largest_and_orthosteric', 'transient',
                'n_zero_frames', 'max_consecutive_zero_frames', 'volume_category',
                'median_pock_volume_all', 'median_pock_volume_open', 'state_pdbid_rep'])


class PocketFileParser:
    """Parses one isovalue's worth of mdpocket/ATClus output (dummy-atom PDBs, descriptor
    files, residue files) across many pocket_dirs into merged dataframes. This is the only
    place that reads the raw pockets/ files - everything downstream works off its output."""

    def __init__(self, pocket_dirs, isovalue=ISOVALUE, dbscan=DBSCAN, saving_loc='.',
                 min_atoms=MIN_ATOMS, verbose=False):
        self.pocket_dirs = pocket_dirs
        self.isovalue = isovalue
        self.dbscan = dbscan
        self.saving_loc = saving_loc
        self.verbose = verbose
        self.min_atoms = min_atoms

    def _normalize_pocket_filenames(self):
        """ATClus/mdpocket always save filenames with the isovalue written as a bare float (e.g.
        mdpout_dens_iso_3.0-out-01.pdb, mdpout_dens_iso_3.5-out-01.pdb) - which breaks every
        downstream dot-free assumption (the -out- search pattern below, and _parse_filename's
        digit-only regex) if left alone. Called first so no filename has a dot anywhere but the
        extension: a non-integer isovalue has its dot replaced with an underscore (iso_3.5 ->
        iso_3_5); an integer-valued one has the dot & trailing zero stripped entirely (iso_3.0
        -> iso_3, matching the "_i3"-style ID convention used everywhere else in the pipeline)"""
        if not isinstance(self.isovalue, float):
            return
        iso_str = str(self.isovalue)
        iso_dot = f"iso_{iso_str}"
        iso_clean = f"iso_{int(self.isovalue)}" if self.isovalue.is_integer() else f"iso_{iso_str.replace('.', '_')}"
        if iso_dot == iso_clean:
            return

        for pocket_path in self.pocket_dirs:
            if os.path.isdir(pocket_path):
                for filename in os.listdir(pocket_path):
                    if iso_dot in filename:
                        new_filename = filename.replace(iso_dot, iso_clean)
                        os.rename(os.path.join(pocket_path, filename), os.path.join(pocket_path, new_filename))
                        if self.verbose:
                            print(f"Renamed: {filename} -> {new_filename}")

    def _parse_filename(self, filename):
        """Parses filename to extract isovalue, pocket number, and type."""
        parsed = {}
        if self.dbscan:
            if filename.startswith("coords_pocket"):
                if 'descriptors' in filename:
                    parsed['type'] = 'descriptor'
                elif 'atoms' in filename:
                    parsed['type'] = 'residue'
                else:
                    parsed['type'] = 'dummy_atom'
                parts = filename.replace('.pdb', '').replace('.txt', '').split('_')
                parsed['pocket_no'] = parts[2]
                parsed['isovalue'] = float(parts[3].replace('_', '.'))
            else:
                parsed['type'] = 'unknown'
        else:
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
                try:
                    parsed['isovalue'] = float(match.group(1).replace('_', '.'))
                    parsed['pocket_no'] = match.group(2)
                except ValueError:
                    parsed['isovalue'] = 'unknown'
                    parsed['pocket_no'] = 'unknown'
            else:
                parsed['isovalue'] = 'unknown'
                parsed['pocket_no'] = 'unknown'
        return parsed

    def _descriptor_extraction(self, file_path, prj, rep, file):
        df_descriptors = pd.read_csv(file_path, sep=r'\s+')
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
            isovalue = float(match.group(1).replace('_', '.'))
            pocket_no = match.group(2)
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
        return df

    def pock_file_parser(self):
        """Returns (result_dict, pockets_to_drop) where result_dict has keys 'dummy_atom_df'
        (per alpha sphere, per frame), 'descriptor_df' (per frame, one row per pocket),
        'res_to_pock_df' (per residue lining each pocket) and 'pock_dummies_traj_paths'."""
        self._normalize_pocket_filenames()  # handle float isovalues
        pocket_dummy_atom_dfs, pock_dummies_traj_paths = [], []
        res_to_pock_dfs, descriptor_dfs = [], []
        exclusion_list, pockets_to_drop = [], []

        if not isinstance(self.isovalue, float):
            isovalue = round(np.float64(self.isovalue), 3)
        elif self.isovalue.is_integer():
            isovalue = int(self.isovalue)
        else:
            isovalue = self.isovalue

        for pocket_path in self.pocket_dirs:
            if not os.path.isdir(pocket_path):
                continue
            path_parts = os.path.normpath(pocket_path).split(os.sep)
            prj = path_parts[-3] if len(path_parts) >= 3 else "unknown_prj"
            rep = path_parts[-2] if len(path_parts) >= 2 else "unknown_rep"

            iso_str = str(isovalue).replace('.', '_')
            files = [f for f in os.listdir(pocket_path) if f"iso_{iso_str}-out-" in f]
            if self.verbose:
                print(f'Reading in pocket files of {prj}_{rep}')

            pocket_dict = {}
            for file in files:
                parsed = self._parse_filename(file)
                if parsed['type'] == 'dummy_atom' and parsed['isovalue'] == isovalue:
                    pocket_dict[parsed['pocket_no']] = os.path.join(pocket_path, file)

            for pock_no, file_path in pocket_dict.items():
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
                df.columns = ['atom', 'atom_no', 'chain', 'molecule_name', 'pocket_number',
                              'x', 'y', 'z', 'n1', 'n2']
                df[['prj', 'rep', 'isovalue']] = [prj, rep, isovalue]
                df['isovalue'] = df['isovalue'].astype(float)
                df['pocket_number'] = pock_no
                unique_id = f"{prj}_{rep}_p{pock_no}_i{isovalue}"
                df.insert(0, 'ID', unique_id)

                if df.shape[0] > self.min_atoms:
                    pocket_dummy_atom_dfs.append(df)
                else:
                    pockets_to_drop.append(unique_id)
                exclusion_list.append(file_path)

            for file in files:
                file_abs = os.path.join(pocket_path, file)
                if file_abs in exclusion_list:
                    continue
                parsed = self._parse_filename(file)
                if parsed['type'] == 'trajectory_pocket':
                    pock_dummies_traj_paths.append(file_abs)
                elif parsed['type'] == 'residue':
                    curr_df = self._extract_amino_acids_of_pocket(file_abs, prj, rep)
                    res_to_pock_dfs.append(curr_df)
                elif parsed['type'] == 'descriptor':
                    descriptor_dfs.append(self._descriptor_extraction(file_abs, prj, rep, file))

        merged_pocket_dummy_atom_df = pd.concat(pocket_dummy_atom_dfs, ignore_index=True)
        merged_descriptor_df = pd.concat(descriptor_dfs, ignore_index=True)
        merged_residue_df = pd.concat(res_to_pock_dfs, ignore_index=True)
        merged_descriptor_df['isovalue'] = merged_descriptor_df['isovalue'].astype(float).round(3)
        merged_descriptor_df = merged_descriptor_df[~merged_descriptor_df['ID'].isin(pockets_to_drop)]
        merged_descriptor_df = merged_descriptor_df[merged_descriptor_df['isovalue'] == isovalue]

        result_dict = {'dummy_atom_df': merged_pocket_dummy_atom_df, 'descriptor_df': merged_descriptor_df,
                       'res_to_pock_df': merged_residue_df, 'pock_dummies_traj_paths': pock_dummies_traj_paths}

        dropped_path = os.path.join(self.saving_loc, f"projects_with_less_than_{self.min_atoms}_atoms.txt")
        os.makedirs(self.saving_loc, exist_ok=True)
        with open(dropped_path, "w") as output:
            output.write(str(pockets_to_drop))
        if self.verbose:
            print(f'Saving the excluded pockets to {dropped_path}')

        return result_dict, pockets_to_drop

    def interpolation_volume(self, df, col_to_interpolate, limit=2, limit_area='inside', method='linear'):
        """Interpolates 0-valued (closed-pocket) frames, up to `limit` (2 by default) consecutive ones, so a
        pocket that briefly closes doesn't register as a volume of exactly 0 for that frame.
        See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html"""
        if self.verbose:
            print(f'Interpolation of the {col_to_interpolate} column')
        else:
            logger.log(f'Interpolation of the {col_to_interpolate} column')
        for col in col_to_interpolate:
            new_col = 'interpolated_' + col
            df[new_col] = df[col].replace(0, np.nan)
            df[new_col] = df[new_col].interpolate(limit=limit, limit_area=limit_area, method=method)
            df[new_col] = df[new_col].fillna(0)
        return df


def _add_identity_columns(df):
    """Local ID (per-experiment pocket identifier - kept isovalue-inclusive, same as ID: pocket
    numbering restarts per isovalue within pocket_search's isovalues=[...] list, so a stripped
    suffix would falsely equate e.g. p01@i2.0 with an unrelated p01@i3.0; only the voxel-IoU
    spatial clustering in global_id_and_comparison.py is allowed to unify pockets across
    isovalues), pdb_id, state and gene - gene from reference_data/hard_coded_gene_dict.txt via
    scripts.gene_selections"""
    df['Local ID'] = df['ID']
    df['state'] = df['prj'].str.extract(r'^(apo|holo)', expand=False)
    df['pdb_id'] = df['prj'].str.extract(r'^(?:apo|holo)([A-Za-z0-9]+)', expand=False)
    gene_map = gene_selections.gene_dict()
    df['gene'] = df['pdb_id'].map(gene_map)
    df['state_pdbid_rep'] = df['ID'].str.extract(r'^([a-z]+[A-Za-z0-9]+_\d+)', expand=False)
    return df


def _add_orthosteric_flag(df, saving_loc):
    """is_orthosteric via orthosteric_filter_ligand_based.classify_binding_site()"""
    ortho = classify_binding_site(df, holo_base=conf.HOLO_BASE_DIR, saving_loc=saving_loc)
    ortho = ortho.assign(rep=ortho['rep'].astype(str), pocket_number=ortho['pocket_number'].astype(str))

    df = df.assign(rep=df['rep'].astype(str), pocket_number=df['pocket_number'].astype(str))
    df = df.merge(ortho[['prj', 'rep', 'pocket_number', 'is_binding_site']]
                  .drop_duplicates(['prj', 'rep', 'pocket_number']),
                  on=['prj', 'rep', 'pocket_number'], how='left')
    df['is_orthosteric'] = df['is_binding_site'].fillna(False)
    return df.drop(columns=['is_binding_site'])


def _add_largest_pocket_flags(df, volume_col='interpolated_pock_volume'):
    """is_largest_pocket: the biggest pocket (by median interpolated volume) in each
    (state, PDB ID, replicate) experiment. is_largest_and_orthosteric: both at once."""
    pocket_median_vol = df.groupby('ID')[volume_col].median().reset_index()
    pocket_median_vol['state_pdbid_rep'] = pocket_median_vol['ID'].str.extract(
        r'^([a-z]+[A-Za-z0-9]+_\d+)', expand=False)
    largest_ids = pocket_median_vol.loc[
        pocket_median_vol.groupby('state_pdbid_rep')[volume_col].idxmax(), 'ID']
    df['is_largest_pocket'] = df['ID'].isin(largest_ids)
    df['is_largest_and_orthosteric'] = df['is_largest_pocket'] & df['is_orthosteric']
    return df


def _add_volume_category(df, volume_col='interpolated_pock_volume'):
    """median_pock_volume_all/_open and volume_category, mapped onto every row of a pocket."""
    median_all = df.groupby('ID')[volume_col].median()
    median_open = df.loc[df[volume_col] > 0].groupby('ID')[volume_col].median()
    df['median_pock_volume_all'] = df['ID'].map(median_all)
    df['median_pock_volume_open'] = df['ID'].map(median_open).fillna(0.0)
    df['volume_category'] = df['median_pock_volume_open'].apply(vis.categorize_volume)
    return df


def build_pocket_dataframes(pocket_dirs, saving_loc=conf.META_ANALYSIS_DIR, isovalue=ISOVALUE,
                            min_atoms=MIN_ATOMS, formats=('csv', 'parquet'), make_plots=True,
                            bw_file_loc=None, pdb_file_loc=None, verbose=True,
                            use_raw_checkpoint=True):
    """Runs the full Step 2.1 pipeline and writes all_pockets / pocket_summary (+
    orthosteric_perframe_volumes.csv). Returns {'all_pockets', 'pocket_summary',
    'transient_dict'} for immediate reuse (e.g. from a notebook) without re-reading from disk.

    use_raw_checkpoint (default True): cache the raw, pre-identity-columns `all_pockets` merge -
    the output of the expensive part, reading every one of `pocket_dirs` to
    `{saving_loc}/_checkpoint_raw_all_pockets.parquet`, and reuse it on a later call instead of re-reading from disk.
    This means a failure or retry in any of the cheap, in-memory work after the read
    (identity columns, orthosteric flag, ...) doesn't force paying the full read again.
    Caution: this does NOT detect if pocket_dirs' underlying files changed since the checkpoint was written
    (e.g. Step 1 was re-run) - delete _checkpoint_raw_all_pockets.parquet` (or pass use_raw_checkpoint=False)
    after any real change to Step 1 output."""
    os.makedirs(saving_loc, exist_ok=True)
    raw_checkpoint_path = os.path.join(saving_loc, '_checkpoint_raw_all_pockets.parquet')

    if use_raw_checkpoint and os.path.exists(raw_checkpoint_path):
        print(f'CHECKPOINT: loading raw pocket read from {raw_checkpoint_path} instead of '
             f're-reading {len(pocket_dirs)} replicates - delete this file (or pass '
             f'use_raw_checkpoint=False) if Step 1 output has changed since it was written.')
        all_pockets = pd.read_parquet(raw_checkpoint_path)
    else:
        parser = PocketFileParser(pocket_dirs, isovalue=isovalue, saving_loc=saving_loc,
                                  min_atoms=min_atoms, verbose=verbose)
        result_dict, pockets_to_drop = parser.pock_file_parser()

        intpol_descr_df = parser.interpolation_volume(result_dict['descriptor_df'],
                                                       col_to_interpolate=['pock_volume'])
        all_pockets = pd.merge(result_dict['dummy_atom_df'], intpol_descr_df,
                               on=['ID', 'pocket_number', 'prj', 'rep', 'isovalue'], how='outer')
        if use_raw_checkpoint:
            all_pockets.to_parquet(raw_checkpoint_path)
            print(f'CHECKPOINT: saved raw pocket read to {raw_checkpoint_path} '
                 f'({len(all_pockets)} rows) - a retry from here will load this instead of '
                 f're-reading {len(pocket_dirs)} replicates.')

    all_pockets = _add_identity_columns(all_pockets)
    all_pockets = _add_orthosteric_flag(all_pockets, saving_loc)
    all_pockets = _add_largest_pocket_flags(all_pockets)
    all_pockets, transient_dict = classify_transient(all_pockets, volume_col='interpolated_pock_volume')
    all_pockets = _add_volume_category(all_pockets)

    all_pockets = all_pockets[[c for c in KEEP_COLS if c in all_pockets.columns]]
    first_frame_idx = all_pockets.groupby('ID')['snapshot'].idxmin()
    pocket_summary = all_pockets.loc[first_frame_idx].reset_index(drop=True)
    pocket_summary['stability'] = pocket_summary['transient'].map({True: 'Transient', False: 'Stable'})

    pocket_io.save_table(all_pockets, saving_loc, 'all_pockets', formats=formats)
    pocket_io.save_table(pocket_summary, saving_loc, 'pocket_summary', formats=formats)
    orthosteric_perframe = all_pockets[all_pockets['is_orthosteric']]
    orthosteric_perframe.to_csv(os.path.join(saving_loc, 'orthosteric_perframe_volumes.csv'), index=False)

    message = (f'{len(pocket_summary)} pockets parsed ({pocket_summary["transient"].sum()} transient, '
              f'{pocket_summary["is_orthosteric"].sum()} orthosteric)')
    if verbose:
        print(message)
    else:
        logger.log(message)

    if make_plots:
        _make_plots(all_pockets, pocket_summary, transient_dict, saving_loc,
                   bw_file_loc=bw_file_loc, pdb_file_loc=pdb_file_loc)

    return {'all_pockets': all_pockets, 'pocket_summary': pocket_summary, 'transient_dict': transient_dict}


def _make_plots(all_pockets, pocket_summary, transient_dict, saving_loc, bw_file_loc=None,
                pdb_file_loc=None, n_cols=4, greyscale=False):
    largest_df = all_pockets[all_pockets['is_largest_pocket']].reset_index(drop=True)
    ortho_df = all_pockets[all_pockets['is_orthosteric']].reset_index(drop=True)

    for label, sub_df, plot_dir in (('largest', largest_df, 'plots_largest'),
                                    ('orthosteric', ortho_df, 'plots_orthosteric')):
        out_dir = os.path.join(saving_loc, plot_dir)
        os.makedirs(out_dir, exist_ok=True)
        if bw_file_loc and pdb_file_loc:
            try:
                vis.plot_largest_pockets(
                    saving_loc=saving_loc, pocket_df=sub_df,
                    pdb_file=os.path.join(pdb_file_loc, 'aligned_top.pdb'),
                    bw_csv_file=os.path.join(bw_file_loc, 'taar9_mouse.csv'),
                    out_html=f'{label}_pocket_3d.html', backbone=True, export_pdfs=True,
                    pdf_output_dir=os.path.join(saving_loc, f'pdfs_{label}'), use_cdn=True,
                    title=f'{label.capitalize()} Pocket Analysis')
            except Exception as e:
                print(f"  WARNING: could not create {label} 3D plot: {e}")
        try:
            vis.create_all_distribution_plots(pocket_df=sub_df, transient_dict=transient_dict,
                                              saving_loc=out_dir, volume_col='interpolated_pock_volume',
                                              n_cols=n_cols, greyscale=greyscale)
        except Exception as e:
            print(f"  WARNING: could not create {label} distribution plots: {e}")

    try:
        create_comparison_plots(pocket_summary, saving_loc, greyscale=greyscale)
    except Exception as e:
        print(f"  WARNING: could not create largest-vs-orthosteric comparison plots: {e}")

    try:
        vis.vis_apo_holo_number_pockets(pocket_summary, figsize=(14, 6), version='median',
                                        save_path=os.path.join(saving_loc, 'apo_holo_pockets_median.png'))
        vis.vis_apo_holo_number_pockets(pocket_summary, figsize=(14, 6), version='replicates',
                                        save_path=os.path.join(saving_loc, 'apo_holo_pockets_replicates.png'))
    except Exception as e:
        print(f"  WARNING: could not create apo/holo pocket-count plots: {e}")

    try:
        vis.plot_binding_site_volume_violin(pocket_comparison_df=pocket_summary, summary_df=all_pockets,
                                            saving_loc=saving_loc, summary_csv_path=None,
                                            create_individual_plots=True)
    except Exception as e:
        print(f"  WARNING: could not create binding-site violin plots: {e}")


def create_comparison_plots(summary_df, saving_loc, greyscale=False):
    """Largest-vs-orthosteric comparison plots: overlap, stability, volume category, and
    whether the largest pocket of each experiment is also its orthosteric one."""

    stability_colors = vis.get_color_dict('stability', greyscale)
    volume_colors = vis.get_color_dict('volume', greyscale)
    comparison_dir = os.path.join(saving_loc, 'plots_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    largest_only = ((summary_df['is_largest_pocket']) & (~summary_df['is_orthosteric'])).sum()
    ortho_only = ((~summary_df['is_largest_pocket']) & (summary_df['is_orthosteric'])).sum()
    both = summary_df['is_largest_and_orthosteric'].sum()
    neither = ((~summary_df['is_largest_pocket']) & (~summary_df['is_orthosteric'])).sum()

    fig1 = go.Figure(data=[go.Bar(x=['Largest Only', 'Orthosteric Only', 'Both', 'Neither'],
                                  y=[largest_only, ortho_only, both, neither],
                                  marker_color=vis.get_viridis_colors(4, greyscale=greyscale),
                                  text=[largest_only, ortho_only, both, neither], textposition='auto')])
    fig1.update_layout(title=dict(text='Pocket Classification Overlap', x=0.5, font=dict(size=18)),
                       xaxis_title='Category', yaxis_title='Number of Pockets', template='plotly_white',
                       height=500, width=700)
    fig1.write_html(os.path.join(comparison_dir, 'pocket_overlap.html'), include_plotlyjs='cdn')

    largest_stability = summary_df[summary_df['is_largest_pocket']]['stability'].value_counts()
    ortho_stability = summary_df[summary_df['is_orthosteric']]['stability'].value_counts()
    x_labels = ['Largest Pockets', 'Orthosteric Pockets']
    fig2 = go.Figure()
    for stability in vis.STABILITY_ORDER:
        fig2.add_trace(go.Bar(name=stability, x=x_labels,
                              y=[largest_stability.get(stability, 0), ortho_stability.get(stability, 0)],
                              marker_color=stability_colors[stability],
                              text=[largest_stability.get(stability, 0), ortho_stability.get(stability, 0)],
                              textposition='auto'))
    fig2.update_layout(barmode='stack', title=dict(text='Stability: Largest vs Orthosteric Pockets', x=0.5,
                                                    font=dict(size=18)), xaxis_title='Pocket Type',
                       yaxis_title='Number of Pockets', legend_title='Stability', template='plotly_white',
                       height=500, width=600)
    fig2.write_html(os.path.join(comparison_dir, 'stability_comparison.html'), include_plotlyjs='cdn')

    largest_volume = summary_df[summary_df['is_largest_pocket']]['volume_category'].value_counts()
    ortho_volume = summary_df[summary_df['is_orthosteric']]['volume_category'].value_counts()
    fig3 = go.Figure()
    for cat in vis.VOLUME_CATEGORY_ORDER:
        fig3.add_trace(go.Bar(name=cat, x=x_labels, y=[largest_volume.get(cat, 0), ortho_volume.get(cat, 0)],
                              marker_color=volume_colors[cat],
                              text=[largest_volume.get(cat, 0), ortho_volume.get(cat, 0)], textposition='auto'))
    fig3.update_layout(barmode='stack', title=dict(text='Volume Categories: Largest vs Orthosteric Pockets',
                                                    x=0.5, font=dict(size=18)), xaxis_title='Pocket Type',
                       yaxis_title='Number of Pockets', legend_title='Volume Category', template='plotly_white',
                       height=500, width=600)
    fig3.write_html(os.path.join(comparison_dir, 'volume_comparison.html'), include_plotlyjs='cdn')

    experiment_summary = summary_df[summary_df['is_largest_pocket']].copy()
    experiment_summary['largest_is_ortho'] = experiment_summary['is_orthosteric'].map({True: 'Yes', False: 'No'})
    by_state = experiment_summary.groupby(['state', 'largest_is_ortho']).size().unstack(fill_value=0)
    colors_yn = {'Yes': vis.get_viridis_colors(2, greyscale=greyscale)[0],
                'No': vis.get_viridis_colors(2, greyscale=greyscale)[1]}
    fig4 = go.Figure()
    for answer in ['Yes', 'No']:
        if answer in by_state.columns:
            fig4.add_trace(go.Bar(name=f'Largest is Orthosteric: {answer}', x=by_state.index, y=by_state[answer],
                                  marker_color=colors_yn[answer], text=by_state[answer], textposition='auto'))
    fig4.update_layout(barmode='group', title=dict(text='Is the Largest Pocket also Orthosteric?', x=0.5,
                                                    font=dict(size=18)), xaxis_title='State',
                       yaxis_title='Number of Experiments', template='plotly_white', height=500, width=600)
    fig4.write_html(os.path.join(comparison_dir, 'largest_is_orthosteric.html'), include_plotlyjs='cdn')

    comparison_summary = pd.DataFrame({
        'Category': ['Largest only', 'Orthosteric only', 'Both (largest & orthosteric)', 'Neither'],
        'Count': [largest_only, ortho_only, both, neither],
        'Percentage': [100 * largest_only / len(summary_df), 100 * ortho_only / len(summary_df),
                       100 * both / len(summary_df), 100 * neither / len(summary_df)]})
    comparison_summary.to_csv(os.path.join(comparison_dir, 'comparison_summary.csv'), index=False)


def pocket_dirs_for(state_dirs=(conf.APO_RESULTS_DIR, conf.HOLO_RESULTS_DIR), isovalue=None,
                    isovalues=conf.ISOVALUES, pdb_ids=None):
    """[<APO_RESULTS_DIR or HOLO_RESULTS_DIR>/<state><PDBID>/<rep>/pockets, ...] for every
    replicate that has a Step 1 pockets/ output under it - or, when isovalues has more than one
    entry, the isovalue_<X.X> subdirectory for the given isovalue (see conf.isovalue_subpath()).

    pdb_ids: None (default) includes every PDB ID; otherwise an iterable of PDB IDs restricts
    this to just those - a general-purpose filter (e.g. a quick test parse of one structure).
    Step 2's own scoping of Global ID clustering to a meaningful subset (conf.
    REPRESENTATIVE_PDB_IDS) happens downstream, in global_id_and_comparison.run_global_id_states'
    own pdb_ids= - not here. Leave this at None in the normal Step 2.1 pipeline run so
    all_pockets/pocket_summary stay the complete dataset regardless of which subset any given
    Global ID run is scoped to."""
    pocket_dirs = []
    for structures_dir in state_dirs:
        if not os.path.isdir(structures_dir):
            continue
        for folder in sorted(os.listdir(structures_dir)):
            folder_path = os.path.join(structures_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if pdb_ids is not None and re.sub(r'^(apo|holo)', '', folder) not in pdb_ids:
                continue
            for rep in sorted(os.listdir(folder_path)):
                pockets_path = conf.isovalue_subpath(os.path.join(folder_path, rep, conf.POCKETS_DIRNAME),
                                                     isovalue, isovalues)
                if os.path.isdir(pockets_path):
                    pocket_dirs.append(pockets_path)
    return pocket_dirs


def main():
    bw_file_loc = os.path.join(conf.REFERENCE_DATA_DIR, 'TAARs_numbered')
    pdb_file_loc = os.path.join(conf.HOLO_RESULTS_DIR, 'holo8ITF', '1')

    isovalues = conf.require_float_isovalues(conf.ISOVALUES)
    for isovalue in isovalues:
        saving_loc = conf.isovalue_meta_analysis_dir(isovalue, isovalues)
        pocket_dirs = pocket_dirs_for(isovalue=isovalue, isovalues=isovalues)
        build_pocket_dataframes(pocket_dirs, saving_loc=saving_loc, isovalue=isovalue,
                                bw_file_loc=bw_file_loc, pdb_file_loc=pdb_file_loc)


if __name__ == '__main__':
    main()
