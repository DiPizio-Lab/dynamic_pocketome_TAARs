"""
Runs the comparative_study driver body
Options: apo only = apo, holo only = holo, both individually = both, and apo and holo as one dataset = combined

Each run gets its own complete, non-colliding result set, including:
    detailed_pocket_clusters_voxel_group_id_qc.html
    3D_coordinates_frequency_colour_coded.html
    for_plot_3D_coordinates_frequency_colour_coded.csv
    first_frame_summary_df.csv
    summary_df_3d_coords.csv
    pocket_comparison_table.csv
    local_to_globalVoxelID.txt / reduced_local_to_globalVoxelID.csv
    unique_to_gene_ / unique_to_structure_ / shared_in_all_gene_comparison.csv

"""

import os, sys, ast, glob, argparse, traceback
import pandas as pd

# repo root on sys.path so `config` and the `scripts` package below resolve
# regardless of the current working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR

# config
ROOT       = INPUT_DIR    # {state}_structures/results/... are inputs, read from input/
OUT_ROOT   = OUTPUT_DIR   # global_ID_{state}_only run directories are written under output/

SUBDIR_FMT = "global_ID_{state}_only"       # -> global_ID_apo_only / global_ID_holo_only
SUBDIR_BOTH = "global_ID_both_states"

# example path:
#   {ROOT}/holo_structures/results/holo8ITF/1/pockets_dens
DIR_GLOB   = "{root}/{state}_structures/results/{state}*/*/pockets_dens"

ISOVALUE   = 3.0
MIN_ATOMS  = 10
DBSCAN     = False
VERBOSE    = False

MODULE_PATH = os.path.join(PROJECT_ROOT, "pipeline")   # meta_analysis_class.py / comparative_study.py live here

# - imports
if MODULE_PATH and MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)
import meta_analysis_class as meta
try:
    from comparative_study import (within_pdb_plot, make_pocket_summary, annotate_uniqueness,
                                   upset_plot_pockets_shared, barchart_pocket_count, plot_3d_heatmap)
except ImportError:
     from comparative_study import (within_pdb_plot, make_pocket_summary, annotate_uniqueness,
                                           upset_plot_pockets_shared, barchart_pocket_count, plot_3d_heatmap)

try:
    from scripts import logging as log
except ImportError:
    class log:                                   # no-op fallback
        @staticmethod
        def log(*a, **k): pass

SHORT_GENE_NAMES = {
    '8ITF': 'mTAAR9', '8IW4': 'mTAAR9', '8IW7': 'mTAAR9', '8IWM': 'mTAAR7f',
    '8JLJ': 'mTAAR1', '8JLK': 'mTAAR1', '8JLN': 'mTAAR9', '8JLO': 'hTAAR1',
    '8JLP': 'hTAAR1', '8JLQ': 'hTAAR1', '8JLR': 'hTAAR1', '8JSO': 'hTAAR1',
    '8PM2': 'mTAAR7f', '8W87': 'hTAAR1', '8W88': 'hTAAR1', '8W89': 'hTAAR1',
    '8W8A': 'hTAAR1', '8WC3': 'mTAAR1', '8WC4': 'mTAAR1', '8WC5': 'mTAAR1',
    '8WC6': 'mTAAR1', '8WC7': 'hTAAR1', '8WC8': 'hTAAR1', '8WC9': 'mTAAR9',
    '8WCB': 'mTAAR1', '8WCC': 'hTAAR1'}

MAESTRO_BIND_SITE_STR = {
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


# wrapper
def _run(prj_ls, saving_loc, label):
    """Your comparative_study driver body. Running with a provided subset (=prj_ls)
    to generate a run dependent global ID"""

    pdbs = sorted({d.split(os.sep)[-3] for d in prj_ls})
    print(f"\n{'='*70}\n[{label}] {len(prj_ls)} pocket dirs | {len(pdbs)} PDBs: {', '.join(pdbs)}"
          f"\n[{label}] saving_loc -> {saving_loc}\n{'='*70}")

    meta_obj = meta.MetaAnalysis(pocket_dirs=prj_ls, saving_loc=saving_loc,
                                 isovalue=ISOVALUE, verbose=VERBOSE, dbscan=DBSCAN, min_atoms=MIN_ATOMS)

    result_dict, pockets_to_drop = meta_obj.pock_file_parser()
    intpol_descr_df = meta_obj.interpolation_volume(df=result_dict['descriptor_df'],
                                                    col_to_interpolate=['pock_volume'], limit=2,
                                                    limit_area='inside', method='linear')
    summary_df = pd.merge(result_dict['dummy_atom_df'], intpol_descr_df,
                          on=['ID', 'pocket_number', 'prj', 'rep', 'isovalue'],
                          how='outer')

    # Assign global cluster
    voxelized_df = meta_obj.voxel_intersection_over_union_global_id(summary_df=summary_df)

    # read in the mapping between local and global ID and append global ID column in summary df
    mapping_global_to_local = {}
    with open(os.path.join(saving_loc, 'local_to_globalVoxelID.txt')) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key_str, val_str = line.split(':', 1)
                mapping_global_to_local[key_str.strip()] = ast.literal_eval(val_str.strip())
    inverted_mapping_dict = {}
    for key, values in mapping_global_to_local.items():
        for val in values:
            inverted_mapping_dict[val] = key
    summary_df['Local ID'] = summary_df['ID'].str.replace('_i3.0$', '', regex=True)
    summary_df['Global ID'] = summary_df['Local ID'].map(inverted_mapping_dict)

    for pdb_id, pdb_df in summary_df.groupby('prj'):
        within_pdb_plot(pdb_df, pdb_id=pdb_id, cluster_col='Global ID', saving_loc=saving_loc)

    summary_df['clean_id'] = summary_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)
    summary_df['gene'] = summary_df['clean_id'].map(SHORT_GENE_NAMES)
    summary_df['state'] = summary_df['prj'].str.extract(r'^(apo|holo)', expand=False)
    summary_df.to_csv(os.path.join(saving_loc, 'summary_df_3d_coords.csv'), index=False)

    small_df = summary_df[['prj', 'rep', 'ID', 'Global ID', 'gene', 'clean_id']]
    pocket_summary = make_pocket_summary(summary_df)
    annotated_df = annotate_uniqueness(small_df)
    annotated_df = annotated_df[['prj', 'rep', 'Global ID', 'gene', 'n_replicates_within_prj',
                                 'unique_in_one_rep', 'genes_list', 'n_distinct_genes', 'unique_to_one_gene']]
    summary_df_upd = pocket_summary.merge(annotated_df, on=['prj', 'rep', 'Global ID', 'gene'], how='outer')

    # filter orthosteric binding site pockets and merge the boolean to the df
    merged_residue_df = result_dict['res_to_pock_df']
    merged_residue_df['residue_id'] = merged_residue_df['residue_id'].astype(int)
    merged_residue_df['clean_id'] = merged_residue_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)
    merged_residue_df['gene'] = merged_residue_df['clean_id'].map(SHORT_GENE_NAMES)

    pocket_df_ls = []
    for gene, gene_df in merged_residue_df.groupby('gene'):
        pocket_df_ls.append(meta_obj.filter_ortho_site(gene_df, MAESTRO_BIND_SITE_STR, min_overlap=0.8))
    ortho_pockets = pd.concat(pocket_df_ls)

    bind_cols = [col for col in ortho_pockets.columns if col.startswith('is_binding_site')]
    ortho_pockets = ortho_pockets.loc[ortho_pockets[bind_cols].any(axis=1)]
    ortho_pockets['is_binding_site_any'] = ortho_pockets[bind_cols].any(axis=1)

    # Match orthosteric pockets on (prj, rep, pocket_number) with pocket_number
    # normalised to a plain int. See notes: the parser
    # collapses isovalue 3.0 -> 'i3' in the ID (meta_analysis_class, is_integer),
    # and the wrapper's '_i3.0$' strip therefore never fires.
    def _pnum(s):
        return s.astype(str).str.extract(r'(\d+)')[0].astype(int)

    ortho_pockets = ortho_pockets.assign(
        rep=ortho_pockets['rep'].astype(str),
        pocket_number=_pnum(ortho_pockets['pocket_number']))
    summary_df_upd = summary_df_upd.assign(
        rep=summary_df_upd['rep'].astype(str),
        pocket_number=summary_df_upd['Local Pocket ID'].str.extract(r'_p(\d+)')[0].astype(int))

    summary_df_upd = summary_df_upd.merge(
        ortho_pockets[['prj', 'rep', 'pocket_number', 'is_binding_site_any', 'residues']]
            .drop_duplicates(['prj', 'rep', 'pocket_number']),
        on=['prj', 'rep', 'pocket_number'], how='left')
    summary_df_upd = summary_df_upd.rename(
        columns={'is_binding_site_any': 'is_binding_site', 'residues': 'pocket_residues'})
    summary_df_upd['is_binding_site'] = summary_df_upd['is_binding_site'].fillna(False)
    summary_df_upd['state'] = summary_df_upd['prj'].str.extract(r'^(apo|holo)', expand=False)
    summary_df_upd = summary_df_upd.drop_duplicates(['Local Pocket ID'])
    summary_df_upd.to_csv(os.path.join(saving_loc, 'pocket_comparison_table.csv'), index=False)

    # comparisons
    for comp_label, mask, fname in [
            ('unique to a gene',          summary_df_upd['unique_to_one_gene'] == True,  'unique_to_gene_comparison.csv'),
            ('unique to a PDB structure', summary_df_upd['unique_in_one_rep'] == True,   'unique_to_structure_comparison.csv'),
            ('present in all 4 genes',    summary_df_upd['n_distinct_genes'] == 4,       'shared_in_all_gene_comparison.csv')]:
        sub = summary_df_upd.loc[mask]
        sub.to_csv(os.path.join(saving_loc, fname), index=False)
        msg = (f"[{comp_label}] {comp_label}: {sub.shape} | local IDs {sub['Local Pocket ID'].nunique()} "
               f"| global IDs {sub['Global ID'].nunique()}")
        print(msg); log.log(msg)

    # plots (subsampled frame df, as in your driver)
    df_firstframe = pd.read_csv(os.path.join(saving_loc, 'summary_df_3d_coords.csv'),
                                skiprows=lambda i: i > 0 and (i - 1) % 1000 != 0)
    df_firstframe = df_firstframe.loc[:, ['ID', 'pocket_number', 'Global ID', 'x', 'y', 'z', 'gene']]
    df_firstframe = df_firstframe.reset_index(drop=True)
    df_firstframe.to_csv(os.path.join(saving_loc, 'first_frame_summary_df.csv'))

    for fn, args in [(upset_plot_pockets_shared, (summary_df_upd, saving_loc)),
                     (barchart_pocket_count, (summary_df_upd, saving_loc))]:
        try:
            fn(*args)
        except Exception as e:
            print(f"[{label}] Error in {fn.__name__}: {e}")
    try:
        plot_3d_heatmap(df=df_firstframe, saving_loc=saving_loc)
    except Exception as e:
        print(f"[{label}] Error in plot_3d_heatmap: {e}")

    n_global = summary_df['Global ID'].nunique()
    n_local = summary_df['Local ID'].nunique()
    print(f"\n[{label}] DONE -- {n_local} local pockets -> {n_global} global IDs")
    print(f"[{label}] QC: {os.path.join(saving_loc, 'detailed_pocket_clusters_voxel_group_id_qc.html')}")

def run_state(state):
    prj_ls = sorted(d for d in glob.glob(DIR_GLOB.format(root=ROOT, state=state)) if os.path.isdir(d))
    if not prj_ls:
        print(f"[{state}] NO pocket_dens dirs matched: {DIR_GLOB.format(root=ROOT, state=state)}"); return
    saving_loc = os.path.join(OUT_ROOT, SUBDIR_FMT.format(state=state))
    os.makedirs(saving_loc, exist_ok=True)
    _run(prj_ls, saving_loc, label=state)

def run_combined(states=("apo", "holo")):
    """One global-ID space over apo AND holo together -> IDs are comparable across states."""
    prj_ls = sorted(d for st in states
                    for d in glob.glob(DIR_GLOB.format(root=ROOT, state=st)) if os.path.isdir(d))
    if not prj_ls:
        print("[both] NO pocket_dens dirs matched"); return
    saving_loc = os.path.join(OUT_ROOT, SUBDIR_BOTH)
    os.makedirs(saving_loc, exist_ok=True)
    _run(prj_ls, saving_loc, label="both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=["apo", "holo", "both", "combined"], default="both")
    args = ap.parse_args()
    if args.state == "combined":
        run_combined()
    else:
        for state in (["apo", "holo"] if args.state == "both" else [args.state]):
            run_state(state)

    print("\nAll requested states finished.")


if __name__ == "__main__":
    main()