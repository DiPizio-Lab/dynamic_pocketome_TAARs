#!/usr/bin/env python3
"""
resume_after_crash.py  --  finish an interrupted run_global_id_per_state.py state.

Your driver crashed just after writing summary_df_3d_coords.csv. Everything
expensive is already on disk and is NOT recomputed here:
    * pock_file_parser results  -> not needed for STAGE 1
    * voxel IoU global clustering (the costly O(n^2) step) -> already done,
      and its Global IDs are baked into summary_df_3d_coords.csv
    * within_pdb_plot htmls     -> present

STAGE 1 (cheap, no re-parse, no full load) -- restores the Blender input:
    first_frame_summary_df.csv
    3D_coordinates_frequency_colour_coded.html
    for_plot_3D_coordinates_frequency_colour_coded.csv   <-- Blender CSV_PATH

STAGE 2 (needs a pocket re-parse for res_to_pock_df, ~minutes) -- restores:
    pocket_comparison_table.csv
    unique_to_gene_comparison.csv
    unique_to_structure_comparison.csv
    shared_in_all_gene_comparison.csv

Idempotent: existing outputs are skipped unless FORCE = True.

Usage (run headless -- PyCharm is what died on memory):
    nohup python resume_after_crash.py --stage 1 > resume_s1.log 2>&1 &
    nohup python resume_after_crash.py --stage 2 > resume_s2.log 2>&1 &
    python resume_after_crash.py --stage both
"""

import os, sys, glob, argparse, traceback
import pandas as pd

# --------------------------------------------------------------------------- config
ROOT        = "/mnt/cpm_crienaecker/Z/RienaeckerC/example_case_global_ID"
STATE       = "holo"                        # the state that crashed
SAVING_LOC  = os.path.join(ROOT, f"global_ID_{STATE}_only")
DIR_GLOB    = "{root}/{state}_structures/results/{state}*/*/pockets_dens"
FORCE       = False                         # True = rewrite even if output exists

ISOVALUE, MIN_ATOMS, DBSCAN, VERBOSE = 3.0, 10, False, False
MODULE_PATH = "/mnt/cpm_crienaecker/Z/RienaeckerC/scripts"

if MODULE_PATH and MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

SUMMARY_CSV = os.path.join(SAVING_LOC, "summary_df_3d_coords.csv")

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


def need(path):
    if os.path.exists(path) and not FORCE:
        print(f"  skip (exists): {os.path.basename(path)}")
        return False
    return True


# --------------------------------------------------------------------------- stage 1
def stage1():
    """Rebuild the plot inputs. Reads only every 1000th row -- no full load, no re-parse."""
    from comparative_study import plot_3d_heatmap

    print(f"\n[{STATE}] STAGE 1 -- plot inputs from {os.path.basename(SUMMARY_CSV)}")
    if not os.path.exists(SUMMARY_CSV):
        raise SystemExit(f"missing {SUMMARY_CSV} -- cannot resume, rerun the state.")
    print(f"  source size: {os.path.getsize(SUMMARY_CSV)/1024**3:.2f} GB")

    ff_path = os.path.join(SAVING_LOC, "first_frame_summary_df.csv")
    if need(ff_path):
        # identical subsample to your driver
        df_firstframe = pd.read_csv(SUMMARY_CSV, skiprows=lambda i: i > 0 and (i - 1) % 1000 != 0)
        df_firstframe = df_firstframe.loc[:, ['ID', 'pocket_number', 'Global ID', 'x', 'y', 'z', 'gene']]
        df_firstframe = df_firstframe.reset_index(drop=True)
        df_firstframe.to_csv(ff_path)
        print(f"  wrote first_frame_summary_df.csv ({len(df_firstframe):,} rows)")
    else:
        df_firstframe = pd.read_csv(ff_path, index_col=0)

    print(f"  Global IDs present: {df_firstframe['Global ID'].nunique()} | "
          f"genes: {sorted(df_firstframe['gene'].dropna().unique())}")

    plot_path = os.path.join(SAVING_LOC, "for_plot_3D_coordinates_frequency_colour_coded.csv")
    if need(plot_path):
        plot_3d_heatmap(df=df_firstframe, saving_loc=SAVING_LOC)
        print("  wrote 3D_coordinates_frequency_colour_coded.html + for_plot_...csv")
    print(f"[{STATE}] STAGE 1 done -> Blender CSV_PATH = {plot_path}")


# --------------------------------------------------------------------------- stage 2
def stage2():
    """Rebuild the annotation/comparison tables. Re-parses pocket files for res_to_pock_df only;
    the global clustering is NOT rerun -- Global IDs are read back from summary_df_3d_coords.csv."""
    try:
        import meta_analysis_class as meta
    except ImportError:
        from scripts import meta_analysis as meta
    from comparative_study import make_pocket_summary, annotate_uniqueness

    out_tbl = os.path.join(SAVING_LOC, "pocket_comparison_table.csv")
    print(f"\n[{STATE}] STAGE 2 -- annotation tables")

    prj_ls = sorted(d for d in glob.glob(DIR_GLOB.format(root=ROOT, state=STATE)) if os.path.isdir(d))
    print(f"  {len(prj_ls)} pocket dirs")
    meta_obj = meta.MetaAnalysis(pocket_dirs=prj_ls, saving_loc=SAVING_LOC, isovalue=ISOVALUE,
                                 verbose=VERBOSE, dbscan=DBSCAN, min_atoms=MIN_ATOMS)

    print("  re-parsing pocket files (needed for res_to_pock_df) ...")
    result_dict, _ = meta_obj.pock_file_parser()

    print("  reading summary_df_3d_coords.csv ...")
    summary_df = pd.read_csv(SUMMARY_CSV)
    print(f"  {len(summary_df):,} rows | Global IDs: {summary_df['Global ID'].nunique()}")

    small_df = summary_df[['prj', 'rep', 'ID', 'Global ID', 'gene', 'clean_id']]
    pocket_summary = make_pocket_summary(summary_df)
    annotated_df = annotate_uniqueness(small_df)
    annotated_df = annotated_df[['prj', 'rep', 'Global ID', 'gene', 'n_replicates_within_prj',
                                 'unique_in_one_rep', 'genes_list', 'n_distinct_genes', 'unique_to_one_gene']]
    summary_df_upd = pocket_summary.merge(annotated_df, on=['prj', 'rep', 'Global ID', 'gene'], how='outer')
    del summary_df

    merged_residue_df = result_dict['res_to_pock_df']
    merged_residue_df['residue_id'] = merged_residue_df['residue_id'].astype(int)
    merged_residue_df['clean_id'] = merged_residue_df['prj'].str.replace(r'^(apo|holo)', '', regex=True)
    merged_residue_df['gene'] = merged_residue_df['clean_id'].map(SHORT_GENE_NAMES)

    pocket_df_ls = [meta_obj.filter_ortho_site(g_df, MAESTRO_BIND_SITE_STR, min_overlap=0.8)
                    for _, g_df in merged_residue_df.groupby('gene')]
    ortho_pockets = pd.concat(pocket_df_ls)

    bind_cols = [c for c in ortho_pockets.columns if c.startswith('is_binding_site')]
    ortho_pockets = ortho_pockets.loc[ortho_pockets[bind_cols].any(axis=1)]
    ortho_pockets['Local Pocket ID'] = (ortho_pockets['prj'] + '_' + ortho_pockets['rep'] +
                                        '_p' + ortho_pockets['pocket_number'])
    ortho_pockets['is_binding_site_any'] = ortho_pockets[bind_cols].any(axis=1)

    summary_df_upd['is_binding_site'] = summary_df_upd['Local Pocket ID'].map(
        dict(zip(ortho_pockets['Local Pocket ID'], ortho_pockets['is_binding_site_any'])))
    summary_df_upd['pocket_residues'] = summary_df_upd['Local Pocket ID'].map(
        dict(zip(ortho_pockets['Local Pocket ID'], ortho_pockets['residues'])))
    summary_df_upd = summary_df_upd.drop_duplicates(['Local Pocket ID'])
    summary_df_upd.to_csv(out_tbl, index=False)
    print(f"  wrote pocket_comparison_table.csv ({len(summary_df_upd)} pockets)")

    for label, mask, fname in [
            ('unique to a gene',          summary_df_upd['unique_to_one_gene'] == True, 'unique_to_gene_comparison.csv'),
            ('unique to a PDB structure', summary_df_upd['unique_in_one_rep'] == True,  'unique_to_structure_comparison.csv'),
            ('present in all 4 genes',    summary_df_upd['n_distinct_genes'] == 4,      'shared_in_all_gene_comparison.csv')]:
        sub = summary_df_upd.loc[mask]
        sub.to_csv(os.path.join(SAVING_LOC, fname), index=False)
        print(f"  {label}: {sub.shape} | local {sub['Local Pocket ID'].nunique()} "
              f"| global {sub['Global ID'].nunique()}")

    for fn_name in ('upset_plot_pockets_shared', 'barchart_pocket_count'):
        try:
            import comparative_study as cs
            getattr(cs, fn_name)(summary_df_upd, SAVING_LOC)
        except Exception as e:
            print(f"  Error in {fn_name}: {e}")
    print(f"[{STATE}] STAGE 2 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()
    print(f"resuming {STATE} in {SAVING_LOC}")
    if args.stage in ("1", "both"):
        try: stage1()
        except Exception: print(f"STAGE 1 FAILED:\n{traceback.format_exc()}")
    if args.stage in ("2", "both"):
        try: stage2()
        except Exception: print(f"STAGE 2 FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
